"""Pipeline orchestration: plan → research → claims → fact-check → write.

Production patterns implemented here, all in deterministic code:

- **Fan-out/fan-in research**: one retrieval worker per sub-question on a
  thread pool; results fan back in submission order. Shared evidence
  state is written under a lock; the corpus/index are read-only.
- **Checkpoint/resume**: after every stage the full pipeline state is
  written to a JSON checkpoint (atomically). ``run(resume=True)`` loads
  it and continues from the first incomplete stage.
- **Per-stage retries**: transient failures retry with exponential
  backoff; every attempt lands in the ledger. ``inject_failure=True``
  makes the research stage fail once so the retry path is provable.
- **Crash simulation**: ``crash_after_stage`` raises after checkpointing
  that stage, so the demo can show a real crash + resume cycle.

In live mode the plan and claims stages call Gemini (see live.py); all
other stages are identical deterministic code.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .claims import Claim, VerifiedClaim, extract_claims, verify_claims
from .corpus import Corpus
from .decomposition import decompose
from .ledger import Ledger, LedgerEntry, estimate_cost, estimate_tokens
from .report import assemble_report
from .retrieval import SearchResult, TfidfIndex

STAGES = ["plan", "research", "claims", "factcheck", "write"]
CHECKPOINT_VERSION = 1


class TransientError(RuntimeError):
    """A retryable stage failure (rate limit, timeout, 5xx)."""


class SimulatedCrash(RuntimeError):
    """Raised by the demo's crash simulation after a checkpoint write."""


@dataclass
class PipelineConfig:
    mode: str = "mock"  # "mock" | "live"
    top_k: int = 3
    keep_unverified: bool = False
    sabotage: bool = True  # mock: deterministically mis-cite two claims
    inject_failure: bool = False  # research stage fails its first attempt
    crash_after_stage: str | None = None
    max_attempts: int = 3
    backoff_s: float = 0.2
    max_workers: int = 4
    checkpoint_path: str | Path | None = "research_checkpoint.json"
    ledger_path: str | Path | None = None


@dataclass
class RunResult:
    question: str
    sub_questions: list[str]
    evidence: dict[str, list[SearchResult]]
    claims: list[Claim]
    verified: list[VerifiedClaim]
    report: str
    ledger_summary: dict
    resumed_from: str | None = None
    completed_stages: list[str] = field(default_factory=list)


def _result_from_dict(data: dict) -> SearchResult:
    return SearchResult(
        doc_id=data["doc_id"],
        title=data["title"],
        snippet=data["snippet"],
        score=float(data["score"]),
    )


class ResearchPipeline:
    """Stateful pipeline; stage methods are also the ADK tools' backend."""

    def __init__(
        self,
        corpus: Corpus | None = None,
        config: PipelineConfig | None = None,
    ):
        self.corpus = corpus or Corpus.load_default()
        self.index = TfidfIndex(self.corpus)
        self.config = config or PipelineConfig()
        self.ledger = Ledger(self.config.ledger_path)
        self._lock = threading.Lock()
        self._fail_once_done = False
        # Pipeline state (checkpointed after each stage).
        self._question: str = ""
        self._sub_questions: list[str] = []
        self._evidence: dict[str, list[SearchResult]] = {}
        self._claims: list[Claim] = []
        self._verified: list[VerifiedClaim] = []
        self._report: str = ""

    # -- stage implementations -------------------------------------------
    # Each returns (result, usage) where usage feeds the ledger.

    def _mock_usage(self, input_text: str, output_text: str, detail: str = "") -> dict:
        return {
            "model": "mock",
            "input_tokens": estimate_tokens(input_text),
            "output_tokens": estimate_tokens(output_text),
            "detail": detail,
        }

    def _do_plan(self) -> tuple[list[str], dict]:
        if self.config.mode == "live":
            from .live import live_decompose, live_model_name

            sub_questions, usage = live_decompose(self._question)
            usage = {"model": live_model_name(), "detail": "live planner", **usage}
            return sub_questions, usage
        sub_questions = decompose(self._question)
        return sub_questions, self._mock_usage(
            self._question, " ".join(sub_questions)
        )

    def _do_research(self) -> tuple[dict[str, list[SearchResult]], dict]:
        if self.config.inject_failure and not self._fail_once_done:
            self._fail_once_done = True
            raise TransientError("injected transient failure (simulated rate limit)")
        evidence: dict[str, list[SearchResult]] = {}

        def worker(sub_question: str) -> None:
            results = self.index.search(sub_question, top_k=self.config.top_k)
            with self._lock:
                evidence[sub_question] = results

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            list(pool.map(worker, self._sub_questions))
        # Fan-in in plan order, regardless of completion order.
        ordered = {sq: evidence.get(sq, []) for sq in self._sub_questions}
        snippets = " ".join(
            r.snippet for results in ordered.values() for r in results
        )
        return ordered, self._mock_usage(" ".join(self._sub_questions), snippets)

    def _do_claims(self) -> tuple[list[Claim], dict]:
        if self.config.mode == "live":
            from .live import live_draft_claims, live_model_name

            claims, usage = live_draft_claims(self._question, self._evidence)
            if claims:
                usage = {"model": live_model_name(), "detail": "live writer", **usage}
                return claims, usage
            fallback = extract_claims(self._evidence, corpus=self.corpus, sabotage=False)
            usage = {
                "model": live_model_name(),
                "detail": "live draft unparseable; fell back to extraction",
                **usage,
            }
            return fallback, usage
        claims = extract_claims(
            self._evidence, corpus=self.corpus, sabotage=self.config.sabotage
        )
        snippets = " ".join(
            r.snippet for results in self._evidence.values() for r in results
        )
        return claims, self._mock_usage(
            snippets, " ".join(c.text for c in claims)
        )

    def _do_factcheck(self) -> tuple[list[VerifiedClaim], dict]:
        verified = verify_claims(self._claims, self.corpus)
        return verified, self._mock_usage(
            " ".join(c.text for c in self._claims),
            " ".join(v.reason for v in verified),
        )

    def _do_write(self) -> tuple[str, dict]:
        report = assemble_report(
            self._question,
            self._sub_questions,
            self._verified,
            self.corpus,
            keep_unverified=self.config.keep_unverified,
        )
        return report, self._mock_usage(
            " ".join(v.claim.text for v in self._verified), report
        )

    _STAGE_FNS = {
        "plan": _do_plan,
        "research": _do_research,
        "claims": _do_claims,
        "factcheck": _do_factcheck,
        "write": _do_write,
    }

    def _apply(self, stage: str, result) -> None:
        if stage == "plan":
            self._sub_questions = result
        elif stage == "research":
            self._evidence = result
        elif stage == "claims":
            self._claims = result
        elif stage == "factcheck":
            self._verified = result
        elif stage == "write":
            self._report = result

    # -- retry wrapper -----------------------------------------------------

    def _run_stage(self, stage: str):
        fn = self._STAGE_FNS[stage]
        attempt = 0
        while True:
            attempt += 1
            start = time.monotonic()
            try:
                result, usage = fn(self)
            except TransientError as exc:
                latency = int((time.monotonic() - start) * 1000)
                self.ledger.record(
                    LedgerEntry(
                        stage=stage,
                        model="mock" if self.config.mode == "mock" else "live",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency,
                        cost_usd=0.0,
                        attempt=attempt,
                        status="failed",
                        detail=str(exc),
                    )
                )
                if attempt >= self.config.max_attempts:
                    raise
                time.sleep(self.config.backoff_s * (2 ** (attempt - 1)))
                continue
            latency = int((time.monotonic() - start) * 1000)
            self.ledger.record(
                LedgerEntry(
                    stage=stage,
                    model=usage["model"],
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    latency_ms=latency,
                    cost_usd=estimate_cost(
                        usage["model"], usage["input_tokens"], usage["output_tokens"]
                    ),
                    attempt=attempt,
                    status="ok",
                    detail=usage.get("detail", ""),
                )
            )
            return result

    # -- checkpointing -----------------------------------------------------

    def _checkpoint_data(self, completed: list[str]) -> dict:
        return {
            "version": CHECKPOINT_VERSION,
            "question": self._question,
            "mode": self.config.mode,
            "completed_stages": completed,
            "state": {
                "sub_questions": self._sub_questions,
                "evidence": {
                    sq: [r.to_dict() for r in results]
                    for sq, results in self._evidence.items()
                },
                "claims": [c.to_dict() for c in self._claims],
                "verified": [v.to_dict() for v in self._verified],
                "report": self._report,
            },
        }

    def _write_checkpoint(self, completed: list[str]) -> None:
        if not self.config.checkpoint_path:
            return
        path = Path(self.config.checkpoint_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._checkpoint_data(completed), indent=2))
        os.replace(tmp, path)  # atomic: a crash never leaves a torn file

    def _load_checkpoint(self) -> list[str]:
        path = Path(self.config.checkpoint_path)  # type: ignore[arg-type]
        data = json.loads(path.read_text())
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {data.get('version')}")
        if data["question"] != self._question:
            raise ValueError(
                "checkpoint is for a different question: "
                f"{data['question']!r} != {self._question!r}"
            )
        state = data["state"]
        self._sub_questions = list(state["sub_questions"])
        self._evidence = {
            sq: [_result_from_dict(r) for r in results]
            for sq, results in state["evidence"].items()
        }
        self._claims = [Claim.from_dict(c) for c in state["claims"]]
        self._verified = [VerifiedClaim.from_dict(v) for v in state["verified"]]
        self._report = state["report"]
        return list(data["completed_stages"])

    # -- public API --------------------------------------------------------

    def run(self, question: str, resume: bool = False) -> RunResult:
        """Run the full pipeline; with ``resume=True`` skip finished stages."""
        self._question = question
        completed: list[str] = []
        resumed_from: str | None = None
        if resume:
            if not self.config.checkpoint_path or not Path(
                self.config.checkpoint_path
            ).exists():
                raise FileNotFoundError(
                    f"no checkpoint to resume from: {self.config.checkpoint_path}"
                )
            completed = self._load_checkpoint()
            remaining = [s for s in STAGES if s not in completed]
            resumed_from = remaining[0] if remaining else None
        for stage in STAGES:
            if stage in completed:
                continue
            result = self._run_stage(stage)
            self._apply(stage, result)
            completed.append(stage)
            self._write_checkpoint(completed)
            if self.config.crash_after_stage == stage:
                raise SimulatedCrash(
                    f"simulated crash after stage '{stage}' "
                    f"(checkpoint written: {self.config.checkpoint_path})"
                )
        return RunResult(
            question=self._question,
            sub_questions=list(self._sub_questions),
            evidence=dict(self._evidence),
            claims=list(self._claims),
            verified=list(self._verified),
            report=self._report,
            ledger_summary=self.ledger.summary(),
            resumed_from=resumed_from,
            completed_stages=list(completed),
        )

    # Stage-level entry points used by the ADK tools (stateful, sequential).

    def plan(self, question: str) -> list[str]:
        self._question = question
        result = self._run_stage("plan")
        self._apply("plan", result)
        return list(self._sub_questions)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        results = self.index.search(query, top_k=top_k)
        with self._lock:
            self._evidence[query] = results
        return results

    def extract_and_verify(self) -> list[VerifiedClaim]:
        # Top up evidence for sub-questions no researcher covered (the ADK
        # composition has a fixed researcher pool; the plan may be longer).
        for sq in self._sub_questions:
            if sq not in self._evidence:
                self.search(sq, top_k=self.config.top_k)
        self._apply("claims", self._run_stage("claims"))
        self._apply("factcheck", self._run_stage("factcheck"))
        return list(self._verified)

    def assemble(self) -> str:
        self._apply("write", self._run_stage("write"))
        return self._report
