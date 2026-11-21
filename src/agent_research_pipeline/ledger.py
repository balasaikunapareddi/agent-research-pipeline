"""Per-stage token/cost/latency ledger (JSONL).

Every stage attempt — including failed ones that triggered a retry — is
recorded: stage, model, tokens, latency, estimated cost, attempt number,
and status. In mock mode token counts are estimates (~1.33 tokens/word,
the house convention); in live mode they come from the API's usage
metadata where reported. Costs apply a small price table to the token
counts and are always labeled estimates.

Run ``python -m agent_research_pipeline.ledger <ledger.jsonl>`` to print
a run summary for a ledger file.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TOKENS_PER_WORD = 1.33
# USD per 1M tokens (input, output) — flash-lite-class pricing, estimates.
PRICE_TABLE = {
    "mock": (0.10, 0.40),
    "gemini-3.5-flash-lite": (0.10, 0.40),
}
DEFAULT_PRICE = (0.10, 0.40)


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * TOKENS_PER_WORD))


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICE_TABLE.get(model, DEFAULT_PRICE)
    return round(
        (input_tokens * price_in + output_tokens * price_out) / 1_000_000, 6
    )


@dataclass
class LedgerEntry:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    attempt: int = 1
    status: str = "ok"  # ok | failed
    detail: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)


class Ledger:
    """JSONL-backed ledger. ``path=None`` keeps entries in memory only."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._entries: list[LedgerEntry] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._entries.append(LedgerEntry(**json.loads(line)))

    def record(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(entry.to_dict()) + "\n")

    def entries(self) -> list[LedgerEntry]:
        return list(self._entries)

    def summary(self) -> dict:
        entries = self._entries
        by_stage: dict[str, dict] = {}
        for e in entries:
            s = by_stage.setdefault(
                e.stage,
                {"attempts": 0, "input_tokens": 0, "output_tokens": 0,
                 "latency_ms": 0, "cost_usd": 0.0},
            )
            s["attempts"] += 1
            s["input_tokens"] += e.input_tokens
            s["output_tokens"] += e.output_tokens
            s["latency_ms"] += e.latency_ms
            s["cost_usd"] = round(s["cost_usd"] + e.cost_usd, 6)
        return {
            "stages": by_stage,
            "total_attempts": len(entries),
            "retries": sum(1 for e in entries if e.attempt > 1),
            "failures": sum(1 for e in entries if e.status == "failed"),
            "total_input_tokens": sum(e.input_tokens for e in entries),
            "total_output_tokens": sum(e.output_tokens for e in entries),
            "total_latency_ms": sum(e.latency_ms for e in entries),
            "total_cost_usd": round(sum(e.cost_usd for e in entries), 6),
        }


def format_summary(summary: dict) -> str:
    lines = ["Run ledger (per stage):"]
    for stage, s in summary["stages"].items():
        lines.append(
            f"  {stage:<10} attempts={s['attempts']} "
            f"in={s['input_tokens']} tok out={s['output_tokens']} tok "
            f"latency={s['latency_ms']} ms cost=${s['cost_usd']:.6f}"
        )
    lines.append(
        f"  TOTAL      attempts={summary['total_attempts']} "
        f"retries={summary['retries']} failures={summary['failures']} "
        f"in={summary['total_input_tokens']} tok "
        f"out={summary['total_output_tokens']} tok "
        f"latency={summary['total_latency_ms']} ms "
        f"cost=${summary['total_cost_usd']:.6f} (estimated)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: research-pipeline-ledger <ledger.jsonl>", file=sys.stderr)
        return 2
    ledger = Ledger(argv[0])
    print(format_summary(ledger.summary()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
