from agent_research_pipeline.ledger import (
    Ledger,
    LedgerEntry,
    estimate_cost,
    estimate_tokens,
    format_summary,
)


def _entry(stage="plan", attempt=1, status="ok", **kw) -> LedgerEntry:
    base = dict(
        stage=stage, model="mock", input_tokens=100, output_tokens=50,
        latency_ms=10, cost_usd=0.001, attempt=attempt, status=status,
    )
    base.update(kw)
    return LedgerEntry(**base)


def test_estimate_tokens_and_cost():
    assert estimate_tokens("one two three") == 4
    assert estimate_tokens("") == 1
    assert estimate_cost("mock", 1_000_000, 1_000_000) == 0.5
    assert estimate_cost("unknown-model", 0, 0) == 0.0


def test_summary_aggregates_by_stage():
    ledger = Ledger()
    ledger.record(_entry("plan"))
    ledger.record(_entry("research", status="failed", cost_usd=0.0))
    ledger.record(_entry("research", attempt=2))
    summary = ledger.summary()
    assert summary["total_attempts"] == 3
    assert summary["retries"] == 1
    assert summary["failures"] == 1
    assert summary["stages"]["research"]["attempts"] == 2
    assert summary["total_input_tokens"] == 300
    assert summary["total_cost_usd"] == 0.002


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    ledger.record(_entry("plan"))
    ledger.record(_entry("write", attempt=2))
    reloaded = Ledger(path)
    assert len(reloaded.entries()) == 2
    assert reloaded.entries()[1].stage == "write"
    assert reloaded.summary()["total_attempts"] == 2


def test_format_summary_mentions_totals():
    ledger = Ledger()
    ledger.record(_entry("plan"))
    text = format_summary(ledger.summary())
    assert "plan" in text
    assert "TOTAL" in text
    assert "(estimated)" in text
