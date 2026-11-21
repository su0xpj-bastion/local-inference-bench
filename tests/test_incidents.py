"""Regression tests that close the action items in docs/ops/actions.json.

Each test is named by an action item's `closed_by` field. tools/ops_review.py refuses to
mark an action closed unless its test exists and passes, so a fix that is later undone
reopens the incident instead of staying "done" on paper.
"""
from __future__ import annotations

import json
from pathlib import Path

from localbench.checks import Ctx, run_all
from localbench.client import make_client
from mockserve.server import serve

SLO = json.loads((Path(__file__).resolve().parent.parent / "slo.json").read_text())["mock"]


def context_check(faults: set[str]):
    server, _ = serve(0, faults)
    try:
        ctx = Ctx(make_client(f"http://127.0.0.1:{server.server_port}/v1"), "gemma-mock-1b-q8", None, SLO)
        return run_all(ctx, {"context_integrity"})[0]
    finally:
        server.shutdown()


# INC-001: silent front truncation must surface as a failure, never as a quiet 200.
def test_inc001_silent_truncation_is_caught():
    r = context_check({"context_truncation"})
    assert r.status == "fail"
    assert r.metrics["recalled"] is False
    assert r.metrics["prompt_tokens"] < SLO["min_prompt_token_ratio"] * r.metrics["expected"]


def test_inc001_truncation_caught_with_dense_tokenizer_too():
    r = context_check({"context_truncation", "dense_tokenizer"})
    assert r.status == "fail"


# INC-002: a correct prompt must not raise a truncation alarm because the tokenizer is denser
# than the harness assumed.
def test_inc002_dense_tokenizer_does_not_false_alarm():
    r = context_check({"dense_tokenizer"})
    assert r.status == "pass", r.detail
    assert abs(r.metrics["prompt_tokens"] / r.metrics["expected"] - 1) < 0.05


def test_inc002_fixed_estimate_would_have_false_alarmed():
    """Reproduces the original bug: 4 chars/token would have called this healthy prompt cut."""
    r = context_check({"dense_tokenizer"})
    naive_expected = r.metrics["prompt_chars"] / 4
    assert r.metrics["prompt_tokens"] < SLO["min_prompt_token_ratio"] * naive_expected


# INC-001: the live profile must probe past Ollama's default context, or the check can't see the bug.
def test_inc001_live_probe_exceeds_ollama_default_context():
    live = json.loads((Path(__file__).resolve().parent.parent / "slo.json").read_text())["ollama-m1-8gb"]
    assert live["context_probe_tokens"] > 4096


# INC-001: the runbook must carry the fix an operator needs, and how to verify it.
def test_inc001_runbook_has_fix_and_verification():
    text = (Path(__file__).resolve().parent.parent / "docs/ops/RUNBOOK.md").read_text()
    assert "OLLAMA_CONTEXT_LENGTH" in text and "--only context_integrity" in text
