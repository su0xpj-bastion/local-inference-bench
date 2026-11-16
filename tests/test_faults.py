"""Every check has a fault it must catch, and a healthy engine must pass everything.

Each fault runs the full suite against the mock engine and pins the exact set of checks
that fail, so a detector that silently stops working (or starts firing on the wrong
fault) breaks the build.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from localbench.checks import CHECKS, Ctx, run_all
from localbench.client import make_client
from mockserve.server import serve

SLO = json.loads((Path(__file__).resolve().parent.parent / "slo.json").read_text())["mock"]

FAULT_CATCH = {
    "slow_prefill": {"prefill_scaling"},
    "decode_throttle": {"decode_sustained", "decode_rate"},
    "no_prefix_cache": {"prefix_cache"},
    "serialized": {"concurrency"},
    "quant_degraded": {"quant_delta"},
    "nondeterministic": {"determinism"},
    "context_truncation": {"context_integrity"},
    "keep_alive_zero": {"warm_ttft", "prefix_cache"},   # an unloaded model also loses its prefix cache
    "usage_missing": {"usage_reported"},
    "ignore_max_tokens": {"max_tokens_honoured"},
}


def run(faults: set[str]):
    server, _ = serve(0, faults)
    try:
        ctx = Ctx(make_client(f"http://127.0.0.1:{server.server_port}/v1"),
                  "gemma-mock-1b-q8", "gemma-mock-1b-q4", SLO)
        return run_all(ctx)
    finally:
        server.shutdown()


def test_healthy_engine_passes_every_check():
    results = run(set())
    assert {r.name: r.status for r in results} == {fn.__name__: "pass" for fn in CHECKS}


@pytest.mark.parametrize("fault", sorted(FAULT_CATCH))
def test_fault_is_caught_by_exactly_its_checks(fault):
    failed = {r.name for r in run({fault}) if r.status == "fail"}
    assert failed == FAULT_CATCH[fault]


def test_every_check_has_a_fault_that_catches_it():
    caught = set().union(*FAULT_CATCH.values())
    assert caught == {fn.__name__ for fn in CHECKS}


def test_quant_delta_reports_how_many_items_flipped():
    r = next(r for r in run({"quant_degraded"}) if r.name == "quant_delta")
    assert r.metrics["flipped"] >= 5 and r.metrics["drop_pts"] > SLO["max_quant_drop_pts"]


def test_ungated_check_warns_instead_of_failing():
    slo = {**SLO, "ungated": ["concurrency"]}
    server, _ = serve(0, {"serialized"})
    try:
        ctx = Ctx(make_client(f"http://127.0.0.1:{server.server_port}/v1"), "gemma-mock-1b-q8", None, slo)
        r = run_all(ctx, {"concurrency"})[0]
    finally:
        server.shutdown()
    assert (r.status, r.gated) == ("warn", False)


def test_quant_delta_skips_without_second_model():
    server, _ = serve(0, set())
    try:
        ctx = Ctx(make_client(f"http://127.0.0.1:{server.server_port}/v1"), "gemma-mock-1b-q8", None, SLO)
        r = run_all(ctx, {"quant_delta"})[0]
    finally:
        server.shutdown()
    assert r.status == "skip"
