from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import openai
import pytest

from localbench.checks import pct, text_of
from localbench.client import Call, chat, make_client
from localbench.quality import ANSWERS, ITEMS, correct, normalize
from mockserve.server import _truncate, ntokens, serve

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def engine():
    server, eng = serve(0, set())
    yield make_client(f"http://127.0.0.1:{server.server_port}/v1"), eng
    server.shutdown()


# ---- grading ----

@pytest.mark.parametrize("got,want,ok", [
    ("Tokyo", "Tokyo", True), ("tokyo.", "Tokyo", True), ("The answer is 43", "43", True),
    ("It's Tokyo.", "Tokyo", True), ("**72**", "72", True), ("430", "43", False),
    ("Kyoto", "Tokyo", False), ("Tokyo is the capital, but Osaka and Kyoto are also big cities in Japan", "Tokyo", False),
])
def test_correct(got, want, ok):
    assert correct(got, want) is ok


def test_quality_set_is_unique_and_answers_are_short():
    assert len(ITEMS) == len(ANSWERS) == 40
    assert all(len(normalize(a)) <= 16 for _, a in ITEMS)


# ---- helpers ----

def test_pct_and_filler():
    assert pct([1, 2, 3, 4, 5], 50) == 3 and pct([1, 2, 3, 4, 5], 95) == 5
    a, b = text_of(500), text_of(500)
    assert a != b and abs(ntokens(a) - 500) < 60          # nonce makes every prompt unique


def test_decode_tps_uses_usage_and_ignores_first_token():
    c = Call(ttft=0.1, wall=1.1, completion_tokens=101, chunks=50)
    assert c.decode_tps == pytest.approx(100.0)
    assert Call(ttft=0.1, wall=0.1, chunks=1).decode_tps is None


def test_truncate_keeps_the_end():
    msgs = [{"role": "user", "content": "A" * 4000 + "END"}]
    out = _truncate(msgs, 100)
    assert out[0]["content"].endswith("END") and ntokens(out[0]["content"]) <= 100


# ---- client against the mock ----

def test_stream_timing_and_usage(engine):
    client, _ = engine
    c = chat(client, "gemma-mock-1b-q8", "hello", max_tokens=10)
    assert c.usage_seen and c.completion_tokens == c.chunks == 10
    assert c.ttft is not None and len(c.gaps) == 9 and c.wall >= c.ttft
    assert c.finish_reason == "length"


def test_unknown_model_raises_not_found(engine):
    client, _ = engine
    with pytest.raises(openai.NotFoundError):
        chat(client, "no-such-model", "hi")


def test_prompt_over_context_window_is_a_400(engine):
    client, _ = engine
    with pytest.raises(openai.BadRequestError):
        chat(client, "gemma-mock-1b-q8", "x " * 40000)


def test_models_endpoint(engine):
    client, _ = engine
    assert {m.id for m in client.models.list()} == {"gemma-mock-1b-q8", "gemma-mock-1b-q4"}


# ---- CLI ----

def test_cli_writes_reports_and_exit_code(tmp_path):
    out = tmp_path / "r"
    ok = subprocess.run([sys.executable, "-m", "localbench", "run", "--target", "mock", "--only",
                         "usage_reported,max_tokens_honoured", "--out", str(out)], cwd=ROOT, capture_output=True)
    assert ok.returncode == 0
    report = json.loads((out / "report.json").read_text())
    assert report["meta"]["verdict"] == "PASS" and len(report["checks"]) == 2
    bad = subprocess.run([sys.executable, "-m", "localbench", "run", "--target", "mock", "--faults",
                          "ignore_max_tokens", "--only", "max_tokens_honoured", "--out", str(out)],
                         cwd=ROOT, capture_output=True)
    assert bad.returncode == 1 and "FAIL" in (out / "report.md").read_text()


def test_live_requires_model(tmp_path):
    r = subprocess.run([sys.executable, "-m", "localbench", "run", "--target", "live", "--out", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "--model is required" in r.stderr
