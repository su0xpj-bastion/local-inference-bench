"""python -m localbench run --target mock|live [--base-url URL] --model M [--model-b M2] [--profile P]"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .checks import Ctx, run_all
from .client import make_client

ROOT = Path(__file__).resolve().parent.parent
MOCK_MODEL, MOCK_MODEL_B = "gemma-mock-1b-q8", "gemma-mock-1b-q4"
ICON = {"pass": "✅", "fail": "❌", "warn": "⚠️", "skip": "⏭️", "info": "ℹ️"}


def host_info() -> dict:
    info = {"os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
            "python": platform.python_version()}
    if sys.platform == "darwin":
        try:
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                         text=True, timeout=5).stdout.strip()
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout
            info["ram_gb"] = round(int(mem) / 2**30)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return info


def render(results, meta: dict) -> str:
    lines = [f"# local-inference-bench: {meta['target']} · {meta['model']}"
             + (f" vs {meta['model_b']}" if meta.get("model_b") else ""), "",
             f"Engine `{meta['base_url']}` · profile `{meta['profile']}` · {meta['requests']} requests · "
             f"{meta['seconds']:.1f} s · host {meta['host'].get('cpu', meta['host']['machine'])}"
             + (f", {meta['host']['ram_gb']} GB" if "ram_gb" in meta["host"] else ""), "",
             "| | Area | Check | Result |", "|---|---|---|---|"]
    for r in results:
        gated = "" if r.gated else " *(not gated)*"
        lines.append(f"| {ICON[r.status]} | {r.area} | {r.name} | {r.detail}{gated} |")
    lines += ["", f"**Verdict: {meta['verdict']}**", ""]
    return "\n".join(lines)


def run(args) -> int:
    slo = json.loads((ROOT / "slo.json").read_text())[args.profile]
    server = None
    if args.target == "mock":
        from mockserve.server import serve
        server, _ = serve(0, {f for f in args.faults.split(",") if f})
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        model, model_b = args.model or MOCK_MODEL, args.model_b or MOCK_MODEL_B
    else:
        base_url, model, model_b = args.base_url, args.model, args.model_b
        if not model:
            print("[ERROR] --model is required for --target live", file=sys.stderr)
            return 2
    client = make_client(base_url, os.environ.get("LOCAL_API_KEY", "local"))
    ctx = Ctx(client, model, model_b, slo)
    start = time.perf_counter()
    results = run_all(ctx, set(args.only.split(",")) if args.only else None)
    seconds = time.perf_counter() - start
    if server:
        server.shutdown()
    failed = [r for r in results if r.status == "fail" and r.gated]
    meta = {"target": args.target, "base_url": base_url, "model": model, "model_b": model_b,
            "profile": args.profile, "faults": args.faults, "requests": ctx.calls, "seconds": seconds,
            "host": host_info(), "verdict": "FAIL" if failed else "PASS"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render(results, meta)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"meta": meta, "checks": [asdict(r) for r in results]},
                                                indent=2, default=str), encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    print(md)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="localbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the checks")
    r.add_argument("--target", choices=["mock", "live"], default="mock")
    r.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="OpenAI-compatible base URL")
    r.add_argument("--model")
    r.add_argument("--model-b", help="a second quantization of the same model, for quant_delta")
    r.add_argument("--profile", default=None, help="slo.json profile (default: mock or ollama-m1-8gb)")
    r.add_argument("--faults", default="", help="mock only: comma-separated faults")
    r.add_argument("--only", default="", help="comma-separated check names")
    r.add_argument("--out", default="reports/latest")
    args = ap.parse_args()
    args.profile = args.profile or ("mock" if args.target == "mock" else "ollama-m1-8gb")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
