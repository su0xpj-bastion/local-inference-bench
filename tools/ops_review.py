"""Ops review: incidents, action items, SLO status and live-run evidence in one page.

    python tools/ops_review.py                 # verify + print the review
    python tools/ops_review.py --out docs/ops/REVIEW.md

A closed action item must name a regression test (`closed_by`). This script runs those tests;
if one is missing or failing, the action is reported as REOPENED and the exit code is 1.
Open actions need an owner and a due date; past-due ones are flagged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / "docs" / "ops"


def run_tests(node_ids: list[str]) -> dict[str, str]:
    """Run the given pytest node ids once; return {node_id: passed|failed|missing}."""
    if not node_ids:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        f"--junitxml={xml}", *node_ids], cwd=ROOT, capture_output=True, text=True)
        seen: dict[str, str] = {}
        if xml.exists():
            for case in ET.parse(xml).iter("testcase"):
                name = case.get("name", "")
                bad = case.find("failure") is not None or case.find("error") is not None
                seen[name] = "failed" if bad else "passed"
    return {n: seen.get(n.split("::")[-1], "missing") for n in node_ids}


def live_runs() -> list[dict]:
    rows = []
    for f in sorted(ROOT.glob("results/*/report.json")):
        d = json.loads(f.read_text())
        ci = next((c for c in d["checks"] if c["name"] == "context_integrity"), None)
        rows.append({"run": f.parent.name, "model": d["meta"]["model"],
                     "verdict": d["meta"]["verdict"], "context": ci["status"] if ci else "n/a"})
    return rows


def review(today: dt.date) -> tuple[str, list[str]]:
    data = json.loads((OPS / "actions.json").read_text())
    problems: list[str] = []

    for inc in data["incidents"]:
        if not (OPS / inc["postmortem"]).exists():
            problems.append(f"{inc['id']}: post-mortem {inc['postmortem']} is missing")

    closed = [a for a in data["actions"] if a["status"] == "closed"]
    for a in closed:
        if not a.get("closed_by"):
            problems.append(f"{a['id']}: closed without a regression test")
    results = run_tests([a["closed_by"] for a in closed if a.get("closed_by")])

    lines = [f"# Ops review ({today.isoformat()})", ""]
    lines += ["## Incidents", "", "| ID | Severity | Title | Open actions | Post-mortem |", "|---|---|---|---|---|"]
    for inc in data["incidents"]:
        n_open = sum(1 for a in data["actions"] if a["incident"] == inc["id"] and a["status"] != "closed")
        lines.append(f"| {inc['id']} | {inc['severity']} | {inc['title']} | {n_open} | [{inc['postmortem']}]({inc['postmortem']}) |")

    lines += ["", "## Action items", "", "| ID | Incident | Type | Action | State | Proof |", "|---|---|---|---|---|---|"]
    for a in data["actions"]:
        if a["status"] == "closed":
            res = results.get(a.get("closed_by", ""), "missing")
            state = "closed" if res == "passed" else "**REOPENED**"
            if res != "passed":
                problems.append(f"{a['id']}: closed, but its test is {res}: {a.get('closed_by')}")
            proof = f"`{a.get('closed_by', '').split('::')[-1]}` {res}"
        else:
            missing = [k for k in ("owner", "due") if not a.get(k)]
            if missing:
                problems.append(f"{a['id']}: open without {', '.join(missing)}")
            due = dt.date.fromisoformat(a["due"]) if a.get("due") else None
            overdue = due is not None and due < today
            state = f"open, due {a.get('due', '?')}" + (" **OVERDUE**" if overdue else "")
            proof = a.get("note", "")
        lines.append(f"| {a['id']} | {a['incident']} | {a['type']} | {a['title']} | {state} | {proof} |")

    runs = live_runs()
    lines += ["", "## Live runs (evidence)", "", "| Run | Model | Verdict | context_integrity |", "|---|---|---|---|"]
    lines += [f"| {r['run']} | {r['model']} | {r['verdict']} | {r['context']} |" for r in runs]
    fixed = [r for r in runs if "ctx16k" in r["run"]]
    lines += ["", f"With the documented config (`OLLAMA_CONTEXT_LENGTH=16384`): "
              f"{sum(r['context'] == 'pass' for r in fixed)}/{len(fixed)} live runs pass context_integrity. "
              f"On defaults: {sum(r['context'] == 'fail' for r in runs if 'default-ctx' in r['run'])} runs caught truncation."]

    n_closed = sum(1 for a in closed if results.get(a.get("closed_by", "")) == "passed")
    lines += ["", "## Summary", "",
              f"- Actions: {n_closed} closed and proven, {len(data['actions']) - len(closed)} open, "
              f"{len(closed) - n_closed} reopened",
              "- Problems: " + ("none" if not problems else "; ".join(problems))]
    return "\n".join(lines) + "\n", problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the review markdown here")
    ap.add_argument("--today", help="override today's date (YYYY-MM-DD), for reproducible reviews")
    args = ap.parse_args()
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    text, problems = review(today)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
