"""
Task 3.iv: before/after comparison of two planner versions.

    V4  "resolution only, strict validation" (archived in baselines/planner_v4.py)
        The LLM maps the perceived scene to canonical names by itself and
        returns one object + one target; any non-canonical name is rejected.
    V5  current planner (task_3/planner.py)
        Deterministic colour grounding -> LLM language understanding ->
        deterministic verification -> plan assembly.

Both versions run on the SAME 35 instructions (the original Task 3.iv
test set, taken from planner_test_cases.py) against the SAME fixed scene
descriptions, and are scored with the same rule as evaluate_planner.py.
The larger stress set is not used here, to keep the API cost down.

Token usage and API latency are measured by wrapping each version's API
client inside this script only: neither planner file is modified.

Usage (from the project root or anywhere):
    python task_3/evaluation/compare_planner_versions.py
    python task_3/evaluation/compare_planner_versions.py --trials 3
    python task_3/evaluation/compare_planner_versions.py --price-in 0.05 --price-out 0.4
        (optional: USD per million input / output tokens, to estimate cost)

Cost: about 70 LLM calls per trial (35 cases x 2 versions, plus any retries).

Outputs (in task_3/evaluation/results/):
    version_comparison_runs.csv      one row per case per version per trial
    version_comparison_summary.json  all metrics
"""

import argparse
import csv
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
for path in (PROJECT_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import task_3.planner as planner_v5
from evaluate_planner import score_plan, _fmt_moves
from planner_test_cases import SCENES, TEST_CASES


def _load_v4():
    spec = importlib.util.spec_from_file_location("planner_v4", HERE / "baselines" / "planner_v4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The original 35-instruction Task 3.iv test set (all core cases).
COMPARISON_IDS = [
    "S01", "S02", "S03", "S04", "S05",
    "P01", "P02", "P03", "P04", "P05", "P06",
    "C01", "C02", "C03", "C04", "C05",
    "A01", "A02", "A03",
    "U01", "U02", "U03", "U04",
    "D01", "D02", "D03", "D04",
    "N01", "N02",
    "M01", "M02", "M03", "M04",
    "V01", "V02",
]
VALID_CATS = ["standard", "paraphrase", "colour_wording", "alt_scene"]
INVALID_CATS = ["unknown_object", "bad_destination", "no_destination", "not_manipulation", "not_in_scene"]


# ===========================================================================
# TOKEN / LATENCY RECORDING (wraps the client object in this script only)
# ===========================================================================

class UsageRecorder:
    def __init__(self):
        self.calls = []

    def wrap(self, client):
        completions = client.chat.completions
        original = completions.create

        def create(*args, **kwargs):
            start = time.perf_counter()
            try:
                response = original(*args, **kwargs)
            except Exception:
                self.calls.append({"prompt_tokens": None, "completion_tokens": None,
                                   "latency_s": time.perf_counter() - start, "ok": False})
                raise
            usage = getattr(response, "usage", None)
            self.calls.append({
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_s": time.perf_counter() - start,
                "ok": True,
            })
            return response

        completions.create = create


def _sum(values):
    values = [v for v in values if v is not None]
    return sum(values) if values else None


# ===========================================================================
# RUNNING
# ===========================================================================

def run_one(version: str, plan_fn, recorder: UsageRecorder, case: dict) -> dict:
    first_call = len(recorder.calls)
    start = time.perf_counter()
    try:
        plan = plan_fn(case["instruction"], scene_info=SCENES[case["scene"]])
        outcome, got_moves, detail = score_plan(case, plan)
    except Exception as error:  # V4 lets API errors escape; record, don't stop
        plan, outcome, got_moves = None, "error", None
        detail = f"{type(error).__name__}: {error}"
    latency = time.perf_counter() - start
    calls = recorder.calls[first_call:]

    return {
        "version": version,
        "id": case["id"],
        "category": case["category"],
        "instruction": case["instruction"],
        "expected": _fmt_moves(case["expected"]),
        "got": "error" if outcome in {"error", "malformed"} else _fmt_moves(got_moves),
        "outcome": outcome,
        "correct": outcome == "correct",
        "detail": detail,
        "llm_calls": len(calls),
        "prompt_tokens": _sum(c["prompt_tokens"] for c in calls),
        "completion_tokens": _sum(c["completion_tokens"] for c in calls),
        "latency_s": round(latency, 3),
        "plan": plan,
    }


def summarise(rows: list) -> dict:
    def rate(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    prompt = [r["prompt_tokens"] for r in rows if r["prompt_tokens"] is not None]
    completion = [r["completion_tokens"] for r in rows if r["completion_tokens"] is not None]
    latencies = sorted(r["latency_s"] for r in rows)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]

    return {
        "runs": len(rows),
        "action_planning_accuracy": rate(rows),
        "valid_request_accuracy": rate([r for r in rows if r["category"] in VALID_CATS]),
        "invalid_request_rejection_rate": rate([r for r in rows if r["category"] in INVALID_CATS]),
        "per_category_accuracy": {c: rate(by_cat[c]) for c in VALID_CATS + INVALID_CATS if by_cat[c]},
        "outcome_counts": dict(Counter(r["outcome"] for r in rows)),
        "mean_llm_calls_per_instruction": round(statistics.mean(r["llm_calls"] for r in rows), 3),
        "mean_prompt_tokens_per_instruction": round(statistics.mean(prompt), 1) if prompt else None,
        "mean_completion_tokens_per_instruction": round(statistics.mean(completion), 1) if completion else None,
        "total_prompt_tokens": sum(prompt) if prompt else None,
        "total_completion_tokens": sum(completion) if completion else None,
        "mean_latency_s": round(statistics.mean(latencies), 3),
        "median_latency_s": round(statistics.median(latencies), 3),
        "p95_latency_s": round(p95, 3),
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Compare planner V4 and V5 on the 35-case set")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--price-in", type=float, help="USD per million input tokens (optional)")
    parser.add_argument("--price-out", type=float, help="USD per million output tokens (optional)")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()

    by_id = {c["id"]: c for c in TEST_CASES}
    cases = [by_id[i] for i in COMPARISON_IDS]

    planner_v4 = _load_v4()
    rec_v4, rec_v5 = UsageRecorder(), UsageRecorder()
    rec_v4.wrap(planner_v4.client)
    rec_v5.wrap(planner_v5._get_client())

    versions = [
        ("V4", planner_v4.plan_from_instruction, rec_v4),
        ("V5", planner_v5.plan_from_instruction, rec_v5),
    ]

    rows, total = [], len(cases) * args.trials * len(versions)
    for trial in range(1, args.trials + 1):
        for case in cases:
            for name, fn, rec in versions:
                row = run_one(name, fn, rec, case)
                row["trial"] = trial
                rows.append(row)
                mark = "OK  " if row["correct"] else "FAIL"
                print(f"[{len(rows):>3}/{total}] {name} [{mark}] {case['id']} "
                      f"{case['instruction']!r} -> {row['got']}")

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": planner_v5.MODEL,
        "cases": len(cases),
        "trials_per_case": args.trials,
        "V4": summarise([r for r in rows if r["version"] == "V4"]),
        "V5": summarise([r for r in rows if r["version"] == "V5"]),
    }
    if args.price_in is not None and args.price_out is not None:
        for v in ("V4", "V5"):
            s = summary[v]
            if s["total_prompt_tokens"] is not None:
                cost = (s["total_prompt_tokens"] * args.price_in
                        + s["total_completion_tokens"] * args.price_out) / 1e6
                s["estimated_cost_usd"] = round(cost, 6)
                s["estimated_cost_per_instruction_usd"] = round(cost / s["runs"], 8)

    # Cases whose outcome differs between the versions (first trial shown)
    changed = []
    for case in cases:
        v4 = [r for r in rows if r["version"] == "V4" and r["id"] == case["id"]]
        v5 = [r for r in rows if r["version"] == "V5" and r["id"] == case["id"]]
        if [r["correct"] for r in v4] != [r["correct"] for r in v5]:
            changed.append({
                "id": case["id"], "instruction": case["instruction"], "expected": v4[0]["expected"],
                "V4_correct": f"{sum(r['correct'] for r in v4)}/{len(v4)}",
                "V4_first": f"{v4[0]['outcome']}: {v4[0]['detail']}",
                "V5_correct": f"{sum(r['correct'] for r in v5)}/{len(v5)}",
                "V5_first": f"{v5[0]['outcome']}: {v5[0]['detail']}",
            })
    summary["cases_with_different_results"] = changed

    # ---- report --------------------------------------------------------------
    def pct(x):
        return "   n/a" if x is None else f"{100 * x:5.1f}%"

    s4, s5 = summary["V4"], summary["V5"]
    print("\n" + "=" * 66)
    print("TASK 3.iv  BEFORE / AFTER: planner V4 vs V5")
    print("=" * 66)
    print(f"Model: {summary['model']}   Cases: {len(cases)}   Trials/case: {args.trials}\n")
    print(f"{'Metric':<40}{'V4':>12}{'V5':>12}")
    print("-" * 64)
    for label, key in [("Action Planning Accuracy", "action_planning_accuracy"),
                       ("  valid requests planned correctly", "valid_request_accuracy"),
                       ("  invalid requests rejected", "invalid_request_rejection_rate")]:
        print(f"{label:<40}{pct(s4[key]):>12}{pct(s5[key]):>12}")
    for cat in VALID_CATS + INVALID_CATS:
        print(f"{'  ' + cat:<40}{pct(s4['per_category_accuracy'].get(cat)):>12}"
              f"{pct(s5['per_category_accuracy'].get(cat)):>12}")
    print("-" * 64)
    for label, key in [("LLM calls per instruction", "mean_llm_calls_per_instruction"),
                       ("Prompt tokens per instruction", "mean_prompt_tokens_per_instruction"),
                       ("Completion tokens per instruction", "mean_completion_tokens_per_instruction"),
                       ("Total prompt tokens", "total_prompt_tokens"),
                       ("Total completion tokens", "total_completion_tokens"),
                       ("Mean latency (s)", "mean_latency_s"),
                       ("Median latency (s)", "median_latency_s"),
                       ("95th percentile latency (s)", "p95_latency_s"),
                       ("Estimated cost (USD)", "estimated_cost_usd")]:
        print(f"{label:<40}{str(s4.get(key, '-')):>12}{str(s5.get(key, '-')):>12}")
    print("\nOutcomes V4:", s4["outcome_counts"])
    print("Outcomes V5:", s5["outcome_counts"])

    if changed:
        print(f"\nCases with different results ({len(changed)}):")
        for c in changed:
            print(f"  [{c['id']}] {c['instruction']!r}  expected {c['expected']}")
            print(f"        V4 {c['V4_correct']}  {c['V4_first']}")
            print(f"        V5 {c['V5_correct']}  {c['V5_first']}")

    # ---- save ----------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "version_comparison_runs.csv"
    fields = ["trial", "version", "id", "category", "instruction", "expected", "got", "outcome",
              "correct", "detail", "llm_calls", "prompt_tokens", "completion_tokens", "latency_s"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out / "version_comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {csv_path}\nSaved: {json_path}")


if __name__ == "__main__":
    main()
