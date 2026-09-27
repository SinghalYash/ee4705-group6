"""
Task 3.iv: evaluate the natural-language planner.

Runs every case in planner_test_cases.py through
task_3.planner.plan_from_instruction() and reports the planner's
success rate (Action Planning Accuracy): overall, for the core set,
for the stress set, and per category. Failures are listed with the
reason, so the report shows weaknesses as well as successes.

Independent of the other tasks: each case gives the planner a fixed
scene description (scene_info), so no MuJoCo rendering, no live Task 2
perception and no Task 4 execution is involved. The only external call
is the planner's own LLM.

A case is CORRECT when:
  - it expects moves  -> feasible = True, the plan passes
                         validate_action_plan(), and it consists of one
                         SEARCH, APPROACH, REACH, GRASP, MOVE_TO, PLACE
                         block per expected (object, target) move,
                         with exactly the expected moves (any order);
  - it expects none   -> feasible = False (a STOP with a reason).

Usage (from the project root or anywhere):
    python task_3/evaluation/evaluate_planner.py
    python task_3/evaluation/evaluate_planner.py --trials 3
    python task_3/evaluation/evaluate_planner.py --group core
    python task_3/evaluation/evaluate_planner.py --category negation --category multi_object

Outputs (in task_3/evaluation/results/):
    planner_eval_runs.csv      one row per case per trial
    planner_eval_summary.json  all metrics, plus every plan produced
"""

import argparse
import csv
import json
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

from task_3.planner import MODEL, plan_from_instruction, validate_action_plan
from planner_test_cases import CATEGORY_ORDER, GROUPS, SCENES, TEST_CASES

MOVE_BLOCK = ["SEARCH", "APPROACH", "REACH", "GRASP", "MOVE_TO", "PLACE"]


# ===========================================================================
# SCORING ONE PLAN
# ===========================================================================

def _fmt_moves(moves) -> str:
    if not moves:
        return "infeasible"
    return ", ".join(f"{o} -> {t}" for o, t in moves)


def extract_moves(actions: list):
    """
    Split a feasible plan into (object, target) moves.
    Returns (moves, error): error is set if the plan is not a clean
    sequence of MOVE_BLOCKs that each refer to one object/target.
    """
    skills = [a.get("skill") for a in actions]
    n = len(MOVE_BLOCK)
    if not skills or len(skills) % n or skills != MOVE_BLOCK * (len(skills) // n):
        return None, f"unexpected skill sequence {skills}"

    moves = []
    for i in range(0, len(actions), n):
        block = actions[i:i + n]
        objs = {a.get("target") for a in block[:4]} | {block[5].get("object")}
        tgts = {block[4].get("target"), block[5].get("target")}
        if len(objs) != 1 or len(tgts) != 1:
            return None, f"inconsistent names inside one move: objects {objs}, targets {tgts}"
        moves.append((objs.pop(), tgts.pop()))
    return moves, None


def score_plan(case: dict, plan: dict) -> tuple:
    """
    Returns (outcome, got_moves, detail). Outcomes:
        correct       matches the expected behaviour
        false_reject  a move was expected, planner said infeasible
        wrong_plan    feasible, but wrong object(s)/target(s)/steps
        false_accept  should have been rejected, planner produced a plan
        malformed     plan fails validate_action_plan()
    """
    ok, error = validate_action_plan(plan)
    if not ok:
        return "malformed", None, error

    actions = plan.get("actions", [])
    expected = case["expected"]

    if not plan.get("feasible"):
        reason = actions[0].get("reason", "") if actions else ""
        if expected:
            return "false_reject", [], reason
        return "correct", [], reason

    moves, error = extract_moves(actions)
    if moves is None:
        return ("false_accept" if not expected else "wrong_plan"), None, error

    if not expected:
        return "false_accept", moves, f"planned {_fmt_moves(moves)}"
    if Counter(moves) != Counter(expected):
        return "wrong_plan", moves, f"planned {_fmt_moves(moves)}"
    return "correct", moves, f"planned {_fmt_moves(moves)}"


# ===========================================================================
# RUNNING
# ===========================================================================

def run_case(case: dict) -> dict:
    start = time.perf_counter()
    try:
        plan = plan_from_instruction(case["instruction"], scene_info=SCENES[case["scene"]])
        outcome, got_moves, detail = score_plan(case, plan)
    except Exception as error:  # the planner should never raise; record it if it does
        plan, outcome, got_moves = None, "error", None
        detail = f"{type(error).__name__}: {error}"

    return {
        "id": case["id"],
        "group": case["group"],
        "category": case["category"],
        "scene": case["scene"],
        "instruction": case["instruction"],
        "expected": _fmt_moves(case["expected"]),
        "got": "error" if outcome in {"error", "malformed"} else _fmt_moves(got_moves),
        "outcome": outcome,
        "correct": outcome == "correct",
        "detail": detail,
        "note": case["note"],
        "latency_s": round(time.perf_counter() - start, 3),
        "plan": plan,
    }


def _rate(rows):
    return (sum(r["correct"] for r in rows) / len(rows)) if rows else None


def summarise(runs: list, cases: list, trials: int) -> dict:
    by_category, by_group, by_case = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in runs:
        by_category[r["category"]].append(r)
        by_group[r["group"]].append(r)
        by_case[r["id"]].append(r)

    expects_move = {c["id"]: bool(c["expected"]) for c in cases}
    move_runs = [r for r in runs if expects_move[r["id"]]]
    reject_runs = [r for r in runs if not expects_move[r["id"]]]
    case_group = {c["id"]: c["group"] for c in cases}

    failed_cases = []
    for cid, rows in by_case.items():
        bad = [r for r in rows if not r["correct"]]
        if bad:
            failed_cases.append({
                "id": cid, "group": case_group[cid], "category": rows[0]["category"],
                "instruction": rows[0]["instruction"], "expected": rows[0]["expected"],
                "failed_trials": f"{len(bad)}/{len(rows)}",
                "outcome": bad[0]["outcome"], "got": bad[0]["got"],
                "detail": bad[0]["detail"], "note": rows[0]["note"],
            })

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "num_cases": len(cases),
        "trials_per_case": trials,
        "num_runs": len(runs),
        "action_planning_accuracy": _rate(runs),
        "per_group_accuracy": {
            g: {"accuracy": _rate(by_group[g]), "runs": len(by_group[g]),
                "cases": sum(c["group"] == g for c in cases)}
            for g in GROUPS if by_group[g]
        },
        "requests_needing_a_plan_accuracy": _rate(move_runs),
        "requests_needing_rejection_accuracy": _rate(reject_runs),
        "per_category_accuracy": {
            cat: {"group": by_category[cat][0]["group"], "accuracy": _rate(by_category[cat]),
                  "runs": len(by_category[cat])}
            for cat in CATEGORY_ORDER if by_category[cat]
        },
        "outcome_counts": dict(Counter(r["outcome"] for r in runs)),
        "cases_fully_correct": f"{sum(all(r['correct'] for r in rows) for rows in by_case.values())}/{len(by_case)}",
        "cases_consistent_across_trials": f"{sum(len({r['outcome'] for r in rows}) == 1 for rows in by_case.values())}/{len(by_case)}",
        "mean_latency_s": round(sum(r["latency_s"] for r in runs) / len(runs), 3),
        "failed_cases": failed_cases,
    }


def print_report(runs: list, summary: dict) -> None:
    def pct(x):
        return "   n/a" if x is None else f"{100 * x:5.1f}%"

    def count(rows):
        return f"{sum(r['correct'] for r in rows)}/{len(rows)}"

    print("\n" + "=" * 70)
    print("TASK 3.iv  PLANNER EVALUATION")
    print("=" * 70)
    print(f"Model: {summary['model']}   Cases: {summary['num_cases']}   "
          f"Trials/case: {summary['trials_per_case']}   Runs: {summary['num_runs']}\n")

    print(f"Action Planning Accuracy (all runs):    {pct(summary['action_planning_accuracy'])}  ({count(runs)})")
    for g, s in summary["per_group_accuracy"].items():
        rows = [r for r in runs if r["group"] == g]
        print(f"  {g:<6} set ({s['cases']:>2} cases):              {pct(s['accuracy'])}  ({count(rows)})")
    print(f"  requests that need a plan:            {pct(summary['requests_needing_a_plan_accuracy'])}")
    print(f"  requests that must be rejected:       {pct(summary['requests_needing_rejection_accuracy'])}\n")

    print(f"{'Group':<8}{'Category':<24}{'Accuracy':>9}{'Runs':>7}")
    print("-" * 48)
    for cat, s in summary["per_category_accuracy"].items():
        print(f"{s['group']:<8}{cat:<24}{pct(s['accuracy']):>9}{s['runs']:>7}")

    print("\nOutcomes:", ", ".join(f"{k}={v}" for k, v in sorted(summary["outcome_counts"].items())))
    print("Cases correct in every trial:", summary["cases_fully_correct"])
    print("Cases with the same outcome in every trial:", summary["cases_consistent_across_trials"])
    print(f"Mean planning latency: {summary['mean_latency_s']} s")

    if summary["failed_cases"]:
        print(f"\nFailed cases ({len(summary['failed_cases'])}):")
        for f in summary["failed_cases"]:
            print(f"  [{f['id']}] {f['group']}/{f['category']}  failed {f['failed_trials']}  {f['instruction']!r}")
            print(f"        expected: {f['expected']}")
            print(f"        got:      {f['outcome']} - {f['detail']}")
            if f["note"]:
                print(f"        note:     {f['note']}")


def save_results(runs: list, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "planner_eval_runs.csv"
    fields = ["trial", "id", "group", "category", "scene", "instruction", "expected",
              "got", "outcome", "correct", "detail", "note", "latency_s"]
    # utf-8-sig so Excel shows the non-English instructions correctly
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    json_path = out_dir / "planner_eval_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "runs": runs}, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles + non-English cases
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Task 3.iv planner evaluation")
    parser.add_argument("--trials", type=int, default=1,
                        help="how many times to run each case (LLM output can vary)")
    parser.add_argument("--group", choices=GROUPS, help="only run the core or the stress set")
    parser.add_argument("--category", action="append", choices=CATEGORY_ORDER,
                        help="only run these categories (repeatable)")
    parser.add_argument("--out", type=Path, default=HERE / "results", help="output folder")
    args = parser.parse_args()

    cases = [c for c in TEST_CASES
             if (not args.group or c["group"] == args.group)
             and (not args.category or c["category"] in args.category)]
    if not cases:
        print("No cases match those filters.")
        return

    runs, total = [], len(cases) * args.trials
    for trial in range(1, args.trials + 1):
        for case in cases:
            result = run_case(case)
            result["trial"] = trial
            runs.append(result)
            mark = "OK  " if result["correct"] else "FAIL"
            print(f"[{len(runs):>3}/{total}] [{mark}] {case['id']} {case['instruction']!r}  -> {result['got']}")

    summary = summarise(runs, cases, args.trials)
    print_report(runs, summary)
    save_results(runs, summary, args.out)


if __name__ == "__main__":
    main()
