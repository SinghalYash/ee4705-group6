"""
Task 3.iv: evaluate the planner on >=20 natural-language instructions,
including paraphrases of the same task and invalid/infeasible requests.
Reports Action Planning Accuracy and saves full results for the report.
"""

import json
from planner import plan_from_instruction

# Each entry: (instruction, expected_feasible)
# "Feasible" means the objects/regions mentioned exist in SCENE_MOCK
# (stone, box, cylinder, red_area). Extend this list once Task 2's real
# grounding is integrated and more objects/regions exist.
TEST_CASES = [
    # --- Valid instructions: 5 standard phrasings ---
    ("Pick up the stone and place it in the red area.", True),
    ("Move the box to the red area.", True),
    ("Take the cylinder and place it in the red area.", True),
    ("Grab the stone and put it in the red zone.", True),
    ("Place the box into the red area.", True),

    # --- Valid instructions: 5 paraphrases of the same underlying tasks ---
    ("Could you move that rock to the red zone?", True),
    ("Please pick up the nearby stone and place it on the red marker.", True),
    ("Grab that cylinder and drop it in the red spot.", True),
    ("Take the box over to the target area.", True),
    ("Find the stone and bring it to the red region.", True),

    # --- Invalid / infeasible instructions: 10 ---
    ("Pick up the banana and place it in the blue area.", False),
    ("Move the robot to Mars.", False),
    ("Grab the stone and place it in the green zone.", False),
    ("Pick up the laptop from the desk.", False),
    ("Put the cylinder in the yellow area.", False),
    ("Fly the drone to the red area.", False),
    ("Grab the chair and place it on the table.", False),
    ("Move the sphere to the target.", False),
    ("Put the stone on the moon.", False),
    ("Delete the box.", False),
]


def run_eval():
    correct = 0
    results = []

    for instruction, expected_feasible in TEST_CASES:
        plan = plan_from_instruction(instruction)
        got_feasible = plan.get("feasible", False)
        is_correct = got_feasible == expected_feasible
        correct += is_correct

        results.append({
            "instruction": instruction,
            "expected_feasible": expected_feasible,
            "got_feasible": got_feasible,
            "correct": is_correct,
            "plan": plan,
        })
        print(f"[{'OK' if is_correct else 'FAIL'}] \"{instruction}\" -> feasible={got_feasible}")

    accuracy = correct / len(TEST_CASES)
    print(f"\nAction Planning Accuracy: {accuracy * 100:.1f}% ({correct}/{len(TEST_CASES)})")

    with open("planner_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Full results saved to planner_eval_results.json")


if __name__ == "__main__":
    run_eval()
