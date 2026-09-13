"""
End-to-end Task 3 demo: spoken instruction -> LLM planner -> mock execution.

Object grounding (Task 2) is not yet implemented, so target positions are
hardcoded to match scene.xml -- see executor.py's MOCK_POSES. This script
demonstrates the language -> action -> movement loop; visual verification
and real grasping come in Tasks 2 and 4.
"""

import json
from voice_input import listen_and_transcribe
from planner import plan_from_instruction
from executor import execute_plan


def main():
    print("=== Voice-Commanded Task Planner (Task 3 demo) ===")
    while True:
        choice = input("\nPress ENTER to speak an instruction, or type 'quit': ").strip()
        if choice.lower() == "quit":
            break

        instruction = listen_and_transcribe()
        if not instruction:
            print("No instruction understood, try again.")
            continue

        plan = plan_from_instruction(instruction)
        print("Generated plan:")
        print(json.dumps(plan, indent=2))

        execute_plan(plan)


if __name__ == "__main__":
    main()
