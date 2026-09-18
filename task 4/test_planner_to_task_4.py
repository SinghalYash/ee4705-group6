import sys
import time
from pathlib import Path

import mujoco


# ==========================================================
# IMPORT PROJECT ROOT
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==========================================================
# IMPORT TASK 3 + TASK 4
# ==========================================================

from planner import plan_from_instruction
from task_4_executor import Task4Executor


# ==========================================================
# TEST INSTRUCTION
# ==========================================================

TEST_INSTRUCTION = (
    "Move the blue box to the red area."
)


# ==========================================================
# HOLD FINAL STATE
# ==========================================================

def hold_final_state(executor):

    robot = executor.robot

    print(
        "\nClose the MuJoCo viewer "
        "when finished."
    )

    while (
        robot.viewer is not None
        and robot.viewer.is_running()
    ):

        for (
            actuator_id,
            command,
        ) in zip(
            robot.arm_actuator_ids,
            robot.joint_commands,
        ):

            robot.data.ctrl[
                actuator_id
            ] = command

        for finger_id in (
            robot.finger_actuator_ids
        ):

            robot.data.ctrl[
                finger_id
            ] = (
                robot.final_gripper_command
            )

        mujoco.mj_step(
            robot.model,
            robot.data,
        )

        if robot.carrying_object:
            robot.update_carried_object()

        robot.viewer.sync()

        time.sleep(
            robot.model.opt.timestep
        )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "=" * 65
    )

    print(
        "TASK 3 PLANNER -> TASK 4 EXECUTOR TEST"
    )

    print(
        "=" * 65
    )

    print(
        "\nInstruction:"
    )

    print(
        TEST_INSTRUCTION
    )

    # ======================================================
    # TASK 3 — GENERATE PLAN
    # ======================================================

    print(
        "\nGenerating structured plan "
        "using Task 3..."
    )

    plan = plan_from_instruction(
        TEST_INSTRUCTION
    )

    print(
        "\n--- TASK 3 PLAN ---"
    )

    print(
        "Feasible:",
        plan.get(
            "feasible"
        ),
    )

    print(
        "Reason:",
        plan.get(
            "reason"
        ),
    )

    actions = plan.get(
        "actions",
        [],
    )

    print(
        "Number of actions:",
        len(actions),
    )

    for index, action in enumerate(
        actions,
        start=1,
    ):

        print(
            f"{index}. {action}"
        )

    # ======================================================
    # CHECK PLAN BEFORE PHYSICAL EXECUTION
    # ======================================================

    if not plan.get(
        "feasible",
        False,
    ):

        print(
            "\nPlanner marked task "
            "as infeasible."
        )

        return

    if not actions:

        print(
            "\nPlanner returned "
            "no executable actions."
        )

        return

    input(
        "\nInspect the generated plan above. "
        "Press ENTER to execute it in MuJoCo..."
    )

    # ======================================================
    # TASK 4 — EXECUTE ACTUAL TASK 3 PLAN
    # ======================================================

    executor = Task4Executor()

    report = executor.execute_plan(
        plan
    )

    # ======================================================
    # FINAL REPORT
    # ======================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "TASK 3 -> TASK 4 FINAL REPORT"
    )

    print(
        "=" * 65
    )

    print(
        "Instruction:",
        TEST_INSTRUCTION,
    )

    print(
        "Plan success:",
        report.success,
    )

    print(
        "Completed actions:",
        report.completed_actions,
        "/",
        report.total_actions,
    )

    print(
        "Grasp attempts:",
        report.grasp_attempts,
    )

    print(
        "Final placement error:",
        report.final_placement_error,
    )

    print(
        "Failure reason:",
        report.failure_reason,
    )

    if report.success:

        print(
            "\n"
            + "=" * 65
        )

        print(
            "REAL TASK 3 -> TASK 4 "
            "INTEGRATION PASSED"
        )

        print(
            "=" * 65
        )

    else:

        print(
            "\n"
            + "=" * 65
        )

        print(
            "REAL TASK 3 -> TASK 4 "
            "INTEGRATION FAILED"
        )

        print(
            "=" * 65
        )

    hold_final_state(
        executor
    )

    executor.robot.close_viewer()


if __name__ == "__main__":
    main()