import time

import mujoco

from task_4_executor import (
    Task4Executor,
)


# ==========================================================
# TASK-3-STYLE TEST PLAN
# ==========================================================

TEST_PLAN = {
    "feasible": True,

    "actions": [

        {
            "skill": "APPROACH",
            "target": "box",
        },

        {
            "skill": "REACH",
            "target": "box",
        },

        {
            "skill": "GRASP",
            "target": "box",
        },

        # Explicit post-GRASP feedback.
        {
            "skill": "VERIFY",
            "verify_type": "GRASP",
            "object": "box",
        },

        {
            "skill": "MOVE_TO",
            "target": "red_area",
        },

        {
            "skill": "PLACE",
            "object": "box",
            "target": "red_area",
        },

        # Explicit post-PLACE feedback.
        {
            "skill": "VERIFY",
            "verify_type": "PLACE",
            "object": "box",
            "target": "red_area",
            "condition": (
                "box in red_area"
            ),
        },

        {
            "skill": "STOP",
            "reason": (
                "Task completed "
                "successfully"
            ),
        },
    ],
}


# ==========================================================
# FINAL HOLD
# ==========================================================

def hold_final_state(
    executor,
):

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
        "========================================"
    )

    print(
        "TASK 3 -> TASK 4 "
        "EXECUTOR INTEGRATION TEST"
    )

    print(
        "========================================"
    )

    print(
        "\nStructured plan:"
    )

    for index, action in enumerate(
        TEST_PLAN["actions"],
        start=1,
    ):

        print(
            f"{index}. {action}"
        )

    input(
        "\nPress ENTER to execute "
        "the complete structured plan..."
    )

    # ======================================================
    # CREATE EXECUTOR
    # ======================================================

    executor = (
        Task4Executor()
    )

    # ======================================================
    # EXECUTE PLAN
    # ======================================================

    report = (
        executor.execute_plan(
            TEST_PLAN
        )
    )

    # ======================================================
    # REPORT
    # ======================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FINAL EXECUTION REPORT"
    )

    print(
        "=" * 65
    )

    print(
        "Plan success:",
        report.success,
    )

    print(
        "Completed actions:",
        report.completed_actions,
    )

    print(
        "Total actions:",
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

    print(
        "\nAction results:"
    )

    for index, result in enumerate(
        report.results,
        start=1,
    ):

        print(
            f"{index}. "
            f"{result.skill} | "
            f"success="
            f"{result.success} | "
            f"error="
            f"{result.error} | "
            f"{result.message}"
        )

    # ======================================================
    # PASS / FAIL
    # ======================================================

    if report.success:

        print(
            "\n"
            + "=" * 65
        )

        print(
            "TASK 3 -> TASK 4 "
            "INTEGRATION PASSED"
        )

        print(
            "=" * 65
        )

        print(
            "\nA structured action plan "
            "successfully controlled "
            "the MuJoCo robot."
        )

    else:

        print(
            "\n"
            + "=" * 65
        )

        print(
            "TASK 3 -> TASK 4 "
            "INTEGRATION FAILED"
        )

        print(
            "=" * 65
        )

    # ======================================================
    # HOLD
    # ======================================================

    hold_final_state(
        executor
    )

    executor.robot.close_viewer()


if __name__ == "__main__":
    main()