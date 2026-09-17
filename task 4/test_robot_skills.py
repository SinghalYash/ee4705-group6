import time

import mujoco

from robot_skills import (
    RobotSkills,
    GRIPPER_OPEN,
)


# ==========================================================
# TEST SETTINGS
# ==========================================================

# Normal Task 4 execution:
FORCE_FIRST_GRASP_FAILURE = False

# Change this to True only when demonstrating the
# recovery behaviour for your report/demo.
#
# FORCE_FIRST_GRASP_FAILURE = True


def print_result(
    title,
    result,
):

    print(
        "\n========================================"
    )

    print(title)

    print(
        "========================================"
    )

    print(
        "Success:",
        result.success,
    )

    print(
        "Skill:",
        result.skill,
    )

    print(
        "Message:",
        result.message,
    )

    print(
        "Error:",
        result.error,
    )


def hold_robot(robot):
    """
    Hold the final robot configuration until the user
    closes the MuJoCo viewer.
    """

    print(
        "\nClose the MuJoCo viewer when finished."
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


def main():

    print(
        "========================================"
    )

    print(
        "TASK 4 — COMPLETE ROBOT SKILLS TEST"
    )

    print(
        "========================================"
    )

    # ======================================================
    # CREATE ROBOT
    # ======================================================

    robot = RobotSkills()

    print(
        "\nMuJoCo model loaded successfully."
    )

    # ======================================================
    # TEST 1 — SIMULATOR-STATE GROUNDING
    # ======================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "TEST 1: SIMULATOR-STATE GROUNDING"
    )

    print(
        "----------------------------------------"
    )

    try:

        print(
            "Stone position:",
            robot.get_object_position(
                "stone"
            ),
        )

        print(
            "Box position:",
            robot.get_object_position(
                "box"
            ),
        )

        print(
            "Cylinder position:",
            robot.get_object_position(
                "cylinder"
            ),
        )

        print(
            "Red area position:",
            robot.get_target_position(
                "red_area"
            ),
        )

        print(
            "\nGROUNDING TEST PASSED"
        )

    except Exception as error:

        print(
            "\nGROUNDING TEST FAILED"
        )

        print(
            "Error:",
            error,
        )

        return

    # ======================================================
    # START VIEWER
    # ======================================================

    print(
        "\nStarting MuJoCo viewer..."
    )

    robot.start_viewer()

    print(
        "Allowing scene to settle..."
    )

    for _ in range(500):

        if not robot.viewer.is_running():
            return

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
            ] = GRIPPER_OPEN

        mujoco.mj_step(
            robot.model,
            robot.data,
        )

        robot.viewer.sync()

        time.sleep(
            robot.model.opt.timestep
        )

    print(
        "\nBox after settling:",
        robot.get_object_position(
            "box"
        ),
    )

    # ======================================================
    # TEST 2 — APPROACH
    # ======================================================

    input(
        "\nPress ENTER to test APPROACH('box')..."
    )

    approach_result = (
        robot.approach(
            "box"
        )
    )

    print_result(
        "APPROACH RESULT",
        approach_result,
    )

    if not approach_result.success:

        robot.safe_stop(
            "APPROACH test failed"
        )

        hold_robot(robot)
        return

    # ======================================================
    # TEST 3 — REACH
    # ======================================================

    input(
        "\nPress ENTER to test REACH('box')..."
    )

    reach_result = (
        robot.reach(
            "box"
        )
    )

    print_result(
        "REACH RESULT",
        reach_result,
    )

    if not reach_result.success:

        robot.safe_stop(
            "REACH test failed"
        )

        hold_robot(robot)
        return

    # ======================================================
    # TEST 4 — GRASP + RECOVERY
    # ======================================================

    input(
        "\nPress ENTER to test GRASP('box')..."
    )

    grasp_result = (
        robot.grasp_with_recovery(
            "box",
            force_first_failure=(
                FORCE_FIRST_GRASP_FAILURE
            ),
        )
    )

    print_result(
        "GRASP RESULT",
        grasp_result,
    )

    if not grasp_result.success:

        print(
            "\nGrasp failed after recovery."
        )

        print(
            "Attempts:",
            robot.grasp_attempts,
        )

        print(
            "Failure reason:",
            robot.last_failure_reason,
        )

        print(
            "Safe stopped:",
            robot.safe_stopped,
        )

        hold_robot(robot)
        return

    # ======================================================
    # TEST 5 — VERIFY GRASP
    # ======================================================

    verify_grasp_result = (
        robot.verify_grasp(
            "box"
        )
    )

    print_result(
        "VERIFY GRASP RESULT",
        verify_grasp_result,
    )

    if not verify_grasp_result.success:

        robot.safe_stop(
            "Post-grasp verification failed"
        )

        hold_robot(robot)
        return

    # ======================================================
    # TEST 6 — MOVE_TO
    # ======================================================

    input(
        "\nPress ENTER to test "
        "MOVE_TO('red_area')..."
    )

    move_result = (
        robot.move_to(
            "red_area"
        )
    )

    print_result(
        "MOVE_TO RESULT",
        move_result,
    )

    if not move_result.success:

        robot.safe_stop(
            "Transport to red_area failed"
        )

        hold_robot(robot)
        return

    # ======================================================
    # TEST 7 — PLACE
    # ======================================================

    input(
        "\nPress ENTER to test "
        "PLACE('box', 'red_area')..."
    )

    place_result = (
        robot.place(
            "box",
            "red_area",
        )
    )

    print_result(
        "PLACE RESULT",
        place_result,
    )

    # ======================================================
    # FINAL SUMMARY
    # ======================================================

    print(
        "\n========================================"
    )

    print(
        "TASK 4 SKILL TEST SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        "Simulator grounding : PASS"
    )

    print(
        "APPROACH             :",
        (
            "PASS"
            if approach_result.success
            else "FAIL"
        ),
    )

    print(
        "REACH                :",
        (
            "PASS"
            if reach_result.success
            else "FAIL"
        ),
    )

    print(
        "GRASP                :",
        (
            "PASS"
            if grasp_result.success
            else "FAIL"
        ),
    )

    print(
        "VERIFY GRASP         :",
        (
            "PASS"
            if verify_grasp_result.success
            else "FAIL"
        ),
    )

    print(
        "MOVE_TO              :",
        (
            "PASS"
            if move_result.success
            else "FAIL"
        ),
    )

    print(
        "PLACE / VERIFY PLACE :",
        (
            "PASS"
            if place_result.success
            else "FAIL"
        ),
    )

    print(
        "Grasp attempts       :",
        robot.grasp_attempts,
    )

    print(
        "Safe stopped         :",
        robot.safe_stopped,
    )

    if place_result.error is not None:

        print(
            "Final placement error:",
            place_result.error,
            "m",
        )

    if place_result.success:

        print(
            "\n========================================"
        )

        print(
            "TASK 4 MANIPULATION PIPELINE PASSED"
        )

        print(
            "========================================"
        )

        print(
            "\nThe robot successfully:"
        )

        print(
            "1. Read simulator-state grounding"
        )

        print(
            "2. Approached the box"
        )

        print(
            "3. Reached the pre-grasp pose"
        )

        print(
            "4. Established and verified a grasp"
        )

        print(
            "5. Transported the box"
        )

        print(
            "6. Placed the box"
        )

        print(
            "7. Verified its final location"
        )

    else:

        print(
            "\n========================================"
        )

        print(
            "TASK 4 PIPELINE NOT FULLY VERIFIED"
        )

        print(
            "========================================"
        )

    # ======================================================
    # HOLD FINAL CONFIGURATION
    # ======================================================

    hold_robot(
        robot
    )

    robot.close_viewer()


if __name__ == "__main__":
    main()