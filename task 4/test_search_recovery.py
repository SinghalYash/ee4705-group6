import time

import mujoco

from robot_skills import (
    RobotSkills,
    GRIPPER_OPEN,
)


# ==========================================================
# TEST SETTINGS
# ==========================================================

# Simulate the target being invisible initially.
FORCE_INITIAL_TARGET_HIDDEN = True

# Simulate Task 2 reacquiring the target after the
# second search viewpoint.
REVEAL_TARGET_AFTER_VIEWPOINT = None


# ==========================================================
# HOLD FINAL STATE
# ==========================================================

def hold_robot(
    robot,
):

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
            ] = GRIPPER_OPEN

        mujoco.mj_step(
            robot.model,
            robot.data,
        )

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
        "TASK 4 — SEARCH RECOVERY TEST"
    )

    print(
        "========================================"
    )

    robot = RobotSkills()

    robot.start_viewer()

    # ======================================================
    # ALLOW SCENE TO SETTLE
    # ======================================================

    print(
        "\nAllowing scene to settle..."
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

    # ======================================================
    # CONFIGURE TEST VISIBILITY
    # ======================================================

    if FORCE_INITIAL_TARGET_HIDDEN:

        robot.visibility_override = (
            False
        )

    else:

        robot.visibility_override = (
            None
        )

    robot.search_reveal_after = (
        REVEAL_TARGET_AFTER_VIEWPOINT
    )

    print(
        "\nTest configuration:"
    )

    print(
        "Initially hidden:",
        FORCE_INITIAL_TARGET_HIDDEN,
    )

    print(
        "Reveal after viewpoint:",
        REVEAL_TARGET_AFTER_VIEWPOINT,
    )

    input(
        "\nPress ENTER to start "
        "search recovery test..."
    )

    # ======================================================
    # RUN SEARCH + APPROACH
    # ======================================================

    result = (
        robot.approach_with_search(
            "box"
        )
    )

    # ======================================================
    # REPORT
    # ======================================================

    print(
        "\n========================================"
    )

    print(
        "SEARCH RECOVERY RESULT"
    )

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

    print(
        "Search used:",
        robot.search_used,
    )

    print(
        "Search viewpoints visited:",
        robot.search_attempts,
    )

    print(
        "Safe stopped:",
        robot.safe_stopped,
    )

    if (
        result.success
        and robot.search_used
        and robot.search_attempts
        == REVEAL_TARGET_AFTER_VIEWPOINT
    ):

        print(
            "\n========================================"
        )

        print(
            "SEARCH RECOVERY TEST PASSED"
        )

        print(
            "========================================"
        )

        print(
            "\nThe target was initially "
            "unavailable, the robot searched "
            "through multiple viewpoints, "
            "reacquired the target, and "
            "continued the approach."
        )

    else:

        print(
            "\n========================================"
        )

        print(
            "SEARCH RECOVERY TEST "
            "NOT VERIFIED"
        )

        print(
            "========================================"
        )

    hold_robot(
        robot
    )

    robot.close_viewer()


if __name__ == "__main__":
    main()