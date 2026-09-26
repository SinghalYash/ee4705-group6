import time

import mujoco

from robot_skills import (
    RobotSkills as BaseRobotSkills,
    ActionResult,
    ARM_JOINTS,
    ARM_ACTUATORS,
    FINGER_ACTUATORS,
    FINGER_GEOMS,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    POSITION_TOLERANCE,
    MAX_DQ,
    DAMPING,
    MAX_GRASP_ATTEMPTS,
    RECOVERY_RETREAT_HEIGHT,
    SEARCH_WAYPOINTS,
    TARGET_RADIUS,
    RELEASE_CLEARANCE,
    OBJECT_PROPERTIES,
    OBJECT_BODY_NAMES,
    OBJECT_GEOM_NAMES,
    TARGET_GEOM_NAMES,
)


class _HeadlessViewer:
    """
    Viewer-compatible dummy object for headless evaluation.

    The normal RobotSkills implementation expects:
        viewer.is_running()
        viewer.sync()
        viewer.close()

    This provides the same interface without opening
    the MuJoCo GUI.
    """

    def __init__(self):

        self._running = True

    def is_running(self):

        return self._running

    def sync(self):

        pass

    def close(self):

        self._running = False


class RobotSkills(
    BaseRobotSkills
):
    """
    Evaluation version of RobotSkills.

    All actual robot behaviours are inherited directly
    from robot_skills.py so the evaluation uses the same:

        APPROACH
        REACH
        GRASP
        grasp recovery
        VERIFY
        MOVE_TO
        PLACE
        search recovery
        SAFE STOP

    The only difference is that the MuJoCo graphical
    viewer is optional.
    """

    def __init__(
        self,
        show_viewer=False,
        realtime=False,
    ):

        self.show_viewer = (
            show_viewer
        )

        self.realtime = (
            realtime
        )

        super().__init__()

    # ======================================================
    # VIEWER
    # ======================================================

    def start_viewer(self):

        if self.show_viewer:

            super().start_viewer()

        else:

            if self.viewer is None:

                self.viewer = (
                    _HeadlessViewer()
                )

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = GRIPPER_OPEN

    def close_viewer(self):

        if self.viewer is not None:

            self.viewer.close()

            self.viewer = None

    # ======================================================
    # SETTLING
    # ======================================================

    def settle(
        self,
        steps=500,
    ):
        """
        Allow objects to settle under MuJoCo physics
        before an evaluation trial starts.
        """

        if self.viewer is None:

            self.start_viewer()

        for _ in range(
            steps
        ):

            if not self.viewer.is_running():

                return False

            # Hold arm.
            for (
                actuator_id,
                command,
            ) in zip(
                self.arm_actuator_ids,
                self.joint_commands,
            ):

                self.data.ctrl[
                    actuator_id
                ] = command

            # Hold current gripper command.
            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = (
                    self.final_gripper_command
                )

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.viewer.sync()

            if self.realtime:

                time.sleep(
                    self.model.opt.timestep
                )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return True