import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "scene.xml"
)


# ==========================================================
# ROBOT CONFIGURATION
# ==========================================================

ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_pitch",
]

ARM_ACTUATORS = [
    "act_shoulder_pan",
    "act_shoulder_lift",
    "act_elbow",
    "act_wrist_pitch",
]

FINGER_ACTUATORS = [
    "act_finger_1",
    "act_finger_2",
    "act_finger_3",
]

FINGER_GEOMS = [
    "finger_1_geom",
    "finger_2_geom",
    "finger_3_geom",
]


# ==========================================================
# CONTROL SETTINGS
# ==========================================================

GRIPPER_OPEN = 0.025
GRIPPER_CLOSED = 0.0

POSITION_TOLERANCE = 0.01
MAX_DQ = 0.004
DAMPING = 0.08


# ==========================================================
# RECOVERY SETTINGS
# ==========================================================

MAX_GRASP_ATTEMPTS = 3

RECOVERY_RETREAT_HEIGHT = 0.10

# ==========================================================
# SEARCH RECOVERY SETTINGS
# ==========================================================

# Safe Cartesian viewpoints used while searching.
#
# These are currently gripper/wrist viewpoints.
# When Task 2 is integrated, the wrist-camera detector
# will check for the requested object at each viewpoint.

SEARCH_WAYPOINTS = [
    np.array([
        0.28,
        -0.18,
        0.30,
    ]),

    np.array([
        0.32,
        -0.08,
        0.30,
    ]),

    np.array([
        0.30,
        0.05,
        0.30,
    ]),

    np.array([
        0.24,
        0.12,
        0.30,
    ]),
]


# ==========================================================
# OBJECT / TARGET SETTINGS
# ==========================================================

TARGET_RADIUS = 0.12
RELEASE_CLEARANCE = 0.008


# Object dimensions are taken directly from scene.xml.
#
# placement_half_height:
#     Height of object centre above the floor when resting.
#
# placement_radius:
#     Conservative XY footprint used when determining
#     whether the whole object is inside the target area.
#
# pregrasp_offset:
#     Desired vertical offset between object centre and
#     grasp_site during REACH.

OBJECT_PROPERTIES = {

    "stone": {
        "placement_half_height": 0.04,
        "placement_radius": 0.04,
        "pregrasp_offset": 0.025,
    },

    "box": {
        "placement_half_height": 0.03,
        "placement_radius": 0.03,
        "pregrasp_offset": 0.025,
    },

    "cylinder": {
        "placement_half_height": 0.05,
        "placement_radius": 0.03,
        "pregrasp_offset": 0.025,
    },
}
# ==========================================================
# SCENE MAPPING
# ==========================================================

OBJECT_BODY_NAMES = {
    "stone": "stone",
    "box": "box_obj",
    "cylinder": "cylinder_obj",
}

OBJECT_GEOM_NAMES = {
    "stone": "stone_geom",
    "box": "box_geom",
    "cylinder": "cylinder_geom",
}

TARGET_GEOM_NAMES = {
    "red_area": "target_area",
}


# ==========================================================
# ACTION RESULT
# ==========================================================

@dataclass
class ActionResult:
    """
    Standard result returned by Task 4 behaviours.

    The Task 4 executor can later use this result to
    decide whether to continue, retry, recover, or stop.
    """

    success: bool
    skill: str
    message: str
    error: float | None = None


# ==========================================================
# ROBOT SKILLS
# ==========================================================

class RobotSkills:

    def __init__(self):

        # --------------------------------------------------
        # Load MuJoCo
        # --------------------------------------------------

        self.model = mujoco.MjModel.from_xml_path(
            MODEL_PATH
        )

        self.data = mujoco.MjData(
            self.model
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # --------------------------------------------------
        # TASK 2 PERCEPTION CALLBACK
        # --------------------------------------------------

        # Task 5 can replace this with a function that uses
        # Task 2 visual perception.
        #
        # Expected form:
        #
        #     callback(target_name) -> bool
        #
        # If None, Task 4 keeps using its existing
        # simulator-state visibility fallback.

        self.perception_callback = None

        # --------------------------------------------------
        # Grasp site
        # --------------------------------------------------

        self.grasp_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )

        if self.grasp_site_id < 0:

            raise RuntimeError(
                "Could not find grasp_site in scene.xml"
            )

        # --------------------------------------------------
        # Arm joint addresses
        # --------------------------------------------------

        self.qpos_ids = []
        self.dof_ids = []

        for joint_name in ARM_JOINTS:

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:

                raise RuntimeError(
                    f"Could not find joint: "
                    f"{joint_name}"
                )

            self.qpos_ids.append(
                self.model.jnt_qposadr[
                    joint_id
                ]
            )

            self.dof_ids.append(
                self.model.jnt_dofadr[
                    joint_id
                ]
            )

        # --------------------------------------------------
        # Arm actuators
        # --------------------------------------------------

        self.arm_actuator_ids = []

        for actuator_name in ARM_ACTUATORS:

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )

            if actuator_id < 0:

                raise RuntimeError(
                    f"Could not find actuator: "
                    f"{actuator_name}"
                )

            self.arm_actuator_ids.append(
                actuator_id
            )

        # --------------------------------------------------
        # Finger actuators
        # --------------------------------------------------

        self.finger_actuator_ids = []

        for actuator_name in FINGER_ACTUATORS:

            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )

            if actuator_id < 0:

                raise RuntimeError(
                    f"Could not find actuator: "
                    f"{actuator_name}"
                )

            self.finger_actuator_ids.append(
                actuator_id
            )

        # --------------------------------------------------
        # Initial robot state
        # --------------------------------------------------

        self.initial_joint_positions = np.array([
            self.data.qpos[qid]
            for qid in self.qpos_ids
        ])

        self.initial_grasp_position = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        self.joint_commands = (
            self.initial_joint_positions.copy()
        )

        # --------------------------------------------------
        # Persistent Task 4 state
        # --------------------------------------------------

        self.current_object = None

        self.grasped_object = None

        self.carrying_object = False

        self.object_offset = None

        self.object_qpos_adr = None
        self.object_dof_adr = None

        self.final_gripper_command = (
            GRIPPER_OPEN
        )

        # --------------------------------------------------
        # Feedback / recovery state
        # --------------------------------------------------

        self.last_action_success = None
        self.last_error = None

        self.grasp_attempts = 0

        self.last_failure_reason = None

        self.safe_stopped = False

        # --------------------------------------------------
        # Search recovery state
        # --------------------------------------------------

        self.search_used = False
        self.search_attempts = 0

        # --------------------------------------------------
        # Temporary visibility test controls
        # --------------------------------------------------
        #
        # None:
        #     use normal temporary simulator visibility
        #
        # False:
        #     pretend Task 2 cannot currently see target
        #
        # True:
        #     pretend target is visible
        #
        # These are ONLY for testing the search recovery
        # before Task 2 vision is integrated.

        self.visibility_override = None

        # If set to an integer, the target will become
        # "visible" after this many search viewpoints.
        #
        # Example:
        #     2 = target found at viewpoint 2
        #
        # This is test-only.
        self.search_reveal_after = None

        # --------------------------------------------------
        # Viewer
        # --------------------------------------------------

        self.viewer = None

    # ======================================================
    # VIEWER
    # ======================================================

    def start_viewer(self):

        if self.viewer is None:

            self.viewer = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )
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
    # SIMULATOR-STATE GROUNDING
    # ======================================================

    def get_object_position(
        self,
        object_name,
    ):

        if object_name not in OBJECT_BODY_NAMES:

            raise ValueError(
                f"Unknown object: {object_name}"
            )

        body_name = OBJECT_BODY_NAMES[
            object_name
        ]

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id < 0:

            raise RuntimeError(
                f"Body '{body_name}' "
                f"not found in scene.xml"
            )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.data.xpos[
            body_id
        ].copy()

    def get_target_position(
        self,
        target_name,
    ):

        if target_name not in TARGET_GEOM_NAMES:

            raise ValueError(
                f"Unknown target: {target_name}"
            )

        geom_name = TARGET_GEOM_NAMES[
            target_name
        ]

        geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if geom_id < 0:

            raise RuntimeError(
                f"Geom '{geom_name}' "
                f"not found in scene.xml"
            )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return self.data.geom_xpos[
            geom_id
        ].copy()

    # ======================================================
    # TARGET VISIBILITY
    # ======================================================

    def is_target_visible(
        self,
        target_name,
    ):
        """
        Return whether the requested target is currently
        visible.

        CURRENT IMPLEMENTATION:
            simulator-state / test visibility

        FUTURE TASK 2 IMPLEMENTATION:
            wrist-camera object detection

        Keeping this as a separate interface means Task 2
        can later replace the visibility source without
        changing the Task 4 recovery logic.
        """

        # ----------------------------------------------
        # Test override
        # ----------------------------------------------

        if (
            self.visibility_override
            is not None
        ):

            return bool(
                self.visibility_override
            )

        # ----------------------------------------------
        # Task 2 perception
        # ----------------------------------------------

        if self.perception_callback is not None:

            return bool(
                self.perception_callback(
                    target_name
                )
            )

        # ----------------------------------------------
        # Temporary simulator-state fallback
        # ----------------------------------------------

        try:

            self.get_object_position(
                target_name
            )

            return True

        except Exception:

            return False

    # ======================================================
    # CONTACT DETECTION
    # ======================================================

    def get_finger_contacts(
        self,
        object_name,
    ):

        if object_name not in OBJECT_GEOM_NAMES:

            raise ValueError(
                f"Unknown object: {object_name}"
            )

        object_geom = OBJECT_GEOM_NAMES[
            object_name
        ]

        contacts = [
            False,
            False,
            False,
        ]

        for i in range(
            self.data.ncon
        ):

            contact = self.data.contact[i]

            geom1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom1,
            )

            geom2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                contact.geom2,
            )

            pair = {
                geom1,
                geom2,
            }

            if object_geom not in pair:
                continue

            for j, finger_geom in enumerate(
                FINGER_GEOMS
            ):

                if finger_geom in pair:
                    contacts[j] = True

        return contacts

    # ======================================================
    # OBJECT FREE-JOINT HELPERS
    # ======================================================

    def get_object_freejoint_addresses(
        self,
        object_name,
    ):

        if object_name not in OBJECT_BODY_NAMES:

            raise ValueError(
                f"Unknown object: {object_name}"
            )

        body_name = OBJECT_BODY_NAMES[
            object_name
        ]

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id < 0:

            raise RuntimeError(
                f"Could not find body: "
                f"{body_name}"
            )

        joint_id = self.model.body_jntadr[
            body_id
        ]

        if joint_id < 0:

            raise RuntimeError(
                f"{body_name} does not "
                f"have a joint"
            )

        qpos_adr = self.model.jnt_qposadr[
            joint_id
        ]

        dof_adr = self.model.jnt_dofadr[
            joint_id
        ]

        return (
            qpos_adr,
            dof_adr,
        )

    # ======================================================
    # KINEMATIC CARRY
    # ======================================================

    def start_carrying(
        self,
        object_name,
    ):

        if (
            self.grasped_object
            != object_name
        ):

            return ActionResult(
                success=False,
                skill="CARRY",
                message=(
                    f"{object_name} has not "
                    f"been verified as grasped"
                ),
            )

        object_pos = (
            self.get_object_position(
                object_name
            )
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        grasp_pos = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        self.object_offset = (
            object_pos
            - grasp_pos
        )

        (
            self.object_qpos_adr,
            self.object_dof_adr,
        ) = self.get_object_freejoint_addresses(
            object_name
        )

        self.carrying_object = True

        print(
            "\n--- CARRY INITIALISED ---"
        )

        print(
            "Object:",
            object_name,
        )

        print(
            "Object position:",
            object_pos,
        )

        print(
            "Grasp position:",
            grasp_pos,
        )

        print(
            "Object/grasp offset:",
            self.object_offset,
        )

        return ActionResult(
            success=True,
            skill="CARRY",
            message=(
                f"{object_name} entered "
                f"transport state"
            ),
        )

    def update_carried_object(self):

        if not self.carrying_object:
            return

        if (
            self.object_offset is None
            or self.object_qpos_adr is None
            or self.object_dof_adr is None
        ):

            return

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        grasp_pos = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        desired_object_pos = (
            grasp_pos
            + self.object_offset
        )

        # Free-joint qpos:
        # x, y, z, qw, qx, qy, qz
        #
        # Only change XYZ.
        self.data.qpos[
            self.object_qpos_adr:
            self.object_qpos_adr + 3
        ] = desired_object_pos

        # Remove residual object velocity.
        self.data.qvel[
            self.object_dof_adr:
            self.object_dof_adr + 6
        ] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    # ======================================================
    # CLOSED-LOOP CARTESIAN CONTROL
    # ======================================================

    def move_cartesian(
        self,
        target_pos,
        label,
        tolerance=POSITION_TOLERANCE,
        max_steps=2500,
        gripper_command=GRIPPER_OPEN,
        carry_object=False,
    ):

        if self.viewer is None:

            return ActionResult(
                success=False,
                skill="MOVE",
                message=(
                    "MuJoCo viewer has not "
                    "been started"
                ),
            )

        print(
            "\n" + "=" * 55
        )

        print(label)

        print(
            "Target:",
            target_pos,
        )

        print(
            "=" * 55
        )

        for step in range(
            max_steps
        ):

            if not self.viewer.is_running():

                return ActionResult(
                    success=False,
                    skill="MOVE",
                    message=(
                        f"{label}: viewer closed"
                    ),
                )

            # ----------------------------------------------
            # Actual physical position
            # ----------------------------------------------

            mujoco.mj_forward(
                self.model,
                self.data,
            )

            current_pos = (
                self.data.site_xpos[
                    self.grasp_site_id
                ].copy()
            )

            error = (
                target_pos
                - current_pos
            )

            error_norm = np.linalg.norm(
                error
            )

            # ----------------------------------------------
            # Success
            # ----------------------------------------------

            if error_norm < tolerance:

                self.last_action_success = True
                self.last_error = error_norm

                print(
                    f"\n{label} SUCCESS"
                )

                print(
                    "Target:",
                    target_pos,
                )

                print(
                    "Actual:",
                    current_pos,
                )

                print(
                    "Error:",
                    error_norm,
                    "m",
                )

                return ActionResult(
                    success=True,
                    skill="MOVE",
                    message=(
                        f"{label} reached"
                    ),
                    error=error_norm,
                )

            # ----------------------------------------------
            # Jacobian
            # ----------------------------------------------

            jacp = np.zeros(
                (3, self.model.nv)
            )

            jacr = np.zeros(
                (3, self.model.nv)
            )

            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacp,
                jacr,
                self.grasp_site_id,
            )

            J = jacp[
                :,
                self.dof_ids,
            ]

            # ----------------------------------------------
            # Damped least-squares IK
            # ----------------------------------------------

            dq = (
                J.T
                @ np.linalg.solve(
                    J @ J.T
                    + DAMPING**2
                    * np.eye(3),
                    error,
                )
            )

            dq = np.clip(
                dq,
                -MAX_DQ,
                MAX_DQ,
            )

            self.joint_commands += dq

            # ----------------------------------------------
            # Joint limits
            # ----------------------------------------------

            for i, joint_name in enumerate(
                ARM_JOINTS
            ):

                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )

                if self.model.jnt_limited[
                    joint_id
                ]:

                    lower = (
                        self.model.jnt_range[
                            joint_id,
                            0,
                        ]
                    )

                    upper = (
                        self.model.jnt_range[
                            joint_id,
                            1,
                        ]
                    )

                    self.joint_commands[i] = (
                        np.clip(
                            self.joint_commands[i],
                            lower,
                            upper,
                        )
                    )

            # ----------------------------------------------
            # Arm commands
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Finger commands
            # ----------------------------------------------

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = gripper_command

            # ----------------------------------------------
            # Physics
            # ----------------------------------------------

            for _ in range(5):

                mujoco.mj_step(
                    self.model,
                    self.data,
                )

                if (
                    carry_object
                    and self.carrying_object
                ):

                    self.update_carried_object()

            self.viewer.sync()

            if step % 100 == 0:

                print(
                    f"step {step:4d} | "
                    f"error="
                    f"{error_norm:.4f} m | "
                    f"site="
                    f"{np.round(current_pos, 3)}"
                )

            time.sleep(
                self.model.opt.timestep
                * 5
            )

        # ----------------------------------------------
        # Failed to converge
        # ----------------------------------------------

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        final_pos = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        final_error = np.linalg.norm(
            target_pos
            - final_pos
        )

        self.last_action_success = False
        self.last_error = final_error

        print(
            f"\n{label} FAILED"
        )

        print(
            "Target:",
            target_pos,
        )

        print(
            "Actual:",
            final_pos,
        )

        print(
            "Final error:",
            final_error,
            "m",
        )

        return ActionResult(
            success=False,
            skill="MOVE",
            message=(
                f"{label} failed to converge"
            ),
            error=final_error,
        )

    # ======================================================
    # APPROACH
    # ======================================================

    def approach(
        self,
        target_name,
    ):

        print(
            f"\n[TASK 4] APPROACH {target_name}"
        )

        # Use position estimated from Task 2 + RGB-D.
        if (
            hasattr(self, "perceived_target_position")
            and hasattr(self, "perceived_target_name")
            and self.perceived_target_name == target_name
        ):

            object_pos = (
                self.perceived_target_position.copy()
            )

        else:

            return ActionResult(
                success=False,
                skill="APPROACH",
                message=(
                    f"No RGB-D position available "
                    f"for {target_name}"
                ),
            )

        print(
            "RGB-D estimated object position:",
            object_pos,
        )

        # ----------------------------------------------
        # Safe pose
        # ----------------------------------------------

        safe_target = np.array([
            0.22,
            -0.12,
            0.34,
        ])

        result = self.move_cartesian(
            safe_target,
            label="APPROACH: SAFE POSE",
            gripper_command=GRIPPER_OPEN,
        )

        if not result.success:

            return ActionResult(
                success=False,
                skill="APPROACH",
                message=(
                    "Failed to reach safe pose"
                ),
                error=result.error,
            )

        object_pos = (
            self.perceived_target_position.copy()
        )

        # ----------------------------------------------
        # Above object
        # ----------------------------------------------

        above_target = np.array([
            object_pos[0],
            object_pos[1],
            0.34,
        ])

        result = self.move_cartesian(
            above_target,
            label="APPROACH: ABOVE OBJECT",
            gripper_command=GRIPPER_OPEN,
        )

        if not result.success:

            return ActionResult(
                success=False,
                skill="APPROACH",
                message=(
                    "Failed to move above object"
                ),
                error=result.error,
            )

        object_pos = (
            self.perceived_target_position.copy()
        )

        # ----------------------------------------------
        # Descend
        # ----------------------------------------------

        approach_target = np.array([
            object_pos[0],
            object_pos[1],
            0.22,
        ])

        result = self.move_cartesian(
            approach_target,
            label="APPROACH: DESCEND",
            gripper_command=GRIPPER_OPEN,
        )

        if not result.success:

            return ActionResult(
                success=False,
                skill="APPROACH",
                message=(
                    "Failed during "
                    "approach descent"
                ),
                error=result.error,
            )

        self.current_object = (
            target_name
        )

        self.last_action_success = True
        self.last_error = result.error

        return ActionResult(
            success=True,
            skill="APPROACH",
            message=(
                f"Successfully approached "
                f"{target_name}"
            ),
            error=result.error,
        )

    # ======================================================
    # REACH
    # ======================================================

    def reach(
        self,
        target_name,
    ):

        print(
            f"\n[TASK 4] REACH {target_name}"
        )

        if (
            hasattr(self, "perceived_target_position")
            and hasattr(self, "perceived_target_name")
            and self.perceived_target_name == target_name
        ):

            object_pos = (
                self.perceived_target_position.copy()
            )

        else:

            return ActionResult(
                success=False,
                skill="REACH",
                message=(
                    f"No RGB-D position available "
                    f"for {target_name}"
                ),
            )

        print(
            "RGB-D estimated object position:",
            object_pos,
        )

         

        object_properties = (
            OBJECT_PROPERTIES[
                target_name
            ]
        )

        pregrasp_target = np.array([
            object_pos[0],
            object_pos[1],
            object_pos[2],
        ])

        

        result = self.move_cartesian(
            pregrasp_target,
            label="REACH: PRE-GRASP",
            tolerance=0.005,
            max_steps=3000,
            gripper_command=GRIPPER_OPEN,
        )

        if not result.success:

            return ActionResult(
                success=False,
                skill="REACH",
                message=(
                    f"Failed to reach "
                    f"{target_name}"
                ),
                error=result.error,
            )

        # ----------------------------------------------
        # Closed-loop reach verification
        # ----------------------------------------------

        object_pos = (
            self.get_object_position(
                target_name
            )
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        grasp_pos = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        object_distance = np.linalg.norm(
            object_pos
            - grasp_pos
        )

        print(
            "\n--- REACH VERIFICATION ---"
        )

        print(
            "Object position:",
            object_pos,
        )

        print(
            "Grasp-site position:",
            grasp_pos,
        )

        print(
            "Object/grasp distance:",
            object_distance,
            "m",
        )

        if object_distance > 0.04:

            return ActionResult(
                success=False,
                skill="REACH",
                message=(
                    "Object remains too far "
                    "from grasp centre"
                ),
                error=object_distance,
            )

        self.current_object = (
            target_name
        )

        return ActionResult(
            success=True,
            skill="REACH",
            message=(
                f"Pre-grasp position "
                f"reached for {target_name}"
            ),
            error=object_distance,
        )

    # ======================================================
    # GRASP
    # ======================================================

    def grasp(
        self,
        target_name,
        duration=4.0,
    ):

        print(
            f"\n[TASK 4] GRASP {target_name}"
        )

        if (
            self.current_object
            != target_name
        ):

            return ActionResult(
                success=False,
                skill="GRASP",
                message=(
                    f"{target_name} has not "
                    f"been reached yet"
                ),
            )

        total_steps = int(
            duration
            / self.model.opt.timestep
        )

        contact_command = None

        print(
            "\nClosing gripper..."
        )

        # ----------------------------------------------
        # Phase A — close gradually
        # ----------------------------------------------

        for step in range(
            total_steps
        ):

            if not self.viewer.is_running():

                return ActionResult(
                    success=False,
                    skill="GRASP",
                    message="Viewer closed",
                )

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

            alpha = (
                step + 1
            ) / total_steps

            finger_command = (
                GRIPPER_OPEN
                + alpha
                * (
                    GRIPPER_CLOSED
                    - GRIPPER_OPEN
                )
            )

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = finger_command

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.viewer.sync()

            mujoco.mj_forward(
                self.model,
                self.data,
            )

            contacts = (
                self.get_finger_contacts(
                    target_name
                )
            )

            number_contacts = sum(
                contacts
            )

            object_pos = (
                self.get_object_position(
                    target_name
                )
            )

            grasp_pos = (
                self.data.site_xpos[
                    self.grasp_site_id
                ].copy()
            )

            object_distance = (
                np.linalg.norm(
                    object_pos
                    - grasp_pos
                )
            )

            if step % 100 == 0:

                print(
                    f"step {step:4d} | "
                    f"finger="
                    f"{finger_command:.4f} | "
                    f"contacts="
                    f"{contacts} | "
                    f"distance="
                    f"{object_distance:.4f}"
                )

            if (
                number_contacts >= 2
                and object_distance < 0.030
            ):

                contact_command = (
                    finger_command
                )

                print(
                    "\nCAPTURE DETECTED"
                )

                print(
                    "Contacts:",
                    contacts,
                )

                print(
                    "Finger command:",
                    contact_command,
                )

                break

            if object_distance > 0.10:

                return ActionResult(
                    success=False,
                    skill="GRASP",
                    message=(
                        "Object escaped "
                        "during grasp"
                    ),
                    error=object_distance,
                )

            time.sleep(
                self.model.opt.timestep
            )

        if contact_command is None:

            return ActionResult(
                success=False,
                skill="GRASP",
                message=(
                    "No stable multi-finger "
                    "capture detected"
                ),
            )

        # ----------------------------------------------
        # Phase B — stability
        # ----------------------------------------------

        print(
            "\nChecking grasp stability..."
        )

        contact_counts = [
            0,
            0,
            0,
        ]

        required_samples = 15
        stability_steps = 400

        for step in range(
            stability_steps
        ):

            if not self.viewer.is_running():

                return ActionResult(
                    success=False,
                    skill="GRASP",
                    message="Viewer closed",
                )

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

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = contact_command

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.viewer.sync()

            contacts = (
                self.get_finger_contacts(
                    target_name
                )
            )

            for i, touching in enumerate(
                contacts
            ):

                if touching:
                    contact_counts[i] += 1

            object_pos = (
                self.get_object_position(
                    target_name
                )
            )

            grasp_pos = (
                self.data.site_xpos[
                    self.grasp_site_id
                ].copy()
            )

            object_distance = (
                np.linalg.norm(
                    object_pos
                    - grasp_pos
                )
            )

            if step % 25 == 0:

                print(
                    f"stability {step:3d} | "
                    f"contacts={contacts} | "
                    f"counts={contact_counts} | "
                    f"distance="
                    f"{object_distance:.4f}"
                )

            if object_distance > 0.04:

                return ActionResult(
                    success=False,
                    skill="GRASP",
                    message=(
                        "Object escaped during "
                        "stability check"
                    ),
                    error=object_distance,
                )

            stable_fingers = sum(
                count >= required_samples
                for count in contact_counts
            )

            if stable_fingers >= 2:

                self.grasped_object = (
                    target_name
                )

                self.final_gripper_command = (
                    contact_command
                )

                self.last_action_success = True

                print(
                    "\nSTABLE GRASP CONFIRMED"
                )

                print(
                    "Contact samples:",
                    contact_counts,
                )

                return ActionResult(
                    success=True,
                    skill="GRASP",
                    message=(
                        f"Stable grasp of "
                        f"{target_name} confirmed"
                    ),
                    error=object_distance,
                )

            time.sleep(
                self.model.opt.timestep
            )

        return ActionResult(
            success=False,
            skill="GRASP",
            message=(
                "Grasp contact detected "
                "but did not remain stable"
            ),
        )

    # ======================================================
    # VERIFY GRASP
    # ======================================================

    def verify_grasp(
        self,
        target_name,
    ):

        if (
            self.grasped_object
            != target_name
        ):

            return ActionResult(
                success=False,
                skill="VERIFY",
                message=(
                    f"{target_name} is not "
                    f"registered as grasped"
                ),
            )

        object_pos = (
            self.get_object_position(
                target_name
            )
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        grasp_pos = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        distance = np.linalg.norm(
            object_pos
            - grasp_pos
        )

        contacts = (
            self.get_finger_contacts(
                target_name
            )
        )

        number_contacts = sum(
            contacts
        )

        print(
            "\n--- GRASP VERIFICATION ---"
        )

        print(
            "Contacts:",
            contacts,
        )

        print(
            "Object/grasp distance:",
            distance,
            "m",
        )

        success = (
            distance < 0.04
            and number_contacts >= 1
        )

        return ActionResult(
            success=success,
            skill="VERIFY",
            message=(
                "Grasp verified"
                if success
                else
                "Grasp verification failed"
            ),
            error=distance,
        )

    # ======================================================
    # RECOVERY — OPEN GRIPPER
    # ======================================================

    def open_gripper(
        self,
        duration=1.0,
    ):

        print(
            "\n[RECOVERY] Opening gripper..."
        )

        if (
            self.viewer is None
            or not self.viewer.is_running()
        ):

            return ActionResult(
                success=False,
                skill="OPEN_GRIPPER",
                message="Viewer is not running",
            )

        current_command = np.mean([
            self.data.ctrl[fid]
            for fid in self.finger_actuator_ids
        ])

        total_steps = max(
            1,
            int(
                duration
                / self.model.opt.timestep
            ),
        )

        for step in range(
            total_steps
        ):

            if not self.viewer.is_running():

                return ActionResult(
                    success=False,
                    skill="OPEN_GRIPPER",
                    message="Viewer closed",
                )

            alpha = (
                step + 1
            ) / total_steps

            finger_command = (
                current_command
                + alpha
                * (
                    GRIPPER_OPEN
                    - current_command
                )
            )

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

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = finger_command

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.viewer.sync()

            time.sleep(
                self.model.opt.timestep
            )

        self.final_gripper_command = (
            GRIPPER_OPEN
        )

        self.grasped_object = None
        self.carrying_object = False

        print(
            "[RECOVERY] Gripper open."
        )

        return ActionResult(
            success=True,
            skill="OPEN_GRIPPER",
            message="Gripper opened",
        )

    
    # ======================================================
    # RECOVERY — RETREAT
    # ======================================================

    def retreat(
        self,
        height=RECOVERY_RETREAT_HEIGHT,
    ):

        print(
            "\n[RECOVERY] Retreating upward..."
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        current_pos = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        retreat_target = (
            current_pos.copy()
        )

        retreat_target[2] += height

        print(
            "Current grasp position:",
            current_pos,
        )

        print(
            "Retreat target:",
            retreat_target,
        )

        result = self.move_cartesian(
            retreat_target,
            label="RECOVERY: RETREAT",
            tolerance=0.01,
            max_steps=2500,
            gripper_command=GRIPPER_OPEN,
        )

        if result.success:

            print(
                "[RECOVERY] Retreat successful."
            )

            return ActionResult(
                success=True,
                skill="RETREAT",
                message=(
                    "Recovery retreat successful"
                ),
                error=result.error,
            )

        print(
            "[RECOVERY] Retreat failed."
        )

        return ActionResult(
            success=False,
            skill="RETREAT",
            message=(
                "Recovery retreat failed"
            ),
            error=result.error,
        )
    

    # ======================================================
    # SAFE STOP
    # ======================================================

    def safe_stop(
        self,
        reason,
    ):

        print(
            "\n========================================"
        )

        print(
            "SAFE STOP"
        )

        print(
            "========================================"
        )

        print(
            "Reason:",
            reason,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # Hold actual current joint configuration.
        self.joint_commands = np.array([
            self.data.qpos[qid]
            for qid in self.qpos_ids
        ])

        self.safe_stopped = True

        self.last_action_success = False
        self.last_failure_reason = reason

        if (
            self.viewer is not None
            and self.viewer.is_running()
        ):

            for _ in range(200):

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

                time.sleep(
                    self.model.opt.timestep
                )

        return ActionResult(
            success=False,
            skill="STOP",
            message=reason,
        )

    # ======================================================
    # NORMAL TASK STOP
    # ======================================================

    def stop(
        self,
        reason="task complete",
    ):
        """
        Normal successful task termination.

        This is different from safe_stop():

        stop()
            = expected completion

        safe_stop()
            = abnormal failure
        """

        print(
            "\n========================================"
        )

        print(
            "TASK STOP"
        )

        print(
            "========================================"
        )

        print(
            "Reason:",
            reason,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # Hold the robot at its current physical pose.
        self.joint_commands = np.array([
            self.data.qpos[qid]
            for qid in self.qpos_ids
        ])

        self.safe_stopped = False
        self.last_action_success = True
        self.last_failure_reason = None

        return ActionResult(
            success=True,
            skill="STOP",
            message=reason,
        )

    # ======================================================
    # GRASP WITH RECOVERY
    # ======================================================

    def grasp_with_recovery(
        self,
        target_name,
        max_attempts=MAX_GRASP_ATTEMPTS,
        force_first_failure=False,
    ):

        print(
            "\n========================================"
        )

        print(
            f"GRASP WITH RECOVERY: "
            f"{target_name}"
        )

        print(
            "========================================"
        )

        self.grasp_attempts = 0
        self.safe_stopped = False
        self.last_failure_reason = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            self.grasp_attempts = attempt

            print(
                "\n----------------------------------------"
            )

            print(
                f"GRASP ATTEMPT "
                f"{attempt}/{max_attempts}"
            )

            print(
                "----------------------------------------"
            )

            # ------------------------------------------
            # Ensure valid pre-grasp state
            # ------------------------------------------

            if (
                self.current_object
                != target_name
            ):

                reach_result = (
                    self.reach(
                        target_name
                    )
                )

                if not reach_result.success:

                    if attempt >= max_attempts:

                        return self.safe_stop(
                            (
                                f"Unable to reach "
                                f"{target_name} "
                                f"for grasp"
                            )
                        )

                    continue

            # ------------------------------------------
            # Test-only forced failure
            # ------------------------------------------

            if (
                force_first_failure
                and attempt == 1
            ):

                print(
                    "\n[TEST] Intentionally "
                    "forcing first grasp "
                    "attempt to fail."
                )

                grasp_result = ActionResult(
                    success=False,
                    skill="GRASP",
                    message=(
                        "Forced first-attempt "
                        "failure for recovery test"
                    ),
                )

            else:

                grasp_result = self.grasp(
                    target_name
                )

            # ------------------------------------------
            # Verify successful grasp
            # ------------------------------------------

            if grasp_result.success:

                verify_result = (
                    self.verify_grasp(
                        target_name
                    )
                )

                if verify_result.success:

                    self.last_action_success = True
                    self.last_failure_reason = None

                    print(
                        "\n========================================"
                    )

                    print(
                        "GRASP RECOVERY "
                        "SEQUENCE SUCCESSFUL"
                    )

                    print(
                        "========================================"
                    )

                    print(
                        "Attempts required:",
                        attempt,
                    )

                    return ActionResult(
                        success=True,
                        skill="GRASP",
                        message=(
                            f"Grasp of "
                            f"{target_name} "
                            f"verified after "
                            f"{attempt} attempt(s)"
                        ),
                        error=verify_result.error,
                    )

                failure_reason = (
                    "Post-grasp "
                    "verification failed"
                )

            else:

                failure_reason = (
                    grasp_result.message
                )

            print(
                "\nGRASP ATTEMPT FAILED"
            )

            print(
                "Reason:",
                failure_reason,
            )

            self.last_failure_reason = (
                failure_reason
            )

            # ------------------------------------------
            # Attempts exhausted
            # ------------------------------------------

            if attempt >= max_attempts:

                return self.safe_stop(
                    (
                        f"Grasp of {target_name} "
                        f"failed after "
                        f"{max_attempts} attempts. "
                        f"Last failure: "
                        f"{failure_reason}"
                    )
                )

            # ------------------------------------------
            # Recovery
            # ------------------------------------------

            print(
                "\nStarting recovery..."
            )

            open_result = (
                self.open_gripper()
            )

            if not open_result.success:

                return self.safe_stop(
                    "Recovery failed: "
                    "could not open gripper"
                )

            retreat_result = (
                self.retreat()
            )

            if not retreat_result.success:

                return self.safe_stop(
                    "Recovery failed: "
                    "could not retreat safely"
                )

            try:

                object_pos = (
                    self.get_object_position(
                        target_name
                    )
                )

                print(
                    "\n[RECOVERY] Updated "
                    "object position:",
                    object_pos,
                )

            except Exception as error:

                return self.safe_stop(
                    (
                        "Recovery failed: "
                        "object state unavailable: "
                        f"{error}"
                    )
                )

            reach_result = (
                self.reach(
                    target_name
                )
            )

            if reach_result.success:

                print(
                    "\n[RECOVERY] Object "
                    "reached again successfully."
                )

            else:

                self.current_object = None

                print(
                    "\n[RECOVERY] "
                    "Re-reach failed."
                )

        return self.safe_stop(
            "Grasp recovery "
            "exited unexpectedly"
        )

    # ======================================================
    # MOVE_TO
    # ======================================================

    def move_to(
        self,
        target_name,
    ):

        print(
            f"\n[TASK 4] MOVE_TO "
            f"{target_name}"
        )

        if self.grasped_object is None:

            return ActionResult(
                success=False,
                skill="MOVE_TO",
                message=(
                    "Cannot transport: "
                    "no object is grasped"
                ),
            )

        # ----------------------------------------------
        # Initialise transport state
        # ----------------------------------------------

        if not self.carrying_object:

            carry_result = (
                self.start_carrying(
                    self.grasped_object
                )
            )

            if not carry_result.success:
                return carry_result

        # ----------------------------------------------
        # Read current target
        # ----------------------------------------------

        try:

            target_pos = (
                self.get_target_position(
                    target_name
                )
            )

        except Exception as error:

            return ActionResult(
                success=False,
                skill="MOVE_TO",
                message=str(error),
            )

        # ----------------------------------------------
        # Lift vertically
        # ----------------------------------------------

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        current_grasp = (
            self.data.site_xpos[
                self.grasp_site_id
            ].copy()
        )

        transport_height = max(
            current_grasp[2] + 0.10,
            0.20,
        )

        lift_target = np.array([
            current_grasp[0],
            current_grasp[1],
            transport_height,
        ])

        lift_result = (
            self.move_cartesian(
                lift_target,
                label="MOVE_TO: LIFT",
                tolerance=0.01,
                max_steps=3000,
                gripper_command=(
                    self.final_gripper_command
                ),
                carry_object=True,
            )
        )

        if not lift_result.success:

            return ActionResult(
                success=False,
                skill="MOVE_TO",
                message=(
                    "Failed to lift object "
                    "for transport"
                ),
                error=lift_result.error,
            )

        # ----------------------------------------------
        # Re-read target before horizontal transport
        # ----------------------------------------------

        target_pos = (
            self.get_target_position(
                target_name
            )
        )

        # Correct for the preserved object/grasp offset
        # so that the OBJECT centre ends above target.
        above_target = np.array([
            target_pos[0]
            - self.object_offset[0],

            target_pos[1]
            - self.object_offset[1],

            transport_height,
        ])

        transport_result = (
            self.move_cartesian(
                above_target,
                label="MOVE_TO: ABOVE TARGET",
                tolerance=0.01,
                max_steps=3500,
                gripper_command=(
                    self.final_gripper_command
                ),
                carry_object=True,
            )
        )

        if not transport_result.success:

            return ActionResult(
                success=False,
                skill="MOVE_TO",
                message=(
                    f"Failed to transport "
                    f"object to {target_name}"
                ),
                error=transport_result.error,
            )

        # ----------------------------------------------
        # Closed-loop transport verification
        # ----------------------------------------------

        object_pos = (
            self.get_object_position(
                self.grasped_object
            )
        )

        target_pos = (
            self.get_target_position(
                target_name
            )
        )

        xy_error = np.linalg.norm(
            object_pos[:2]
            - target_pos[:2]
        )

        print(
            "\n--- TRANSPORT VERIFICATION ---"
        )

        print(
            "Object position:",
            object_pos,
        )

        print(
            "Target position:",
            target_pos,
        )

        print(
            "Object/target XY error:",
            xy_error,
            "m",
        )

        success = (
            xy_error < 0.03
        )

        return ActionResult(
            success=success,
            skill="MOVE_TO",
            message=(
                f"Object transported above "
                f"{target_name}"
                if success
                else
                "Transport completed but "
                "object is not aligned "
                "with target"
            ),
            error=xy_error,
        )

    # ======================================================
    # PLACE
    # ======================================================

    def place(
        self,
        object_name,
        target_name,
    ):

        print(
            f"\n[TASK 4] PLACE "
            f"{object_name} -> {target_name}"
        )

        if (
            self.grasped_object
            != object_name
        ):

            return ActionResult(
                success=False,
                skill="PLACE",
                message=(
                    f"{object_name} is not "
                    f"currently grasped"
                ),
            )

        if not self.carrying_object:

            return ActionResult(
                success=False,
                skill="PLACE",
                message=(
                    "Object is not in "
                    "transport state"
                ),
            )

        try:

            target_pos = (
                self.get_target_position(
                    target_name
                )
            )

        except Exception as error:

            return ActionResult(
                success=False,
                skill="PLACE",
                message=str(error),
            )

        # ----------------------------------------------
        # Placement pose
        # ----------------------------------------------

        object_properties = (
            OBJECT_PROPERTIES[
                object_name
            ]
        )

        desired_object_z = (
            object_properties[
                "placement_half_height"
            ]
            + RELEASE_CLEARANCE
        )

        desired_grasp_z = (
            desired_object_z
            - self.object_offset[2]
        )

        release_target = np.array([
            target_pos[0]
            - self.object_offset[0],

            target_pos[1]
            - self.object_offset[1],

            desired_grasp_z,
        ])

        print(
            "Release target:",
            release_target,
        )

        # ----------------------------------------------
        # Lower
        # ----------------------------------------------

        lower_result = (
            self.move_cartesian(
                release_target,
                label="PLACE: LOWER",
                tolerance=0.006,
                max_steps=3500,
                gripper_command=(
                    self.final_gripper_command
                ),
                carry_object=True,
            )
        )

        if not lower_result.success:

            return ActionResult(
                success=False,
                skill="PLACE",
                message=(
                    "Failed to lower object "
                    "to placement pose"
                ),
                error=lower_result.error,
            )

        # ----------------------------------------------
        # Stop kinematic carrying before release
        # ----------------------------------------------

        self.carrying_object = False

        self.data.qvel[
            self.object_dof_adr:
            self.object_dof_adr + 6
        ] = 0.0

        # ----------------------------------------------
        # Gradual release
        # ----------------------------------------------

        print(
            "\nReleasing object..."
        )

        release_steps = 600

        start_command = (
            self.final_gripper_command
        )

        for step in range(
            release_steps
        ):

            if not self.viewer.is_running():

                return ActionResult(
                    success=False,
                    skill="PLACE",
                    message=(
                        "Viewer closed "
                        "during release"
                    ),
                )

            alpha = (
                step + 1
            ) / release_steps

            finger_command = (
                start_command
                + alpha
                * (
                    GRIPPER_OPEN
                    - start_command
                )
            )

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

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = finger_command

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.viewer.sync()

            if step % 100 == 0:

                object_pos = (
                    self.get_object_position(
                        object_name
                    )
                )

                print(
                    f"release {step:3d} | "
                    f"finger="
                    f"{finger_command:.4f} | "
                    f"object="
                    f"{np.round(object_pos, 3)}"
                )

            time.sleep(
                self.model.opt.timestep
            )

        self.final_gripper_command = (
            GRIPPER_OPEN
        )

        self.grasped_object = None

        # ----------------------------------------------
        # Allow object to settle
        # ----------------------------------------------

        print(
            "\nAllowing object to settle..."
        )

        for _ in range(1000):

            if not self.viewer.is_running():
                break

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

            for finger_id in (
                self.finger_actuator_ids
            ):

                self.data.ctrl[
                    finger_id
                ] = GRIPPER_OPEN

            mujoco.mj_step(
                self.model,
                self.data,
            )

            self.viewer.sync()

            time.sleep(
                self.model.opt.timestep
            )

        # ----------------------------------------------
        # Closed-loop placement verification
        # ----------------------------------------------

        verify_result = self.verify_place(
            object_name,
            target_name,
        )

        # PLACE itself reports whether the physical
        # placement succeeded. An explicit VERIFY action
        # can still check the state again afterwards.
        return ActionResult(
            success=verify_result.success,
            skill="PLACE",
            message=(
                f"{object_name} placed and verified "
                f"in {target_name}"
                if verify_result.success
                else
                f"{object_name} placement in "
                f"{target_name} failed verification"
            ),
            error=verify_result.error,
        )

    # ======================================================
    # VERIFY PLACE
    # ======================================================

    def verify_place(
        self,
        object_name,
        target_name,
    ):

        object_pos = (
            self.get_object_position(
                object_name
            )
        )

        target_pos = (
            self.get_target_position(
                target_name
            )
        )

        xy_error = np.linalg.norm(
            object_pos[:2]
            - target_pos[:2]
        )
 

        object_properties = (
            OBJECT_PROPERTIES[
                object_name
            ]
        )

        placement_limit = (
            TARGET_RADIUS
            - object_properties[
                "placement_radius"
            ]
        )

        success = (
            xy_error
            < placement_limit
        )

        print(
            "\n--- PLACE VERIFICATION ---"
        )

        print(
            "Object position:",
            object_pos,
        )

        print(
            "Target centre:",
            target_pos,
        )

        print(
            "XY placement error:",
            xy_error,
            "m",
        )

        print(
            "Acceptance radius:",
            placement_limit,
            "m",
        )

        print(
            "Placement verified:",
            success,
        )

        self.last_action_success = success
        self.last_error = xy_error

        if not success:

            self.last_failure_reason = (
                "Object outside target region"
            )

        return ActionResult(
            success=success,
            skill="VERIFY",
            message=(
                f"{object_name} successfully "
                f"placed in {target_name}"
                if success
                else
                f"{object_name} is outside "
                f"{target_name}"
            ),
            error=xy_error,
        )

    # ======================================================
    # TASK 4 RECOVERY — SEARCH
    # ======================================================

    def search(
        self,
        target_name,
        max_search_poses=None,
    ):
        """
        Search for a target by moving through a sequence
        of safe Cartesian viewpoints.

        At each viewpoint the system checks whether the
        target is visible.

        If the target is found, its current simulator
        position is read again before execution continues.

        If all viewpoints are exhausted, SEARCH fails.
        """

        print(
            "\n========================================"
        )

        print(
            f"SEARCH FOR TARGET: "
            f"{target_name}"
        )

        print(
            "========================================"
        )

        self.search_used = True
        self.search_attempts = 0

        if max_search_poses is None:

            max_search_poses = len(
                SEARCH_WAYPOINTS
            )

        max_search_poses = min(
            max_search_poses,
            len(SEARCH_WAYPOINTS),
        )

        # ----------------------------------------------
        # Check current view first
        # ----------------------------------------------

        if self.is_target_visible(
            target_name
        ):

            try:

                position = (
                    self.get_object_position(
                        target_name
                    )
                )

            except Exception:

                position = None

            print(
                "Target already visible."
            )

            if position is not None:

                print(
                    "Target position:",
                    position,
                )

            return ActionResult(
                success=True,
                skill="SEARCH",
                message=(
                    f"{target_name} "
                    f"already visible"
                ),
            )

        print(
            "Target not visible "
            "from current viewpoint."
        )

        # ----------------------------------------------
        # Search viewpoints
        # ----------------------------------------------

        for index in range(
            max_search_poses
        ):

            viewpoint_number = (
                index + 1
            )

            waypoint = (
                SEARCH_WAYPOINTS[
                    index
                ].copy()
            )

            self.search_attempts = (
                viewpoint_number
            )

            print(
                "\n----------------------------------------"
            )

            print(
                f"SEARCH VIEWPOINT "
                f"{viewpoint_number}/"
                f"{max_search_poses}"
            )

            print(
                "----------------------------------------"
            )

            print(
                "Search waypoint:",
                waypoint,
            )

            result = self.move_cartesian(
                waypoint,
                label=(
                    f"SEARCH: VIEWPOINT "
                    f"{viewpoint_number}"
                ),
                tolerance=0.015,
                max_steps=2500,
                gripper_command=(
                    GRIPPER_OPEN
                ),
            )

            if not result.success:

                print(
                    "Search viewpoint "
                    "could not be reached."
                )

                continue

            # ==========================================
            # TEST-ONLY simulated reacquisition
            # ==========================================

            if (
                self.search_reveal_after
                is not None
                and viewpoint_number
                >= self.search_reveal_after
            ):

                self.visibility_override = (
                    True
                )

            # ==========================================
            # Check visibility
            # ==========================================

            visible = (
                self.is_target_visible(
                    target_name
                )
            )

            print(
                "Target visible:",
                visible,
            )

            if visible:

                try:

                    target_position = (
                        self.get_object_position(
                            target_name
                        )
                    )

                except Exception as error:

                    return ActionResult(
                        success=False,
                        skill="SEARCH",
                        message=(
                            "Target appeared "
                            "visible but its "
                            "position could not "
                            f"be obtained: {error}"
                        ),
                    )

                print(
                    "\nTARGET FOUND"
                )

                print(
                    "Updated target position:",
                    target_position,
                )

                self.current_object = (
                    target_name
                )

                self.last_action_success = (
                    True
                )

                return ActionResult(
                    success=True,
                    skill="SEARCH",
                    message=(
                        f"{target_name} found "
                        f"after "
                        f"{viewpoint_number} "
                        f"search viewpoint(s)"
                    ),
                )

        # ----------------------------------------------
        # Search exhausted
        # ----------------------------------------------

        self.last_action_success = False

        self.last_failure_reason = (
            f"{target_name} not found "
            f"after {max_search_poses} "
            f"search viewpoints"
        )

        print(
            "\nSEARCH FAILED"
        )

        print(
            self.last_failure_reason
        )

        return ActionResult(
            success=False,
            skill="SEARCH",
            message=(
                self.last_failure_reason
            ),
        )

    # ======================================================
    # APPROACH WITH SEARCH RECOVERY
    # ======================================================

    def approach_with_search(
        self,
        target_name,
    ):
        """
        Attempt to approach a target.

        If the target is not initially visible, perform
        SEARCH first. Continue with APPROACH only after
        the target has been reacquired.
        """

        print(
            "\n========================================"
        )

        print(
            f"APPROACH WITH SEARCH: "
            f"{target_name}"
        )

        print(
            "========================================"
        )

        visible = (
            self.is_target_visible(
                target_name
            )
        )

        print(
            "Initially visible:",
            visible,
        )

        # ----------------------------------------------
        # Search recovery
        # ----------------------------------------------

        if not visible:

            print(
                "\nTarget is not initially "
                "visible."
            )

            print(
                "Starting search recovery..."
            )

            search_result = (
                self.search(
                    target_name
                )
            )

            if not search_result.success:

                return self.safe_stop(
                    (
                        "Search recovery failed: "
                        f"{search_result.message}"
                    )
                )

            print(
                "\nSearch recovery successful."
            )

            print(
                "Reacquiring target state "
                "before approach..."
            )

            try:

                updated_position = (
                    self.get_object_position(
                        target_name
                    )
                )

            except Exception as error:

                return self.safe_stop(
                    (
                        "Target was found but "
                        "its updated state could "
                        f"not be read: {error}"
                    )
                )

            print(
                "Updated target position:",
                updated_position,
            )

        # ----------------------------------------------
        # Continue normal approach
        # ----------------------------------------------

        return self.approach(
            target_name
        )