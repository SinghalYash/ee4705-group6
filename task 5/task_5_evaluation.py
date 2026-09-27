"""
Task 5: Full-System End-to-End Evaluation

Pipeline:
    Language Instruction
        -> Visual Understanding / Target Grounding
        -> Action Planning
        -> Robot Execution
        -> Verification
        -> Task Feedback

Outputs:
    task 5/results/task_5_final_results.csv
    task 5/results/task_5_summary.csv

IMPORTANT:
MuJoCo ground truth is used ONLY for:
    - experimental scene randomisation
    - evaluation/scoring
    - localisation-error measurement

The manipulation controller does NOT receive the true
MuJoCo object position.

Task 2 provides visually estimated X/Y coordinates.
The known physical object geometry provides the expected
centre height Z for manipulation.
"""

# ==========================================================
# STANDARD IMPORTS
# ==========================================================

import csv
import sys
import time
import traceback

from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from PIL import Image


# ==========================================================
# PROJECT PATHS
# ==========================================================

TASK5_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

PROJECT_ROOT = (
    TASK5_DIR.parent
)

TASK4_DIR = (
    PROJECT_ROOT
    / "task_4"
)


for path in [
    PROJECT_ROOT,
    TASK4_DIR,
    TASK5_DIR,
]:

    path_string = str(
        path
    )

    if path_string not in sys.path:

        sys.path.insert(
            0,
            path_string,
        )


# ==========================================================
# TASK IMPORTS
# ==========================================================

from task_2.perception import (
    ground_object,
)

from task_3.planner import (
    plan_from_instruction,
)

from task_4.task_4_executor import (
    Task4Executor,
)

from task_4.robot_skills import (
    ARM_JOINTS,
    GRIPPER_OPEN,
    OBJECT_PROPERTIES,
)

from camera_grounding import (
    bbox_depth_to_world,
)


# ==========================================================
# EVALUATION SETTINGS
# ==========================================================

# ----------------------------------------------------------
# SMOKE TEST
#
# Keep at 3 until this updated perception -> manipulation
# interface is verified.
#
# FINAL evaluation:
#
#     N_TRIALS = 24
# ----------------------------------------------------------

N_TRIALS = 24

# Reproducible experiment.
RANDOM_SEED = 4705


# Automated evaluation should normally be headless.
SHOW_VIEWER = False


# Physics settling before execution.
SETTLE_STEPS = 500


# ==========================================================
# CAMERA SETTINGS
# ==========================================================

# Global scene understanding for Task 3.
PLANNING_CAMERA = (
    "overhead_cam"
)


# Initial object grounding.
INITIAL_GROUNDING_CAMERA = (
    "overhead_cam"
)


# Search/reacquisition should use a moving camera.
SEARCH_CAMERA = (
    "wrist_cam"
)


# ==========================================================
# OUTPUT PATHS
# ==========================================================

RESULTS_DIR = (
    TASK5_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


TRIAL_CSV = (
    RESULTS_DIR
    / "task_5_final_results.csv"
)


SUMMARY_CSV = (
    RESULTS_DIR
    / "task_5_summary.csv"
)


# ==========================================================
# OBJECT INFORMATION
# ==========================================================

OBJECTS = [
    "box",
    "cylinder",
    "stone",
]


OBJECT_BODY_NAMES = {

    "box":
        "box_obj",

    "cylinder":
        "cylinder_obj",

    "stone":
        "stone",
}


OBJECT_NOMINAL_POSITIONS = {

    "box":
        np.array([
            0.32,
            -0.08,
            0.03,
        ]),

    "cylinder":
        np.array([
            0.30,
            0.08,
            0.05,
        ]),

    "stone":
        np.array([
            0.28,
            -0.18,
            0.04,
        ]),
}


OBJECT_REST_HEIGHTS = {

    "box":
        0.03,

    "cylinder":
        0.05,

    "stone":
        0.04,
}


VISUAL_TARGET_NAMES = {

    "box":
        "blue cube",

    "stone":
        "grey stone",

    "cylinder":
        "green cylinder",
}


# ==========================================================
# TARGET INFORMATION
# ==========================================================

TARGET_GEOM_NAME = (
    "target_area"
)


TARGET_NOMINAL_POSITION = (
    np.array([
        0.25,
        0.20,
        0.001,
    ])
)


# ==========================================================
# RANDOMISATION LIMITS
# ==========================================================

OBJECT_XY_JITTER = (
    0.025
)


TARGET_XY_JITTER = (
    0.025
)


# Order:
# shoulder_pan
# shoulder_lift
# elbow
# wrist_pitch

ARM_OFFSET_LIMITS = (
    np.array([
        0.04,
        0.04,
        0.03,
        0.02,
    ])
)


# ==========================================================
# INSTRUCTION VARIATIONS
# ==========================================================

INSTRUCTION_TEMPLATES = {

    "box": [

        (
            "direct",
            "Move the blue box to the red area."
        ),

        (
            "direct",
            "Put the box in the red area."
        ),

        (
            "paraphrase",
            "Place the blue cube on the red target."
        ),

        (
            "paraphrase",
            (
                "Take the blue object and "
                "move it to the red zone."
            )
        ),

        (
            "colour",
            (
                "Move the blue one to "
                "the red area."
            )
        ),

        (
            "informal",
            (
                "Can you put the blue block "
                "on the red spot?"
            )
        ),
    ],


    "cylinder": [

        (
            "direct",
            (
                "Move the green cylinder "
                "to the red area."
            )
        ),

        (
            "direct",
            (
                "Put the cylinder in "
                "the red area."
            )
        ),

        (
            "paraphrase",
            (
                "Place the green cylinder "
                "on the red target."
            )
        ),

        (
            "paraphrase",
            (
                "Take the green object and "
                "move it to the red zone."
            )
        ),

        (
            "colour",
            (
                "Move the green one to "
                "the red area."
            )
        ),

        (
            "informal",
            (
                "Can you put the green thing "
                "on the red spot?"
            )
        ),
    ],


    "stone": [

        (
            "direct",
            (
                "Move the grey stone "
                "to the red area."
            )
        ),

        (
            "direct",
            (
                "Put the stone in "
                "the red area."
            )
        ),

        (
            "paraphrase",
            (
                "Place the rock on "
                "the red target."
            )
        ),

        (
            "paraphrase",
            (
                "Take the grey object and "
                "move it to the red zone."
            )
        ),

        (
            "colour",
            (
                "Move the grey one to "
                "the red area."
            )
        ),

        (
            "informal",
            (
                "Can you put the rock "
                "on the red spot?"
            )
        ),
    ],
}


# ==========================================================
# CSV FIELDS
# ==========================================================

FIELDNAMES = [

    # Trial
    "trial",
    "object",
    "target",
    "instruction",
    "phrase_type",

    # Object initial state
    "object_start_x",
    "object_start_y",
    "object_start_z",

    # Target initial state
    "target_start_x",
    "target_start_y",
    "target_start_z",

    # Robot initial state
    "shoulder_pan_start",
    "shoulder_lift_start",
    "elbow_start",
    "wrist_pitch_start",

    # Task 2
    "grounding_success",
    "grounding_correct",

    # Raw RGB-D surface point
    "perceived_x",
    "perceived_y",
    "perceived_z",

    # Position supplied to Task 4
    "manipulation_x",
    "manipulation_y",
    "manipulation_z",

    # Evaluation-only ground truth
    "true_x",
    "true_y",
    "true_z",

    # Raw RGB-D error
    "grounding_error_x_m",
    "grounding_error_y_m",
    "grounding_error_z_m",

    "grounding_xy_error_m",
    "grounding_localisation_error_m",

    # Task 3
    "plan_feasible",
    "planning_correct",
    "planned_object",
    "planned_target",
    "planned_actions",

    # Task 4 stages
    "search_success",
    "approach_success",
    "reach_success",

    "execution_attempted",
    "manipulation_attempted",

    "grasp_success",
    "grasp_attempts",
    "grasp_recovery_used",

    "move_success",

    "place_success",
    "placement_error_m",

    "manipulation_success",

    # Overall
    "end_to_end_success",

    "failure_stage",
    "failure_reason",

    # Timing
    "grounding_time_s",
    "planning_time_s",
    "execution_time_s",
    "total_time_s",

    # AI calls
    "task2_calls",
    "task3_calls",
]


# ==========================================================
# EMPTY RESULT ROW
# ==========================================================

def make_empty_row(
    trial_number,
    object_name,
    instruction,
    phrase_type,
):

    row = {
        field: None
        for field in FIELDNAMES
    }

    row.update({

        "trial":
            trial_number,

        "object":
            object_name,

        "target":
            "red_area",

        "instruction":
            instruction,

        "phrase_type":
            phrase_type,

        "grounding_success":
            False,

        "grounding_correct":
            False,

        "plan_feasible":
            False,

        "planning_correct":
            False,

        "search_success":
            False,

        "approach_success":
            False,

        "reach_success":
            False,

        "execution_attempted":
            False,

        "manipulation_attempted":
            False,

        "grasp_success":
            False,

        "grasp_attempts":
            0,

        "grasp_recovery_used":
            False,

        "move_success":
            False,

        "place_success":
            False,

        "manipulation_success":
            False,

        "end_to_end_success":
            False,

        "grounding_time_s":
            0.0,

        "task2_calls":
            0,

        "task3_calls":
            0,
    })

    return row


# ==========================================================
# RGB-D CAPTURE
# ==========================================================

def capture_rgbd(
    robot,
    camera_name,
):

    renderer = mujoco.Renderer(
        robot.model,
        height=480,
        width=640,
    )

    # ------------------------------------------------------
    # RGB
    # ------------------------------------------------------

    renderer.update_scene(
        robot.data,
        camera=camera_name,
    )

    rgb_array = (
        renderer.render()
    )

    rgb_image = (
        Image.fromarray(
            rgb_array
        )
    )

    # ------------------------------------------------------
    # DEPTH
    # ------------------------------------------------------

    renderer.enable_depth_rendering()

    renderer.update_scene(
        robot.data,
        camera=camera_name,
    )

    depth_image = (
        renderer.render()
    )

    renderer.disable_depth_rendering()

    renderer.close()

    return (
        rgb_image,
        depth_image,
    )


# ==========================================================
# PERCEPTION CAMERA
# ==========================================================

def choose_perception_camera(
    robot,
):

    # Once SEARCH has moved to one of its viewpoints,
    # use the wrist camera so the visual viewpoint
    # actually changes.

    if (
        getattr(
            robot,
            "search_attempts",
            0,
        )
        > 0
    ):

        return SEARCH_CAMERA

    return INITIAL_GROUNDING_CAMERA


# ==========================================================
# TASK 2 -> TASK 4 PERCEPTION CALLBACK
# ==========================================================

def install_evaluation_perception_callback(
    robot,
    expected_object,
    row,
):

    def detect_target(
        target_name,
    ):

        row[
            "task2_calls"
        ] += 1

        visual_target = (
            VISUAL_TARGET_NAMES.get(
                target_name,
                target_name,
            )
        )

        camera_name = (
            choose_perception_camera(
                robot
            )
        )

        print(
            "\nTask 2 checking for:",
            visual_target,
        )

        print(
            "Camera:",
            camera_name,
        )

        perception_start = (
            time.perf_counter()
        )

        # ==================================================
        # CAPTURE
        # ==================================================

        (
            image,
            depth_image,
        ) = capture_rgbd(
            robot,
            camera_name=camera_name,
        )

        # ==================================================
        # TASK 2
        # ==================================================

        result = ground_object(
            None,
            None,
            image,
            visual_target,
        )

        row[
            "grounding_time_s"
        ] += float(
            time.perf_counter()
            - perception_start
        )

        print(
            "Perception status:",
            result.status,
        )

        print(
            "Grounded target:",
            result.target,
        )

        success = (
            result.status == "success"
            and result.target is not None
        )

        if not success:

            return False

        # ==================================================
        # TARGET IDENTITY
        # ==================================================

        row[
            "grounding_success"
        ] = True

        row[
            "grounding_correct"
        ] = (
            target_name
            == expected_object
        )

        # ==================================================
        # BOUNDING BOX
        # ==================================================

        bbox = (
            result.target.bbox
        )

        if (
            bbox is None
            or len(
                bbox
            ) != 4
        ):

            print(
                "Task 2 returned no "
                "usable bounding box."
            )

            return False

        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = bbox

        centre_x = round(
            (
                x_min
                + x_max
            )
            / 2.0
        )

        centre_y = round(
            (
                y_min
                + y_max
            )
            / 2.0
        )

        centre_x = int(
            np.clip(
                centre_x,
                0,
                depth_image.shape[1] - 1,
            )
        )

        centre_y = int(
            np.clip(
                centre_y,
                0,
                depth_image.shape[0] - 1,
            )
        )

        depth_value = (
            depth_image[
                centre_y,
                centre_x,
            ]
        )

        # ==================================================
        # RAW RGB-D WORLD POSITION
        # ==================================================

        estimated_position = (
            bbox_depth_to_world(
                model=robot.model,
                data=robot.data,
                bbox=bbox,
                depth=depth_value,
                camera_name=camera_name,
            )
        )

        # ==================================================
        # SURFACE POINT -> MANIPULATION REFERENCE
        # ==================================================
        #
        # RGB-D returns the visible surface point.
        #
        # Task 4's APPROACH / REACH logic expects the
        # object-centre position.
        #
        # X/Y remain visually estimated.
        #
        # Z comes from the known physical geometry of the
        # requested object resting on the floor.
        #
        # This is NOT MuJoCo ground-truth object position.
        # It is a known object-property prior.

        manipulation_position = (
            estimated_position.copy()
        )

        if (
            target_name
            in OBJECT_PROPERTIES
        ):

            manipulation_position[2] = (
                OBJECT_PROPERTIES[
                    target_name
                ][
                    "placement_half_height"
                ]
            )

        # ==================================================
        # PROVIDE TASK 4 WITH VISUAL POSITION
        # ==================================================

        robot.perceived_target_position = (
            manipulation_position.copy()
        )

        robot.perceived_target_name = (
            target_name
        )

        # ==================================================
        # STORE RAW + CONTROL POSITIONS
        # ==================================================

        row[
            "perceived_x"
        ] = float(
            estimated_position[0]
        )

        row[
            "perceived_y"
        ] = float(
            estimated_position[1]
        )

        row[
            "perceived_z"
        ] = float(
            estimated_position[2]
        )

        row[
            "manipulation_x"
        ] = float(
            manipulation_position[0]
        )

        row[
            "manipulation_y"
        ] = float(
            manipulation_position[1]
        )

        row[
            "manipulation_z"
        ] = float(
            manipulation_position[2]
        )

        print(
            "Raw RGB-D surface XYZ:",
            estimated_position,
        )

        print(
            "Task 4 manipulation XYZ:",
            manipulation_position,
        )

        # ==================================================
        # EVALUATION-ONLY GROUND TRUTH
        # ==================================================

        try:

            true_position = (
                robot.get_object_position(
                    expected_object
                )
            )

            error_vector = (
                estimated_position
                - true_position
            )

            error_x = float(
                error_vector[0]
            )

            error_y = float(
                error_vector[1]
            )

            error_z = float(
                error_vector[2]
            )

            xy_error = float(
                np.linalg.norm(
                    error_vector[:2]
                )
            )

            xyz_error = float(
                np.linalg.norm(
                    error_vector
                )
            )

            row[
                "true_x"
            ] = float(
                true_position[0]
            )

            row[
                "true_y"
            ] = float(
                true_position[1]
            )

            row[
                "true_z"
            ] = float(
                true_position[2]
            )

            row[
                "grounding_error_x_m"
            ] = error_x

            row[
                "grounding_error_y_m"
            ] = error_y

            row[
                "grounding_error_z_m"
            ] = error_z

            row[
                "grounding_xy_error_m"
            ] = xy_error

            row[
                "grounding_localisation_error_m"
            ] = xyz_error

            print(
                "True MuJoCo centre XYZ:",
                true_position,
            )

            print(
                "Raw RGB-D error vector:",
                error_vector,
            )

            print(
                "Raw RGB-D XY error:",
                xy_error,
                "m",
            )

            print(
                "Raw RGB-D 3D error:",
                xyz_error,
                "m",
            )

        except Exception as error:

            print(
                "Could not calculate "
                "evaluation-only ground truth:",
                error,
            )

        return True

    robot.perception_callback = (
        detect_target
    )


# ==========================================================
# FREE-JOINT HELPERS
# ==========================================================

def get_freejoint_addresses(
    robot,
    body_name,
):

    body_id = mujoco.mj_name2id(
        robot.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:

        raise RuntimeError(
            f"Body not found: "
            f"{body_name}"
        )

    joint_id = (
        robot.model.body_jntadr[
            body_id
        ]
    )

    if joint_id < 0:

        raise RuntimeError(
            f"{body_name} has no joint"
        )

    if (
        robot.model.jnt_type[
            joint_id
        ]
        != mujoco.mjtJoint.mjJNT_FREE
    ):

        raise RuntimeError(
            f"{body_name} does not "
            "have a free joint"
        )

    qpos_adr = (
        robot.model.jnt_qposadr[
            joint_id
        ]
    )

    dof_adr = (
        robot.model.jnt_dofadr[
            joint_id
        ]
    )

    return (
        qpos_adr,
        dof_adr,
    )


# ==========================================================
# RANDOMISE OBJECT
# ==========================================================

def randomise_object(
    robot,
    object_name,
    rng,
):

    body_name = (
        OBJECT_BODY_NAMES[
            object_name
        ]
    )

    (
        qpos_adr,
        dof_adr,
    ) = get_freejoint_addresses(
        robot,
        body_name,
    )

    position = (
        OBJECT_NOMINAL_POSITIONS[
            object_name
        ].copy()
    )

    position[0] += rng.uniform(
        -OBJECT_XY_JITTER,
        OBJECT_XY_JITTER,
    )

    position[1] += rng.uniform(
        -OBJECT_XY_JITTER,
        OBJECT_XY_JITTER,
    )

    position[2] = (
        OBJECT_REST_HEIGHTS[
            object_name
        ]
    )

    robot.data.qpos[
        qpos_adr:
        qpos_adr + 3
    ] = position

    # Identity quaternion.
    robot.data.qpos[
        qpos_adr + 3:
        qpos_adr + 7
    ] = np.array([
        1.0,
        0.0,
        0.0,
        0.0,
    ])

    # Zero free-body velocity.
    robot.data.qvel[
        dof_adr:
        dof_adr + 6
    ] = 0.0

    return position


# ==========================================================
# RANDOMISE TARGET
# ==========================================================

def randomise_target_area(
    robot,
    rng,
):

    geom_id = mujoco.mj_name2id(
        robot.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        TARGET_GEOM_NAME,
    )

    if geom_id < 0:

        raise RuntimeError(
            "target_area geom not found"
        )

    position = (
        TARGET_NOMINAL_POSITION.copy()
    )

    position[0] += rng.uniform(
        -TARGET_XY_JITTER,
        TARGET_XY_JITTER,
    )

    position[1] += rng.uniform(
        -TARGET_XY_JITTER,
        TARGET_XY_JITTER,
    )

    robot.model.geom_pos[
        geom_id
    ] = position

    return position


# ==========================================================
# RANDOMISE ROBOT
# ==========================================================

def randomise_robot(
    robot,
    rng,
):

    nominal = (
        robot.initial_joint_positions.copy()
    )

    offsets = rng.uniform(
        -ARM_OFFSET_LIMITS,
        ARM_OFFSET_LIMITS,
    )

    initial = (
        nominal
        + offsets
    )

    for index, joint_name in enumerate(
        ARM_JOINTS
    ):

        joint_id = mujoco.mj_name2id(
            robot.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if (
            robot.model.jnt_limited[
                joint_id
            ]
        ):

            lower = (
                robot.model.jnt_range[
                    joint_id,
                    0,
                ]
            )

            upper = (
                robot.model.jnt_range[
                    joint_id,
                    1,
                ]
            )

            initial[
                index
            ] = np.clip(
                initial[
                    index
                ],
                lower,
                upper,
            )

    for (
        qpos_id,
        value,
    ) in zip(
        robot.qpos_ids,
        initial,
    ):

        robot.data.qpos[
            qpos_id
        ] = value

    robot.joint_commands = (
        initial.copy()
    )

    return initial


# ==========================================================
# RESET TASK STATE
# ==========================================================

def reset_task_state(
    robot,
):

    robot.current_object = None

    robot.grasped_object = None

    robot.carrying_object = False

    robot.object_offset = None

    robot.object_qpos_adr = None

    robot.object_dof_adr = None

    robot.final_gripper_command = (
        GRIPPER_OPEN
    )

    robot.last_action_success = None

    robot.last_error = None

    robot.grasp_attempts = 0

    robot.last_failure_reason = None

    robot.safe_stopped = False

    robot.search_used = False

    robot.search_attempts = 0

    robot.visibility_override = None

    robot.search_reveal_after = None

    if hasattr(
        robot,
        "perceived_target_position",
    ):

        delattr(
            robot,
            "perceived_target_position",
        )

    if hasattr(
        robot,
        "perceived_target_name",
    ):

        delattr(
            robot,
            "perceived_target_name",
        )

    for finger_id in (
        robot.finger_actuator_ids
    ):

        robot.data.ctrl[
            finger_id
        ] = GRIPPER_OPEN


# ==========================================================
# SETTLE SCENE
# ==========================================================

def settle_scene(
    robot,
    steps=SETTLE_STEPS,
):

    for _ in range(
        steps
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

        if (
            SHOW_VIEWER
            and robot.viewer is not None
        ):

            robot.viewer.sync()

    mujoco.mj_forward(
        robot.model,
        robot.data,
    )


# ==========================================================
# PLAN HELPERS
# ==========================================================

def extract_plan_object(
    plan,
):

    for action in plan.get(
        "actions",
        [],
    ):

        if action.get(
            "skill"
        ) in {
            "SEARCH",
            "APPROACH",
            "REACH",
            "GRASP",
        }:

            target = action.get(
                "target"
            )

            if target in OBJECTS:

                return target

    return None


def extract_plan_target(
    plan,
):

    for action in plan.get(
        "actions",
        [],
    ):

        if (
            action.get(
                "skill"
            )
            == "MOVE_TO"
        ):

            return action.get(
                "target"
            )

    for action in plan.get(
        "actions",
        [],
    ):

        if (
            action.get(
                "skill"
            )
            == "PLACE"
        ):

            return action.get(
                "target"
            )

    return None


def expected_action_sequence(
    object_name,
    target_name,
):

    return [

        {
            "skill":
                "SEARCH",

            "target":
                object_name,
        },

        {
            "skill":
                "APPROACH",

            "target":
                object_name,
        },

        {
            "skill":
                "REACH",

            "target":
                object_name,
        },

        {
            "skill":
                "GRASP",

            "target":
                object_name,
        },

        {
            "skill":
                "MOVE_TO",

            "target":
                target_name,
        },

        {
            "skill":
                "PLACE",

            "object":
                object_name,

            "target":
                target_name,
        },
    ]


def is_plan_correct(
    plan,
    object_name,
    target_name,
):

    if not plan.get(
        "feasible",
        False,
    ):

        return False

    return (
        plan.get(
            "actions",
            [],
        )
        == expected_action_sequence(
            object_name,
            target_name,
        )
    )


# ==========================================================
# ACTION RESULT HELPER
# ==========================================================

def get_action_result(
    report,
    skill,
):

    if report is None:

        return None

    for result in (
        report.results
    ):

        if (
            result.skill
            == skill
        ):

            return result

    return None


# ==========================================================
# FAILURE CLASSIFICATION
# ==========================================================

def determine_execution_failure_stage(
    report,
):

    if report is None:

        return "EXECUTION"

    for result in (
        report.results
    ):

        if result.success:

            continue

        skill = (
            result.skill
        )

        message = (
            result.message
            or ""
        ).lower()

        if skill == "SEARCH":

            return (
                "GROUNDING_SEARCH"
            )

        if skill == "APPROACH":

            return "APPROACH"

        if skill == "REACH":

            return "REACH"

        if skill == "GRASP":

            return "GRASP"

        if skill == "MOVE_TO":

            return "TRANSPORT"

        if skill in {
            "PLACE",
            "VERIFY",
        }:

            return "PLACE"

        # --------------------------------------------------
        # Recovery routines may ultimately return STOP.
        #
        # Classify the underlying failure rather than
        # reporting every bounded recovery as simply STOP.
        # --------------------------------------------------

        if skill == "STOP":

            if (
                "grasp"
                in message
                or "reach"
                in message
            ):

                return (
                    "GRASP_RECOVERY"
                )

            if (
                "search"
                in message
                or "not found"
                in message
            ):

                return (
                    "GROUNDING_SEARCH"
                )

            return "SAFE_STOP"

        return skill

    return "EXECUTION"


# ==========================================================
# CSV SAVE
# ==========================================================

def save_trial_csv(
    rows,
):

    if not rows:

        return

    with open(
        TRIAL_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# BALANCED OBJECT ORDER
# ==========================================================
# ==========================================================
# BALANCED FINAL TRIAL SCHEDULE
# ==========================================================

def build_trial_schedule(
    rng,
):
    """
    Build 24 balanced Task 5 trials.

    Balance:
        Objects:
            8 box
            8 cylinder
            8 stone

        Instruction styles:
            6 direct
            6 paraphrase
            6 colour
            6 informal

    Scene position and robot configuration are still
    randomly perturbed separately inside each trial.
    """

    if N_TRIALS != 24:

        raise ValueError(
            "The balanced final schedule expects "
            "N_TRIALS = 24."
        )

    # ------------------------------------------------------
    # Eight trials for each object.
    #
    # Each object receives:
    #
    # direct      x2
    # paraphrase  x2
    # colour      x2
    # informal    x2
    #
    # Across all three objects:
    #
    # direct      = 6
    # paraphrase  = 6
    # colour      = 6
    # informal    = 6
    # ------------------------------------------------------

    schedule = []

    phrase_pattern = [
        "direct",
        "direct",
        "paraphrase",
        "paraphrase",
        "colour",
        "colour",
        "informal",
        "informal",
    ]

    for object_name in OBJECTS:

        for phrase_type in phrase_pattern:

            # Find all templates for this object that
            # belong to the requested phrase category.

            candidates = [

                instruction

                for (
                    template_type,
                    instruction,
                ) in INSTRUCTION_TEMPLATES[
                    object_name
                ]

                if (
                    template_type
                    == phrase_type
                )
            ]

            if not candidates:

                raise RuntimeError(
                    f"No instruction template for "
                    f"{object_name} / {phrase_type}"
                )

            instruction = (
                candidates[
                    int(
                        rng.integers(
                            0,
                            len(
                                candidates
                            ),
                        )
                    )
                ]
            )

            schedule.append({

                "object":
                    object_name,

                "phrase_type":
                    phrase_type,

                "instruction":
                    instruction,
            })

    # Randomise trial order while preserving the
    # balanced composition.

    rng.shuffle(
        schedule
    )

    return schedule

# ==========================================================
# ONE TRIAL
# ==========================================================

def run_trial(
    trial_number,
    object_name,
    phrase_type,
    instruction,
    rng,
):

    row = make_empty_row(
        trial_number,
        object_name,
        instruction,
        phrase_type,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"TASK 5 TRIAL "
        f"{trial_number}/"
        f"{N_TRIALS}"
    )

    print(
        "=" * 75
    )

    print(
        "Object:",
        object_name,
    )

    print(
        "Instruction style:",
        phrase_type,
    )

    print(
        "Instruction:",
        instruction,
    )


    trial_start = (
        time.perf_counter()
    )

    executor = None
    report = None

    try:

        # ==================================================
        # FRESH SIMULATION
        # ==================================================

        executor = (
            Task4Executor()
        )

        robot = (
            executor.robot
        )

        if SHOW_VIEWER:

            executor.start()

        # ==================================================
        # RANDOMISE TRIAL
        # ==================================================

        reset_task_state(
            robot
        )

        randomise_object(
            robot,
            object_name,
            rng,
        )

        randomise_target_area(
            robot,
            rng,
        )

        arm_position = (
            randomise_robot(
                robot,
                rng,
            )
        )

        mujoco.mj_forward(
            robot.model,
            robot.data,
        )

        settle_scene(
            robot
        )

        # Evaluation-only state after settling.

        object_position = (
            robot.get_object_position(
                object_name
            )
        )

        target_position = (
            robot.get_target_position(
                "red_area"
            )
        )

        row.update({

            "object_start_x":
                float(
                    object_position[0]
                ),

            "object_start_y":
                float(
                    object_position[1]
                ),

            "object_start_z":
                float(
                    object_position[2]
                ),

            "target_start_x":
                float(
                    target_position[0]
                ),

            "target_start_y":
                float(
                    target_position[1]
                ),

            "target_start_z":
                float(
                    target_position[2]
                ),

            "shoulder_pan_start":
                float(
                    arm_position[0]
                ),

            "shoulder_lift_start":
                float(
                    arm_position[1]
                ),

            "elbow_start":
                float(
                    arm_position[2]
                ),

            "wrist_pitch_start":
                float(
                    arm_position[3]
                ),
        })

        # ==================================================
        # TASK 2 CALLBACK
        # ==================================================

        install_evaluation_perception_callback(
            robot,
            expected_object=object_name,
            row=row,
        )

        # ==================================================
        # TASK 3 PLANNING IMAGE
        # ==================================================

        (
            planning_image,
            _,
        ) = capture_rgbd(
            robot,
            camera_name=(
                PLANNING_CAMERA
            ),
        )

        # ==================================================
        # TASK 3
        # ==================================================

        planning_start = (
            time.perf_counter()
        )

        row[
            "task3_calls"
        ] += 1

        plan = (
            plan_from_instruction(
                instruction,
                image=planning_image,
            )
        )

        row[
            "planning_time_s"
        ] = float(
            time.perf_counter()
            - planning_start
        )

        row[
            "plan_feasible"
        ] = bool(
            plan.get(
                "feasible",
                False,
            )
        )

        row[
            "planned_object"
        ] = (
            extract_plan_object(
                plan
            )
        )

        row[
            "planned_target"
        ] = (
            extract_plan_target(
                plan
            )
        )

        row[
            "planned_actions"
        ] = str(
            plan.get(
                "actions",
                [],
            )
        )

        row[
            "planning_correct"
        ] = (
            is_plan_correct(
                plan,
                object_name,
                "red_area",
            )
        )

        # ==================================================
        # PLAN FAILURE
        # ==================================================

        if not row[
            "plan_feasible"
        ]:

            row[
                "failure_stage"
            ] = (
                "PLANNING"
            )

            actions = (
                plan.get(
                    "actions",
                    [],
                )
            )

            if actions:

                row[
                    "failure_reason"
                ] = (
                    actions[0].get(
                        "reason",
                        "Plan infeasible",
                    )
                )

            else:

                row[
                    "failure_reason"
                ] = (
                    "Plan infeasible"
                )

            return row

        if not row[
            "planning_correct"
        ]:

            row[
                "failure_stage"
            ] = (
                "PLANNING"
            )

            row[
                "failure_reason"
            ] = (
                "Generated plan did not "
                "match expected object "
                "and target"
            )

            return row

        # ==================================================
        # TASK 4 EXECUTION
        # ==================================================

        row[
            "execution_attempted"
        ] = True

        execution_start = (
            time.perf_counter()
        )

        report = (
            executor.execute_plan(
                plan
            )
        )

        row[
            "execution_time_s"
        ] = float(
            time.perf_counter()
            - execution_start
        )

        # ==================================================
        # STAGE RESULTS
        # ==================================================

        search_result = (
            get_action_result(
                report,
                "SEARCH",
            )
        )

        approach_result = (
            get_action_result(
                report,
                "APPROACH",
            )
        )

        reach_result = (
            get_action_result(
                report,
                "REACH",
            )
        )

        grasp_result = (
            get_action_result(
                report,
                "GRASP",
            )
        )

        move_result = (
            get_action_result(
                report,
                "MOVE_TO",
            )
        )

        place_result = (
            get_action_result(
                report,
                "PLACE",
            )
        )

        if search_result is not None:

            row[
                "search_success"
            ] = bool(
                search_result.success
            )

        if approach_result is not None:

            row[
                "approach_success"
            ] = bool(
                approach_result.success
            )

        if reach_result is not None:

            row[
                "reach_success"
            ] = bool(
                reach_result.success
            )

        # ==================================================
        # MANIPULATION ATTEMPT
        # ==================================================
        #
        # SEARCH failure is a perception/search failure,
        # not a manipulation failure.
        #
        # Manipulation begins only once SEARCH and APPROACH
        # succeeded and REACH/GRASP was actually entered.

        row[
            "manipulation_attempted"
        ] = bool(
            row[
                "search_success"
            ]
            and row[
                "approach_success"
            ]
            and (
                reach_result
                is not None
                or grasp_result
                is not None
            )
        )

        # ==================================================
        # GRASP
        # ==================================================

        row[
            "grasp_attempts"
        ] = int(
            report.grasp_attempts
        )

        row[
            "grasp_recovery_used"
        ] = bool(
            report.grasp_attempts
            > 1
        )

        if grasp_result is not None:

            row[
                "grasp_success"
            ] = bool(
                grasp_result.success
            )

        # ==================================================
        # TRANSPORT
        # ==================================================

        if move_result is not None:

            row[
                "move_success"
            ] = bool(
                move_result.success
            )

        # ==================================================
        # PLACE
        # ==================================================

        if place_result is not None:

            row[
                "place_success"
            ] = bool(
                place_result.success
            )

        if (
            report.final_placement_error
            is not None
        ):

            row[
                "placement_error_m"
            ] = float(
                report
                .final_placement_error
            )

        # ==================================================
        # MANIPULATION SUCCESS
        # ==================================================

        row[
            "manipulation_success"
        ] = bool(
            row[
                "manipulation_attempted"
            ]
            and row[
                "grasp_success"
            ]
            and row[
                "move_success"
            ]
            and row[
                "place_success"
            ]
            and report.success
        )

        # ==================================================
        # END-TO-END SUCCESS
        # ==================================================

        row[
            "end_to_end_success"
        ] = bool(
            row[
                "grounding_correct"
            ]
            and row[
                "planning_correct"
            ]
            and row[
                "manipulation_success"
            ]
        )

        # ==================================================
        # FAILURE CLASSIFICATION
        # ==================================================

        if not report.success:

            row[
                "failure_stage"
            ] = (
                determine_execution_failure_stage(
                    report
                )
            )

            row[
                "failure_reason"
            ] = (
                report.failure_reason
            )

        elif not row[
            "grounding_correct"
        ]:

            row[
                "failure_stage"
            ] = (
                "GROUNDING"
            )

            row[
                "failure_reason"
            ] = (
                "Task 2 did not correctly "
                "ground the intended object"
            )

        return row

    # ======================================================
    # EXCEPTION
    # ======================================================

    except Exception as error:

        row[
            "failure_stage"
        ] = (
            "UNHANDLED_EXCEPTION"
        )

        row[
            "failure_reason"
        ] = str(
            error
        )

        print(
            "\nUNHANDLED TRIAL ERROR:"
        )

        traceback.print_exc()

        return row

    # ======================================================
    # CLEANUP / TIMING
    # ======================================================

    finally:

        row[
            "total_time_s"
        ] = float(
            time.perf_counter()
            - trial_start
        )

        if (
            executor is not None
            and executor.robot.viewer
            is not None
        ):

            try:

                executor.robot.viewer.close()

            except Exception:

                pass


# ==========================================================
# RATE
# ==========================================================

def rate(
    numerator,
    denominator,
):

    if denominator == 0:

        return None

    return (
        100.0
        * numerator
        / denominator
    )


# ==========================================================
# SUMMARY CSV
# ==========================================================

def save_summary_csv(
    rows,
):

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "metric",
                "value",
            ],
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# FINAL SUMMARY
# ==========================================================

def print_summary(
    rows,
):

    total = len(
        rows
    )

    # ======================================================
    # REQUIRED METRICS
    # ======================================================

    grounding_correct = sum(
        bool(
            row[
                "grounding_correct"
            ]
        )
        for row in rows
    )

    planning_correct = sum(
        bool(
            row[
                "planning_correct"
            ]
        )
        for row in rows
    )

    manipulation_rows = [
        row
        for row in rows
        if row[
            "manipulation_attempted"
        ]
    ]

    manipulation_successes = sum(
        bool(
            row[
                "manipulation_success"
            ]
        )
        for row in manipulation_rows
    )

    e2e_successes = sum(
        bool(
            row[
                "end_to_end_success"
            ]
        )
        for row in rows
    )

    grounding_accuracy = rate(
        grounding_correct,
        total,
    )

    planning_accuracy = rate(
        planning_correct,
        total,
    )

    manipulation_rate = rate(
        manipulation_successes,
        len(
            manipulation_rows
        ),
    )

    e2e_rate = rate(
        e2e_successes,
        total,
    )

    # ======================================================
    # TIMING
    # ======================================================

    all_times = [

        float(
            row[
                "total_time_s"
            ]
        )

        for row in rows

        if row[
            "total_time_s"
        ] is not None
    ]

    success_times = [

        float(
            row[
                "total_time_s"
            ]
        )

        for row in rows

        if (
            row[
                "end_to_end_success"
            ]
            and row[
                "total_time_s"
            ]
            is not None
        )
    ]

    mean_all_time = (
        float(
            np.mean(
                all_times
            )
        )
        if all_times
        else None
    )

    mean_success_time = (
        float(
            np.mean(
                success_times
            )
        )
        if success_times
        else None
    )

    # ======================================================
    # RAW RGB-D LOCALISATION
    # ======================================================

    x_errors = [

        abs(
            float(
                row[
                    "grounding_error_x_m"
                ]
            )
        )

        for row in rows

        if row[
            "grounding_error_x_m"
        ] is not None
    ]

    y_errors = [

        abs(
            float(
                row[
                    "grounding_error_y_m"
                ]
            )
        )

        for row in rows

        if row[
            "grounding_error_y_m"
        ] is not None
    ]

    z_errors = [

        abs(
            float(
                row[
                    "grounding_error_z_m"
                ]
            )
        )

        for row in rows

        if row[
            "grounding_error_z_m"
        ] is not None
    ]

    xy_errors = [

        float(
            row[
                "grounding_xy_error_m"
            ]
        )

        for row in rows

        if row[
            "grounding_xy_error_m"
        ] is not None
    ]

    xyz_errors = [

        float(
            row[
                "grounding_localisation_error_m"
            ]
        )

        for row in rows

        if row[
            "grounding_localisation_error_m"
        ] is not None
    ]

    mean_abs_x_error = (
        float(
            np.mean(
                x_errors
            )
        )
        if x_errors
        else None
    )

    mean_abs_y_error = (
        float(
            np.mean(
                y_errors
            )
        )
        if y_errors
        else None
    )

    mean_abs_z_error = (
        float(
            np.mean(
                z_errors
            )
        )
        if z_errors
        else None
    )

    mean_xy_error = (
        float(
            np.mean(
                xy_errors
            )
        )
        if xy_errors
        else None
    )

    mean_xyz_error = (
        float(
            np.mean(
                xyz_errors
            )
        )
        if xyz_errors
        else None
    )

    # ======================================================
    # PLACEMENT
    # ======================================================

    placement_errors = [

        float(
            row[
                "placement_error_m"
            ]
        )

        for row in rows

        if (
            row[
                "place_success"
            ]
            and row[
                "placement_error_m"
            ]
            is not None
        )
    ]

    mean_placement_error = (
        float(
            np.mean(
                placement_errors
            )
        )
        if placement_errors
        else None
    )

    # ======================================================
    # RECOVERY / CALLS
    # ======================================================

    recovery_used = sum(
        bool(
            row[
                "grasp_recovery_used"
            ]
        )
        for row in rows
    )

    task2_calls = sum(
        int(
            row[
                "task2_calls"
            ]
            or 0
        )
        for row in rows
    )

    task3_calls = sum(
        int(
            row[
                "task3_calls"
            ]
            or 0
        )
        for row in rows
    )

    # ======================================================
    # PRINT
    # ======================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TASK 5 FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        "\nTotal trials:",
        total,
    )

    print(
        "\nREQUIRED METRICS"
    )

    print(
        "-" * 55
    )

    print(
        "Target Grounding Accuracy:",
        (
            f"{grounding_accuracy:.1f}% "
            f"({grounding_correct}/{total})"
            if grounding_accuracy
            is not None
            else "N/A"
        ),
    )

    print(
        "Action Planning Accuracy:",
        (
            f"{planning_accuracy:.1f}% "
            f"({planning_correct}/{total})"
            if planning_accuracy
            is not None
            else "N/A"
        ),
    )

    print(
        "Manipulation Success Rate:",
        (
            f"{manipulation_rate:.1f}% "
            f"({manipulation_successes}/"
            f"{len(manipulation_rows)})"
            if manipulation_rate
            is not None
            else "N/A"
        ),
    )

    print(
        "End-to-End Task Success Rate:",
        (
            f"{e2e_rate:.1f}% "
            f"({e2e_successes}/{total})"
            if e2e_rate
            is not None
            else "N/A"
        ),
    )

    print(
        "Average Task Completion Time "
        "(all trials):",
        (
            f"{mean_all_time:.2f} s"
            if mean_all_time
            is not None
            else "N/A"
        ),
    )

    print(
        "Average Task Completion Time "
        "(successful trials):",
        (
            f"{mean_success_time:.2f} s"
            if mean_success_time
            is not None
            else "N/A"
        ),
    )

    # ======================================================
    # LOCALISATION
    # ======================================================

    print(
        "\nRAW RGB-D LOCALISATION DIAGNOSTICS"
    )

    print(
        "-" * 55
    )

    print(
        "Mean absolute X error:",
        (
            f"{100 * mean_abs_x_error:.2f} cm"
            if mean_abs_x_error
            is not None
            else "N/A"
        ),
    )

    print(
        "Mean absolute Y error:",
        (
            f"{100 * mean_abs_y_error:.2f} cm"
            if mean_abs_y_error
            is not None
            else "N/A"
        ),
    )

    print(
        "Mean absolute Z error:",
        (
            f"{100 * mean_abs_z_error:.2f} cm"
            if mean_abs_z_error
            is not None
            else "N/A"
        ),
    )

    print(
        "Mean XY localisation error:",
        (
            f"{100 * mean_xy_error:.2f} cm"
            if mean_xy_error
            is not None
            else "N/A"
        ),
    )

    print(
        "Mean 3D localisation error:",
        (
            f"{100 * mean_xyz_error:.2f} cm"
            if mean_xyz_error
            is not None
            else "N/A"
        ),
    )

    # ======================================================
    # SUPPORTING METRICS
    # ======================================================

    print(
        "\nSUPPORTING METRICS"
    )

    print(
        "-" * 55
    )

    print(
        "Mean successful placement error:",
        (
            f"{100 * mean_placement_error:.2f} cm"
            if mean_placement_error
            is not None
            else "N/A"
        ),
    )

    print(
        "Trials using grasp recovery:",
        f"{recovery_used}/{total}",
    )

    print(
        "Task 2 calls:",
        task2_calls,
    )

    print(
        "Task 3 calls:",
        task3_calls,
    )

    # ======================================================
    # ROBUSTNESS BY OBJECT
    # ======================================================

    print(
        "\nROBUSTNESS BY OBJECT"
    )

    print(
        "-" * 55
    )

    for object_name in OBJECTS:

        subset = [
            row
            for row in rows
            if row[
                "object"
            ] == object_name
        ]

        if not subset:

            continue

        successes = sum(
            bool(
                row[
                    "end_to_end_success"
                ]
            )
            for row in subset
        )

        subset_rate = rate(
            successes,
            len(
                subset
            ),
        )

        print(
            f"{object_name:10s}: "
            f"{successes}/"
            f"{len(subset)} "
            f"({subset_rate:.1f}%)"
        )

    # ======================================================
    # ROBUSTNESS BY INSTRUCTION
    # ======================================================

    print(
        "\nROBUSTNESS BY INSTRUCTION PHRASING"
    )

    print(
        "-" * 55
    )

    phrase_types = sorted(
        {
            row[
                "phrase_type"
            ]
            for row in rows
        }
    )

    for phrase_type in phrase_types:

        subset = [
            row
            for row in rows
            if row[
                "phrase_type"
            ] == phrase_type
        ]

        successes = sum(
            bool(
                row[
                    "end_to_end_success"
                ]
            )
            for row in subset
        )

        subset_rate = rate(
            successes,
            len(
                subset
            ),
        )

        print(
            f"{phrase_type:12s}: "
            f"{successes}/"
            f"{len(subset)} "
            f"({subset_rate:.1f}%)"
        )

    # ======================================================
    # FAILURE BREAKDOWN
    # ======================================================

    print(
        "\nFAILURE BREAKDOWN"
    )

    print(
        "-" * 55
    )

    failure_counts = Counter(
        row[
            "failure_stage"
        ]
        for row in rows
        if row[
            "failure_stage"
        ]
    )

    if not failure_counts:

        print(
            "No failures observed."
        )

    else:

        for (
            stage,
            count,
        ) in sorted(
            failure_counts.items()
        ):

            print(
                f"{stage}: {count}"
            )

        print(
            "\nDetailed failures:"
        )

        for row in rows:

            if not row[
                "failure_stage"
            ]:

                continue

            print(
                f"Trial {row['trial']} | "
                f"{row['object']} | "
                f"{row['failure_stage']} | "
                f"{row['failure_reason']}"
            )

    # ======================================================
    # SUMMARY CSV
    # ======================================================

    summary_rows = [

        {
            "metric":
                "Number of trials",

            "value":
                total,
        },

        {
            "metric":
                "Target Grounding Accuracy (%)",

            "value":
                grounding_accuracy,
        },

        {
            "metric":
                "Action Planning Accuracy (%)",

            "value":
                planning_accuracy,
        },

        {
            "metric":
                "Manipulation Success Rate (%)",

            "value":
                manipulation_rate,
        },

        {
            "metric":
                "End-to-End Task Success Rate (%)",

            "value":
                e2e_rate,
        },

        {
            "metric":
                (
                    "Average Task Completion "
                    "Time - All Trials (s)"
                ),

            "value":
                mean_all_time,
        },

        {
            "metric":
                (
                    "Average Task Completion "
                    "Time - Successful Trials (s)"
                ),

            "value":
                mean_success_time,
        },

        {
            "metric":
                (
                    "Mean Absolute RGB-D "
                    "X Error (m)"
                ),

            "value":
                mean_abs_x_error,
        },

        {
            "metric":
                (
                    "Mean Absolute RGB-D "
                    "Y Error (m)"
                ),

            "value":
                mean_abs_y_error,
        },

        {
            "metric":
                (
                    "Mean Absolute RGB-D "
                    "Z Error (m)"
                ),

            "value":
                mean_abs_z_error,
        },

        {
            "metric":
                (
                    "Mean RGB-D XY "
                    "Localisation Error (m)"
                ),

            "value":
                mean_xy_error,
        },

        {
            "metric":
                (
                    "Mean RGB-D 3D "
                    "Localisation Error (m)"
                ),

            "value":
                mean_xyz_error,
        },

        {
            "metric":
                "Mean Placement Error (m)",

            "value":
                mean_placement_error,
        },

        {
            "metric":
                "Trials Using Grasp Recovery",

            "value":
                recovery_used,
        },

        {
            "metric":
                "Task 2 Calls",

            "value":
                task2_calls,
        },

        {
            "metric":
                "Task 3 Calls",

            "value":
                task3_calls,
        },
    ]

    save_summary_csv(
        summary_rows
    )

    print(
        "\nTrial CSV:"
    )

    print(
        TRIAL_CSV
    )

    print(
        "\nSummary CSV:"
    )

    print(
        SUMMARY_CSV
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "=" * 75
    )

    print(
        "TASK 5 — END-TO-END EVALUATION"
    )

    print(
        "=" * 75
    )

    print(
        "Trials:",
        N_TRIALS,
    )

    print(
        "Random seed:",
        RANDOM_SEED,
    )

    print(
        "Viewer:",
        SHOW_VIEWER,
    )

    print(
        "Planning camera:",
        PLANNING_CAMERA,
    )

    print(
        "Initial grounding camera:",
        INITIAL_GROUNDING_CAMERA,
    )

    print(
        "Search camera:",
        SEARCH_CAMERA,
    )

    print(
        "\nEvaluation variations:"
    )

    print(
        "- target object"
    )

    print(
        "- object XY position"
    )

    print(
        "- robot initial condition"
    )

    print(
        "- target-area XY position"
    )

    print(
        "- instruction phrasing"
    )

    rng = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )

    trial_schedule = (
        build_trial_schedule(
            rng
        )
    )

    rows = []

    for (
        trial_number,
        trial_config,
    ) in enumerate(
        trial_schedule,
        start=1,
    ):

        row = run_trial(
            trial_number=trial_number,
            object_name=(
                trial_config[
                    "object"
                ]
            ),
            phrase_type=(
                trial_config[
                    "phrase_type"
                ]
            ),
            instruction=(
                trial_config[
                    "instruction"
                ]
            ),
            rng=rng,
        )

        rows.append(
            row
        )

        # Preserve partial results after every trial.
        save_trial_csv(
            rows
        )

        print(
            "\n--- TRIAL RESULT ---"
        )

        print(
            "Object:",
            row[
                "object"
            ],
        )

        print(
            "Grounding correct:",
            row[
                "grounding_correct"
            ],
        )

        print(
            "Planning correct:",
            row[
                "planning_correct"
            ],
        )

        print(
            "Execution attempted:",
            row[
                "execution_attempted"
            ],
        )

        print(
            "Manipulation attempted:",
            row[
                "manipulation_attempted"
            ],
        )

        print(
            "Reach success:",
            row[
                "reach_success"
            ],
        )

        print(
            "Grasp success:",
            row[
                "grasp_success"
            ],
        )

        print(
            "Place success:",
            row[
                "place_success"
            ],
        )

        print(
            "Manipulation success:",
            row[
                "manipulation_success"
            ],
        )

        print(
            "End-to-end success:",
            row[
                "end_to_end_success"
            ],
        )

        print(
            "Completion time:",
            (
                f"{row['total_time_s']:.2f} s"
                if row[
                    "total_time_s"
                ]
                is not None
                else "N/A"
            ),
        )

        print(
            "Failure stage:",
            (
                row[
                    "failure_stage"
                ]
                or "None"
            ),
        )

        if row[
            "failure_reason"
        ]:

            print(
                "Failure reason:",
                row[
                    "failure_reason"
                ],
            )

    # ======================================================
    # SUMMARY
    # ======================================================

    print_summary(
        rows
    )


if __name__ == "__main__":

    main()