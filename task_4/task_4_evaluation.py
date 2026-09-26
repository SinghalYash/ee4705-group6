import csv
from collections import Counter
from pathlib import Path

import numpy as np
import mujoco
import traceback

from robot_skills_evaluation import (
    RobotSkills,
    GRIPPER_OPEN,
    ARM_JOINTS,
)


# ==========================================================
# EVALUATION MODE
# ==========================================================

# True:
#     Run only the first baseline trial.
#
# False:
#     Run all 12 evaluation trials.

SMOKE_TEST_ONLY = False

SHOW_VIEWER = False
REALTIME = False


# ==========================================================
# OUTPUT
# ==========================================================

RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

CSV_PATH = (
    RESULTS_DIR
    / "task_4_trial_results.csv"
)


# ==========================================================
# NOMINAL OBJECT STARTING POSITIONS
# ==========================================================

# Taken from scene.xml.

NOMINAL_OBJECT_XY = {

    "box": (
        0.32,
        -0.08,
    ),

    "cylinder": (
        0.30,
        0.08,
    ),

    "stone": (
        0.28,
        -0.18,
    ),
}


# ==========================================================
# 12 FIXED EVALUATION TRIALS
# ==========================================================

# Four trials per object.
#
# The robot and object are both varied.
#
# arm_offset order:
#
# shoulder_pan
# shoulder_lift
# elbow
# wrist_pitch

ALL_TRIAL_CONFIGS = [

    # ======================================================
    # BOX — TRIALS 1–4
    # ======================================================

    {
        "name": "box_baseline",
        "object": "box",

        "object_xy": (
            0.32,
            -0.08,
        ),

        "arm_offset": (
            0.00,
            0.00,
            0.00,
            0.00,
        ),
    },

    {
        "name": "box_shift_x",
        "object": "box",

        "object_xy": (
            0.30,
            -0.08,
        ),

        "arm_offset": (
            0.04,
            0.00,
            0.00,
            0.00,
        ),
    },

    {
        "name": "box_shift_y",
        "object": "box",

        "object_xy": (
            0.32,
            -0.10,
        ),

        "arm_offset": (
            0.00,
            -0.04,
            0.00,
            0.00,
        ),
    },

    {
        "name": "box_shift_xy",
        "object": "box",

        "object_xy": (
            0.34,
            -0.06,
        ),

        "arm_offset": (
            -0.03,
            0.03,
            0.02,
            0.00,
        ),
    },


    # ======================================================
    # CYLINDER — TRIALS 5–8
    # ======================================================

    {
        "name": "cylinder_baseline",
        "object": "cylinder",

        "object_xy": (
            0.30,
            0.08,
        ),

        "arm_offset": (
            0.00,
            0.00,
            0.00,
            0.00,
        ),
    },

    {
        "name": "cylinder_shift_x",
        "object": "cylinder",

        "object_xy": (
            0.28,
            0.08,
        ),

        "arm_offset": (
            0.04,
            0.00,
            0.00,
            0.00,
        ),
    },

    {
        "name": "cylinder_shift_y",
        "object": "cylinder",

        "object_xy": (
            0.30,
            0.06,
        ),

        "arm_offset": (
            0.00,
            0.04,
            0.00,
            0.00,
        ),
    },

    {
        "name": "cylinder_shift_xy",
        "object": "cylinder",

        "object_xy": (
            0.32,
            0.06,
        ),

        "arm_offset": (
            -0.03,
            0.03,
            -0.02,
            0.02,
        ),
    },


    # ======================================================
    # STONE — TRIALS 9–12
    # ======================================================

    {
        "name": "stone_baseline",
        "object": "stone",

        "object_xy": (
            0.28,
            -0.18,
        ),

        "arm_offset": (
            0.00,
            0.00,
            0.00,
            0.00,
        ),
    },

    {
        "name": "stone_shift_x",
        "object": "stone",

        "object_xy": (
            0.30,
            -0.18,
        ),

        "arm_offset": (
            0.03,
            0.00,
            0.00,
            0.00,
        ),
    },

    {
        "name": "stone_shift_y",
        "object": "stone",

        "object_xy": (
            0.28,
            -0.16,
        ),

        "arm_offset": (
            0.00,
            -0.03,
            0.02,
            0.00,
        ),
    },

    {
        "name": "stone_shift_xy",
        "object": "stone",

        "object_xy": (
            0.30,
            -0.16,
        ),

        "arm_offset": (
            -0.02,
            0.02,
            -0.02,
            0.02,
        ),
    },
]


if SMOKE_TEST_ONLY:

    TRIAL_CONFIGS = [
        ALL_TRIAL_CONFIGS[10]
    ]

else:

    TRIAL_CONFIGS = (
        ALL_TRIAL_CONFIGS
    )


# ==========================================================
# RESULT ROW
# ==========================================================

def make_result_row(
    trial_number,
    config,
):

    return {

        "trial": trial_number,

        "configuration": (
            config[
                "name"
            ]
        ),

        "object": (
            config[
                "object"
            ]
        ),

        "requested_object_x": (
            config[
                "object_xy"
            ][0]
        ),

        "requested_object_y": (
            config[
                "object_xy"
            ][1]
        ),

        "object_start_x": None,
        "object_start_y": None,
        "object_start_z": None,

        "shoulder_pan_start": None,
        "shoulder_lift_start": None,
        "elbow_start": None,
        "wrist_pitch_start": None,

        "approach_success": False,

        "reach_success": False,

        "grasp_success": False,

        "grasp_attempts": 0,

        "first_attempt_grasp_success": False,

        "grasp_recovery_used": False,

        "verify_grasp_success": False,

        "move_success": False,

        "place_success": False,

        "verify_place_success": False,

        "placement_error_m": None,

        "execution_success": False,

        "failure_stage": "",

        "failure_reason": "",
    }


# ==========================================================
# CONFIGURE TRIAL
# ==========================================================

def configure_trial(
    robot,
    config,
):

    object_name = (
        config[
            "object"
        ]
    )

    # ======================================================
    # OBJECT POSITION
    # ======================================================

    (
        object_qpos_adr,
        object_dof_adr,
    ) = (
        robot.get_object_freejoint_addresses(
            object_name
        )
    )

    object_x, object_y = (
        config[
            "object_xy"
        ]
    )

    nominal_position = (
        robot.get_object_position(
            object_name
        )
    )

    # Preserve nominal Z from scene.xml.
    robot.data.qpos[
        object_qpos_adr:
        object_qpos_adr + 3
    ] = np.array(
        [
            object_x,
            object_y,
            nominal_position[2],
        ],
        dtype=float,
    )

    # Reset object velocity.
    robot.data.qvel[
        object_dof_adr:
        object_dof_adr + 6
    ] = 0.0

    # ======================================================
    # ARM CONFIGURATION
    # ======================================================

    arm_offset = np.array(
        config[
            "arm_offset"
        ],
        dtype=float,
    )

    trial_joint_positions = (
        robot.initial_joint_positions
        + arm_offset
    )

    # Respect XML joint limits.
    for i, joint_name in enumerate(
        ARM_JOINTS
    ):

        joint_id = mujoco.mj_name2id(
            robot.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if robot.model.jnt_limited[
            joint_id
        ]:

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

            trial_joint_positions[i] = (
                np.clip(
                    trial_joint_positions[i],
                    lower,
                    upper,
                )
            )

    for qid, value in zip(
        robot.qpos_ids,
        trial_joint_positions,
    ):

        robot.data.qpos[
            qid
        ] = value

    robot.joint_commands = (
        trial_joint_positions.copy()
    )

    # ======================================================
    # RESET TASK STATE
    # ======================================================

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

    for finger_id in (
        robot.finger_actuator_ids
    ):

        robot.data.ctrl[
            finger_id
        ] = GRIPPER_OPEN

    mujoco.mj_forward(
        robot.model,
        robot.data,
    )

    return (
        trial_joint_positions
    )


# ==========================================================
# FAILURE HELPER
# ==========================================================

def record_failure(
    row,
    stage,
    result,
):

    row[
        "failure_stage"
    ] = stage

    row[
        "failure_reason"
    ] = (
        result.message
    )

    return row


# ==========================================================
# ONE TRIAL
# ==========================================================

def run_trial(
    trial_number,
    config,
):

    object_name = (
        config[
            "object"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"TRIAL "
        f"{trial_number}/"
        f"{len(TRIAL_CONFIGS)}"
    )

    print(
        "Configuration:",
        config[
            "name"
        ],
    )

    print(
        "Object:",
        object_name,
    )

    print(
        "=" * 70
    )

    row = make_result_row(
        trial_number,
        config,
    )

    # Fresh MuJoCo instance for every trial.
    robot = RobotSkills(
        show_viewer=SHOW_VIEWER,
        realtime=REALTIME,
    )

    robot.start_viewer()

    try:

        # ==================================================
        # INITIALISE
        # ==================================================

        trial_joint_positions = (
            configure_trial(
                robot,
                config,
            )
        )

        if not robot.settle(
            steps=500
        ):

            row[
                "failure_stage"
            ] = (
                "SETTLE_FAILED"
            )

            row[
                "failure_reason"
            ] = (
                "Simulation stopped while "
                "settling initial state"
            )

            return row

        initial_object = (
            robot.get_object_position(
                object_name
            )
        )

        row[
            "object_start_x"
        ] = float(
            initial_object[0]
        )

        row[
            "object_start_y"
        ] = float(
            initial_object[1]
        )

        row[
            "object_start_z"
        ] = float(
            initial_object[2]
        )

        row[
            "shoulder_pan_start"
        ] = float(
            trial_joint_positions[0]
        )

        row[
            "shoulder_lift_start"
        ] = float(
            trial_joint_positions[1]
        )

        row[
            "elbow_start"
        ] = float(
            trial_joint_positions[2]
        )

        row[
            "wrist_pitch_start"
        ] = float(
            trial_joint_positions[3]
        )

        print(
            "Initial object position:",
            initial_object,
        )

        print(
            "Initial arm:",
            trial_joint_positions,
        )

        # ==================================================
        # APPROACH
        # ==================================================

        result = (
            robot.approach(
                object_name
            )
        )

        row[
            "approach_success"
        ] = bool(
            result.success
        )

        if not result.success:

            return record_failure(
                row,
                "APPROACH_FAILED",
                result,
            )

        # ==================================================
        # REACH
        # ==================================================

        result = (
            robot.reach(
                object_name
            )
        )

        row[
            "reach_success"
        ] = bool(
            result.success
        )

        if not result.success:

            return record_failure(
                row,
                "REACH_FAILED",
                result,
            )

        # ==================================================
        # GRASP + RECOVERY
        # ==================================================

        result = (
            robot.grasp_with_recovery(
                object_name
            )
        )

        row[
            "grasp_success"
        ] = bool(
            result.success
        )

        row[
            "grasp_attempts"
        ] = int(
            robot.grasp_attempts
        )

        row[
            "first_attempt_grasp_success"
        ] = bool(
            result.success
            and robot.grasp_attempts == 1
        )

        row[
            "grasp_recovery_used"
        ] = bool(
            robot.grasp_attempts > 1
        )

        if not result.success:

            if (
                robot.grasp_attempts
                >= 3
            ):

                row[
                    "failure_stage"
                ] = (
                    "GRASP_RECOVERY_EXHAUSTED"
                )

            else:

                row[
                    "failure_stage"
                ] = (
                    "GRASP_FAILED"
                )

            row[
                "failure_reason"
            ] = (
                robot.last_failure_reason
                or result.message
            )

            return row

        # ==================================================
        # VERIFY GRASP
        # ==================================================

        result = (
            robot.verify_grasp(
                object_name
            )
        )

        row[
            "verify_grasp_success"
        ] = bool(
            result.success
        )

        if not result.success:

            row[
                "grasp_success"
            ] = False

            row[
                "first_attempt_grasp_success"
            ] = False

            return record_failure(
                row,
                "VERIFY_GRASP_FAILED",
                result,
            )

        # ==================================================
        # MOVE TO TARGET
        # ==================================================

        result = (
            robot.move_to(
                "red_area"
            )
        )

        row[
            "move_success"
        ] = bool(
            result.success
        )

        if not result.success:

            return record_failure(
                row,
                "TRANSPORT_FAILED",
                result,
            )

        # ==================================================
        # PLACE SELECTED OBJECT
        # ==================================================

        result = (
            robot.place(
                object_name,
                "red_area",
            )
        )

        row[
            "place_success"
        ] = bool(
            result.success
        )

        if result.error is not None:

            row[
                "placement_error_m"
            ] = float(
                result.error
            )

        if not result.success:

            return record_failure(
                row,
                "PLACE_FAILED",
                result,
            )

        # ==================================================
        # FINAL PLACE VERIFICATION
        # ==================================================

        result = (
            robot.verify_place(
                object_name,
                "red_area",
            )
        )

        row[
            "verify_place_success"
        ] = bool(
            result.success
        )

        if result.error is not None:

            row[
                "placement_error_m"
            ] = float(
                result.error
            )

        if not result.success:

            row[
                "place_success"
            ] = False

            return record_failure(
                row,
                "PLACE_OUTSIDE_TARGET",
                result,
            )

        # ==================================================
        # SUCCESS
        # ==================================================

        row[
            "execution_success"
        ] = True

        return row

    finally:

        robot.close_viewer()


# ==========================================================
# CSV
# ==========================================================

def save_csv(
    rows,
):

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ==========================================================
# PER-OBJECT SUMMARY
# ==========================================================

def print_object_summary(
    rows,
    object_name,
):

    object_rows = [
        row
        for row in rows
        if (
            row[
                "object"
            ]
            == object_name
        )
    ]

    total = len(
        object_rows
    )

    if total == 0:
        return

    grasp_successes = sum(
        bool(
            row[
                "grasp_success"
            ]
        )
        for row in object_rows
    )

    place_successes = sum(
        bool(
            row[
                "place_success"
            ]
        )
        for row in object_rows
    )

    first_attempt = sum(
        bool(
            row[
                "first_attempt_grasp_success"
            ]
        )
        for row in object_rows
    )

    errors = np.array(
        [
            float(
                row[
                    "placement_error_m"
                ]
            )
            for row in object_rows
            if (
                row[
                    "place_success"
                ]
                and row[
                    "placement_error_m"
                ]
                is not None
            )
        ],
        dtype=float,
    )

    print(
        f"\n{object_name.upper()}"
    )

    print(
        "-" * 40
    )

    print(
        "Trials:",
        total,
    )

    print(
        "Grasp success:",
        (
            f"{100 * grasp_successes / total:.1f}% "
            f"({grasp_successes}/{total})"
        ),
    )

    print(
        "First-attempt grasp:",
        (
            f"{100 * first_attempt / total:.1f}% "
            f"({first_attempt}/{total})"
        ),
    )

    print(
        "Place success:",
        (
            f"{100 * place_successes / total:.1f}% "
            f"({place_successes}/{total})"
        ),
    )

    if errors.size > 0:

        print(
            "Mean placement error:",
            (
                f"{100 * np.mean(errors):.2f} cm"
            ),
        )

        print(
            "Maximum placement error:",
            (
                f"{100 * np.max(errors):.2f} cm"
            ),
        )

    else:

        print(
            "Mean placement error: N/A"
        )


# ==========================================================
# OVERALL SUMMARY
# ==========================================================

def print_summary(
    rows,
):

    total = len(
        rows
    )

    if total == 0:
        return

    grasp_successes = sum(
        bool(
            row[
                "grasp_success"
            ]
        )
        for row in rows
    )

    first_attempt_grasps = sum(
        bool(
            row[
                "first_attempt_grasp_success"
            ]
        )
        for row in rows
    )

    place_successes = sum(
        bool(
            row[
                "place_success"
            ]
        )
        for row in rows
    )

    execution_successes = sum(
        bool(
            row[
                "execution_success"
            ]
        )
        for row in rows
    )

    recovery_trials = sum(
        bool(
            row[
                "grasp_recovery_used"
            ]
        )
        for row in rows
    )

    placement_errors = np.array(
        [
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
        ],
        dtype=float,
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

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TASK 4 EVALUATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nTotal trials:",
        total,
    )

    print(
        "\nOVERALL RESULTS"
    )

    print(
        "-" * 40
    )

    print(
        "Grasp Success Rate:",
        (
            f"{100 * grasp_successes / total:.1f}% "
            f"({grasp_successes}/{total})"
        ),
    )

    print(
        "First-Attempt Grasp Rate:",
        (
            f"{100 * first_attempt_grasps / total:.1f}% "
            f"({first_attempt_grasps}/{total})"
        ),
    )

    print(
        "Place Success Rate:",
        (
            f"{100 * place_successes / total:.1f}% "
            f"({place_successes}/{total})"
        ),
    )

    if grasp_successes > 0:

        print(
            "Conditional Place Success:",
            (
                f"{100 * place_successes / grasp_successes:.1f}% "
                f"({place_successes}/{grasp_successes})"
            ),
        )

    print(
        "Overall Execution Success:",
        (
            f"{100 * execution_successes / total:.1f}% "
            f"({execution_successes}/{total})"
        ),
    )

    print(
        "Trials requiring grasp recovery:",
        (
            f"{recovery_trials}/{total}"
        ),
    )

    if placement_errors.size > 0:

        print(
            "\nPLACEMENT ERROR"
        )

        print(
            "-" * 40
        )

        print(
            "Mean:",
            (
                f"{100 * np.mean(placement_errors):.2f} cm"
            ),
        )

        print(
            "Standard deviation:",
            (
                f"{100 * np.std(placement_errors):.2f} cm"
            ),
        )

        print(
            "Minimum:",
            (
                f"{100 * np.min(placement_errors):.2f} cm"
            ),
        )

        print(
            "Maximum:",
            (
                f"{100 * np.max(placement_errors):.2f} cm"
            ),
        )

    # ======================================================
    # BY OBJECT
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RESULTS BY OBJECT"
    )

    print(
        "=" * 70
    )

    for object_name in [
        "box",
        "cylinder",
        "stone",
    ]:

        print_object_summary(
            rows,
            object_name,
        )

    # ======================================================
    # FAILURE BREAKDOWN
    # ======================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TYPICAL EXECUTION FAILURES"
    )

    print(
        "=" * 70
    )

    if not failure_counts:

        print(
            "No failures observed."
        )

    else:

        for (
            failure_type,
            count,
        ) in sorted(
            failure_counts.items()
        ):

            print(
                f"{failure_type}: "
                f"{count}"
            )

        print(
            "\nDetailed failures:"
        )

        for row in rows:

            if row[
                "failure_stage"
            ]:

                print(
                    f"Trial "
                    f"{row['trial']} | "
                    f"{row['object']} | "
                    f"{row['failure_stage']} | "
                    f"{row['failure_reason']}"
                )

    print(
        "\nResults CSV:"
    )

    print(
        CSV_PATH
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "=" * 70
    )

    print(
        "TASK 4 — MULTI-OBJECT "
        "MANIPULATION EVALUATION"
    )

    print(
        "=" * 70
    )

    if SMOKE_TEST_ONLY:

        print(
            "\nMODE: ONE-TRIAL SMOKE TEST"
        )

    else:

        print(
            "\nMODE: FULL 12-TRIAL EVALUATION"
        )

    print(
        "Trials:",
        len(
            TRIAL_CONFIGS
        ),
    )

    print(
        "MuJoCo GUI:",
        SHOW_VIEWER,
    )

    rows = []

    for (
        trial_number,
        config,
    ) in enumerate(
        TRIAL_CONFIGS,
        start=1,
    ):

        try:

            row = run_trial(
                trial_number,
                config,
            )

        

        except Exception as error:

            print(
                "\nUNHANDLED ERROR "
                f"DURING TRIAL "
                f"{trial_number}:"
            )

            print(
                error
            )

            print(
                "\nFULL TRACEBACK:"
            )

            traceback.print_exc()

            row = make_result_row(
                trial_number,
                config,
            )

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

        rows.append(
            row
        )

        # Preserve partial experiment results.
        save_csv(
            rows
        )

        print(
            "\n--- TRIAL "
            f"{trial_number} RESULT ---"
        )

        print(
            "Object:",
            row[
                "object"
            ],
        )

        print(
            "Grasp success:",
            row[
                "grasp_success"
            ],
        )

        print(
            "Grasp attempts:",
            row[
                "grasp_attempts"
            ],
        )

        print(
            "Place success:",
            row[
                "place_success"
            ],
        )

        print(
            "Placement error:",
            row[
                "placement_error_m"
            ],
        )

        print(
            "Execution success:",
            row[
                "execution_success"
            ],
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

    print_summary(
        rows
    )


if __name__ == "__main__":
    main()