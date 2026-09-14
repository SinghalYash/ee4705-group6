import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"


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
# GRIPPER SETTINGS
# ==========================================================

# q = 0.025 -> fully open
# q = 0.0   -> fully closed
GRIPPER_OPEN = 0.025
GRIPPER_CLOSED = 0.0


# ==========================================================
# CARTESIAN CONTROLLER SETTINGS
# ==========================================================

POSITION_TOLERANCE = 0.01
MAX_DQ = 0.004
DAMPING = 0.08


# ==========================================================
# OBJECT / TARGET SETTINGS
# ==========================================================

# Blue box is 6 cm x 6 cm x 6 cm.
BOX_HALF_HEIGHT = 0.03
BOX_HALF_WIDTH = 0.03

# Red target radius from scene.xml.
TARGET_RADIUS = 0.12

# Release box slightly above the floor.
RELEASE_CLEARANCE = 0.008


# ==========================================================
# CONTACT DETECTION
# ==========================================================

def get_finger_contacts(model, data):
    """
    Return contact state of the three fingers with box_geom.

    Example:
        [True, False, True]
    """

    contacts = [
        False,
        False,
        False,
    ]

    for i in range(data.ncon):

        contact = data.contact[i]

        geom1 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )

        geom2 = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )

        pair = {
            geom1,
            geom2,
        }

        if "box_geom" not in pair:
            continue

        for j, finger_geom in enumerate(
            FINGER_GEOMS
        ):

            if finger_geom in pair:
                contacts[j] = True

    return contacts


# ==========================================================
# FREE-JOINT HELPERS
# ==========================================================

def get_freejoint_qpos_address(
    model,
    body_name,
):
    """
    Return qpos address for a body's free joint.
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    joint_id = model.body_jntadr[
        body_id
    ]

    return model.jnt_qposadr[
        joint_id
    ]


def get_freejoint_dof_address(
    model,
    body_name,
):
    """
    Return qvel / DoF address for a body's free joint.
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    joint_id = model.body_jntadr[
        body_id
    ]

    return model.jnt_dofadr[
        joint_id
    ]


# ==========================================================
# KINEMATIC CARRY
# ==========================================================

def update_carried_object(
    model,
    data,
    site_id,
    object_qpos_adr,
    object_dof_adr,
    object_offset,
):
    """
    Keep the box at the same XYZ offset from grasp_site.

    Used only AFTER Stage 5 confirms a stable grasp.
    """

    mujoco.mj_forward(
        model,
        data,
    )

    grasp_position = data.site_xpos[
        site_id
    ].copy()

    desired_object_position = (
        grasp_position
        + object_offset
    )

    # Free-joint qpos:
    # [x, y, z, qw, qx, qy, qz]
    #
    # Change XYZ only and preserve orientation.
    data.qpos[
        object_qpos_adr:
        object_qpos_adr + 3
    ] = desired_object_position

    # Remove residual linear/angular velocity.
    data.qvel[
        object_dof_adr:
        object_dof_adr + 6
    ] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


# ==========================================================
# CLOSED-LOOP CARTESIAN MOVEMENT
# ==========================================================

def move_cartesian(
    model,
    data,
    viewer,
    site_id,
    target_pos,
    qpos_ids,
    dof_ids,
    actuator_ids,
    finger_ids,
    joint_commands,
    label,
    tolerance=POSITION_TOLERANCE,
    max_steps=2500,
    gripper_command=GRIPPER_OPEN,
    carry_object=False,
    object_qpos_adr=None,
    object_dof_adr=None,
    object_offset=None,
):
    """
    Move grasp_site toward a Cartesian target using
    damped least-squares Jacobian control.

    carry_object=True makes a successfully grasped box
    follow grasp_site during transport.
    """

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

        if not viewer.is_running():

            return (
                joint_commands,
                False,
            )

        # --------------------------------------------------
        # Measure physical grasp-site position
        # --------------------------------------------------

        mujoco.mj_forward(
            model,
            data,
        )

        current_pos = data.site_xpos[
            site_id
        ].copy()

        error = (
            target_pos
            - current_pos
        )

        error_norm = np.linalg.norm(
            error
        )

        # --------------------------------------------------
        # Success
        # --------------------------------------------------

        if error_norm < tolerance:

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

            return (
                joint_commands,
                True,
            )

        # --------------------------------------------------
        # Jacobian
        # --------------------------------------------------

        jacp = np.zeros(
            (3, model.nv)
        )

        jacr = np.zeros(
            (3, model.nv)
        )

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J = jacp[
            :,
            dof_ids,
        ]

        # --------------------------------------------------
        # Damped least-squares IK
        # --------------------------------------------------

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

        joint_commands += dq

        # --------------------------------------------------
        # Joint limits
        # --------------------------------------------------

        for i, joint_name in enumerate(
            ARM_JOINTS
        ):

            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if model.jnt_limited[
                joint_id
            ]:

                lower = model.jnt_range[
                    joint_id,
                    0,
                ]

                upper = model.jnt_range[
                    joint_id,
                    1,
                ]

                joint_commands[i] = np.clip(
                    joint_commands[i],
                    lower,
                    upper,
                )

        # --------------------------------------------------
        # Arm commands
        # --------------------------------------------------

        for actuator_id, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[
                actuator_id
            ] = command

        # --------------------------------------------------
        # Gripper command
        # --------------------------------------------------

        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = gripper_command

        # --------------------------------------------------
        # Physics
        # --------------------------------------------------

        for _ in range(5):

            mujoco.mj_step(
                model,
                data,
            )

            if carry_object:

                update_carried_object(
                    model,
                    data,
                    site_id,
                    object_qpos_adr,
                    object_dof_adr,
                    object_offset,
                )

        viewer.sync()

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"error={error_norm:.4f} m | "
                f"site={np.round(current_pos, 3)}"
            )

        time.sleep(
            model.opt.timestep
            * 5
        )

    # ------------------------------------------------------
    # Failed
    # ------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )

    current_pos = data.site_xpos[
        site_id
    ].copy()

    final_error = np.linalg.norm(
        target_pos
        - current_pos
    )

    print(
        f"\n{label} FAILED"
    )

    print(
        "Actual:",
        current_pos,
    )

    print(
        "Final error:",
        final_error,
        "m",
    )

    return (
        joint_commands,
        False,
    )


# ==========================================================
# STAGE 5 — GRASP
# ==========================================================

def close_gripper_until_contact(
    model,
    data,
    viewer,
    finger_ids,
    actuator_ids,
    joint_commands,
    object_id,
    grasp_site_id,
    duration=4.0,
):
    """
    Slowly close all three fingers.

    Stop when:
      - at least two fingers contact the box, and
      - the box is within 3 cm of grasp_site.

    Then verify stable contact.
    """

    print(
        "\n" + "=" * 55
    )

    print(
        "STAGE 5: 3-PRONG GRASP"
    )

    print(
        "=" * 55
    )

    total_steps = int(
        duration
        / model.opt.timestep
    )

    contact_command = None

    print(
        "\nClosing all three prongs..."
    )

    # ======================================================
    # PHASE A — CLOSE
    # ======================================================

    for step in range(
        total_steps
    ):

        if not viewer.is_running():

            return (
                False,
                GRIPPER_OPEN,
            )

        # Hold arm.
        for actuator_id, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[
                actuator_id
            ] = command

        # Slowly close.
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

        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = finger_command

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        mujoco.mj_forward(
            model,
            data,
        )

        contacts = get_finger_contacts(
            model,
            data,
        )

        number_contacts = sum(
            contacts
        )

        object_now = data.xpos[
            object_id
        ].copy()

        grasp_now = data.site_xpos[
            grasp_site_id
        ].copy()

        object_distance = np.linalg.norm(
            object_now
            - grasp_now
        )

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"finger={finger_command:.4f} | "
                f"contacts={contacts} | "
                f"n={number_contacts} | "
                f"object_dist="
                f"{object_distance:.4f}"
            )

        # Stop squeezing immediately at useful capture.
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
                "Box distance:",
                object_distance,
            )

            print(
                "Freezing finger command at:",
                contact_command,
            )

            break

        if object_distance > 0.10:

            print(
                "\nGRASP ABORTED: "
                "box moved too far "
                "from grasp site."
            )

            return (
                False,
                GRIPPER_OPEN,
            )

        time.sleep(
            model.opt.timestep
        )

    if contact_command is None:

        print(
            "\nGRASP FAILED: "
            "useful capture was never achieved."
        )

        return (
            False,
            GRIPPER_OPEN,
        )

    # ======================================================
    # PHASE B — STABILITY
    # ======================================================

    print(
        "\nChecking grasp stability..."
    )

    contact_counts = [
        0,
        0,
        0,
    ]

    required_contact_count = 15
    max_stability_steps = 400
    max_stable_distance = 0.04

    for step in range(
        max_stability_steps
    ):

        if not viewer.is_running():

            return (
                False,
                GRIPPER_OPEN,
            )

        # Hold arm.
        for actuator_id, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[
                actuator_id
            ] = command

        # Hold captured finger spacing.
        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = contact_command

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        mujoco.mj_forward(
            model,
            data,
        )

        contacts = get_finger_contacts(
            model,
            data,
        )

        for i, touching in enumerate(
            contacts
        ):

            if touching:
                contact_counts[i] += 1

        object_now = data.xpos[
            object_id
        ].copy()

        grasp_now = data.site_xpos[
            grasp_site_id
        ].copy()

        object_distance = np.linalg.norm(
            object_now
            - grasp_now
        )

        if step % 25 == 0:

            print(
                f"stability {step:3d} | "
                f"contacts={contacts} | "
                f"counts={contact_counts} | "
                f"object_dist="
                f"{object_distance:.4f}"
            )

        if (
            object_distance
            > max_stable_distance
        ):

            print(
                "\nGRASP FAILED: "
                "box moved outside "
                "stable grasp region."
            )

            return (
                False,
                GRIPPER_OPEN,
            )

        successful_prongs = sum(
            count
            >= required_contact_count
            for count
            in contact_counts
        )

        if successful_prongs >= 2:

            print(
                "\nSTABLE GRASP CONFIRMED"
            )

            print(
                "Finger command:",
                contact_command,
            )

            print(
                "Contact samples:",
                contact_counts,
            )

            print(
                "Box distance:",
                object_distance,
            )

            return (
                True,
                contact_command,
            )

        time.sleep(
            model.opt.timestep
        )

    print(
        "\nGRASP FAILED: "
        "insufficient stable contact."
    )

    return (
        False,
        GRIPPER_OPEN,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(
        model
    )

    # ------------------------------------------------------
    # Carry state
    # ------------------------------------------------------

    carrying_box = False

    object_offset = None
    object_qpos_adr = None
    object_dof_adr = None

    # ------------------------------------------------------
    # IDs
    # ------------------------------------------------------

    grasp_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "grasp_site",
    )

    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "box_obj",
    )

    target_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "target_area",
    )

    # ------------------------------------------------------
    # Arm addresses
    # ------------------------------------------------------

    qpos_ids = []
    dof_ids = []

    for joint_name in ARM_JOINTS:

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_ids.append(
            model.jnt_qposadr[
                joint_id
            ]
        )

        dof_ids.append(
            model.jnt_dofadr[
                joint_id
            ]
        )

    actuator_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in ARM_ACTUATORS
    ]

    finger_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in FINGER_ACTUATORS
    ]

    # ------------------------------------------------------
    # Initial state
    # ------------------------------------------------------

    mujoco.mj_forward(
        model,
        data,
    )

    # Save exact initial robot state for Stage 10.
    initial_grasp_position = (
        data.site_xpos[
            grasp_site_id
        ].copy()
    )

    initial_joint_positions = np.array([
        data.qpos[qid]
        for qid in qpos_ids
    ])

    object_pos = data.xpos[
        object_id
    ].copy()

    target_pos = data.geom_xpos[
        target_geom_id
    ].copy()

    print(
        "Initial grasp-site position:",
        initial_grasp_position,
    )

    print(
        "Initial arm joint positions:",
        initial_joint_positions,
    )

    print(
        "Box position:",
        object_pos,
    )

    print(
        "Target area centre:",
        target_pos,
    )

    # ======================================================
    # APPROACH WAYPOINTS
    # ======================================================

    safe_target = np.array([
        0.22,
        -0.12,
        0.34,
    ])

    above_box = np.array([
        object_pos[0],
        object_pos[1],
        0.34,
    ])

    approach_target = np.array([
        object_pos[0],
        object_pos[1],
        0.22,
    ])

    pregrasp_target = np.array([
        object_pos[0],
        object_pos[1],
        object_pos[2] + 0.025,
    ])

    joint_commands = np.array([
        data.qpos[qid]
        for qid in qpos_ids
    ])

    final_gripper_command = (
        GRIPPER_OPEN
    )

    # ======================================================
    # VIEWER
    # ======================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Start open.
        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = GRIPPER_OPEN

        # ==================================================
        # STAGES 1–4 — APPROACH
        # ==================================================

        stages = [
            (
                "STAGE 1: SAFE POSE",
                safe_target,
                0.01,
                2500,
            ),
            (
                "STAGE 2: ABOVE BOX",
                above_box,
                0.01,
                2500,
            ),
            (
                "STAGE 3: DESCEND",
                approach_target,
                0.01,
                2500,
            ),
            (
                "STAGE 4: PRE-GRASP",
                pregrasp_target,
                0.005,
                3000,
            ),
        ]

        ok = True

        for (
            label,
            target,
            tolerance,
            max_steps,
        ) in stages:

            (
                joint_commands,
                ok,
            ) = move_cartesian(
                model,
                data,
                viewer,
                grasp_site_id,
                target,
                qpos_ids,
                dof_ids,
                actuator_ids,
                finger_ids,
                joint_commands,
                label,
                tolerance=tolerance,
                max_steps=max_steps,
            )

            if not ok:

                print(
                    "\nStopping before grasp "
                    "because movement failed."
                )

                break

        # ==================================================
        # STAGE 5 — GRASP
        # ==================================================

        if ok:

            mujoco.mj_forward(
                model,
                data,
            )

            print(
                "\n--- PRE-GRASP CHECK ---"
            )

            print(
                "Box centre:",
                data.xpos[
                    object_id
                ].copy(),
            )

            print(
                "Grasp site:",
                data.site_xpos[
                    grasp_site_id
                ].copy(),
            )

            (
                grasp_ok,
                final_gripper_command,
            ) = close_gripper_until_contact(
                model,
                data,
                viewer,
                finger_ids,
                actuator_ids,
                joint_commands,
                object_id,
                grasp_site_id,
            )

            # ==============================================
            # STAGE 6 — LIFT
            # ==============================================

            if grasp_ok:

                print(
                    "\n================================="
                )

                print(
                    "STAGE 6: LIFT BOX"
                )

                print(
                    "================================="
                )

                mujoco.mj_forward(
                    model,
                    data,
                )

                box_before_lift = data.xpos[
                    object_id
                ].copy()

                grasp_before_lift = (
                    data.site_xpos[
                        grasp_site_id
                    ].copy()
                )

                object_offset = (
                    box_before_lift
                    - grasp_before_lift
                )

                object_qpos_adr = (
                    get_freejoint_qpos_address(
                        model,
                        "box_obj",
                    )
                )

                object_dof_adr = (
                    get_freejoint_dof_address(
                        model,
                        "box_obj",
                    )
                )

                carrying_box = True

                lift_target = (
                    grasp_before_lift.copy()
                )

                lift_target[2] += 0.10

                (
                    joint_commands,
                    lift_ok,
                ) = move_cartesian(
                    model,
                    data,
                    viewer,
                    grasp_site_id,
                    lift_target,
                    qpos_ids,
                    dof_ids,
                    actuator_ids,
                    finger_ids,
                    joint_commands,
                    label="STAGE 6: LIFT",
                    tolerance=0.01,
                    max_steps=3000,
                    gripper_command=(
                        final_gripper_command
                    ),
                    carry_object=True,
                    object_qpos_adr=(
                        object_qpos_adr
                    ),
                    object_dof_adr=(
                        object_dof_adr
                    ),
                    object_offset=(
                        object_offset
                    ),
                )

                mujoco.mj_forward(
                    model,
                    data,
                )

                box_after_lift = data.xpos[
                    object_id
                ].copy()

                grasp_after_lift = (
                    data.site_xpos[
                        grasp_site_id
                    ].copy()
                )

                box_lift_amount = (
                    box_after_lift[2]
                    - box_before_lift[2]
                )

                gripper_lift_amount = (
                    grasp_after_lift[2]
                    - grasp_before_lift[2]
                )

                lift_tracking_error = abs(
                    box_lift_amount
                    - gripper_lift_amount
                )

                print(
                    "\n--- LIFT RESULT ---"
                )

                print(
                    "Box vertical displacement:",
                    box_lift_amount,
                )

                print(
                    "Gripper vertical displacement:",
                    gripper_lift_amount,
                )

                print(
                    "Lift tracking difference:",
                    lift_tracking_error,
                )

                lift_verified = (
                    lift_ok
                    and box_lift_amount > 0.07
                    and lift_tracking_error < 0.01
                )

                if lift_verified:

                    print(
                        "\n================================="
                    )

                    print(
                        "BOX LIFT SUCCESSFUL"
                    )

                    print(
                        "================================="
                    )

                    # ======================================
                    # STAGE 7A — CLEARANCE
                    # ======================================

                    mujoco.mj_forward(
                        model,
                        data,
                    )

                    current_grasp = (
                        data.site_xpos[
                            grasp_site_id
                        ].copy()
                    )

                    transport_height = max(
                        current_grasp[2],
                        0.20,
                    )

                    clearance_target = np.array([
                        current_grasp[0],
                        current_grasp[1],
                        transport_height,
                    ])

                    (
                        joint_commands,
                        clearance_ok,
                    ) = move_cartesian(
                        model,
                        data,
                        viewer,
                        grasp_site_id,
                        clearance_target,
                        qpos_ids,
                        dof_ids,
                        actuator_ids,
                        finger_ids,
                        joint_commands,
                        label=(
                            "STAGE 7A: CLEARANCE"
                        ),
                        tolerance=0.01,
                        max_steps=2500,
                        gripper_command=(
                            final_gripper_command
                        ),
                        carry_object=True,
                        object_qpos_adr=(
                            object_qpos_adr
                        ),
                        object_dof_adr=(
                            object_dof_adr
                        ),
                        object_offset=(
                            object_offset
                        ),
                    )

                    # ======================================
                    # STAGE 7B — ABOVE TARGET
                    # ======================================

                    if clearance_ok:

                        above_target = np.array([
                            target_pos[0],
                            target_pos[1],
                            transport_height,
                        ])

                        (
                            joint_commands,
                            transport_ok,
                        ) = move_cartesian(
                            model,
                            data,
                            viewer,
                            grasp_site_id,
                            above_target,
                            qpos_ids,
                            dof_ids,
                            actuator_ids,
                            finger_ids,
                            joint_commands,
                            label=(
                                "STAGE 7B: "
                                "TRANSPORT ABOVE TARGET"
                            ),
                            tolerance=0.01,
                            max_steps=3500,
                            gripper_command=(
                                final_gripper_command
                            ),
                            carry_object=True,
                            object_qpos_adr=(
                                object_qpos_adr
                            ),
                            object_dof_adr=(
                                object_dof_adr
                            ),
                            object_offset=(
                                object_offset
                            ),
                        )

                    else:

                        transport_ok = False

                    # ======================================
                    # STAGE 8 — LOWER
                    # ======================================

                    if transport_ok:

                        print(
                            "\n================================="
                        )

                        print(
                            "STAGE 8: LOWER BOX"
                        )

                        print(
                            "================================="
                        )

                        desired_box_z = (
                            BOX_HALF_HEIGHT
                            + RELEASE_CLEARANCE
                        )

                        desired_grasp_z = (
                            desired_box_z
                            - object_offset[2]
                        )

                        # Correct for XYZ offset so that
                        # BOX CENTRE lands on target centre.
                        release_target = np.array([
                            target_pos[0]
                            - object_offset[0],

                            target_pos[1]
                            - object_offset[1],

                            desired_grasp_z,
                        ])

                        print(
                            "Release target:",
                            release_target,
                        )

                        (
                            joint_commands,
                            lower_ok,
                        ) = move_cartesian(
                            model,
                            data,
                            viewer,
                            grasp_site_id,
                            release_target,
                            qpos_ids,
                            dof_ids,
                            actuator_ids,
                            finger_ids,
                            joint_commands,
                            label=(
                                "STAGE 8: LOWER"
                            ),
                            tolerance=0.006,
                            max_steps=3500,
                            gripper_command=(
                                final_gripper_command
                            ),
                            carry_object=True,
                            object_qpos_adr=(
                                object_qpos_adr
                            ),
                            object_dof_adr=(
                                object_dof_adr
                            ),
                            object_offset=(
                                object_offset
                            ),
                        )

                    else:

                        lower_ok = False

                    # ======================================
                    # STAGE 9 — RELEASE
                    # ======================================

                    if lower_ok:

                        print(
                            "\n================================="
                        )

                        print(
                            "STAGE 9: RELEASE BOX"
                        )

                        print(
                            "================================="
                        )

                        mujoco.mj_forward(
                            model,
                            data,
                        )

                        print(
                            "Box before release:",
                            data.xpos[
                                object_id
                            ].copy(),
                        )

                        # Stop kinematic carrying.
                        carrying_box = False

                        # Remove residual velocity.
                        data.qvel[
                            object_dof_adr:
                            object_dof_adr + 6
                        ] = 0.0

                        # ----------------------------------
                        # Slowly open fingers
                        # ----------------------------------

                        release_steps = 600

                        for release_step in range(
                            release_steps
                        ):

                            if not viewer.is_running():
                                break

                            alpha = (
                                release_step + 1
                            ) / release_steps

                            finger_command = (
                                final_gripper_command
                                + alpha
                                * (
                                    GRIPPER_OPEN
                                    - final_gripper_command
                                )
                            )

                            # Hold arm.
                            for (
                                actuator_id,
                                command,
                            ) in zip(
                                actuator_ids,
                                joint_commands,
                            ):

                                data.ctrl[
                                    actuator_id
                                ] = command

                            # Open fingers.
                            for finger_id in finger_ids:

                                data.ctrl[
                                    finger_id
                                ] = finger_command

                            mujoco.mj_step(
                                model,
                                data,
                            )

                            viewer.sync()

                            if (
                                release_step
                                % 100
                                == 0
                            ):

                                mujoco.mj_forward(
                                    model,
                                    data,
                                )

                                print(
                                    f"release "
                                    f"{release_step:3d} | "
                                    f"finger="
                                    f"{finger_command:.4f} | "
                                    f"box="
                                    f"{np.round(data.xpos[object_id], 3)}"
                                )

                            time.sleep(
                                model.opt.timestep
                            )

                        final_gripper_command = (
                            GRIPPER_OPEN
                        )

                        # ----------------------------------
                        # Let box settle
                        # ----------------------------------

                        print(
                            "\nAllowing box "
                            "to settle..."
                        )

                        settle_steps = 1000

                        for _ in range(
                            settle_steps
                        ):

                            if not viewer.is_running():
                                break

                            for (
                                actuator_id,
                                command,
                            ) in zip(
                                actuator_ids,
                                joint_commands,
                            ):

                                data.ctrl[
                                    actuator_id
                                ] = command

                            for finger_id in finger_ids:

                                data.ctrl[
                                    finger_id
                                ] = GRIPPER_OPEN

                            mujoco.mj_step(
                                model,
                                data,
                            )

                            viewer.sync()

                            time.sleep(
                                model.opt.timestep
                            )

                        # ----------------------------------
                        # Verify placement
                        # ----------------------------------

                        mujoco.mj_forward(
                            model,
                            data,
                        )

                        box_final = data.xpos[
                            object_id
                        ].copy()

                        target_xy_error = (
                            np.linalg.norm(
                                box_final[:2]
                                - target_pos[:2]
                            )
                        )

                        placement_limit = (
                            TARGET_RADIUS
                            - BOX_HALF_WIDTH
                        )

                        print(
                            "\n--- PLACEMENT RESULT ---"
                        )

                        print(
                            "Target centre:",
                            target_pos,
                        )

                        print(
                            "Final box position:",
                            box_final,
                        )

                        print(
                            "XY distance from "
                            "target centre:",
                            target_xy_error,
                            "m",
                        )

                        print(
                            "Placement acceptance "
                            "radius:",
                            placement_limit,
                            "m",
                        )

                        # ==================================
                        # PLACEMENT SUCCESS
                        # ==================================

                        if (
                            target_xy_error
                            < placement_limit
                        ):

                            print(
                                "\n================================="
                            )

                            print(
                                "BOX PLACEMENT "
                                "SUCCESSFUL"
                            )

                            print(
                                "================================="
                            )

                            # ==================================
                            # STAGE 10 — RETURN HOME
                            # ==================================

                            print(
                                "\n================================="
                            )

                            print(
                                "STAGE 10: RETURN HOME"
                            )

                            print(
                                "================================="
                            )

                            # ------------------------------
                            # Stage 10A — RETREAT UPWARD
                            # ------------------------------

                            mujoco.mj_forward(
                                model,
                                data,
                            )

                            current_grasp = (
                                data.site_xpos[
                                    grasp_site_id
                                ].copy()
                            )

                            retreat_target = (
                                current_grasp.copy()
                            )

                            retreat_target[2] += (
                                0.15
                            )

                            print(
                                "Retreat target:",
                                retreat_target,
                            )

                            (
                                joint_commands,
                                retreat_ok,
                            ) = move_cartesian(
                                model,
                                data,
                                viewer,
                                grasp_site_id,
                                retreat_target,
                                qpos_ids,
                                dof_ids,
                                actuator_ids,
                                finger_ids,
                                joint_commands,
                                label=(
                                    "STAGE 10A: "
                                    "RETREAT"
                                ),
                                tolerance=0.01,
                                max_steps=3000,
                                gripper_command=(
                                    GRIPPER_OPEN
                                ),
                                carry_object=False,
                            )

                            # ------------------------------
                            # Stage 10B —
                            # Exact initial joint pose
                            # ------------------------------

                            if retreat_ok:

                                print(
                                    "\nReturning to exact "
                                    "initial robot "
                                    "configuration..."
                                )

                                return_steps = 1500

                                start_joint_positions = (
                                    np.array([
                                        data.qpos[qid]
                                        for qid
                                        in qpos_ids
                                    ])
                                )

                                for return_step in range(
                                    return_steps
                                ):

                                    if not viewer.is_running():
                                        break

                                    alpha = (
                                        return_step + 1
                                    ) / return_steps

                                    home_command = (
                                        (1.0 - alpha)
                                        * start_joint_positions
                                        + alpha
                                        * initial_joint_positions
                                    )

                                    for (
                                        actuator_id,
                                        command,
                                    ) in zip(
                                        actuator_ids,
                                        home_command,
                                    ):

                                        data.ctrl[
                                            actuator_id
                                        ] = command

                                    # Keep gripper open.
                                    for finger_id in finger_ids:

                                        data.ctrl[
                                            finger_id
                                        ] = GRIPPER_OPEN

                                    mujoco.mj_step(
                                        model,
                                        data,
                                    )

                                    viewer.sync()

                                    if (
                                        return_step
                                        % 250
                                        == 0
                                    ):

                                        print(
                                            f"return "
                                            f"{return_step:4d} | "
                                            f"q="
                                            f"{np.round(home_command, 3)}"
                                        )

                                    time.sleep(
                                        model.opt.timestep
                                    )

                                # Make final hold use
                                # exact initial joint pose.
                                joint_commands = (
                                    initial_joint_positions.copy()
                                )

                                # Allow servo to settle.
                                for _ in range(500):

                                    if not viewer.is_running():
                                        break

                                    for (
                                        actuator_id,
                                        command,
                                    ) in zip(
                                        actuator_ids,
                                        joint_commands,
                                    ):

                                        data.ctrl[
                                            actuator_id
                                        ] = command

                                    for finger_id in finger_ids:

                                        data.ctrl[
                                            finger_id
                                        ] = GRIPPER_OPEN

                                    mujoco.mj_step(
                                        model,
                                        data,
                                    )

                                    viewer.sync()

                                    time.sleep(
                                        model.opt.timestep
                                    )

                                home_ok = True

                            else:

                                home_ok = False

                                print(
                                    "Return-home skipped "
                                    "because retreat failed."
                                )

                            # ------------------------------
                            # Verify home
                            # ------------------------------

                            if home_ok:

                                mujoco.mj_forward(
                                    model,
                                    data,
                                )

                                final_grasp_position = (
                                    data.site_xpos[
                                        grasp_site_id
                                    ].copy()
                                )

                                final_joint_positions = (
                                    np.array([
                                        data.qpos[qid]
                                        for qid
                                        in qpos_ids
                                    ])
                                )

                                home_position_error = (
                                    np.linalg.norm(
                                        final_grasp_position
                                        - initial_grasp_position
                                    )
                                )

                                home_joint_error = (
                                    np.linalg.norm(
                                        final_joint_positions
                                        - initial_joint_positions
                                    )
                                )

                                print(
                                    "\n--- RETURN HOME RESULT ---"
                                )

                                print(
                                    "Initial grasp position:",
                                    initial_grasp_position,
                                )

                                print(
                                    "Final grasp position:",
                                    final_grasp_position,
                                )

                                print(
                                    "Cartesian home error:",
                                    home_position_error,
                                    "m",
                                )

                                print(
                                    "Initial joint positions:",
                                    initial_joint_positions,
                                )

                                print(
                                    "Final joint positions:",
                                    final_joint_positions,
                                )

                                print(
                                    "Joint-space home error:",
                                    home_joint_error,
                                )

                                if (
                                    home_position_error
                                    < 0.01
                                    and home_joint_error
                                    < 0.05
                                ):

                                    print(
                                        "\n"
                                        "================================="
                                    )

                                    print(
                                        "TASK COMPLETE — "
                                        "ROBOT RETURNED HOME"
                                    )

                                    print(
                                        "================================="
                                    )

                                else:

                                    print(
                                        "\nRETURN HOME "
                                        "NOT FULLY VERIFIED"
                                    )

                        else:

                            print(
                                "\n================================="
                            )

                            print(
                                "BOX PLACEMENT "
                                "NOT VERIFIED"
                            )

                            print(
                                "================================="
                            )

        # ==================================================
        # FINAL HOLD
        # ==================================================

        print(
            "\nHolding final configuration..."
        )

        while viewer.is_running():

            # Hold arm at its final command.
            for actuator_id, command in zip(
                actuator_ids,
                joint_commands,
            ):

                data.ctrl[
                    actuator_id
                ] = command

            # Hold current gripper command.
            for finger_id in finger_ids:

                data.ctrl[
                    finger_id
                ] = final_gripper_command

            mujoco.mj_step(
                model,
                data,
            )

            # If execution stopped during carrying,
            # continue holding the box relative to
            # grasp_site.
            if (
                carrying_box
                and object_qpos_adr is not None
                and object_dof_adr is not None
                and object_offset is not None
            ):

                update_carried_object(
                    model,
                    data,
                    grasp_site_id,
                    object_qpos_adr,
                    object_dof_adr,
                    object_offset,
                )

            viewer.sync()

            time.sleep(
                model.opt.timestep
            )


if __name__ == "__main__":
    main()