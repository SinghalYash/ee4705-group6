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

# q = 0.025 -> open
# q = 0.0   -> closed
GRIPPER_OPEN = 0.025
GRIPPER_CLOSED = 0.0


# ==========================================================
# CARTESIAN CONTROLLER SETTINGS
# ==========================================================

POSITION_TOLERANCE = 0.01

# Maximum change in commanded joint position
# per Cartesian-control iteration.
MAX_DQ = 0.004

# Damped least-squares regularisation.
DAMPING = 0.08


# ==========================================================
# CONTACT DETECTION
# ==========================================================

def get_finger_contacts(model, data):
    """
    Return contact state for the three gripper fingers.

    Example:
        [True, False, True]

    means finger 1 and finger 3 are touching box_geom.
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

        # Ignore contacts that do not involve the box.
        if "box_geom" not in pair:
            continue

        for j, finger_geom in enumerate(
            FINGER_GEOMS
        ):

            if finger_geom in pair:
                contacts[j] = True

    return contacts


# ==========================================================
# FREE-JOINT HELPER
# ==========================================================

def get_freejoint_qpos_address(
    model,
    body_name,
):
    """
    Return the qpos address of the free joint belonging
    to the specified movable body.
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
    Return the qvel / DoF address of the free joint
    belonging to the specified movable body.
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
# KINEMATIC CARRY HELPER
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
    Keep the carried object's centre at a fixed XYZ offset
    from grasp_site.

    This is activated only AFTER Stage 5 has physically
    detected a stable grasp.

    The object's orientation is left unchanged.
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

    # Free-joint qpos layout:
    #
    # [x, y, z, qw, qx, qy, qz]
    #
    # Only overwrite XYZ.
    data.qpos[
        object_qpos_adr:
        object_qpos_adr + 3
    ] = desired_object_position

    # Remove residual linear and angular velocity.
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
    Move grasp_site toward target_pos using damped
    least-squares Jacobian control.

    If carry_object=True, the object is kinematically kept
    at its recorded offset from grasp_site after every
    physics step.
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

        # Only use the four arm DoFs.
        J = jacp[
            :,
            dof_ids,
        ]

        # --------------------------------------------------
        # Damped least-squares IK correction
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
        # Respect joint limits
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
        # Send arm commands
        # --------------------------------------------------

        for actuator_id, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[
                actuator_id
            ] = command

        # --------------------------------------------------
        # Maintain requested gripper opening
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

            # During Stage 6+, force the successfully
            # grasped box to follow the gripper.
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
    # Failed to converge
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
# STAGE 5 — CLOSE GRIPPER
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

    Stop closing immediately once:
      - at least two fingers contact the box, and
      - the box remains close to grasp_site.

    Then hold the finger command and verify stability.
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
    # PHASE A — CLOSE UNTIL USEFUL CAPTURE
    # ======================================================

    for step in range(
        total_steps
    ):

        if not viewer.is_running():

            return (
                False,
                GRIPPER_OPEN,
            )

        # Hold arm at pre-grasp pose.
        for actuator_id, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[
                actuator_id
            ] = command

        # --------------------------------------------------
        # Slowly close 0.025 -> 0
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Contact state
        # --------------------------------------------------

        contacts = get_finger_contacts(
            model,
            data,
        )

        number_contacts = sum(
            contacts
        )

        # --------------------------------------------------
        # Box position relative to grasp site
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"finger={finger_command:.4f} | "
                f"contacts={contacts} | "
                f"n={number_contacts} | "
                f"object_dist={object_distance:.4f}"
            )

        # --------------------------------------------------
        # Capture
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Failure
        # --------------------------------------------------

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

    # ------------------------------------------------------
    # No capture occurred
    # ------------------------------------------------------

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
    # PHASE B — STABILITY TEST
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

        # Hold fingers at captured spacing.
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

        # --------------------------------------------------
        # Contact statistics
        # --------------------------------------------------

        contacts = get_finger_contacts(
            model,
            data,
        )

        for i, touching in enumerate(
            contacts
        ):

            if touching:
                contact_counts[i] += 1

        # --------------------------------------------------
        # Object position
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        if step % 25 == 0:

            print(
                f"stability {step:3d} | "
                f"contacts={contacts} | "
                f"counts={contact_counts} | "
                f"object_dist={object_distance:.4f}"
            )

        # --------------------------------------------------
        # Failure
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Success
        # --------------------------------------------------

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

    print(
        "Contact samples:",
        contact_counts,
    )

    return (
        False,
        GRIPPER_OPEN,
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # Load MuJoCo model
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Arm joint addresses
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

    # ------------------------------------------------------
    # Actuator IDs
    # ------------------------------------------------------

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

    object_pos = data.xpos[
        object_id
    ].copy()

    print(
        "Box position:",
        object_pos,
    )

    # ======================================================
    # WAYPOINTS
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

    # Slightly above box centre to give the lower
    # third prong floor clearance.
    pregrasp_target = np.array([
        object_pos[0],
        object_pos[1],
        object_pos[2] + 0.025,
    ])

    print(
        "Safe target:",
        safe_target,
    )

    print(
        "Above-box target:",
        above_box,
    )

    print(
        "Approach target:",
        approach_target,
    )

    print(
        "Pre-grasp target:",
        pregrasp_target,
    )

    # ------------------------------------------------------
    # Initial arm command
    # ------------------------------------------------------

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

        # Start fully open.
        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = GRIPPER_OPEN

        # ==================================================
        # STAGES 1–4
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

            joint_commands, ok = (
                move_cartesian(
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
            )

            if not ok:

                print(
                    "\nStopping before grasp "
                    "because a movement stage failed."
                )

                break

        # ==================================================
        # STAGE 5
        # ==================================================

        if ok:

            mujoco.mj_forward(
                model,
                data,
            )

            grasp_pos = data.site_xpos[
                grasp_site_id
            ].copy()

            object_now = data.xpos[
                object_id
            ].copy()

            print(
                "\n--- PRE-GRASP CHECK ---"
            )

            print(
                "Box centre:",
                object_now,
            )

            print(
                "Grasp site:",
                grasp_pos,
            )

            print(
                "Distance grasp-site to box:",
                np.linalg.norm(
                    grasp_pos
                    - object_now
                ),
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
                    "GRASP CONTACT SUCCESSFUL"
                )

                print(
                    "================================="
                )

                print(
                    "\n================================="
                )

                print(
                    "STAGE 6: LIFT BOX"
                )

                print(
                    "================================="
                )

                # ------------------------------------------
                # Record physical grasp state
                # ------------------------------------------

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

                # Preserve current relative XYZ.
                object_offset = (
                    box_before_lift
                    - grasp_before_lift
                )

                print(
                    "Box before lift:",
                    box_before_lift,
                )

                print(
                    "Grasp site before lift:",
                    grasp_before_lift,
                )

                print(
                    "Box/grasp offset:",
                    object_offset,
                )

                # ------------------------------------------
                # Locate box free joint
                # ------------------------------------------

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

                # Everything required for carrying has
                # now been initialized.
                carrying_box = True

                # ------------------------------------------
                # 10 cm vertical lift
                # ------------------------------------------

                lift_target = (
                    grasp_before_lift.copy()
                )

                lift_target[2] += 0.10

                print(
                    "Lift target:",
                    lift_target,
                )

                # ------------------------------------------
                # Perform lift
                # ------------------------------------------

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

                    # Keep successful finger spacing.
                    gripper_command=(
                        final_gripper_command
                    ),

                    # Make box follow grasp site.
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

                # ------------------------------------------
                # Verify physical lift
                # ------------------------------------------

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

                print(
                    "\n--- LIFT RESULT ---"
                )

                print(
                    "Lift controller success:",
                    lift_ok,
                )

                print(
                    "Box before:",
                    box_before_lift,
                )

                print(
                    "Box after:",
                    box_after_lift,
                )

                print(
                    "Grasp site before:",
                    grasp_before_lift,
                )

                print(
                    "Grasp site after:",
                    grasp_after_lift,
                )

                print(
                    "Box vertical displacement:",
                    box_lift_amount,
                    "m",
                )

                print(
                    "Gripper vertical displacement:",
                    gripper_lift_amount,
                    "m",
                )

                # Difference between how far the box and
                # gripper moved should be tiny.
                lift_tracking_error = abs(
                    box_lift_amount
                    - gripper_lift_amount
                )

                print(
                    "Lift tracking difference:",
                    lift_tracking_error,
                    "m",
                )

                if (
                    lift_ok
                    and box_lift_amount > 0.07
                    and lift_tracking_error < 0.01
                ):

                    print(
                        "\n================================="
                    )

                    print(
                        "BOX LIFT SUCCESSFUL"
                    )

                    print(
                        "================================="
                    )

                else:

                    print(
                        "\n================================="
                    )

                    print(
                        "BOX LIFT NOT VERIFIED"
                    )

                    print(
                        "================================="
                    )

            else:

                print(
                    "\nStopping: grasp failed."
                )

        # ==================================================
        # FINAL HOLD
        # ==================================================

        print(
            "\nHolding final configuration..."
        )

        while viewer.is_running():

            # Hold arm.
            for actuator_id, command in zip(
                actuator_ids,
                joint_commands,
            ):

                data.ctrl[
                    actuator_id
                ] = command

            # Hold gripper.
            for finger_id in finger_ids:

                data.ctrl[
                    finger_id
                ] = final_gripper_command

            # Physics.
            mujoco.mj_step(
                model,
                data,
            )

            # If Stage 6 succeeded far enough to initialize
            # carry state, continue holding the box relative
            # to grasp_site.
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