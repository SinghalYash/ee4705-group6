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


# q = 0      -> closed
# q = 0.025  -> open
GRIPPER_OPEN = 0.025
GRIPPER_CLOSED = 0.0


# ==========================================================
# CARTESIAN CONTROLLER SETTINGS
# ==========================================================

POSITION_TOLERANCE = 0.01

# Maximum joint-command change per Cartesian update.
MAX_DQ = 0.004

# Damped least-squares regularisation.
DAMPING = 0.08


# ==========================================================
# HELPER: DETECT PRONG / BOX CONTACTS
# ==========================================================

def get_finger_contacts(model, data):
    """
    Returns a list:

        [finger_1_contact,
         finger_2_contact,
         finger_3_contact]

    Each entry is True if that finger is currently
    contacting box_geom.
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

        if (
            "box_geom" in pair
            and "finger_1_geom" in pair
        ):
            contacts[0] = True

        if (
            "box_geom" in pair
            and "finger_2_geom" in pair
        ):
            contacts[1] = True

        if (
            "box_geom" in pair
            and "finger_3_geom" in pair
        ):
            contacts[2] = True

    return contacts


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
):
    """
    Move grasp_site toward target_pos using closed-loop
    damped least-squares Jacobian control.

    The controller repeatedly:
      1. measures the actual grasp-site position,
      2. calculates Cartesian error,
      3. evaluates the Jacobian at the physical state,
      4. increments the joint commands,
      5. advances MuJoCo physics.
    """

    print("\n" + "=" * 55)
    print(label)
    print("Target:", target_pos)
    print("=" * 55)

    for step in range(max_steps):

        if not viewer.is_running():
            return joint_commands, False

        # --------------------------------------------------
        # Measure actual grasp-site position
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

            return joint_commands, True

        # --------------------------------------------------
        # Cartesian Jacobian
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

        # Use only the four arm joints.
        J = jacp[:, dof_ids]

        # --------------------------------------------------
        # Damped least-squares correction
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

        # Accumulate command so the controller can
        # compensate for gravity and servo tracking error.
        joint_commands += dq

        # --------------------------------------------------
        # Respect joint limits
        # --------------------------------------------------

        for i, joint_name in enumerate(
            ARM_JOINTS
        ):

            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if model.jnt_limited[jid]:

                lower = model.jnt_range[
                    jid, 0
                ]

                upper = model.jnt_range[
                    jid, 1
                ]

                joint_commands[i] = np.clip(
                    joint_commands[i],
                    lower,
                    upper,
                )

        # --------------------------------------------------
        # Send arm commands
        # --------------------------------------------------

        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[aid] = command

        # Keep all three prongs open during Cartesian motion.
        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = GRIPPER_OPEN

        # --------------------------------------------------
        # Physics
        # --------------------------------------------------

        for _ in range(5):

            mujoco.mj_step(
                model,
                data,
            )

        viewer.sync()

        # --------------------------------------------------
        # Diagnostics
        # --------------------------------------------------

        if step % 100 == 0:

            actual_q = np.array([
                data.qpos[qid]
                for qid in qpos_ids
            ])

            print(
                f"step {step:4d} | "
                f"error={error_norm:.4f} m | "
                f"site="
                f"{np.round(current_pos, 3)}"
            )

            print(
                "    commanded q:",
                np.round(
                    joint_commands,
                    3,
                ),
            )

            print(
                "    actual q:   ",
                np.round(
                    actual_q,
                    3,
                ),
            )

        time.sleep(
            model.opt.timestep * 5
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

    error_norm = np.linalg.norm(
        target_pos
        - current_pos
    )

    print(
        f"\n{label} FAILED"
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
        "Final error:",
        error_norm,
        "m",
    )

    return joint_commands, False


# ==========================================================
# 3-PRONG GRASP
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
    Close all three prongs gradually.

    Phase A:
        close until at least TWO prongs contact the box.

    Phase B:
        freeze that finger command and check whether
        the box remains captured near grasp_site.

    Returns:
        (success, final_finger_command)
    """

    print("\n" + "=" * 55)
    print("STAGE 5: 3-PRONG GRASP")
    print("=" * 55)

    total_steps = int(
        duration
        / model.opt.timestep
    )

    contact_command = None

    # ======================================================
    # PHASE A — CLOSE UNTIL CAPTURE
    # ======================================================

    print(
        "\nClosing all three prongs..."
    )

    two_contact_count = 0

    REQUIRED_CAPTURE_STEPS = 30
    MAX_CAPTURE_DISTANCE = 0.030

    for step in range(total_steps):

        if not viewer.is_running():

            return (
                False,
                GRIPPER_OPEN,
            )

        # Hold arm at pre-grasp pose.
        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[aid] = command

        # ----------------------------------------------
        # Slowly close:
        #
        # 0.025 -> 0.0
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Contacts
        # ----------------------------------------------

        finger_contacts = (
            get_finger_contacts(
                model,
                data,
            )
        )

        number_contacts = sum(
            finger_contacts
        )

        # ----------------------------------------------
        # Object position
        # ----------------------------------------------

        object_now = data.xpos[
            object_id
        ].copy()

        grasp_now = data.site_xpos[
            grasp_site_id
        ].copy()

        object_distance = (
            np.linalg.norm(
                object_now
                - grasp_now
            )
        )

        # ----------------------------------------------
        # Diagnostics
        # ----------------------------------------------

        if step % 100 == 0:

            print(
                f"step {step:4d} | "
                f"finger={finger_command:.4f} | "
                f"contacts={finger_contacts} | "
                f"n={number_contacts} | "
                f"capture_count={two_contact_count} | "
                f"object_dist={object_distance:.4f}"
            )

        # --------------------------------------------------
        # Initial capture
        #
        # Do not freeze at the first transient contact.
        # Continue closing until:
        #   A) all 3 prongs contact the box, OR
        #   B) at least 2 prongs contact once the gripper
        #      has closed sufficiently.
        # --------------------------------------------------
        # --------------------------------------------------
# Detect a stable initial capture
# --------------------------------------------------

        if number_contacts >= 2:
            two_contact_count += 1
        else:
            two_contact_count = 0


        # Best case: all three fingers touch.
        if all(finger_contacts):

            contact_command = finger_command

            print("\n3-PRONG CAPTURE DETECTED")
            print("Contacts:", finger_contacts)
            print(
                "Freezing finger command at:",
                contact_command,
            )

            break


        # Otherwise accept persistent 2-prong contact
        # while the box is still centred.
        if (
            two_contact_count >= REQUIRED_CAPTURE_STEPS
            and object_distance < MAX_CAPTURE_DISTANCE
        ):

            contact_command = finger_command

            print(
                "\nSTABLE 2-PRONG CAPTURE DETECTED"
            )

            print(
                "Contacts:",
                finger_contacts,
            )

            print(
                "Consecutive capture steps:",
                two_contact_count,
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

        if (
            number_contacts >= 2
            and object_distance < 0.025
        ):

            contact_command = finger_command

            print(
                "\n2-PRONG CAPTURE DETECTED"
            )

            print(
                "Contacts:",
                finger_contacts,
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
        
        

        # ----------------------------------------------
        # Object escaped
        # ----------------------------------------------

        if object_distance > 0.12:

            print(
                "\nGRASP ABORTED"
            )

            print(
                "Box moved too far "
                "from grasp site."
            )

            print(
                "Distance:",
                object_distance,
            )

            return (
                False,
                GRIPPER_OPEN,
            )

        time.sleep(
            model.opt.timestep
        )

    # ------------------------------------------------------
    # No capture
    # ------------------------------------------------------

    if contact_command is None:

        print(
            "\nGRASP FAILED"
        )

        print(
            "At least two prongs "
            "never contacted the box."
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

    required_contact_count = 20

    max_stability_steps = 500

    # The object should stay reasonably close
    # to the centre of the gripper.
    MAX_STABLE_DISTANCE = 0.05

    for step in range(
        max_stability_steps
    ):

        if not viewer.is_running():

            return (
                False,
                GRIPPER_OPEN,
            )

        # ----------------------------------------------
        # Hold arm
        # ----------------------------------------------

        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):

            data.ctrl[aid] = command

        # ----------------------------------------------
        # Freeze all three prongs
        # ----------------------------------------------

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

        # ----------------------------------------------
        # Contact state
        # ----------------------------------------------

        finger_contacts = (
            get_finger_contacts(
                model,
                data,
            )
        )

        for i, touching in enumerate(
            finger_contacts
        ):

            if touching:

                contact_counts[i] += 1

        # ----------------------------------------------
        # Object position
        # ----------------------------------------------

        object_now = data.xpos[
            object_id
        ].copy()

        grasp_now = data.site_xpos[
            grasp_site_id
        ].copy()

        object_distance = (
            np.linalg.norm(
                object_now
                - grasp_now
            )
        )

        # ----------------------------------------------
        # Diagnostics
        # ----------------------------------------------

        if step % 25 == 0:

            print(
                f"stability "
                f"{step:3d} | "
                f"contacts="
                f"{finger_contacts} | "
                f"counts="
                f"{contact_counts} | "
                f"object_dist="
                f"{object_distance:.4f}"
            )

        # ----------------------------------------------
        # Failure
        # ----------------------------------------------

        if (
            object_distance
            > MAX_STABLE_DISTANCE
        ):

            print(
                "\nGRASP FAILED"
            )

            print(
                "Box moved outside "
                "the stable grasp region."
            )

            print(
                "Distance:",
                object_distance,
            )

            return (
                False,
                GRIPPER_OPEN,
            )

        # ----------------------------------------------
        # Success criterion
        #
        # Require repeated contact from at least
        # TWO of the three prongs.
        # ----------------------------------------------

        successful_prongs = sum(
            count
            >= required_contact_count
            for count
            in contact_counts
        )

        if successful_prongs >= 2:

            print(
                "\nSTABLE 3-PRONG "
                "GRASP CONFIRMED"
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

    # ------------------------------------------------------
    # Stability window expired
    # ------------------------------------------------------

    print(
        "\nGRASP FAILED"
    )

    print(
        "The box remained nearby, "
        "but sufficient stable contact "
        "was not achieved."
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
    # Load model
    # ------------------------------------------------------

    model = (
        mujoco.MjModel.from_xml_path(
            MODEL_PATH
        )
    )

    data = mujoco.MjData(
        model
    )

    # ------------------------------------------------------
    # Important IDs
    # ------------------------------------------------------

    grasp_site_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )
    )

    object_id = (
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "box_obj",
        )
    )

    # ------------------------------------------------------
    # Arm joint qpos / DoF addresses
    # ------------------------------------------------------

    qpos_ids = []
    dof_ids = []

    for joint_name in ARM_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_ids.append(
            model.jnt_qposadr[
                jid
            ]
        )

        dof_ids.append(
            model.jnt_dofadr[
                jid
            ]
        )

    # ------------------------------------------------------
    # Arm actuators
    # ------------------------------------------------------

    actuator_ids = [

        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        for actuator_name
        in ARM_ACTUATORS
    ]

    # ------------------------------------------------------
    # Three finger actuators
    # ------------------------------------------------------

    finger_ids = [

        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )

        for actuator_name
        in FINGER_ACTUATORS
    ]

    # ------------------------------------------------------
    # Initial forward kinematics
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

    # High and unobstructed initial waypoint.
    safe_target = np.array([
        0.22,
        -0.12,
        0.34,
    ])

    # Directly above box.
    above_box = np.array([
        object_pos[0],
        object_pos[1],
        0.34,
    ])

    # Intermediate descent.
    approach_target = np.array([
        object_pos[0],
        object_pos[1],
        0.22,
    ])

    # ------------------------------------------------------
    # PRE-GRASP
    #
    # Keep grasp_site slightly ABOVE the box centre.
    #
    # This is intentional because the 3-prong gripper
    # contains a lower prong which otherwise risks
    # contacting the floor.
    # ------------------------------------------------------

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
    # Start arm commands from actual state
    # ------------------------------------------------------

    joint_commands = np.array([
        data.qpos[qid]
        for qid
        in qpos_ids
    ])

    final_gripper_command = (
        GRIPPER_OPEN
    )

    grasp_ok = False

    # ======================================================
    # VIEWER
    # ======================================================

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Start with all three prongs open.
        for finger_id in finger_ids:

            data.ctrl[
                finger_id
            ] = GRIPPER_OPEN

        # ==================================================
        # STAGE 1 — SAFE POSE
        # ==================================================

        joint_commands, ok = (
            move_cartesian(
                model,
                data,
                viewer,
                grasp_site_id,
                safe_target,
                qpos_ids,
                dof_ids,
                actuator_ids,
                finger_ids,
                joint_commands,
                label=(
                    "STAGE 1: SAFE POSE"
                ),
            )
        )

        if not ok:

            print(
                "\nStopping: could not "
                "reach safe pose."
            )

        else:

            # ==============================================
            # STAGE 2 — ABOVE BOX
            # ==============================================

            joint_commands, ok = (
                move_cartesian(
                    model,
                    data,
                    viewer,
                    grasp_site_id,
                    above_box,
                    qpos_ids,
                    dof_ids,
                    actuator_ids,
                    finger_ids,
                    joint_commands,
                    label=(
                        "STAGE 2: ABOVE BOX"
                    ),
                )
            )

            if not ok:

                print(
                    "\nStopping: could not "
                    "reach above-box pose."
                )

            else:

                # ==========================================
                # STAGE 3 — DESCEND
                # ==========================================

                joint_commands, ok = (
                    move_cartesian(
                        model,
                        data,
                        viewer,
                        grasp_site_id,
                        approach_target,
                        qpos_ids,
                        dof_ids,
                        actuator_ids,
                        finger_ids,
                        joint_commands,
                        label=(
                            "STAGE 3: DESCEND"
                        ),
                    )
                )

                if not ok:

                    print(
                        "\nStopping: descent "
                        "failed."
                    )

                else:

                    # ======================================
                    # STAGE 4 — PRE-GRASP
                    # ======================================

                    joint_commands, ok = (
                        move_cartesian(
                            model,
                            data,
                            viewer,
                            grasp_site_id,
                            pregrasp_target,
                            qpos_ids,
                            dof_ids,
                            actuator_ids,
                            finger_ids,
                            joint_commands,
                            label=(
                                "STAGE 4: "
                                "PRE-GRASP"
                            ),
                            tolerance=0.005,
                            max_steps=3000,
                        )
                    )

                    if not ok:

                        print(
                            "\nStopping: could "
                            "not reach "
                            "pre-grasp position."
                        )

                    else:

                        print(
                            "\n"
                            "================================="
                        )

                        print(
                            "PRE-GRASP POSITION "
                            "REACHED"
                        )

                        print(
                            "================================="
                        )

                        # ================================
                        # PRE-GRASP DIAGNOSTICS
                        # ================================

                        mujoco.mj_forward(
                            model,
                            data,
                        )

                        grasp_pos = (
                            data.site_xpos[
                                grasp_site_id
                            ].copy()
                        )

                        object_now = (
                            data.xpos[
                                object_id
                            ].copy()
                        )

                        grasp_to_object = (
                            np.linalg.norm(
                                grasp_pos
                                - object_now
                            )
                        )

                        print(
                            "\n--- PRE-GRASP "
                            "CHECK ---"
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
                            "Distance grasp-site "
                            "to box:",
                            grasp_to_object,
                        )

                        # ================================
                        # STAGE 5 — GRASP
                        # ================================

                        (
                            grasp_ok,
                            final_gripper_command,
                        ) = (
                            close_gripper_until_contact(
                                model,
                                data,
                                viewer,
                                finger_ids,
                                actuator_ids,
                                joint_commands,
                                object_id,
                                grasp_site_id,
                            )
                        )

                        if grasp_ok:

                            print(
                                "\n"
                                "================================="
                            )

                            print(
                                "GRASP CONTACT "
                                "SUCCESSFUL"
                            )

                            print(
                                "================================="
                            )

                        else:

                            print(
                                "\nStopping: "
                                "grasp failed."
                            )

        # ==================================================
        # FINAL HOLD
        # ==================================================

        print(
            "\nHolding final configuration..."
        )

        while viewer.is_running():

            # Hold arm pose.
            for aid, command in zip(
                actuator_ids,
                joint_commands,
            ):

                data.ctrl[
                    aid
                ] = command

            # Hold all three prongs at the final
            # successful grasp command.
            for finger_id in finger_ids:

                data.ctrl[
                    finger_id
                ] = (
                    final_gripper_command
                )

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            time.sleep(
                model.opt.timestep
            )


if __name__ == "__main__":
    main()