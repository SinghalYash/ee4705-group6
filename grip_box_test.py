import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"

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

GRIPPER_OPEN = 0.025
GRIPPER_CLOSED = 0.0


# Cartesian controller settings
POSITION_TOLERANCE = 0.01   # 1 cm
MAX_DQ = 0.004              # max joint-command increment
DAMPING = 0.08


def move_cartesian(
    model,
    data,
    viewer,
    site_id,
    target_pos,
    qpos_ids,
    dof_ids,
    actuator_ids,
    left_finger,
    right_finger,
    joint_commands,
    label,
    tolerance=POSITION_TOLERANCE,
    max_steps=2500,
):
    """
    Closed-loop Cartesian movement of grasp_site.

    Repeatedly:
      1. measure actual grasp-site position,
      2. calculate Cartesian error,
      3. calculate Jacobian at current physical state,
      4. increment joint commands,
      5. step MuJoCo physics.
    """

    print("\n" + "=" * 50)
    print(label)
    print("Target:", target_pos)
    print("=" * 50)

    for step in range(max_steps):

        if not viewer.is_running():
            return joint_commands, False

        # --------------------------------------------------
        # Measure actual physical position
        # --------------------------------------------------

        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[
            site_id
        ].copy()

        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        # --------------------------------------------------
        # Success
        # --------------------------------------------------

        if error_norm < tolerance:

            print(f"\n{label} SUCCESS")
            print("Target:", target_pos)
            print("Actual:", current_pos)
            print("Error:", error_norm, "m")

            return joint_commands, True

        # --------------------------------------------------
        # Jacobian at CURRENT physical configuration
        # --------------------------------------------------

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J = jacp[:, dof_ids]

        # --------------------------------------------------
        # Damped least-squares Cartesian correction
        # --------------------------------------------------

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + DAMPING**2 * np.eye(3),
                error,
            )
        )

        # Keep individual updates small.
        dq = np.clip(
            dq,
            -MAX_DQ,
            MAX_DQ,
        )

        # --------------------------------------------------
        # IMPORTANT:
        # Accumulate the desired joint commands.
        #
        # Do NOT use:
        #
        #     joint_commands = actual_q + dq
        #
        # because gravity / tracking error can prevent
        # the command from accumulating.
        # --------------------------------------------------

        joint_commands += dq

        # --------------------------------------------------
        # Respect arm joint limits
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

                lower = model.jnt_range[jid, 0]
                upper = model.jnt_range[jid, 1]

                joint_commands[i] = np.clip(
                    joint_commands[i],
                    lower,
                    upper,
                )

        # --------------------------------------------------
        # Send joint commands
        # --------------------------------------------------

        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):
            data.ctrl[aid] = command

        # Keep fingers open.
        data.ctrl[left_finger] = GRIPPER_OPEN
        data.ctrl[right_finger] = GRIPPER_OPEN

        # --------------------------------------------------
        # Let physics respond
        # --------------------------------------------------

        for _ in range(5):
            mujoco.mj_step(model, data)

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
                f"site={np.round(current_pos, 3)}"
            )

            print(
                "    commanded q:",
                np.round(joint_commands, 3),
            )

            print(
                "    actual q:   ",
                np.round(actual_q, 3),
            )

        time.sleep(
            model.opt.timestep * 5
        )

    # ------------------------------------------------------
    # Failure
    # ------------------------------------------------------

    mujoco.mj_forward(model, data)

    current_pos = data.site_xpos[
        site_id
    ].copy()

    error_norm = np.linalg.norm(
        target_pos - current_pos
    )

    print(f"\n{label} FAILED")
    print("Target:", target_pos)
    print("Actual:", current_pos)
    print("Final error:", error_norm, "m")

    return joint_commands, False

def close_gripper_until_contact(
    model,
    data,
    viewer,
    left_finger,
    right_finger,
    actuator_ids,
    joint_commands,
    duration=4.0,
):
    """
    Slowly close the gripper.

    Once bilateral contact is first detected:
      - stop closing,
      - hold that finger command,
      - verify contact remains stable.
    """

    print("\n" + "=" * 50)
    print("STAGE 5: GRASP")
    print("=" * 50)

    object_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "box_obj",
    )

    grasp_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "grasp_site",
    )

    steps = int(duration / model.opt.timestep)

    contact_command = None

    # --------------------------------------------------
    # PHASE A — gradually close until BOTH fingers touch
    # --------------------------------------------------

    for step in range(steps):

        if not viewer.is_running():
            return False, GRIPPER_OPEN

        # Hold arm pose.
        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):
            data.ctrl[aid] = command

        alpha = (step + 1) / steps

        finger_command = (
            GRIPPER_OPEN
            + alpha
            * (GRIPPER_CLOSED - GRIPPER_OPEN)
        )

        data.ctrl[left_finger] = finger_command
        data.ctrl[right_finger] = finger_command

        mujoco.mj_step(model, data)
        viewer.sync()
        mujoco.mj_forward(model, data)

        # ----------------------------------------------
        # Determine contacts
        # ----------------------------------------------

        left_contact = False
        right_contact = False

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

            pair = {geom1, geom2}

            if (
                "box_geom" in pair
                and (
                    "finger_left_geom" in pair
                    or "finger_left_pad" in pair
                )
            ):
                left_contact = True

            if (
                "box_geom" in pair
                and (
                    "finger_right_geom" in pair
                    or "finger_right_pad" in pair
                )
            ):
                right_contact = True

        object_now = data.xpos[object_id].copy()

        grasp_now = data.site_xpos[
            grasp_site_id
        ].copy()

        object_distance = np.linalg.norm(
            object_now - grasp_now
        )

        if step % 100 == 0:
            print(
                f"step {step:4d} | "
                f"finger={finger_command:.4f} | "
                f"left={left_contact} | "
                f"right={right_contact} | "
                f"object_dist={object_distance:.4f}"
            )

        # Stop closing as soon as bilateral contact occurs.
        if left_contact and right_contact:

            contact_command = finger_command

            print(
                "\nFIRST BILATERAL CONTACT"
            )

            print(
                "Freezing finger command at:",
                contact_command,
            )

            break

        # Sphere has clearly escaped.
        if object_distance > 0.10:

            print("\nGRASP ABORTED")
            print(
                "Box moved too far from grasp site."
            )

            return False, GRIPPER_OPEN

        time.sleep(model.opt.timestep)

    # Never achieved bilateral contact.
    if contact_command is None:

        print("\nGRASP FAILED")
        print(
            "Bilateral contact was never achieved."
        )

        return False, GRIPPER_OPEN

    # --------------------------------------------------
    # PHASE B — hold contact command and test stability
    # --------------------------------------------------

    print("\nChecking grasp stability...")

    # Count how often EACH side contacts the stone.
    # Contact does not need to occur simultaneously every frame.
    left_contact_count = 0
    right_contact_count = 0

    required_contact_count = 30
    max_stability_steps = 500

    # During the stability test, the stone must remain
    # close to the centre of the gripper.
    MAX_STABLE_DISTANCE = 0.025


    for step in range(max_stability_steps):

        if not viewer.is_running():
            return False, GRIPPER_OPEN

        # Hold arm.
        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):
            data.ctrl[aid] = command

        # IMPORTANT:
        # Do NOT continue closing.
        data.ctrl[left_finger] = contact_command
        data.ctrl[right_finger] = contact_command

        mujoco.mj_step(model, data)
        viewer.sync()
        mujoco.mj_forward(model, data)

        left_contact = False
        right_contact = False

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

            pair = {geom1, geom2}

            if (
                "box_geom" in pair
                and (
                    "finger_left_geom" in pair
                    or "finger_left_pad" in pair
                )
            ):
                left_contact = True

            if (
                "box_geom" in pair
                and (
                    "finger_right_geom" in pair
                    or "finger_right_pad" in pair
                )
            ):
                right_contact = True

        object_now = data.xpos[
            object_id
        ].copy()

        grasp_now = data.site_xpos[
            grasp_site_id
        ].copy()

        object_distance = np.linalg.norm(
            object_now - grasp_now
        )

        # Record contact samples independently.
        if left_contact:
            left_contact_count += 1

        if right_contact:
            right_contact_count += 1
            
        if step % 25 == 0:
            print(
                f"stability {step:3d} | "
                f"left={left_contact} | "
                f"right={right_contact} | "
                f"left_count={left_contact_count} | "
                f"right_count={right_contact_count} | "
                f"object_dist={object_distance:.4f}"
            )

        if object_distance > 0.10:

            print("\nGRASP FAILED")
            print(
                "Box escaped during stability test."
            )

            return False, GRIPPER_OPEN

        # The box has not completely escaped, but it has
        # moved too far away to count as a stable grasp.
        if object_distance > MAX_STABLE_DISTANCE:

            print("\nGRASP FAILED")
            print(
                "Box moved outside the stable "
                "grasp region."
            )

            print(
                "Box distance:",
                object_distance,
            )

        if (
            left_contact_count >= required_contact_count
            and right_contact_count >= required_contact_count
            and object_distance < MAX_STABLE_DISTANCE
        ):

            print(
                "\nSTABLE GRASP CONFIRMED"
            )

            print(
                "Finger command:",
                contact_command,
            )

            print(
                "Left contact samples:",
                left_contact_count,
            )

            print(
                "Right contact samples:",
                right_contact_count,
            )

            print(
                "Box distance:",
                object_distance,
            )

            return True, contact_command

        time.sleep(model.opt.timestep)

    # If all stability steps finish without success:
    print("\nGRASP FAILED")

    print(
        "Sufficient left/right contact "
        "was not achieved."
    )

    print(
        "Left contact samples:",
        left_contact_count,
    )

    print(
        "Right contact samples:",
        right_contact_count,
    )

    return False, GRIPPER_OPEN


def main():

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    # ------------------------------------------------------
    # Get IDs
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

    qpos_ids = []
    dof_ids = []

    for joint_name in ARM_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_ids.append(
            model.jnt_qposadr[jid]
        )

        dof_ids.append(
            model.jnt_dofadr[jid]
        )

    actuator_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in ARM_ACTUATORS
    ]

    left_finger = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "act_finger_left",
    )

    right_finger = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "act_finger_right",
    )

    # ------------------------------------------------------
    # Initial state
    # ------------------------------------------------------

    mujoco.mj_forward(model, data)

    object_pos = data.xpos[
        object_id
    ].copy()

    print("Box position:", object_pos)

    # ------------------------------------------------------
    # Cartesian waypoints
    # ------------------------------------------------------

    # Stage 1:
    # Move to a high, unobstructed location.
    safe_target = np.array([
        0.22,
        -0.12,
        0.32,
    ])

    # Stage 2:
    # Move horizontally until directly above box.
    above_box = np.array([
        object_pos[0],
        object_pos[1],
        0.32,
    ])

    approach_target = np.array([
        object_pos[0],
        object_pos[1],
        0.20,
    ])

    pregrasp_target = np.array([
        object_pos[0],
        object_pos[1],
        object_pos[2],
    ])

    print("Safe target:", safe_target)
    print("Above-box target:", above_box)
    print("Approach target:", approach_target)
    print("Pre-grasp target:", pregrasp_target)

    # Start commands from actual physical joint positions.
    joint_commands = np.array([
        data.qpos[qid]
        for qid in qpos_ids
    ])

    final_gripper_command = GRIPPER_OPEN
    grasp_ok = False    

    # ------------------------------------------------------
    # Start simulation
    # ------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Keep gripper open throughout approach.
        data.ctrl[left_finger] = GRIPPER_OPEN
        data.ctrl[right_finger] = GRIPPER_OPEN

        # ==================================================
        # STAGE 1 — SAFE POSE
        # ==================================================

        joint_commands, ok = move_cartesian(
            model,
            data,
            viewer,
            grasp_site_id,
            safe_target,
            qpos_ids,
            dof_ids,
            actuator_ids,
            left_finger,
            right_finger,
            joint_commands,
            label="STAGE 1: SAFE POSE",
        )

        if not ok:
            print("\nStopping: could not reach safe pose.")

        else:

            # ==============================================
            # STAGE 2 — ABOVE BOX
            # ==============================================

            joint_commands, ok = move_cartesian(
                model,
                data,
                viewer,
                grasp_site_id,
                above_box,
                qpos_ids,
                dof_ids,
                actuator_ids,
                left_finger,
                right_finger,
                joint_commands,
                label="STAGE 2: ABOVE BOX",
            )

            if not ok:
                print(
                    "\nStopping: could not reach "
                    "above-box pose."
                )

            else:

                # ==========================================
                # STAGE 3 — CONTROLLED DESCENT
                # ==========================================

                joint_commands, ok = move_cartesian(
                    model,
                    data,
                    viewer,
                    grasp_site_id,
                    approach_target,
                    qpos_ids,
                    dof_ids,
                    actuator_ids,
                    left_finger,
                    right_finger,
                    joint_commands,
                    label="STAGE 3: DESCEND",
                )

                if not ok:
                    print(
                        "\nStopping: descent failed."
                    )

                else:

                    print(
                        "\n================================="
                    )
                    print(
                        "APPROACH SEQUENCE SUCCESSFUL"
                    )
                    print(
                        "================================="
                    )

                    # ======================================
                    # STAGE 4 — PRE-GRASP
                    # ======================================

                    joint_commands, ok = move_cartesian(
                        model,
                        data,
                        viewer,
                        grasp_site_id,
                        pregrasp_target,
                        qpos_ids,
                        dof_ids,
                        actuator_ids,
                        left_finger,
                        right_finger,
                        joint_commands,
                        label="STAGE 4: PRE-GRASP",
                        tolerance=0.005,
                        max_steps=3000,
                    )

                    if not ok:

                        print(
                            "\nStopping: could not reach "
                            "pre-grasp position."
                        )

                    else:

                        print(
                            "\n================================="
                        )
                        print(
                            "PRE-GRASP POSITION REACHED"
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

                        grasp_pos = data.site_xpos[
                            grasp_site_id
                        ].copy()

                        object_now = data.xpos[
                            object_id
                        ].copy()

                        grasp_to_object = np.linalg.norm(
                            grasp_pos - object_now
                        )

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
                            "Distance grasp-site "
                            "to box:",
                            grasp_to_object,
                        )

                    # ======================================
                    # STAGE 5 — GRASP
                    # ======================================

                        grasp_ok, final_gripper_command = (
                            close_gripper_until_contact(
                                model,
                                data,
                                viewer,
                                left_finger,
                                right_finger,
                                actuator_ids,
                                joint_commands,
                            )
                        )

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

                        else:

                            print(
                                "\nStopping: grasp failed."
                            )

        # ==================================================
        # HOLD FINAL CONFIGURATION
        # ==================================================

        while viewer.is_running():

            for aid, command in zip(
                actuator_ids,
                joint_commands,
            ):
                data.ctrl[aid] = command

            # Hold the final gripper position.
            # If grasp succeeded, this keeps gripping the box.
            # If grasp failed/not attempted, this remains GRIPPER_OPEN.
            data.ctrl[left_finger] = final_gripper_command
            data.ctrl[right_finger] = final_gripper_command

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