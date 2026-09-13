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

STONE_RADIUS = 0.04

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
    duration=3.0,
):
    """
    Gradually close both fingers and detect contact
    between each finger and the stone.
    """

    print("\n" + "=" * 50)
    print("STAGE 5: GRASP")
    print("=" * 50)

    steps = int(duration / model.opt.timestep)

    for step in range(steps):

        if not viewer.is_running():
            return False, GRIPPER_OPEN

        # Hold the arm at the successful pre-grasp pose.
        for aid, command in zip(
            actuator_ids,
            joint_commands,
        ):
            data.ctrl[aid] = command

        # Gradually close from 0.025 -> 0.0
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

        # ----------------------------------------------
        # Detect finger/stone contacts
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

            if pair == {
                "finger_left_geom",
                "stone_geom",
            }:
                left_contact = True

            if pair == {
                "finger_right_geom",
                "stone_geom",
            }:
                right_contact = True

        if step % 100 == 0:
            print(
                f"step {step:4d} | "
                f"finger={finger_command:.4f} | "
                f"left={left_contact} | "
                f"right={right_contact}"
            )

        # Both fingers are touching the stone.
        if left_contact and right_contact:

            print("\nBILATERAL CONTACT DETECTED")
            print(
                "Finger command:",
                finger_command,
            )

            # Let the grasp settle while maintaining
            # the same arm pose and finger command.
            for _ in range(500):

                for aid, command in zip(
                    actuator_ids,
                    joint_commands,
                ):
                    data.ctrl[aid] = command

                data.ctrl[left_finger] = finger_command
                data.ctrl[right_finger] = finger_command

                mujoco.mj_step(model, data)
                viewer.sync()

            return True, finger_command

        time.sleep(model.opt.timestep)

    print("\nGRASP FAILED")
    print(
        "Both fingers did not contact the stone."
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

    stone_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "stone",
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

    stone_pos = data.xpos[
        stone_id
    ].copy()

    print("Stone position:", stone_pos)

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
    # Move horizontally until directly above stone.
    above_stone = np.array([
        stone_pos[0],
        stone_pos[1],
        0.32,
    ])

    # Stage 3:
    # Only now descend toward stone.
    approach_target = np.array([
        stone_pos[0],
        stone_pos[1],
        0.20,
    ])

    pregrasp_target = np.array([
        stone_pos[0],
        stone_pos[1],
        stone_pos[2] + 0.02,
    ])

    print("Safe target:", safe_target)
    print("Above-stone target:", above_stone)
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
            # STAGE 2 — ABOVE STONE
            # ==============================================

            joint_commands, ok = move_cartesian(
                model,
                data,
                viewer,
                grasp_site_id,
                above_stone,
                qpos_ids,
                dof_ids,
                actuator_ids,
                left_finger,
                right_finger,
                joint_commands,
                label="STAGE 2: ABOVE STONE",
            )

            if not ok:
                print(
                    "\nStopping: could not reach "
                    "above-stone pose."
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
                        tolerance=0.008,
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

                        stone_now = data.xpos[
                            stone_id
                        ].copy()

                        grasp_to_stone = np.linalg.norm(
                            grasp_pos - stone_now
                        )

                        print(
                            "\n--- PRE-GRASP CHECK ---"
                        )

                        print(
                            "Stone centre:",
                            stone_now,
                        )

                        print(
                            "Grasp site:",
                            grasp_pos,
                        )

                        print(
                            "Distance grasp-site "
                            "to stone:",
                            grasp_to_stone,
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
            # If grasp succeeded, this keeps gripping the stone.
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