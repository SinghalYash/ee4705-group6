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
        0.30,
        -0.20,
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

    print("Safe target:", safe_target)
    print("Above-stone target:", above_stone)
    print("Approach target:", approach_target)

    # Start commands from actual physical joint positions.
    joint_commands = np.array([
        data.qpos[qid]
        for qid in qpos_ids
    ])

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
            print(
                "\nStopping: could not reach "
                "safe pose."
            )

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
                        "\n"
                        "================================="
                    )
                    print(
                        "APPROACH SEQUENCE SUCCESSFUL"
                    )
                    print(
                        "================================="
                    )

        # --------------------------------------------------
        # Final diagnostics
        # --------------------------------------------------

        mujoco.mj_forward(
            model,
            data,
        )

        final_pos = data.site_xpos[
            grasp_site_id
        ].copy()

        current_stone_pos = data.xpos[
            stone_id
        ].copy()

        print(
            "\n--- FINAL PHYSICAL RESULT ---"
        )

        print(
            "Stone:",
            current_stone_pos,
        )

        print(
            "Grasp site:",
            final_pos,
        )

        print(
            "Distance from approach target:",
            np.linalg.norm(
                approach_target
                - final_pos
            ),
        )

        print(
            "\nFinal joint positions:"
        )

        for name, qid in zip(
            ARM_JOINTS,
            qpos_ids,
        ):

            print(
                name,
                "=",
                data.qpos[qid],
            )

        # --------------------------------------------------
        # Hold final configuration for inspection
        # --------------------------------------------------

        while viewer.is_running():

            for aid, command in zip(
                actuator_ids,
                joint_commands,
            ):
                data.ctrl[aid] = command

            data.ctrl[left_finger] = GRIPPER_OPEN
            data.ctrl[right_finger] = GRIPPER_OPEN

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