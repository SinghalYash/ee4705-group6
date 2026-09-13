import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"

# Position above the stone, not touching it yet.
# Stone centre = [0.40, -0.30, 0.05]
TARGET_POS = np.array([0.40, -0.30, 0.12])

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


def main():
    # ---------------------------------------------------------
    # Load MuJoCo model
    # ---------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # End-effector reference site.
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "gripper_site",
    )

    # Stone body, used only for diagnostics.
    stone_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "stone",
    )

    # ---------------------------------------------------------
    # Find joint qpos and DoF indices
    # ---------------------------------------------------------
    qpos_ids = []
    dof_ids = []

    for joint_name in ARM_JOINTS:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        qpos_ids.append(model.jnt_qposadr[jid])
        dof_ids.append(model.jnt_dofadr[jid])

    # Find actuator IDs.
    actuator_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in ARM_ACTUATORS
    ]

    # Initialise forward kinematics.
    mujoco.mj_forward(model, data)

    print("Initial gripper position:", data.site_xpos[site_id])
    print("Stone position:", data.xpos[stone_id])
    print("Target position:", TARGET_POS)

    # ---------------------------------------------------------
    # Position-only inverse kinematics
    # ---------------------------------------------------------
    for iteration in range(500):

        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id].copy()
        error = TARGET_POS - current_pos

        error_norm = np.linalg.norm(error)

        # Stop when end-effector is within 5 mm of target.
        if error_norm < 0.005:
            print(f"IK converged after {iteration} iterations.")
            break

        # Translational Jacobian for the gripper site.
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        # Use only the four arm joints.
        J = jacp[:, dof_ids]

        # Damped least-squares inverse kinematics.
        damping = 0.05

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T + damping**2 * np.eye(3),
                error,
            )
        )

        # Prevent excessively large changes per IK iteration.
        max_step = 0.05
        dq = np.clip(dq, -max_step, max_step)

        # Apply the calculated joint increments.
        for i, qpos_id in enumerate(qpos_ids):
            data.qpos[qpos_id] += dq[i]

    else:
        print("WARNING: IK did not converge within 500 iterations.")

    # Recompute kinematics after final IK update.
    mujoco.mj_forward(model, data)

    # Store resulting joint configuration.
    desired_joint_positions = np.array(
        [data.qpos[qid] for qid in qpos_ids]
    )

    final_gripper_pos = data.site_xpos[site_id].copy()
    final_error = np.linalg.norm(
        TARGET_POS - final_gripper_pos
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------
    print("\n--- IK RESULT ---")
    print("Desired joint positions:", desired_joint_positions)
    print("Stone position:", data.xpos[stone_id])
    print("Target position:", TARGET_POS)
    print("IK gripper position:", final_gripper_pos)
    print("Final IK error:", final_error)

    xy_error = np.linalg.norm(
        TARGET_POS[:2] - final_gripper_pos[:2]
    )

    print("XY error:", xy_error)

    # ---------------------------------------------------------
    # Reset simulation and physically move arm
    # ---------------------------------------------------------
    mujoco.mj_resetData(model, data)

    # Gripper actuators.
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

    # Keep gripper open.
    data.ctrl[left_finger] = 0.0
    data.ctrl[right_finger] = 0.0

    # ---------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # Command arm toward the IK solution.
            for aid, target in zip(
                actuator_ids,
                desired_joint_positions,
            ):
                data.ctrl[aid] = target

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()