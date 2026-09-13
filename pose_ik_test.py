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


def rotation_error(R_current, R_target):
    """
    Small-angle orientation error in world coordinates.
    """
    R_err = R_target @ R_current.T

    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def solve_pose_ik(
    model,
    data,
    site_id,
    target_pos,
    target_rot,
    qpos_ids,
    dof_ids,
):
    """
    Solve grasp-site position + orientation using MuJoCo's
    actual translational and rotational Jacobians.
    """

    best_score = float("inf")
    best_qpos = None

    for iteration in range(1500):

        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id].copy()

        current_rot = (
            data.site_xmat[site_id]
            .reshape(3, 3)
            .copy()
        )

        pos_error = target_pos - current_pos

        rot_error = rotation_error(
            current_rot,
            target_rot,
        )

        pos_norm = np.linalg.norm(pos_error)
        rot_norm = np.linalg.norm(rot_error)

        # Position matters more than orientation.
        score = pos_norm + 0.05 * rot_norm

        if score < best_score:
            best_score = score
            best_qpos = np.array([
                data.qpos[qid]
                for qid in qpos_ids
            ])

        if (
            pos_norm < 0.005
            and rot_norm < 0.08
        ):
            print(
                f"IK converged after "
                f"{iteration} iterations."
            )
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jacp,
            jacr,
            site_id,
        )

        J_pos = jacp[:, dof_ids]
        J_rot = jacr[:, dof_ids]

        # Orientation is deliberately weighted lower.
        orientation_weight = 0.25

        J = np.vstack([
            J_pos,
            orientation_weight * J_rot,
        ])

        error = np.concatenate([
            pos_error,
            orientation_weight * rot_error,
        ])

        damping = 0.08

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + damping**2 * np.eye(6),
                error,
            )
        )

        dq = np.clip(
            dq,
            -0.025,
            0.025,
        )

        for i, qid in enumerate(qpos_ids):
            data.qpos[qid] += dq[i]

    # Restore best configuration found.
    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = best_qpos[i]

    mujoco.mj_forward(model, data)

    final_pos = data.site_xpos[site_id].copy()
    final_rot = (
        data.site_xmat[site_id]
        .reshape(3, 3)
        .copy()
    )

    final_pos_error = np.linalg.norm(
        target_pos - final_pos
    )

    final_rot_error = np.linalg.norm(
        rotation_error(
            final_rot,
            target_rot,
        )
    )

    print("\n--- POSE IK RESULT ---")
    print("Requested position:", target_pos)
    print("Reached position:  ", final_pos)
    print(
        "Position error:     ",
        final_pos_error,
        "m",
    )
    print(
        "Orientation error:  ",
        final_rot_error,
    )

    return best_qpos


def move_smoothly(
    model,
    data,
    viewer,
    actuator_ids,
    target,
    duration=4.0,
):

    start = np.array([
        data.ctrl[aid]
        for aid in actuator_ids
    ])

    steps = int(
        duration / model.opt.timestep
    )

    for step in range(steps):

        alpha = (step + 1) / steps

        alpha = alpha * alpha * (
            3 - 2 * alpha
        )

        command = (
            (1 - alpha) * start
            + alpha * target
        )

        for aid, value in zip(
            actuator_ids,
            command,
        ):
            data.ctrl[aid] = value

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(model.opt.timestep)


def main():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # -----------------------------------------------------
    # IDs
    # -----------------------------------------------------

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

    for name in ARM_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
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

    # -----------------------------------------------------
    # Target
    # -----------------------------------------------------

    stone_pos = data.xpos[
        stone_id
    ].copy()

    target_pos = stone_pos + np.array([
        0.0,
        0.0,
        0.12,
    ])

    print("Stone position:", stone_pos)
    print("Approach target:", target_pos)

    # -----------------------------------------------------
    # Orientation target
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # Rather than inventing an orientation matrix, use the
    # gripper's initial orientation as our first target.
    #
    # This asks IK to move the grasp site while keeping the
    # redesigned gripper approximately in its known-good
    # forward-facing orientation.
    # -----------------------------------------------------

    target_rot = (
        data.site_xmat[grasp_site_id]
        .reshape(3, 3)
        .copy()
    )

    joint_target = solve_pose_ik(
        model,
        data,
        grasp_site_id,
        target_pos,
        target_rot,
        qpos_ids,
        dof_ids,
    )

    print(
        "Joint target:",
        joint_target,
    )

    # -----------------------------------------------------
    # Reset and execute
    # -----------------------------------------------------

    mujoco.mj_resetData(model, data)

    data.ctrl[left_finger] = GRIPPER_OPEN
    data.ctrl[right_finger] = GRIPPER_OPEN

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        print(
            "\nMoving to pose-aware "
            "approach position..."
        )

        move_smoothly(
            model,
            data,
            viewer,
            actuator_ids,
            joint_target,
            duration=4.0,
        )

        # Let the controller settle.
        for _ in range(500):

            for aid, value in zip(
                actuator_ids,
                joint_target,
            ):
                data.ctrl[aid] = value

            data.ctrl[left_finger] = GRIPPER_OPEN
            data.ctrl[right_finger] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            viewer.sync()

        mujoco.mj_forward(model, data)

        actual_pos = data.site_xpos[
            grasp_site_id
        ].copy()

        print("\n--- PHYSICAL RESULT ---")
        print("Target:", target_pos)
        print("Actual:", actual_pos)
        print(
            "Physical position error:",
            np.linalg.norm(
                target_pos - actual_pos
            ),
        )

        while viewer.is_running():

            for aid, value in zip(
                actuator_ids,
                joint_target,
            ):
                data.ctrl[aid] = value

            data.ctrl[left_finger] = GRIPPER_OPEN
            data.ctrl[right_finger] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()