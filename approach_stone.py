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


def solve_ik(
    model,
    data,
    site_id,
    target_pos,
    qpos_ids,
    dof_ids,
    label="TARGET",
):
    best_error = float("inf")
    best_qpos = None

    for iteration in range(1000):

        mujoco.mj_forward(model, data)

        current_pos = data.site_xpos[site_id].copy()
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        # Remember best configuration encountered
        if error_norm < best_error:
            best_error = error_norm
            best_qpos = np.array(
                [data.qpos[qid] for qid in qpos_ids]
            )

        # Success criterion: within 5 mm
        if error_norm < 0.005:
            print(
                f"{label}: IK converged after "
                f"{iteration} iterations."
            )
            print(f"{label}: requested = {target_pos}")
            print(f"{label}: reached   = {current_pos}")
            print(f"{label}: error     = {error_norm:.6f} m")

            return best_qpos, True

        # Translational Jacobian
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

        # Damped least-squares IK
        damping = 0.05

        dq = (
            J.T
            @ np.linalg.solve(
                J @ J.T
                + damping**2 * np.eye(3),
                error,
            )
        )

        # Smaller step for stability
        dq = np.clip(dq, -0.03, 0.03)

        for i, qid in enumerate(qpos_ids):
            data.qpos[qid] += dq[i]

    # --------------------------------------------------
    # IK failed — restore best solution found
    # --------------------------------------------------

    for i, qid in enumerate(qpos_ids):
        data.qpos[qid] = best_qpos[i]

    mujoco.mj_forward(model, data)

    reached = data.site_xpos[site_id].copy()

    print(f"\nWARNING: {label} IK did not converge.")
    print(f"{label}: requested = {target_pos}")
    print(f"{label}: best reached = {reached}")
    print(f"{label}: best error = {best_error:.6f} m")

    return best_qpos, False

def move_to_joint_target(
    model,
    data,
    viewer,
    actuator_ids,
    target,
    duration=2.0,
):

    start_ctrl = np.array(
        [data.ctrl[aid] for aid in actuator_ids]
    )

    steps = int(duration / model.opt.timestep)

    for step in range(steps):

        alpha = (step + 1) / steps

        # Smooth interpolation
        alpha = alpha * alpha * (3 - 2 * alpha)

        command = (
            (1 - alpha) * start_ctrl
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

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # ------------------------------------------------------
    # IDs
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

    for name in ARM_JOINTS:

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        qpos_ids.append(model.jnt_qposadr[jid])
        dof_ids.append(model.jnt_dofadr[jid])

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

    mujoco.mj_forward(model, data)

    # ------------------------------------------------------
    # Determine target positions from actual stone position
    # ------------------------------------------------------

    stone_pos = data.xpos[stone_id].copy()

    print("Stone position:", stone_pos)

    above_stone = stone_pos + np.array([
        0.0,
        0.0,
        0.15,
    ])

    pre_grasp = stone_pos + np.array([
        0.0,
        0.0,
        0.10,
    ])

    print("Above-stone target:", above_stone)
    print("Pre-grasp target:", pre_grasp)

    # ------------------------------------------------------
    # Solve IK offline
    # ------------------------------------------------------

    above_joint_target, above_ok = solve_ik(
        model,
        data,
        grasp_site_id,
        above_stone,
        qpos_ids,
        dof_ids,
        label="ABOVE",
    )

    print(
        "Above-stone joint target:",
        above_joint_target,
    )

    # Start second IK calculation from the first solution.
    pregrasp_joint_target, pregrasp_ok = solve_ik(
        model,
        data,
        grasp_site_id,
        pre_grasp,
        qpos_ids,
        dof_ids,
        label="PRE-GRASP",
    )

    if not above_ok:
        raise RuntimeError(
            "Above-stone target is not reachable. "
            "Execution stopped."
        )

    if not pregrasp_ok:
        raise RuntimeError(
            "Pre-grasp target is not reachable. "
            "Execution stopped."
        )

    print(
        "Pre-grasp joint target:",
        pregrasp_joint_target,
    )

    # ------------------------------------------------------
    # Reset actual simulation
    # ------------------------------------------------------

    mujoco.mj_resetData(model, data)

    # Open gripper.
    data.ctrl[left_finger] = GRIPPER_OPEN
    data.ctrl[right_finger] = GRIPPER_OPEN

    # ------------------------------------------------------
    # Execute trajectory
    # ------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        print("\nMoving ABOVE stone...")

        move_to_joint_target(
            model,
            data,
            viewer,
            actuator_ids,
            above_joint_target,
            duration=3.0,
        )

        time.sleep(1.0)

        print("Descending toward stone...")

        move_to_joint_target(
            model,
            data,
            viewer,
            actuator_ids,
            pregrasp_joint_target,
            duration=3.0,
        )

        print("Reached pre-grasp position.")

        # Hold final position for inspection.
        while viewer.is_running():

            for aid, value in zip(
                actuator_ids,
                pregrasp_joint_target,
            ):
                data.ctrl[aid] = value

            data.ctrl[left_finger] = GRIPPER_OPEN
            data.ctrl[right_finger] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()