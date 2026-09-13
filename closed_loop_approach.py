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

# We stop when the physical grasp site is this close.
POSITION_TOLERANCE = 0.008

# Maximum joint-command change per control iteration.
MAX_DQ = 0.008

# Damped least-squares parameter.
DAMPING = 0.08


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

    stone_pos = data.xpos[stone_id].copy()

    # For the first physical test, stay safely above the stone.
    approach_target = stone_pos + np.array([
        0.0,
        0.0,
        0.13,
    ])

    print("Stone position:", stone_pos)
    print("Approach target:", approach_target)

    # Start arm commands at the actual initial joint positions.
    joint_commands = np.array([
        data.qpos[qid]
        for qid in qpos_ids
    ])

    # ------------------------------------------------------
    # Viewer + closed-loop control
    # ------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        # Open fingers.
        data.ctrl[left_finger] = GRIPPER_OPEN
        data.ctrl[right_finger] = GRIPPER_OPEN

        print("\nStarting closed-loop approach...")

        max_steps = 6000

        for step in range(max_steps):

            if not viewer.is_running():
                break

            # ----------------------------------------------
            # Measure ACTUAL physical position
            # ----------------------------------------------

            mujoco.mj_forward(model, data)

            current_pos = (
                data.site_xpos[
                    grasp_site_id
                ].copy()
            )

            error = (
                approach_target
                - current_pos
            )

            error_norm = np.linalg.norm(error)

            # ----------------------------------------------
            # Success
            # ----------------------------------------------

            if error_norm < POSITION_TOLERANCE:

                print("\nAPPROACH SUCCESS")
                print(
                    "Target:",
                    approach_target,
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

                break

            # ----------------------------------------------
            # Jacobian from CURRENT physical configuration
            # ----------------------------------------------

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
                grasp_site_id,
            )

            J = jacp[:, dof_ids]

            # ----------------------------------------------
            # Damped least-squares Cartesian correction
            # ----------------------------------------------

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

            # IMPORTANT:
            # Build the next command around the ACTUAL
            # physical joint positions, not an offline state.
            actual_q = np.array([
                data.qpos[qid]
                for qid in qpos_ids
            ])

            joint_commands = (
                actual_q + dq
            )

            # ----------------------------------------------
            # Respect joint limits
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Send joint commands
            # ----------------------------------------------

            for aid, command in zip(
                actuator_ids,
                joint_commands,
            ):
                data.ctrl[aid] = command

            # Keep gripper open.
            data.ctrl[left_finger] = GRIPPER_OPEN
            data.ctrl[right_finger] = GRIPPER_OPEN

            # ----------------------------------------------
            # Physics
            # ----------------------------------------------

            # Several physics steps per Cartesian update
            # gives the position servos time to respond.
            for _ in range(5):

                mujoco.mj_step(
                    model,
                    data,
                )

            viewer.sync()

            # ----------------------------------------------
            # Diagnostics
            # ----------------------------------------------

            if step % 100 == 0:

                print(
                    f"step {step:4d} | "
                    f"error = "
                    f"{error_norm:.4f} m | "
                    f"site = "
                    f"{np.round(current_pos, 3)}"
                )

            time.sleep(
                model.opt.timestep * 5
            )

        else:
            print(
                "\nAPPROACH FAILED: "
                "maximum steps reached."
            )

        # --------------------------------------------------
        # Final physical result
        # --------------------------------------------------

        mujoco.mj_forward(
            model,
            data,
        )

        final_pos = (
            data.site_xpos[
                grasp_site_id
            ].copy()
        )

        final_error = np.linalg.norm(
            approach_target
            - final_pos
        )

        print("\n--- FINAL PHYSICAL RESULT ---")
        print("Stone:", data.xpos[stone_id])
        print("Target:", approach_target)
        print("Grasp site:", final_pos)
        print("Error:", final_error)

        print("\nFinal joint positions:")

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
        # Hold final configuration
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