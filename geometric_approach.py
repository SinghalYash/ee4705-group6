import time
import numpy as np
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"

# ---------------------------------------------------------
# Robot geometry from scene.xml
# ---------------------------------------------------------

# Shoulder joint world height:
#
# base z = 0.10
# link1 z = 0.05
# vertical link = 0.20
#
# shoulder_lift is therefore at z ≈ 0.35
SHOULDER_Z = 0.35

# Horizontal arm links
L1 = 0.25
L2 = 0.20

# Distance from end of link2 to grasp region.
#
# wrist body:        0.20 already accounted for as L2
# wrist geom:        ~0.05
# gripper base:      0.08
#
# We initially approximate the final tool offset separately.
TOOL_LENGTH = 0.15


ARM_ACTUATORS = [
    "act_shoulder_pan",
    "act_shoulder_lift",
    "act_elbow",
    "act_wrist_pitch",
]


def get_actuator_id(model, name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )


def solve_geometric_ik(target_xyz):
    """
    Geometric IK for the simple 4-DOF arm.

    Strategy:
      1. shoulder_pan aims toward target in XY.
      2. shoulder_lift + elbow position wrist in vertical plane.
      3. wrist_pitch compensates so gripper stays approximately horizontal.
    """

    x, y, z = target_xyz

    # -----------------------------------------------------
    # 1. Base rotation
    # -----------------------------------------------------

    shoulder_pan = np.arctan2(y, x)

    radial_distance = np.sqrt(x**2 + y**2)

    # -----------------------------------------------------
    # 2. Account for horizontal gripper/tool length
    # -----------------------------------------------------

    wrist_r = radial_distance - TOOL_LENGTH

    wrist_z = z

    # Coordinates relative to shoulder joint.
    r = wrist_r
    h = wrist_z - SHOULDER_Z

    distance = np.sqrt(r**2 + h**2)

    print("\n--- GEOMETRIC IK ---")
    print("Target:", target_xyz)
    print("Radial target distance:", radial_distance)
    print("Desired wrist r:", r)
    print("Desired wrist height relative to shoulder:", h)
    print("Shoulder-to-wrist distance:", distance)

    # -----------------------------------------------------
    # Reachability
    # -----------------------------------------------------

    max_reach = L1 + L2
    min_reach = abs(L1 - L2)

    if distance > max_reach:
        raise ValueError(
            f"Target too far away. Distance={distance:.3f}, "
            f"max reach={max_reach:.3f}"
        )

    if distance < min_reach:
        raise ValueError(
            f"Target too close. Distance={distance:.3f}, "
            f"min reach={min_reach:.3f}"
        )

    # -----------------------------------------------------
    # 3. Elbow angle
    # -----------------------------------------------------

    cos_elbow = (
        r**2
        + h**2
        - L1**2
        - L2**2
    ) / (2 * L1 * L2)

    cos_elbow = np.clip(
        cos_elbow,
        -1.0,
        1.0,
    )

    # Elbow-down configuration
    elbow = np.arccos(cos_elbow)

    # -----------------------------------------------------
    # 4. Shoulder angle
    # -----------------------------------------------------

    shoulder_lift = (
        np.arctan2(-h, r)
        - np.arctan2(
            L2 * np.sin(elbow),
            L1 + L2 * np.cos(elbow),
        )
    )

    # -----------------------------------------------------
    # 5. Wrist compensation
    # -----------------------------------------------------

    # Keep the tool approximately horizontal.
    wrist_pitch = -(shoulder_lift + elbow)

    result = np.array([
        shoulder_pan,
        shoulder_lift,
        elbow,
        wrist_pitch,
    ])

    print("Shoulder pan:", np.degrees(shoulder_pan))
    print("Shoulder lift:", np.degrees(shoulder_lift))
    print("Elbow:", np.degrees(elbow))
    print("Wrist pitch:", np.degrees(wrist_pitch))

    return result


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

        # Smoothstep interpolation
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

        mujoco.mj_step(
            model,
            data,
        )

        viewer.sync()

        time.sleep(
            model.opt.timestep
        )


def main():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    mujoco.mj_forward(
        model,
        data,
    )

    # -----------------------------------------------------
    # Find stone
    # -----------------------------------------------------

    stone_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "stone",
    )

    stone_pos = (
        data.xpos[stone_id].copy()
    )

    print(
        "Stone position:",
        stone_pos,
    )

    # -----------------------------------------------------
    # SIDE-APPROACH TARGET
    # -----------------------------------------------------
    #
    # For now, target a point slightly ABOVE the stone.
    # We are NOT touching it yet.

    approach_target = np.array([
        stone_pos[0],
        stone_pos[1],
        0.16,
    ])

    joint_target = solve_geometric_ik(
        approach_target
    )

    # -----------------------------------------------------
    # Actuator IDs
    # -----------------------------------------------------

    actuator_ids = [
        get_actuator_id(
            model,
            name,
        )
        for name in ARM_ACTUATORS
    ]

    left_finger = get_actuator_id(
        model,
        "act_finger_left",
    )

    right_finger = get_actuator_id(
        model,
        "act_finger_right",
    )

    # -----------------------------------------------------
    # Gripper open
    # -----------------------------------------------------

    data.ctrl[left_finger] = 0.025
    data.ctrl[right_finger] = 0.025

    # -----------------------------------------------------
    # Execute
    # -----------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        print(
            "\nMoving to geometric "
            "approach pose..."
        )

        move_smoothly(
            model,
            data,
            viewer,
            actuator_ids,
            joint_target,
            duration=4.0,
        )

        print(
            "Reached commanded pose."
        )

        grasp_site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "grasp_site",
        )

        mujoco.mj_forward(model, data)

        actual_grasp_pos = data.site_xpos[
            grasp_site_id
        ].copy()

        actual_stone_pos = data.xpos[
            stone_id
        ].copy()

        print("\n--- ACTUAL MUJOCO RESULT ---")
        print(
            "Commanded target:",
            approach_target,
        )
        print(
            "Actual grasp-site:",
            actual_grasp_pos,
        )
        print(
            "Stone:",
            actual_stone_pos,
        )
        print(
            "Position error:",
            np.linalg.norm(
                actual_grasp_pos
                - approach_target
            ),
        )

        # Hold pose
        while viewer.is_running():

            for aid, value in zip(
                actuator_ids,
                joint_target,
            ):
                data.ctrl[aid] = value

            data.ctrl[left_finger] = 0.025
            data.ctrl[right_finger] = 0.025

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