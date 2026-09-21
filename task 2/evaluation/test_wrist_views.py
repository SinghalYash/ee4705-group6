from pathlib import Path

import mujoco
from PIL import Image


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

MODEL_PATH = PROJECT_ROOT / "scene.xml"

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "wrist_views"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# TEST ARM POSES
# ==========================================================

WRIST_POSES = {
    "center": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.4,
        "elbow": 0.8,
        "wrist_pitch": -0.3,
    },

    "left": {
        "shoulder_pan": -0.5,
        "shoulder_lift": -0.4,
        "elbow": 0.8,
        "wrist_pitch": -0.3,
    },

    "right": {
        "shoulder_pan": 0.5,
        "shoulder_lift": -0.4,
        "elbow": 0.8,
        "wrist_pitch": -0.3,
    },

    "close": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.6,
        "elbow": 1.0,
        "wrist_pitch": -0.5,
    },

    "high": {
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.2,
        "elbow": 0.5,
        "wrist_pitch": -0.4,
    },
}


# ==========================================================
# SET JOINT POSITION
# ==========================================================

def set_joint_position(
    model,
    data,
    joint_name,
    value,
):
    """
    Directly set a joint position in MuJoCo.
    """

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )

    qpos_address = model.jnt_qposadr[
        joint_id
    ]

    data.qpos[
        qpos_address
    ] = value


# ==========================================================
# MAIN
# ==========================================================

def main():

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )

    for view_name, pose in WRIST_POSES.items():

        # Return simulation to original state.
        mujoco.mj_resetData(
            model,
            data,
        )

        # Move arm into requested pose.
        for joint_name, value in pose.items():

            set_joint_position(
                model,
                data,
                joint_name,
                value,
            )

        # Recalculate body/camera positions.
        mujoco.mj_forward(
            model,
            data,
        )

        # Render using the REAL wrist camera.
        renderer.update_scene(
            data,
            camera="wrist_cam",
        )

        rgb_image = renderer.render()

        # Save image.
        output_path = (
            OUTPUT_DIR
            / f"wrist_{view_name}.png"
        )

        Image.fromarray(
            rgb_image
        ).save(
            output_path
        )

        print(
            f"Saved {view_name}: "
            f"{output_path}"
        )

    renderer.close()

    print(
        "\nWrist camera view test complete."
    )


if __name__ == "__main__":
    main()