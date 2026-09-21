from pathlib import Path
import random
import csv
import mujoco
from PIL import Image


# ==========================================================
# PATHS
# ==========================================================

# generate_dataset.py
#       ↓ parent
# evaluation/
#       ↓ parent
# task 2/
#       ↓ parent
# project root/
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

MODEL_PATH = PROJECT_ROOT / "scene.xml"

IMAGE_DIR = (
    Path(__file__).resolve().parent
    / "images"
)

GROUND_TRUTH_PATH = (
    Path(__file__).resolve().parent
    / "ground_truth.csv"
)

# Create images folder if it does not exist.
IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# DATASET SETTINGS
# ==========================================================

NUM_TRIALS = 20

OBJECTS = [
    "stone",
    "box_obj",
    "cylinder_obj",
]

# Randomized workspace.
X_RANGE = (0.22, 0.38)
Y_RANGE = (-0.20, 0.15)

# Minimum XY distance between object centres.
MIN_OBJECT_DISTANCE = 0.08

# ==========================================================
# CAMERA VIEWS
# ==========================================================

CAMERA_VIEWS = {
    "standard": {
        "azimuth": 90,
        "elevation": -75,
        "distance": 1.30,
    },

    "left": {
        "azimuth": 65,
        "elevation": -65,
        "distance": 1.30,
    },

    "right": {
        "azimuth": 115,
        "elevation": -65,
        "distance": 1.30,
    },

    "close": {
        "azimuth": 90,
        "elevation": -75,
        "distance": 1.00,
    },

    "far": {
        "azimuth": 90,
        "elevation": -75,
        "distance": 1.60,
    },
}

# ==========================================================
# MUJOCO HELPER
# ==========================================================

def get_freejoint_qpos_address(
    model,
    body_name,
):
    """
    Find where a free object's position begins
    inside MuJoCo's qpos array.

    Free-joint qpos:
    [x, y, z, qw, qx, qy, qz]
    """

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    joint_id = model.body_jntadr[
        body_id
    ]

    return model.jnt_qposadr[
        joint_id
    ]


# ==========================================================
# RANDOM POSITION GENERATOR
# ==========================================================

def generate_valid_position(
    existing_positions,
):
    """
    Generate a random XY position.

    The position is accepted only if it is at least
    MIN_OBJECT_DISTANCE away from all previously
    positioned objects.
    """

    max_attempts = 100

    for _ in range(max_attempts):

        x = random.uniform(
            X_RANGE[0],
            X_RANGE[1],
        )

        y = random.uniform(
            Y_RANGE[0],
            Y_RANGE[1],
        )

        valid = True

        for other_x, other_y in existing_positions:

            distance = (
                (x - other_x) ** 2
                + (y - other_y) ** 2
            ) ** 0.5

            if distance < MIN_OBJECT_DISTANCE:
                valid = False
                break

        if valid:
            return x, y

    raise RuntimeError(
        "Could not generate a valid object position."
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("Generating Task 2 evaluation dataset...")

    # ------------------------------------------------------
    # LOAD MUJOCO
    # ------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )

    camera = mujoco.MjvCamera()

    camera.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )

    # Aim approximately at the manipulation workspace.
    camera.lookat[:] = [
        0.30,
        0.00,
        0.05,
    ]

    # ------------------------------------------------------
    # CREATE CSV
    # ------------------------------------------------------

    with open(
        GROUND_TRUTH_PATH,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(csv_file)

        # CSV column names
        writer.writerow([
            "trial",
            "image",
            "view",
            "azimuth",
            "elevation",
            "distance",
            "object",
            "x",
            "y",
            "z",
        ])

        # --------------------------------------------------
        # GENERATE TRIALS
        # --------------------------------------------------

        for trial in range(
            1,
            NUM_TRIALS + 1,
        ):

            # ----------------------------------------------
            # SELECT CAMERA VIEW
            # ----------------------------------------------

            view_names = list(
                CAMERA_VIEWS.keys()
            )

            trials_per_view = (
                NUM_TRIALS // len(view_names)
            )

            view_index = (
                (trial - 1) // trials_per_view
            )

            view_name = view_names[
                view_index
            ]

            view = CAMERA_VIEWS[
                view_name
            ]

            camera.azimuth = view[
                "azimuth"
            ]

            camera.elevation = view[
                "elevation"
            ]

            camera.distance = view[
                "distance"
            ]

            # Reset simulation to scene.xml defaults.
            mujoco.mj_resetData(
                model,
                data,
            )

            placed_positions = []

            trial_positions = {}

            # ----------------------------------------------
            # RANDOMIZE OBJECTS
            # ----------------------------------------------

            for body_name in OBJECTS:

                qpos_address = (
                    get_freejoint_qpos_address(
                        model,
                        body_name,
                    )
                )

                new_x, new_y = (
                    generate_valid_position(
                        placed_positions
                    )
                )

                data.qpos[
                    qpos_address
                ] = new_x

                data.qpos[
                    qpos_address + 1
                ] = new_y

                placed_positions.append(
                    (new_x, new_y)
                )

            # Update MuJoCo state.
            mujoco.mj_forward(
                model,
                data,
            )

            # ----------------------------------------------
            # GET TRUE OBJECT POSITIONS
            # ----------------------------------------------

            for body_name in OBJECTS:

                body_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name,
                )

                position = (
                    data.xpos[
                        body_id
                    ].copy()
                )

                trial_positions[
                    body_name
                ] = position

            # ----------------------------------------------
            # RENDER CAMERA
            # ----------------------------------------------

            renderer.update_scene(
                data,
                camera=camera,
            )

            rgb_image = (
                renderer.render()
            )

            # ----------------------------------------------
            # SAVE IMAGE
            # ----------------------------------------------

            image_name = (
                f"trial_{trial:03d}.png"
            )

            image_path = (
                IMAGE_DIR
                / image_name
            )

            Image.fromarray(
                rgb_image
            ).save(
                image_path
            )

            # ----------------------------------------------
            # SAVE GROUND TRUTH
            # ----------------------------------------------

            for (
                body_name,
                position,
            ) in trial_positions.items():

                writer.writerow([
                    trial,
                    image_name,
                    view_name,
                    camera.azimuth,
                    camera.elevation,
                    camera.distance,
                    body_name,
                    position[0],
                    position[1],
                    position[2],
                ])

            print(
                f"Generated trial "
                f"{trial:03d}/{NUM_TRIALS}"
            )

    renderer.close()

    print(
        "\nDataset generation complete."
    )

    print(
        "Images:",
        IMAGE_DIR,
    )

    print(
        "Ground truth:",
        GROUND_TRUTH_PATH,
    )
# ==========================================================
# RUN PROGRAM
# ==========================================================

if __name__ == "__main__":
    main()