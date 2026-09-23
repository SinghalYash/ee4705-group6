from pathlib import Path
import random
import csv

import mujoco
import numpy as np
from PIL import Image


# ==========================================================
# PATHS
# ==========================================================

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

# Create image directory if it does not exist.
IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# DATASET SETTINGS
# ==========================================================

NUM_TRIALS = 20

# Fixed random seed so that the same dataset can be
# reproduced every time the script is run.
RANDOM_SEED = 4705


# Bodies that will be randomized.
OBJECTS = [
    "stone",
    "box_obj",
    "cylinder_obj",
]


# Map each body to the geometry that appears in
# MuJoCo's segmentation rendering.
OBJECT_GEOMS = {
    "stone": "stone_geom",
    "box_obj": "box_geom",
    "cylinder_obj": "cylinder_geom",
}


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

    A free joint stores:

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
# TRUE BOUNDING BOX FROM SEGMENTATION
# ==========================================================

def get_true_bbox(
    model,
    segmentation,
    geom_name,
):
    """
    Calculate the exact visible 2D bounding box of a
    MuJoCo geometry using the segmentation image.

    Returns:

        [x_min, y_min, x_max, y_max]

    If the object is not visible, returns None.
    """

    # Get MuJoCo geometry ID.
    geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    # MuJoCo segmentation output:
    #
    # channel 0 -> object ID
    # channel 1 -> object type

    object_ids = segmentation[:, :, 0]

    object_types = segmentation[:, :, 1]


    # Select only pixels belonging to this geometry.
    mask = (
        (object_ids == geom_id)
        &
        (
            object_types
            == mujoco.mjtObj.mjOBJ_GEOM
        )
    )


    # Get coordinates of all matching pixels.
    y_pixels, x_pixels = np.where(
        mask
    )


    # No matching pixels means the object is not visible.
    if len(x_pixels) == 0:

        return None


    # Calculate visible bounding box.
    return [
        int(x_pixels.min()),
        int(y_pixels.min()),
        int(x_pixels.max()),
        int(y_pixels.max()),
    ]


# ==========================================================
# RANDOM POSITION GENERATOR
# ==========================================================

def generate_valid_position(
    existing_positions,
):
    """
    Generate a random XY position.

    A position is accepted only if it is at least
    MIN_OBJECT_DISTANCE away from all objects that
    have already been positioned.
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

    # Make dataset generation reproducible.
    random.seed(
        RANDOM_SEED
    )


    print(
        "Generating Task 2 evaluation dataset..."
    )


    # ------------------------------------------------------
    # LOAD MUJOCO MODEL
    # ------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )


    # ------------------------------------------------------
    # CREATE RENDERER
    # ------------------------------------------------------

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )


    # ------------------------------------------------------
    # CREATE FREE CAMERA
    # ------------------------------------------------------

    camera = mujoco.MjvCamera()

    camera.type = (
        mujoco.mjtCamera.mjCAMERA_FREE
    )


    # Aim at the manipulation workspace.
    camera.lookat[:] = [
        0.30,
        0.00,
        0.05,
    ]


    # ------------------------------------------------------
    # CAMERA VIEW SETUP
    # ------------------------------------------------------

    view_names = list(
        CAMERA_VIEWS.keys()
    )

    trials_per_view = (
        NUM_TRIALS
        // len(view_names)
    )


    # ------------------------------------------------------
    # CREATE GROUND-TRUTH CSV
    # ------------------------------------------------------

    with open(
        GROUND_TRUTH_PATH,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )


        # --------------------------------------------------
        # CSV HEADER
        # --------------------------------------------------

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
            "bbox_xmin",
            "bbox_ymin",
            "bbox_xmax",
            "bbox_ymax",
        ])


        # --------------------------------------------------
        # GENERATE TRIALS
        # --------------------------------------------------

        for trial in range(
            1,
            NUM_TRIALS + 1,
        ):


            # ==============================================
            # SELECT CAMERA VIEW
            # ==============================================

            view_index = (
                (trial - 1)
                // trials_per_view
            )


            # Safety in case NUM_TRIALS is changed later.
            view_index = min(
                view_index,
                len(view_names) - 1,
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


            # ==============================================
            # RESET SIMULATION
            # ==============================================

            mujoco.mj_resetData(
                model,
                data,
            )


            # ==============================================
            # RANDOMIZE OBJECT POSITIONS
            # ==============================================

            placed_positions = []


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


                # Change X position.
                data.qpos[
                    qpos_address
                ] = new_x


                # Change Y position.
                data.qpos[
                    qpos_address + 1
                ] = new_y


                # Z is left unchanged.
                placed_positions.append(
                    (
                        new_x,
                        new_y,
                    )
                )


            # ==============================================
            # UPDATE MUJOCO STATE
            # ==============================================

            mujoco.mj_forward(
                model,
                data,
            )


            # ==============================================
            # RECORD TRUE WORLD POSITIONS
            # ==============================================

            trial_positions = {}


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


            # ==============================================
            # RENDER RGB IMAGE
            # ==============================================

            renderer.update_scene(
                data,
                camera=camera,
            )


            rgb_image = (
                renderer.render()
            )


            # ==============================================
            # RENDER SEGMENTATION
            # ==============================================

            renderer.enable_segmentation_rendering()


            renderer.update_scene(
                data,
                camera=camera,
            )


            segmentation = (
                renderer.render()
            )


            renderer.disable_segmentation_rendering()


            # ==============================================
            # SAVE RGB IMAGE
            # ==============================================

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


            # ==============================================
            # SAVE GROUND TRUTH FOR EACH OBJECT
            # ==============================================

            for (
                body_name,
                position,
            ) in trial_positions.items():


                # ------------------------------------------
                # GET GEOMETRY NAME
                # ------------------------------------------

                geom_name = OBJECT_GEOMS[
                    body_name
                ]


                # ------------------------------------------
                # GET TRUE IMAGE BOUNDING BOX
                # ------------------------------------------

                true_bbox = get_true_bbox(
                    model,
                    segmentation,
                    geom_name,
                )


                # ------------------------------------------
                # HANDLE INVISIBLE OBJECT
                # ------------------------------------------

                if true_bbox is None:

                    bbox_values = [
                        "",
                        "",
                        "",
                        "",
                    ]

                else:

                    bbox_values = (
                        true_bbox
                    )


                # ------------------------------------------
                # WRITE CSV ROW
                # ------------------------------------------

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
                    *bbox_values,
                ])


            print(
                f"Generated trial "
                f"{trial:03d}/{NUM_TRIALS} "
                f"({view_name})"
            )


    # ------------------------------------------------------
    # CLEAN UP
    # ------------------------------------------------------

    renderer.close()


    # ------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------

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