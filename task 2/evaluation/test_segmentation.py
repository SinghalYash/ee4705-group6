from pathlib import Path

import mujoco
import numpy as np

from PIL import (
    Image,
    ImageDraw,
)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
)

MODEL_PATH = (
    PROJECT_ROOT / "scene.xml"
)

OUTPUT_DIR = (
    Path(__file__).resolve().parent
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # LOAD MUJOCO
    # ------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_PATH)
    )

    data = mujoco.MjData(
        model
    )

    mujoco.mj_forward(
        model,
        data,
    )

    renderer = mujoco.Renderer(
        model,
        height=480,
        width=640,
    )


    # ------------------------------------------------------
    # RGB IMAGE
    # ------------------------------------------------------

    renderer.update_scene(
        data,
        camera="overhead_cam",
    )

    rgb = renderer.render()


    # ------------------------------------------------------
    # SEGMENTATION IMAGE
    # ------------------------------------------------------

    renderer.enable_segmentation_rendering()

    renderer.update_scene(
        data,
        camera="overhead_cam",
    )

    segmentation = renderer.render()

    renderer.disable_segmentation_rendering()


    print(
        "Segmentation shape:",
        segmentation.shape,
    )

    print(
        "Segmentation dtype:",
        segmentation.dtype,
    )


    # ------------------------------------------------------
    # FIND BLUE CUBE GEOMETRY ID
    # ------------------------------------------------------

    box_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "box_geom",
    )

    print(
        "box_geom ID:",
        box_geom_id,
    )


    # ------------------------------------------------------
    # FIND PIXELS BELONGING TO BLUE CUBE
    # ------------------------------------------------------

    object_ids = (
        segmentation[:, :, 0]
    )

    object_types = (
        segmentation[:, :, 1]
    )

    box_mask = (
        (object_ids == box_geom_id)
        &
        (
            object_types
            == mujoco.mjtObj.mjOBJ_GEOM
        )
    )


    y_pixels, x_pixels = np.where(
        box_mask
    )


    # ------------------------------------------------------
    # CHECK VISIBILITY
    # ------------------------------------------------------

    if len(x_pixels) == 0:

        print(
            "box_geom is not visible "
            "in the camera."
        )

    else:

        # --------------------------------------------------
        # CALCULATE TRUE BOUNDING BOX
        # --------------------------------------------------

        x_min = int(
            x_pixels.min()
        )

        y_min = int(
            y_pixels.min()
        )

        x_max = int(
            x_pixels.max()
        )

        y_max = int(
            y_pixels.max()
        )

        true_bbox = [
            x_min,
            y_min,
            x_max,
            y_max,
        ]

        print(
            "True cube bounding box:",
            true_bbox,
        )


        # --------------------------------------------------
        # DRAW TRUE BOUNDING BOX
        # --------------------------------------------------

        image = Image.fromarray(
            rgb
        )

        draw = ImageDraw.Draw(
            image
        )

        draw.rectangle(
            true_bbox,
            outline="red",
            width=4,
        )

        draw.text(
            (
                x_min,
                max(
                    0,
                    y_min - 20,
                ),
            ),
            "TRUE blue cube",
            fill="red",
        )


        # --------------------------------------------------
        # SAVE VISUALIZATION
        # --------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / "segmentation_bbox_test.png"
        )

        image.save(
            output_path
        )

        print(
            "Saved bbox visualization:",
            output_path,
        )


    renderer.close()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()