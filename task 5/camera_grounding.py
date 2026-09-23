import numpy as np
import mujoco


# ==========================================================
# CAMERA SETTINGS
# ==========================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def bbox_center(bbox):
    """
    Return the centre pixel of a bounding box.

    bbox format:
        [x_min, y_min, x_max, y_max]
    """

    x_min, y_min, x_max, y_max = bbox

    u = (
        x_min + x_max
    ) / 2.0

    v = (
        y_min + y_max
    ) / 2.0

    return np.array([
        u,
        v,
    ])


def get_camera_info(
    model,
    data,
    camera_name="overhead_cam",
):
    """
    Get the MuJoCo camera's world position,
    orientation matrix and field of view.
    """

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name,
    )

    if camera_id < 0:
        raise ValueError(
            f"Camera '{camera_name}' not found."
        )

    # Camera position in world coordinates.
    camera_position = (
        data.cam_xpos[
            camera_id
        ].copy()
    )

    # Camera orientation in world coordinates.
    camera_rotation = (
        data.cam_xmat[
            camera_id
        ]
        .reshape(3, 3)
        .copy()
    )

    # Vertical field of view in degrees.
    fovy = model.cam_fovy[
        camera_id
    ]

    return (
        camera_position,
        camera_rotation,
        fovy,
    )

def pixel_depth_to_world(
    model,
    data,
    u,
    v,
    depth,
    camera_name="overhead_cam",
    image_width=640,
    image_height=480,
):
    """
    Convert an RGB-D pixel into MuJoCo world coordinates.

    Inputs:
        u, v:
            Pixel coordinates.

        depth:
            Depth value from MuJoCo depth rendering.

    Returns:
        world_position:
            [x, y, z] in MuJoCo world coordinates.
    """

    # ------------------------------------------------------
    # CAMERA INFORMATION
    # ------------------------------------------------------

    (
        camera_position,
        camera_rotation,
        fovy,
    ) = get_camera_info(
        model,
        data,
        camera_name,
    )


    # ------------------------------------------------------
    # CAMERA INTRINSICS
    # ------------------------------------------------------

    fovy_rad = np.deg2rad(
        fovy
    )

    fy = (
        image_height
        / (
            2.0
            * np.tan(
                fovy_rad / 2.0
            )
        )
    )

    # Square pixels.
    fx = fy

    cx = image_width / 2.0
    cy = image_height / 2.0


    # ------------------------------------------------------
    # PIXEL + DEPTH -> CAMERA COORDINATES
    # ------------------------------------------------------

    # MuJoCo camera looks along local -Z.
    x_camera = (
        (u - cx)
        * depth
        / fx
    )

    y_camera = -(
        (v - cy)
        * depth
        / fy
    )

    z_camera = -depth


    point_camera = np.array([
        x_camera,
        y_camera,
        z_camera,
    ])


    # ------------------------------------------------------
    # CAMERA -> WORLD COORDINATES
    # ------------------------------------------------------

    world_position = (
        camera_position
        + camera_rotation
        @ point_camera
    )

    return world_position

def bbox_depth_to_world(
    model,
    data,
    bbox,
    depth,
    camera_name="overhead_cam",
):
    """
    Convert the centre of a detected bounding box
    and its depth into MuJoCo world coordinates.
    """

    u, v = bbox_center(
        bbox
    )

    return pixel_depth_to_world(
        model=model,
        data=data,
        u=u,
        v=v,
        depth=depth,
        camera_name=camera_name,
    )

