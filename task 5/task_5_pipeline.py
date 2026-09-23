import numpy as np
import sys
from pathlib import Path

import mujoco
from PIL import Image

from camera_grounding import bbox_depth_to_world


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

TASK2_DIR = (
    PROJECT_ROOT / "task 2"
)

TASK4_DIR = (
    PROJECT_ROOT / "task 4"
)


for path in [
    PROJECT_ROOT,
    TASK2_DIR,
    TASK4_DIR,
]:

    if str(path) not in sys.path:

        sys.path.insert(
            0,
            str(path),
        )


# ==========================================================
# TASK IMPORTS
# ==========================================================

# Task 2
from qwen_backend import load_model
from perception import ground_object

# Task 3
from planner import plan_from_instruction

# Task 4
from task_4_executor import Task4Executor

# ==========================================================
# CAMERA CAPTURE
# ==========================================================

def capture_rgbd(
    robot,
    camera_name="overhead_cam",
):
    """
    Capture aligned RGB and depth images from the
    same MuJoCo camera.

    Returns:
        rgb_image: PIL RGB image
        depth_image: NumPy depth array
    """

    renderer = mujoco.Renderer(
        robot.model,
        height=480,
        width=640,
    )

    # ------------------------------------------------------
    # RGB
    # ------------------------------------------------------

    renderer.update_scene(
        robot.data,
        camera=camera_name,
    )

    rgb_array = renderer.render()

    rgb_image = Image.fromarray(
        rgb_array
    )


    # ------------------------------------------------------
    # DEPTH
    # ------------------------------------------------------

    renderer.enable_depth_rendering()

    renderer.update_scene(
        robot.data,
        camera=camera_name,
    )

    depth_image = renderer.render()

    renderer.disable_depth_rendering()


    renderer.close()

    return (
        rgb_image,
        depth_image,
    )

# ==========================================================
# VISUAL TARGET NAMES
# ==========================================================

VISUAL_TARGET_NAMES = {
    "box": "blue cube",
    "stone": "grey stone",
    "cylinder": "green cylinder",
}

# ==========================================================
# TASK 2 PERCEPTION CALLBACK
# ==========================================================

def create_perception_callback(
    robot,
    model,
    processor,
):

    def detect_target(
        target_name,
    ):

        # Convert Task 4's canonical name
        # into a visual description.
        visual_target = VISUAL_TARGET_NAMES.get(
            target_name,
            target_name,
        )

        print(
            "\nTask 2 checking for:",
            visual_target,
        )

        # Capture the current simulation image.
        image, depth_image = capture_rgbd(
            robot,
            camera_name="overhead_cam",
        )

        # Run Task 2.
        result = ground_object(
            model,
            processor,
            image,
            visual_target,
        )

        print(
            "Perception status:",
            result.status,
        )

        print(
            "Grounded target:",
            result.target,
        )

        if (
            result.status == "success"
            and result.target is not None
        ):

            x_min, y_min, x_max, y_max = (
                result.target.bbox
            )

            centre_x = round(
                (x_min + x_max) / 2
            )

            centre_y = round(
                (y_min + y_max) / 2
            )

            depth_value = depth_image[
                centre_y,
                centre_x,
            ]
            
            estimated_position = bbox_depth_to_world(
                model=robot.model,
                data=robot.data,
                bbox=result.target.bbox,
                depth=depth_value,
                camera_name="overhead_cam",
            )


            # Ground truth ONLY for evaluation.
            true_position = robot.get_object_position(
                target_name
            )


            position_error = np.linalg.norm(
                estimated_position
                - true_position
            )


            print(
                "RGB-D estimated XYZ:",
                estimated_position,
            )

            print(
                "True MuJoCo XYZ:",
                true_position,
            )

            print(
                "3D localisation error:",
                position_error,
                "m",
            )

            print(
                "Bounding-box centre:",
                (
                    centre_x,
                    centre_y,
                ),
            )

            print(
                "Depth at target centre:",
                depth_value,
                "m",
            )

        # Task 4 only needs True / False
        # from is_target_visible().
        return (
            result.status == "success"
            and result.target is not None
        )

    return detect_target

# ==========================================================
# MAIN TASK 5 PIPELINE
# ==========================================================

if __name__ == "__main__":

    instruction = (
        "Move the blue cube to the red area."
    )


    # ------------------------------------------------------
    # TASK 2 MODEL
    # ------------------------------------------------------

    print(
        "Loading Task 2 model..."
    )

    model, processor = load_model()


    # ------------------------------------------------------
    # TASK 4 EXECUTOR
    # ------------------------------------------------------

    print(
        "Creating Task 4 executor..."
    )

    executor = Task4Executor()


    # ------------------------------------------------------
    # CONNECT TASK 2 TO TASK 4
    # ------------------------------------------------------

    executor.robot.perception_callback = (
        create_perception_callback(
            executor.robot,
            model,
            processor,
        )
    )


    # ------------------------------------------------------
    # TASK 3 PLANNING
    # ------------------------------------------------------

    print(
        "\nGenerating Task 3 plan..."
    )

    plan = plan_from_instruction(
        instruction
    )


    print(
        "\nTask 3 plan:"
    )

    print(
        plan
    )


    # ------------------------------------------------------
    # CHECK PLAN
    # ------------------------------------------------------

    if not plan.get(
        "feasible",
        False,
    ):

        print(
            "\nTask 3 marked the "
            "instruction as infeasible."
        )

        raise SystemExit


    # ------------------------------------------------------
    # TASK 4 EXECUTION
    # ------------------------------------------------------

    print(
        "\nExecuting Task 5 pipeline..."
    )

    report = executor.execute_plan(
        plan
    )


    # ------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TASK 5 FINAL RESULT"
    )

    print(
        "=" * 60
    )

    print(
        "Instruction:",
        instruction,
    )

    print(
        "Success:",
        report.success,
    )

    print(
        "Completed actions:",
        report.completed_actions,
        "/",
        report.total_actions,
    )

    print(
        "Failure reason:",
        report.failure_reason,
    )

    print(
        "Final placement error:",
        report.final_placement_error,
    )