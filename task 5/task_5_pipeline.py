import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image

from camera_grounding import bbox_depth_to_world


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK4_DIR = PROJECT_ROOT / "task 4"

for path in [PROJECT_ROOT, TASK4_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ==========================================================
# TASK IMPORTS
# ==========================================================

# Task 2 and Task 3 are real Python packages (task_2/, task_3/), so
# they are imported by their package-qualified names. This keeps each
# module loaded exactly once -- previously this file added task_2's
# own folder to sys.path and imported "perception" bare, while
# task_3/scene_builder.py separately imported "task_2.perception"
# dotted, which silently created two different copies of the same
# module (and, with it, two different DetectedObject/PerceptionResult
# classes). Importing everything the same way avoids that.
from task_2.perception import ground_object
from task_3.planner import plan_from_instruction

# Task 4 lives in a folder with a space in its name ("task 4"), so it
# cannot be imported as a normal dotted package -- its folder is put
# on sys.path above, and its module is imported directly instead.
from task_4.task_4_executor import Task4Executor

# Optional: lets this pipeline accept a *spoken* instruction as well
# as a typed one.
from task_3.voice_input import listen_and_transcribe


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
):
    """
    Build the callback Task 4 calls (as robot.perception_callback)
    every time it needs to check whether a named target is currently
    visible. Task 2's cloud backend (task_2/qwen_backend.py) needs no
    locally loaded model, so nothing is threaded through here besides
    the robot handle.
    """

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

        # Run Task 2. (model/processor are unused by the cloud
        # backend -- ground_object keeps the parameters only so its
        # signature still matches the older local-model tests.)
        result = ground_object(
            None,
            None,
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

            # Save the RGB-D estimate for Task 4.
            robot.perceived_target_position = (
                estimated_position.copy()
            )

            robot.perceived_target_name = (
                target_name
            )


            # Ground truth ONLY for evaluation -- this comparison is
            # purely diagnostic logging, so if target_name isn't a
            # name the simulator recognises (e.g. Task 4's name
            # normalisation didn't map it to a known canonical name),
            # skip the comparison instead of crashing the whole run.
            # The actual perception result returned below does not
            # depend on any of this.
            try:

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

            except Exception as error:

                print(
                    "(skipping ground-truth comparison -- "
                    f"'{target_name}' not recognised: {error})"
                )

        # Task 4 only needs True / False
        # from is_target_visible().
        return (
            result.status == "success"
            and result.target is not None
        )

    return detect_target


# ==========================================================
# INSTRUCTION INPUT (typed or spoken)
# ==========================================================

def get_instruction():
    """
    Ask the user for an instruction, typed or spoken.

    Speech goes through Task 3's listen_and_transcribe(), which now
    includes a domain-vocabulary correction pass (see
    task_3/voice_input.py) to fix common misheard words such as
    "read" -> "red" before the text ever reaches the planner.
    """

    choice = input(
        "\nType an instruction, or press ENTER to speak it: "
    ).strip()

    if choice:
        return choice

    for attempt in range(3):
        spoken = listen_and_transcribe()
        if spoken:
            return spoken
        print("Didn't catch that -- let's try again.")

    print("No instruction understood after 3 tries; using the typed fallback.")
    return input("Type an instruction: ").strip()


# ==========================================================
# MAIN TASK 5 PIPELINE
# ==========================================================

if __name__ == "__main__":

  # ------------------------------------------------------
    # CREATE TASK 4 EXECUTOR ONCE
    # ------------------------------------------------------

    print("Creating Task 4 executor...")

    executor = Task4Executor()

    # Connect Task 2 perception to Task 4.
    executor.robot.perception_callback = (
        create_perception_callback(
            executor.robot,
        )
    )

    print("\nTask 5 system ready.")
    print("Type 'quit' or 'exit' to stop.")


    # ======================================================
    # CONTINUOUS TASK 5 LOOP
    # ======================================================

    while True:

        # --------------------------------------------------
        # GET NEXT INSTRUCTION
        # --------------------------------------------------

        instruction = get_instruction()

        if not instruction:
            continue

        if instruction.lower() in {
            "quit",
            "exit",
            "q",
        }:
            print("\nStopping Task 5.")
            break


        # --------------------------------------------------
        # CAPTURE CURRENT SCENE
        # --------------------------------------------------

        print(
            "\nCapturing scene for Task 3 planning..."
        )

        planning_image, _ = capture_rgbd(
            executor.robot,
            camera_name="overhead_cam",
        )


        # --------------------------------------------------
        # TASK 3 PLANNING
        # --------------------------------------------------

        print(
            "\nGenerating Task 3 plan..."
        )

        plan = plan_from_instruction(
            instruction,
            image=planning_image,
        )

        print(
            "\nTask 3 plan:"
        )

        print(plan)


        # --------------------------------------------------
        # CHECK PLAN
        # --------------------------------------------------

        if not plan.get(
            "feasible",
            False,
        ):

            print(
                "\nTask 3 marked the "
                "instruction as infeasible."
            )

            # IMPORTANT:
            # Do not exit the entire program.
            # Go back and ask for another instruction.
            continue


        # --------------------------------------------------
        # TASK 4 EXECUTION
        # --------------------------------------------------

        print(
            "\nExecuting Task 5 pipeline..."
        )

        report = executor.execute_plan(
            plan
        )


        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

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

        print(
            "\nReady for next instruction."
        )