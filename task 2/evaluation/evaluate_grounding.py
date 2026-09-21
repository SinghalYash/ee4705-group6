from pathlib import Path
import csv

from PIL import Image

import sys


# ==========================================================
# PATHS
# ==========================================================

TASK2_DIR = (
    Path(__file__).resolve().parent.parent
)

IMAGE_DIR = (
    Path(__file__).resolve().parent
    / "images"
)

RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "evaluation_results.csv"
)


# Allow imports from task 2/
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(TASK2_DIR),
    )


# ==========================================================
# IMPORT TASK 2
# ==========================================================

from qwen_local import (
    load_model,
    ask_qwen,
)


# ==========================================================
# EVALUATION TARGETS
# ==========================================================

TARGETS = [
    "blue cube",
    "grey stone",
    "green cylinder",
]


NUM_TRIALS = 20

def main():

    print(
        "Loading Qwen for evaluation..."
    )

    model, processor = load_model()

    print(
        "\nStarting grounding evaluation..."
    )

    with open(
        RESULTS_PATH,
        "w",
        newline="",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "trial",
            "image",
            "target",
            "response",
        ])

        for trial in range(
            1,
            NUM_TRIALS + 1,
        ):

            image_name = (
                f"trial_{trial:03d}.png"
            )

            image_path = (
                IMAGE_DIR
                / image_name
            )

            image = Image.open(
                image_path
            ).convert("RGB")


            # Rotate through target objects.
            target = TARGETS[
                (trial - 1)
                % len(TARGETS)
            ]


            query = f"""
Locate the {target} in this image.

If the target is visible, return its bounding box.

Return JSON only:

{{
    "status": "success",
    "target": "{target}",
    "bbox": [x_min, y_min, x_max, y_max]
}}

If the target is not visible:

{{
    "status": "not_found",
    "target": "{target}"
}}
"""


            response = ask_qwen(
                model,
                processor,
                image,
                query,
            )


            writer.writerow([
                trial,
                image_name,
                target,
                response,
            ])


            print(
                f"Trial {trial:03d}: "
                f"{target}"
            )

            print(
                response
            )

            print(
                "-" * 50
            )

    print(
        "\nEvaluation complete."
    )

    print(
        "Results saved to:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()