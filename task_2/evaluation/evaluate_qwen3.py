from pathlib import Path
import csv
import json
import re
import sys

from PIL import Image


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
    / "qwen3_results.csv"
)


# Allow imports from task 2/
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(TASK2_DIR),
    )


# ==========================================================
# IMPORT QWEN3
# ==========================================================

from qwen3_local import (
    load_model,
    ask_qwen,
    qwen3_bbox_to_pixels,
)


# ==========================================================
# SETTINGS
# ==========================================================

TARGETS = [
    "blue cube",
    "grey stone",
    "green cylinder",
]

NUM_TRIALS = 20


# ==========================================================
# PARSE QWEN RESPONSE
# ==========================================================

def parse_response(response):

    cleaned = re.sub(
        r"```(?:json)?|```",
        "",
        response,
    ).strip()

    try:
        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        return None


# ==========================================================
# MAIN
# ==========================================================

def main():

    print(
        "Loading Qwen3-VL-8B for evaluation..."
    )

    model, processor = load_model()

    print(
        "\nStarting Qwen3 grounding evaluation..."
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
            "status",
            "bbox",
            "raw_response",
        ])


        for trial in range(
            1,
            NUM_TRIALS + 1,
        ):

            # ----------------------------------------------
            # IMAGE
            # ----------------------------------------------

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


            # ----------------------------------------------
            # TARGET
            # ----------------------------------------------

            target = TARGETS[
                (trial - 1)
                % len(TARGETS)
            ]


            # ----------------------------------------------
            # PROMPT
            # ----------------------------------------------

            query = f"""
Locate the {target} in this image.

Return the bounding box using normalized coordinates
from 0 to 1000 in this order:

[x_min, y_min, x_max, y_max]

Return JSON only:

{{
    "status": "success",
    "target": "{target}",
    "bbox": [x_min, y_min, x_max, y_max]
}}

If the target is not visible, return:

{{
    "status": "not_found",
    "target": "{target}"
}}
"""


            # ----------------------------------------------
            # RUN QWEN3
            # ----------------------------------------------

            response = ask_qwen(
                model,
                processor,
                image,
                query,
            )


            # ----------------------------------------------
            # PARSE RESPONSE
            # ----------------------------------------------

            parsed = parse_response(
                response
            )


            if parsed is None:

                status = "invalid_response"
                pixel_bbox = None

            else:

                status = parsed.get(
                    "status",
                    "unknown",
                )

                qwen_bbox = parsed.get(
                    "bbox"
                )


                if (
                    status == "success"
                    and isinstance(
                        qwen_bbox,
                        list,
                    )
                    and len(qwen_bbox) == 4
                ):

                    pixel_bbox = (
                        qwen3_bbox_to_pixels(
                            qwen_bbox,
                            image.width,
                            image.height,
                        )
                    )

                else:

                    pixel_bbox = None


            # ----------------------------------------------
            # SAVE RESULT
            # ----------------------------------------------

            writer.writerow([
                trial,
                image_name,
                target,
                status,
                (
                    json.dumps(pixel_bbox)
                    if pixel_bbox
                    else ""
                ),
                response,
            ])


            # ----------------------------------------------
            # DISPLAY
            # ----------------------------------------------

            print(
                f"Trial {trial:03d}: "
                f"{target}"
            )

            print(
                "Status:",
                status,
            )

            print(
                "Pixel bbox:",
                pixel_bbox,
            )

            print(
                "-" * 50
            )


    print(
        "\nQwen3 evaluation complete."
    )

    print(
        "Results saved to:",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()