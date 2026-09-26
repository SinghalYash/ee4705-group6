from pathlib import Path
import csv
import json
import re


# ==========================================================
# PATHS
# ==========================================================

EVALUATION_DIR = (
    Path(__file__).resolve().parent
)

GROUND_TRUTH_PATH = (
    EVALUATION_DIR / "ground_truth.csv"
)

PREDICTIONS_PATH = (
    EVALUATION_DIR / "evaluation_results.csv"
)

FINAL_RESULTS_PATH = (
    EVALUATION_DIR / "final_results.csv"
)


# ==========================================================
# SETTINGS
# ==========================================================

IOU_THRESHOLD = 0.5


# Convert user-friendly target names to MuJoCo body names.
TARGET_TO_OBJECT = {
    "blue cube": "box_obj",
    "grey stone": "stone",
    "gray stone": "stone",
    "green cylinder": "cylinder_obj",
}


# ==========================================================
# JSON PARSER
# ==========================================================

def parse_qwen_response(response):
    """
    Extract JSON from Qwen's response.

    Returns a Python dictionary, or None if the
    response cannot be parsed.
    """

    # Remove Markdown code fences.
    cleaned = re.sub(
        r"```(?:json)?|```",
        "",
        response,
    ).strip()

    # Extract the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1:
        return None

    cleaned = cleaned[
        start:end + 1
    ]

    try:
        return json.loads(
            cleaned
        )

    except json.JSONDecodeError:
        return None


# ==========================================================
# IOU
# ==========================================================

def calculate_iou(
    box_a,
    box_b,
):
    """
    Calculate Intersection over Union between
    two bounding boxes.

    Format:
    [x_min, y_min, x_max, y_max]
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b


    # Intersection rectangle.
    intersection_x1 = max(
        ax1,
        bx1,
    )

    intersection_y1 = max(
        ay1,
        by1,
    )

    intersection_x2 = min(
        ax2,
        bx2,
    )

    intersection_y2 = min(
        ay2,
        by2,
    )


    intersection_width = max(
        0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1,
    )


    intersection_area = (
        intersection_width
        * intersection_height
    )


    # Individual box areas.
    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )


    union_area = (
        area_a
        + area_b
        - intersection_area
    )


    if union_area == 0:
        return 0.0


    return (
        intersection_area
        / union_area
    )


# ==========================================================
# LOAD GROUND TRUTH
# ==========================================================

def load_ground_truth():

    ground_truth = {}


    with open(
        GROUND_TRUTH_PATH,
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )


        for row in reader:

            trial = int(
                row["trial"]
            )

            object_name = row[
                "object"
            ]


            # Skip invisible objects.
            if not row[
                "bbox_xmin"
            ]:

                continue


            bbox = [
                int(row["bbox_xmin"]),
                int(row["bbox_ymin"]),
                int(row["bbox_xmax"]),
                int(row["bbox_ymax"]),
            ]


            ground_truth[
                (trial, object_name)
            ] = {
                "bbox": bbox,
                "view": row["view"],
            }


    return ground_truth


# ==========================================================
# MAIN
# ==========================================================

def main():

    ground_truth = (
        load_ground_truth()
    )


    results = []


    with open(
        PREDICTIONS_PATH,
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )


        for row in reader:

            trial = int(
                row["trial"]
            )

            image_name = row[
                "image"
            ]

            target = row[
                "target"
            ]


            object_name = TARGET_TO_OBJECT.get(
                target
            )


            gt = ground_truth.get(
                (
                    trial,
                    object_name,
                )
            )


            # If ground truth itself is unavailable.
            if gt is None:

                results.append({
                    "trial": trial,
                    "image": image_name,
                    "target": target,
                    "view": "",
                    "qwen_status": "unknown",
                    "iou": 0.0,
                    "correct": False,
                    "failure_type": "ground_truth_unavailable",
                })

                continue


            parsed = parse_qwen_response(
                row["response"]
            )


            # Qwen returned something we could not parse.
            if parsed is None:

                results.append({
                    "trial": trial,
                    "image": image_name,
                    "target": target,
                    "view": gt["view"],
                    "qwen_status": "invalid_response",
                    "iou": 0.0,
                    "correct": False,
                    "failure_type": "invalid_response",
                })

                continue


            qwen_status = parsed.get(
                "status",
                "unknown",
            )


            # Qwen says target wasn't visible.
            if qwen_status == "not_found":

                results.append({
                    "trial": trial,
                    "image": image_name,
                    "target": target,
                    "view": gt["view"],
                    "qwen_status": qwen_status,
                    "iou": 0.0,
                    "correct": False,
                    "failure_type": "missed_target",
                })

                continue


            predicted_bbox = parsed.get(
                "bbox"
            )


            # Success response but no usable bbox.
            if (
                not isinstance(
                    predicted_bbox,
                    list,
                )
                or len(
                    predicted_bbox
                ) != 4
            ):

                results.append({
                    "trial": trial,
                    "image": image_name,
                    "target": target,
                    "view": gt["view"],
                    "qwen_status": qwen_status,
                    "iou": 0.0,
                    "correct": False,
                    "failure_type": "invalid_bbox",
                })

                continue


            predicted_bbox = [
                float(value)
                for value
                in predicted_bbox
            ]


            iou = calculate_iou(
                predicted_bbox,
                gt["bbox"],
            )


            correct = (
                iou >= IOU_THRESHOLD
            )


            results.append({
                "trial": trial,
                "image": image_name,
                "target": target,
                "view": gt["view"],
                "qwen_status": qwen_status,
                "iou": iou,
                "correct": correct,
                "failure_type": (
                    ""
                    if correct
                    else "poor_localization"
                ),
            })


    # ======================================================
    # SAVE FINAL RESULTS
    # ======================================================

    with open(
        FINAL_RESULTS_PATH,
        "w",
        newline="",
    ) as csv_file:

        fieldnames = [
            "trial",
            "image",
            "target",
            "view",
            "qwen_status",
            "iou",
            "correct",
            "failure_type",
        ]


        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )


        writer.writeheader()

        writer.writerows(
            results
        )


    # ======================================================
    # SUMMARY
    # ======================================================

    total = len(
        results
    )

    correct_count = sum(
        result["correct"]
        for result in results
    )


    grounding_accuracy = (
        correct_count
        / total
        * 100
        if total
        else 0
    )


    mean_iou = (
        sum(
            result["iou"]
            for result in results
        )
        / total
        if total
        else 0
    )

    # ======================================================
    # IOU DISTRIBUTION
    # ======================================================

    iou_values = [
        result["iou"]
        for result in results
    ]

    sorted_ious = sorted(
        iou_values
    )


    # Median IoU
    middle = len(sorted_ious) // 2

    if len(sorted_ious) % 2 == 0:

        median_iou = (
            sorted_ious[middle - 1]
            + sorted_ious[middle]
        ) / 2

    else:

        median_iou = sorted_ious[
            middle
        ]


    best_iou = max(
        iou_values
    )

    worst_iou = min(
        iou_values
    )


    # Count results at several thresholds.
    thresholds = [
        0.25,
        0.50,
        0.75,
    ]

    threshold_results = {}

    for threshold in thresholds:

        count = sum(
            iou >= threshold
            for iou in iou_values
        )

        threshold_results[
            threshold
        ] = count


    print(
        "\n"
        + "=" * 55
    )

    print(
        "TASK 2 GROUNDING EVALUATION"
    )

    print(
        "=" * 55
    )

    print(
        "Total trials:",
        total,
    )

    print(
        "Correctly grounded:",
        correct_count,
    )

    print(
        "Target Grounding Accuracy:",
        f"{grounding_accuracy:.1f}%",
    )

    print(
        "Mean IoU:",
        f"{mean_iou:.3f}",
    )

    print(
        "Median IoU:",
        f"{median_iou:.3f}",
    )

    print(
        "Best IoU:",
        f"{best_iou:.3f}",
    )

    print(
        "Worst IoU:",
        f"{worst_iou:.3f}",
    )


    print(
        "\nIoU distribution:"
    )

    for threshold, count in threshold_results.items():

        percentage = (
            count
            / total
            * 100
        )

        print(
            f"  IoU >= {threshold:.2f}: "
            f"{count}/{total} "
            f"({percentage:.1f}%)"
        )


    # ======================================================
    # PERFORMANCE BY VIEW
    # ======================================================

    print(
        "\nAccuracy by camera view:"
    )


    views = sorted(
        {
            result["view"]
            for result in results
            if result["view"]
        }
    )


    for view in views:

        view_results = [
            result
            for result in results
            if result["view"] == view
        ]


        view_correct = sum(
            result["correct"]
            for result in view_results
        )


        view_accuracy = (
            view_correct
            / len(view_results)
            * 100
        )


        print(
            f"  {view}: "
            f"{view_accuracy:.1f}% "
            f"({view_correct}/"
            f"{len(view_results)})"
        )


    # ======================================================
    # PERFORMANCE BY TARGET
    # ======================================================

    print(
        "\nAccuracy by target:"
    )


    for target in TARGET_TO_OBJECT:

        # Avoid duplicate gray/grey stone reporting.
        if target == "gray stone":
            continue


        target_results = [
            result
            for result in results
            if result["target"] == target
        ]


        if not target_results:
            continue


        target_correct = sum(
            result["correct"]
            for result in target_results
        )


        target_accuracy = (
            target_correct
            / len(target_results)
            * 100
        )


        print(
            f"  {target}: "
            f"{target_accuracy:.1f}% "
            f"({target_correct}/"
            f"{len(target_results)})"
        )


    print(
        "\nDetailed results:",
        FINAL_RESULTS_PATH,
    )


if __name__ == "__main__":
    main()