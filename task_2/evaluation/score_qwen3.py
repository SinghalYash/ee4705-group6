from pathlib import Path
import csv
import json


# ==========================================================
# PATHS
# ==========================================================

EVALUATION_DIR = Path(__file__).resolve().parent

GROUND_TRUTH_PATH = (
    EVALUATION_DIR / "ground_truth.csv"
)

QWEN3_RESULTS_PATH = (
    EVALUATION_DIR / "qwen3_results.csv"
)


# ==========================================================
# SETTINGS
# ==========================================================

IOU_THRESHOLD = 0.5

TARGET_TO_OBJECT = {
    "blue cube": "box_obj",
    "grey stone": "stone",
    "green cylinder": "cylinder_obj",
}


# ==========================================================
# IOU
# ==========================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0,
        ix2 - ix1,
    )

    intersection_height = max(
        0,
        iy2 - iy1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union == 0:
        return 0.0

    return intersection / union


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

            if not row["bbox_xmin"]:
                continue

            trial = int(
                row["trial"]
            )

            object_name = row[
                "object"
            ]

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

    ground_truth = load_ground_truth()

    results = []


    with open(
        QWEN3_RESULTS_PATH,
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            trial = int(
                row["trial"]
            )

            target = row[
                "target"
            ]

            status = row[
                "status"
            ]

            object_name = TARGET_TO_OBJECT[
                target
            ]

            gt = ground_truth.get(
                (trial, object_name)
            )


            if gt is None:

                iou = 0.0
                correct = False

            elif (
                status != "success"
                or not row["bbox"]
            ):

                iou = 0.0
                correct = False

            else:

                predicted_bbox = json.loads(
                    row["bbox"]
                )

                iou = calculate_iou(
                    predicted_bbox,
                    gt["bbox"],
                )

                correct = (
                    iou >= IOU_THRESHOLD
                )


            results.append({
                "trial": trial,
                "target": target,
                "view": (
                    gt["view"]
                    if gt
                    else ""
                ),
                "iou": iou,
                "correct": correct,
            })


    # ======================================================
    # SUMMARY
    # ======================================================

    total = len(results)

    correct_count = sum(
        result["correct"]
        for result in results
    )

    accuracy = (
        correct_count
        / total
        * 100
    )

    ious = [
        result["iou"]
        for result in results
    ]

    mean_iou = (
        sum(ious)
        / total
    )

    sorted_ious = sorted(
        ious
    )

    middle = total // 2

    median_iou = (
        (
            sorted_ious[middle - 1]
            + sorted_ious[middle]
        )
        / 2
        if total % 2 == 0
        else sorted_ious[middle]
    )


    print(
        "\n"
        + "=" * 55
    )

    print(
        "QWEN3-VL-8B GROUNDING EVALUATION"
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
        f"{accuracy:.1f}%",
    )

    print(
        "Mean IoU:",
        f"{mean_iou:.3f}",
    )

    print(
        "Median IoU:",
        f"{median_iou:.3f}",
    )


    # ======================================================
    # IOU DISTRIBUTION
    # ======================================================

    print(
        "\nIoU distribution:"
    )

    for threshold in [
        0.25,
        0.50,
        0.75,
    ]:

        count = sum(
            iou >= threshold
            for iou in ious
        )

        print(
            f"  IoU >= {threshold:.2f}: "
            f"{count}/{total} "
            f"({count / total * 100:.1f}%)"
        )


    # ======================================================
    # BY TARGET
    # ======================================================

    print(
        "\nAccuracy by target:"
    )

    for target in TARGET_TO_OBJECT:

        target_results = [
            result
            for result in results
            if result["target"] == target
        ]

        count = sum(
            result["correct"]
            for result in target_results
        )

        print(
            f"  {target}: "
            f"{count / len(target_results) * 100:.1f}% "
            f"({count}/{len(target_results)})"
        )


    # ======================================================
    # BY VIEW
    # ======================================================

    print(
        "\nAccuracy by camera view:"
    )

    views = sorted(
        set(
            result["view"]
            for result in results
            if result["view"]
        )
    )

    for view in views:

        view_results = [
            result
            for result in results
            if result["view"] == view
        ]

        count = sum(
            result["correct"]
            for result in view_results
        )

        print(
            f"  {view}: "
            f"{count / len(view_results) * 100:.1f}% "
            f"({count}/{len(view_results)})"
        )


if __name__ == "__main__":
    main()