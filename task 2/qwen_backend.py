from qwen3_local import (
    load_model,
    ask_qwen,
    qwen3_bbox_to_pixels,
)


def bbox_to_pixels(
    bbox,
    image_width,
    image_height,
):
    return qwen3_bbox_to_pixels(
        bbox,
        image_width,
        image_height,
    )