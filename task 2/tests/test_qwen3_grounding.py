from pathlib import Path
from PIL import Image

from qwen3_local import (
    load_model,
    ask_qwen,
)


# ==========================================
# PATH
# ==========================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "task 2"
    / "evaluation"
    / "images"
    / "trial_001.png"
)


# ==========================================
# LOAD MODEL
# ==========================================

model, processor = load_model()


# ==========================================
# LOAD IMAGE
# ==========================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ==========================================
# GROUND TARGET
# ==========================================

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


answer = ask_qwen(
    model,
    processor,
    image,
    query,
)


print("\nQwen3-VL response:")
print(answer)