import json
import re

from PIL import Image, ImageDraw

from qwen_local import (
    load_model,
    ask_qwen,
)


# ==========================================================
# LOAD MODEL
# ==========================================================

model, processor = load_model()


# ==========================================================
# LOAD IMAGE
# ==========================================================

IMAGE_PATH = "camera_test_overhead.png"

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ==========================================================
# ASK QWEN FOR BOUNDING BOX
# ==========================================================

query = """
Locate the blue cube in this image.

Return the bounding box of the blue cube.

Return JSON only using exactly this format:

{
    "label": "cube",
    "color": "blue",
    "bbox": [x_min, y_min, x_max, y_max]
}

Do not include explanations.
"""


answer = ask_qwen(
    model,
    processor,
    image,
    query,
)


print("\nQwen grounding response:")
print(answer)


# ==========================================================
# REMOVE MARKDOWN CODE BLOCK IF PRESENT
# ==========================================================

clean_answer = re.sub(
    r"```(?:json)?|```",
    "",
    answer,
).strip()


# ==========================================================
# CONVERT JSON TEXT -> PYTHON DICTIONARY
# ==========================================================

result = json.loads(
    clean_answer
)

bbox = result["bbox"]
label = result["label"]
color = result["color"]


print("\nParsed bounding box:")
print(bbox)


# ==========================================================
# DRAW BOUNDING BOX
# ==========================================================

draw = ImageDraw.Draw(
    image
)

x_min, y_min, x_max, y_max = bbox


draw.rectangle(
    [x_min, y_min, x_max, y_max],
    outline="red",
    width=4,
)


draw.text(
    (x_min, max(0, y_min - 20)),
    f"{color} {label}",
    fill="red",
)


# ==========================================================
# SAVE RESULT
# ==========================================================

OUTPUT_PATH = "grounding_result.png"

image.save(
    OUTPUT_PATH
)


print(
    f"\nSaved grounding visualization to "
    f"{OUTPUT_PATH}"
)