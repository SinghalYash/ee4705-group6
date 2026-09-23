from pathlib import Path
from PIL import Image

from qwen3_local import load_model
from perception import ground_object


# ==========================================================
# PATH
# ==========================================================

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


# ==========================================================
# LOAD QWEN3
# ==========================================================

model, processor = load_model()


# ==========================================================
# LOAD IMAGE
# ==========================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# ==========================================================
# RUN TASK 2
# ==========================================================

result = ground_object(
    model,
    processor,
    image,
    "blue cube",
)


# ==========================================================
# DISPLAY RESULT
# ==========================================================

print("\n===== TASK 2 RESULT =====")

print("Status:", result.status)
print("Query:", result.query)
print("Answer:", result.answer)
print("Target:", result.target)
print("Reason:", result.reason)