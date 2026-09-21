from PIL import Image

from qwen_local import load_model
from perception import ground_object


# Load Qwen
model, processor = load_model()


# Load MuJoCo overhead camera image
image = Image.open(
    "camera_test_overhead.png"
).convert("RGB")


# Choose what we want the robot to find
target = "blue cube"


# Ask Task 2 to ground the target
result = ground_object(
    model,
    processor,
    image,
    target,
)


# Display structured result
print("\n===== PERCEPTION RESULT =====")

print("Status:", result.status)
print("Query:", result.query)
print("Answer:", result.answer)
print("Target:", result.target)
print("Reason:", result.reason)