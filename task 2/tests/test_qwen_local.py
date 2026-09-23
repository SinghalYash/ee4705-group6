from PIL import Image

from qwen_local import (
    load_model,
    ask_qwen,
)


# ==========================================
# LOAD QWEN
# ==========================================

model, processor = load_model()


# ==========================================
# LOAD TEST IMAGE
# ==========================================

image = Image.open(
    "camera_test_overhead.png"
)


# ==========================================
# ASK VISUAL QUESTION
# ==========================================

query = (
    "What task-relevant objects are "
    "visible in this image?"
)

answer = ask_qwen(
    model,
    processor,
    image,
    query,
)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\nQuestion:")
print(query)

print("\nQwen response:")
print(answer)