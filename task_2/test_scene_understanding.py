from PIL import Image
from perception import understand_scene
# from qwen_backend import ask_qwen

image = Image.open("C:/Yash/NUS/Year 4/EE4705/Project/camera_test_overhead.png")

result = understand_scene(image,"What objects and target areas are visible?")

print("\nStatus:")
print(result.status)

print("\nAnswer:")
print(result.answer)

print("\nObjects:")

for obj in result.objects:
    print(
        f"- {obj.color} {obj.label}"
    )

# print("\nTarget regions:")

# for region in result.target_regions:
#     print(    
#         f"- {region}"
#     )