from perception_interface import DetectedObject, PerceptionResult


# Pretend that a future VLM detected these objects
blue_cube = DetectedObject(
    label="cube",
    color="blue",
    bbox=[210, 270, 265, 325],
    confidence=0.95,
)

stone = DetectedObject(
    label="stone",
    color="grey",
    bbox=[120, 230, 170, 280],
    confidence=0.91,
)

green_cylinder = DetectedObject(
    label="cylinder",
    color="green",
    bbox=[330, 220, 380, 300],
    confidence=0.93,
)


# Pretend the user asked this question
query = "Where is the blue cube?"


# Pretend this is the VLM's final structured response
result = PerceptionResult(
    status="success",
    query=query,
    answer="The blue cube is visible.",
    objects=[
        blue_cube,
        stone,
        green_cylinder,
    ],
    target=blue_cube,
)


print("Status:", result.status)
print("Query:", result.query)
print("Answer:", result.answer)

print("\nDetected objects:")

for obj in result.objects:
    print(
        f"- {obj.color} {obj.label}, "
        f"bbox={obj.bbox}, "
        f"confidence={obj.confidence}"
    )

print("\nGrounded target:")
print(result.target)