import json
import re
from task_2.perception_interface import DetectedObject, PerceptionResult
from task_2.qwen_backend import ask_qwen,bbox_to_pixels

def clean_json_response(response: str) -> str:
    """
    Remove Markdown code fences that Qwen may add.
    """

    return re.sub(
        r"```(?:json)?|```",
        "",
        response,
    ).strip()

def understand_scene(image, query: str) -> PerceptionResult:
    """
    Ask Qwen a general question about the current scene.

    Examples:
        "What objects are visible?"
        "What is in the scene?"
        "Which objects are near the robot?"
        "Which object is inside the red area?"

    Returns structured scene information.
    """

    prompt = f"""
You are the visual perception system of a tabletop manipulation robot,
viewing the scene from a fixed overhead/angled camera.

Question: "{query}"

WHAT IS IN THE IMAGE:
- A table with a small number of movable objects, and up to one flat
  target region marked on the table surface.
- The robot's own arm, gripper fingers, and mounting base. These are
  the robot, NOT scene objects or target regions -- never report any
  part of the arm/gripper/base as an object or a target region, even
  if a part of it is a distinct color.
- A flat colored circle/marker painted on the table is a TARGET
  REGION, not an object, even if lighting makes it look raised.

IDENTIFYING OBJECTS:
- Because the camera looks down at a steep angle, a short cylinder, a
  sphere, and a rounded stone can all look like a plain circle from
  this view. When the exact shape is unclear, identify the object by
  its COLOR rather than guessing the 3D shape.
- Report the shape label you actually see, plus the color. If shape
  is genuinely ambiguous, prefer a generic label such as "round
  object" over a confident but possibly wrong shape guess -- color is
  what matters most for downstream matching.

OUTPUT (JSON only, exactly this structure):
{{
    "status": "success",
    "answer": "short description of the scene",
    "objects": [
        {{"label": "object type", "color": "object color"}}
    ],
    "target_regions": [
        {{"label": "region type", "color": "region color"}}
    ]
}}

RULES:
- Each physical object appears exactly once in "objects"; never list
  its color as a separate entry (a green sphere is one entry:
  {{"label": "sphere", "color": "green"}}, not two).
- Only report items actually visible in the image. Do not invent or
  duplicate objects. Empty lists are fine if nothing is visible.
"""

    response = ask_qwen(image, prompt)
    clean_response = clean_json_response(response)

    try:
        result = json.loads(clean_response)

    except json.JSONDecodeError:

        return PerceptionResult(
            status="error",
            query=query,
            answer="Qwen returned invalid JSON.",
            objects=[],
            target=None,
            target_regions=[],
            reason=clean_response,
        )

    # CONVERT OBJECTS INTO DetectedObject INSTANCES

    detected_objects = []

    for obj in result.get("objects", []):

        detected_objects.append(
            DetectedObject(
                label=obj.get(
                    "label",
                    "unknown",
                ),
                color=obj.get(
                    "color"
                ),
                bbox=None,
                confidence=None,
            )
        )
    # RETURN STRUCTURED RESULT

    return PerceptionResult(
        status=result.get(
            "status",
            "success",
        ),
        query=query,
        answer=result.get(
            "answer",
            "",
        ),
        objects=detected_objects,
        target=None,
        target_regions=result.get(
            "target_regions",
            [],
        ),
        reason=None,
    )


def ground_object(
    model,
    processor,
    image,
    target: str,
) -> PerceptionResult:
    """
    Locate a requested object in an image using Qwen3-VL.

    Always returns a PerceptionResult.
    """

    # ------------------------------------------------------
    # BUILD GROUNDING PROMPT
    # ------------------------------------------------------

    query = f"""
Locate the {target} in this image.

The robot's own arm, gripper fingers, and mounting base are not
objects -- ignore them completely, even if part of the arm happens
to share a similar color with the target.

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


    # ------------------------------------------------------
    # ASK QWEN3
    # ------------------------------------------------------

    response = ask_qwen(image,query)


    # ------------------------------------------------------
    # CLEAN RESPONSE
    # ------------------------------------------------------

    clean_response = clean_json_response(
        response
    )


    # ------------------------------------------------------
    # PARSE JSON
    # ------------------------------------------------------

    try:

        result = json.loads(
            clean_response
        )

    except json.JSONDecodeError:

        return PerceptionResult(
            status="error",
            query=target,
            answer="Qwen returned invalid JSON.",
            objects=[],
            target=None,
            reason=clean_response,
        )


    # ------------------------------------------------------
    # TARGET NOT FOUND
    # ------------------------------------------------------

    if result.get("status") == "not_found":

        return PerceptionResult(
            status="not_found",
            query=target,
            answer=f"{target} was not found.",
            objects=[],
            target=None,
            reason="Target not visible.",
        )


    # ------------------------------------------------------
    # TARGET FOUND
    # ------------------------------------------------------

    if result.get("status") == "success":

        qwen_bbox = result.get(
            "bbox"
        )


        # Check that Qwen returned a valid bbox.
        if (
            not isinstance(
                qwen_bbox,
                list,
            )
            or len(qwen_bbox) != 4
        ):

            return PerceptionResult(
                status="error",
                query=target,
                answer="Qwen returned an invalid bounding box.",
                objects=[],
                target=None,
                reason=str(qwen_bbox),
            )


        # Convert Qwen3's normalized coordinates
        # into actual image pixels.
        pixel_bbox = bbox_to_pixels(
            qwen_bbox,
            image.width,
            image.height,
        )


        detected_target = DetectedObject(
            label=target,
            color=None,
            bbox=pixel_bbox,
            confidence=None,
        )


        return PerceptionResult(
            status="success",
            query=target,
            answer=f"{target} was grounded successfully.",
            objects=[
                detected_target
            ],
            target=detected_target,
            reason=None,
        )


    # ------------------------------------------------------
    # UNEXPECTED RESPONSE
    # ------------------------------------------------------

    return PerceptionResult(
        status="error",
        query=target,
        answer="Unexpected perception response.",
        objects=[],
        target=None,
        reason=str(result),
    )