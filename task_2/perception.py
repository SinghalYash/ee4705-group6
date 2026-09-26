import json
import re
from perception_interface import DetectedObject, PerceptionResult
from qwen_backend import ask_qwen,bbox_to_pixels

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

    prompt = f"""You are the visual perception system of a robot.

        Look carefully at the provided camera image and answer
        the following question:

        "{query}"

        Identify task-relevant objects and target regions that
        are actually visible in the image.

        Return JSON only using this exact structure:

        {{
            "status": "success",
            "answer": "short natural-language answer",
            "objects": [
                {{
                    "label": "object name",
                    "color": "object color"
                }}
            ],
            "target_regions": [
                "region name"
            ]
        }}

        Rules:

        - Only report objects that are actually visible.
        - Do not invent objects.
        - Use simple object names where possible.
        - Include the color when it is visually identifiable.
        - Target regions are designated areas such as a red area,
        blue area, marker, zone, or platform.
        - If no relevant objects are visible, return an empty
        objects list.
        - If no target regions are visible, return an empty
        target_regions list.
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
            # target_regions=[],
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