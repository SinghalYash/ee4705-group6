import json
import re

from perception_interface import (
    DetectedObject,
    PerceptionResult,
)

from qwen_local import ask_qwen


def clean_json_response(response: str) -> str:
    """
    Remove Markdown code fences that Qwen may add.
    """

    return re.sub(
        r"```(?:json)?|```",
        "",
        response,
    ).strip()

def ground_object(
    model,
    processor,
    image,
    target: str,
) -> PerceptionResult:

    # Build prompt
    query = f"""
    ...
    """

    # Ask Qwen
    response = ask_qwen(
        model,
        processor,
        image,
        query,
    )

    clean_response = clean_json_response(
        response
    )

    # Parse JSON
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

    # Target not found
    if result.get("status") == "not_found":

        return PerceptionResult(
            status="not_found",
            query=target,
            answer=f"{target} was not found.",
            objects=[],
            target=None,
            reason=result.get(
                "reason",
                "Target not visible.",
            ),
        )

    # Target found
    if result.get("status") == "success":

        detected_target = DetectedObject(
            label=result.get(
                "label",
                target,
            ),
            color=result.get(
                "color"
            ),
            bbox=result.get(
                "bbox",
                [],
            ),
            confidence=None,
        )

        return PerceptionResult(
            status="success",
            query=target,
            answer=f"{target} was grounded successfully.",
            objects=[detected_target],
            target=detected_target,
            reason=None,
        )

    # Anything unexpected
    return PerceptionResult(
        status="error",
        query=target,
        answer="Unexpected perception response.",
        objects=[],
        target=None,
        reason=str(result),
    )