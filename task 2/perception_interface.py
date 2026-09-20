from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectedObject:
    """
    One object detected in the camera image.
    """

    label: str
    color: Optional[str]
    bbox: list[int]
    confidence: Optional[float] = None

@dataclass
class PerceptionResult:
    """
    Structured output returned by the perception system.
    """

    status: str
    query: str
    answer: str
    objects: list[DetectedObject]
    target: Optional[DetectedObject] = None
    reason: Optional[str] = None

def perceive(image, query: str) -> PerceptionResult:
    """
    Main visual perception interface.

    Inputs
    ------
    image:
        RGB image captured from the robot camera.

    query:
        Natural-language question or instruction.

    Returns
    -------
    PerceptionResult:
        Structured perception information.
    """

    raise NotImplementedError(
        "VLM connection will be implemented in Task 2(ii)."
    )   