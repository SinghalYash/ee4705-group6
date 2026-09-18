"""
Task 3: Natural-language -> structured action plan.

Uses a MOCKED scene description (SCENE_MOCK below) instead of real VLM
grounding, since Task 2 isn't built yet. Swap SCENE_MOCK for the actual
output of Task 2's grounding module once it's ready -- the rest of this
file (prompt, validation, plan_from_instruction) does not need to change.
"""

import json
import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# Provider configuration. Default: OpenAI. Uncomment ONE alternative block
# below instead if your team is using Qwen-VL (free quota) or local Ollama.
# ---------------------------------------------------------------------------

# client = OpenAI()  # reads OPENAI_API_KEY from your environment
# MODEL = "gpt-5-mini"

# --- Qwen (Alibaba Cloud, Singapore region, free quota) ---
client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                 base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen3-vl-flash"

# --- Local Ollama (free, offline, no API key needed) ---
# client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
# MODEL = "qwen2.5vl:3b"


ALLOWED_SKILLS = ["SEARCH", "APPROACH", "REACH", "GRASP", "MOVE_TO", "PLACE", "VERIFY", "STOP"]

# Mocked scene info -- stand-in until Task 2 (VLM grounding) is ready.
# Names match the objects/target defined in scene.xml.
SCENE_MOCK = {
    "objects": ["stone", "box", "cylinder"],
    "target_regions": ["red_area"],
}

SYSTEM_PROMPT = f"""
You are the task planner for a simulated robot arm with
a three-prong gripper.

Objects currently in the scene:
{", ".join(SCENE_MOCK["objects"])}

Target regions currently in the scene:
{", ".join(SCENE_MOCK["target_regions"])}

Your job is to convert the user's natural-language
instruction into a structured JSON robot action plan.

You may ONLY use these skills:

SEARCH
APPROACH
REACH
GRASP
MOVE_TO
PLACE
VERIFY
STOP


============================================================
CANONICAL SCENE NAMES
============================================================

Always use the following canonical names in the JSON output:

Objects:
- stone
- box
- cylinder

Target regions:
- red_area

Convert natural-language descriptions to these names.

Examples:

"blue box" -> "box"
"blue cube" -> "box"

"rock" -> "stone"
"sphere" -> "stone"
"grey stone" -> "stone"

"green cylinder" -> "cylinder"

"red area" -> "red_area"
"red zone" -> "red_area"
"red marker" -> "red_area"
"red spot" -> "red_area"


============================================================
ACTION SCHEMA
============================================================

SEARCH:
{{"skill": "SEARCH", "target": "<object>"}}

APPROACH:
{{"skill": "APPROACH", "target": "<object>"}}

REACH:
{{"skill": "REACH", "target": "<object>"}}

GRASP:
{{"skill": "GRASP", "target": "<object>"}}

MOVE_TO:
{{"skill": "MOVE_TO", "target": "<target_region>"}}

PLACE:
{{
    "skill": "PLACE",
    "object": "<object>",
    "target": "<target_region>"
}}

VERIFY:
{{
    "skill": "VERIFY",
    "verify_type": "<GRASP or PLACE>",
    "object": "<object>",
    "target": "<target_region when required>"
}}

STOP:
{{
    "skill": "STOP",
    "reason": "<reason>"
}}


============================================================
MANIPULATION SEQUENCE
============================================================

For an instruction that asks the robot to move an object
to a target region, use this sequence:

1. SEARCH for the object.
2. APPROACH the object.
3. REACH the object's pre-grasp position.
4. GRASP the object.
5. MOVE_TO the destination.
6. PLACE the object at the destination.

Do NOT omit REACH between APPROACH and GRASP.

For PLACE, ALWAYS provide BOTH:

- "object"
- "target"

Do not put the object name in the PLACE "target" field.

Correct:
{{
    "skill": "PLACE",
    "object": "box",
    "target": "red_area"
}}

Incorrect:
{{
    "skill": "PLACE",
    "target": "box"
}}

Do not invent fields such as:

"location"
"destination"
"item"

Use only the fields defined above.


============================================================
FEASIBILITY
============================================================

Only reference objects and target regions that exist in
the scene.

If the requested object or target does not exist, or the
instruction is impossible or nonsensical:

- set "feasible" to false
- return one STOP action
- explain the reason in the STOP action

Do not guess or invent objects.


============================================================
EXAMPLE
============================================================

User instruction:

Move the blue box to the red area.

Correct output:

{{
    "feasible": true,
    "actions": [
        {{
            "skill": "SEARCH",
            "target": "box"
        }},
        {{
            "skill": "APPROACH",
            "target": "box"
        }},
        {{
            "skill": "REACH",
            "target": "box"
        }},
        {{
            "skill": "GRASP",
            "target": "box"
        }},
        {{
            "skill": "MOVE_TO",
            "target": "red_area"
        }},
        {{
            "skill": "PLACE",
            "object": "box",
            "target": "red_area"
        }}
    ]
}}

Respond ONLY with valid JSON.

Do not include markdown.
Do not include explanations outside the JSON.
"""

def validate_action_plan(
    plan,
):
    """
    Validate the structure of the action plan returned
    by the language model.
    """

    if not isinstance(
        plan,
        dict,
    ):

        return (
            False,
            "Planner output is not a dictionary",
        )

    actions = plan.get(
        "actions"
    )

    if not isinstance(
        actions,
        list,
    ):

        return (
            False,
            "Planner actions are not a list",
        )

    valid_objects = {
        "stone",
        "box",
        "cylinder",
    }

    valid_targets = {
        "red_area",
    }

    for index, action in enumerate(
        actions,
        start=1,
    ):

        if not isinstance(
            action,
            dict,
        ):

            return (
                False,
                f"Action {index} is not a dictionary",
            )

        skill = action.get(
            "skill"
        )

        if skill not in ALLOWED_SKILLS:

            return (
                False,
                (
                    f"Action {index} uses "
                    f"invalid skill: {skill}"
                ),
            )

        # ----------------------------------------------
        # Object-directed skills
        # ----------------------------------------------

        if skill in {
            "SEARCH",
            "APPROACH",
            "REACH",
            "GRASP",
        }:

            target = action.get(
                "target"
            )

            if target not in valid_objects:

                return (
                    False,
                    (
                        f"Action {index} has "
                        f"invalid object target: "
                        f"{target}"
                    ),
                )

        # ----------------------------------------------
        # MOVE_TO
        # ----------------------------------------------

        if skill == "MOVE_TO":

            target = action.get(
                "target"
            )

            if target not in valid_targets:

                return (
                    False,
                    (
                        f"Action {index} has "
                        f"invalid destination: "
                        f"{target}"
                    ),
                )

        # ----------------------------------------------
        # PLACE
        # ----------------------------------------------

        if skill == "PLACE":

            object_name = action.get(
                "object"
            )

            target = action.get(
                "target"
            )

            if object_name not in valid_objects:

                return (
                    False,
                    (
                        f"Action {index} PLACE "
                        f"has invalid object: "
                        f"{object_name}"
                    ),
                )

            if target not in valid_targets:

                return (
                    False,
                    (
                        f"Action {index} PLACE "
                        f"has invalid target: "
                        f"{target}"
                    ),
                )

    return (
        True,
        None,
    )

def plan_from_instruction(instruction: str) -> dict:
    """Call the LLM planner and return a validated action-plan dict."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
    )
    raw = response.choices[0].message.content

    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        plan = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {
            "feasible": False,
            "actions": [{"skill": "STOP", "reason": "Could not parse planner output"}],
        }

    # Basic schema validation: reject any plan using an undefined skill.
    for action in plan.get("actions", []):
        if action.get("skill") not in ALLOWED_SKILLS:
            return {
                "feasible": False,
                "actions": [{"skill": "STOP", "reason": f"Invalid skill: {action.get('skill')}"}],
            }

    return plan


if __name__ == "__main__":
    while True:
        instruction = input("\nType an instruction (or 'quit'): ").strip()
        if instruction.lower() == "quit":
            break
        plan = plan_from_instruction(instruction)
        print(json.dumps(plan, indent=2))
