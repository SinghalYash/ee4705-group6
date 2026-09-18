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

SYSTEM_PROMPT = f"""You are the task planner for a simulated robot arm with a 2-finger gripper.

Objects currently in the scene: {", ".join(SCENE_MOCK['objects'])}
Target regions currently in the scene: {", ".join(SCENE_MOCK['target_regions'])}

Convert the user's natural-language instruction into a JSON action plan using ONLY these skills:
SEARCH(target), APPROACH(target), REACH(target), GRASP(target), MOVE_TO(target),
PLACE(object, target), VERIFY(condition), STOP(reason)

Rules:
- Only reference objects/regions that exist in the scene listed above.
- Treat paraphrases the same way, e.g. "the rock" / "that stone" both mean "stone",
  and "red zone" / "red marker" / "red spot" all mean "red_area".
- If the instruction references something NOT in the scene, or is otherwise impossible
  or nonsensical, set "feasible" to false and return a single STOP action with a "reason"
  explaining why. Do not guess or invent objects that were not listed above.
- Respond ONLY with JSON, no extra text, in exactly this format:
{{"feasible": true, "actions": [{{"skill": "...", "target": "..."}}, ...]}}
"""


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
