"""
Task 3: Natural-language instruction -> structured action plan.

Pipeline for one instruction
----------------------------
1. PERCEIVE  - Task 2 (task_3.scene_builder.build_scene) describes the
               live camera frame, e.g. objects ["blue square",
               "green circle", "gray sphere"], regions ["red circle"].
2. GROUND    - Each perceived entry is bound, deterministically, to one
               of the robot's canonical names (box / stone / cylinder /
               red_area) using its COLOUR, which the VLM reports
               reliably. Its SHAPE word is not trusted (from the
               overhead camera a cylinder often looks like a circle,
               a cube like a square). Colours come from scene.xml.
3. RESOLVE   - The LLM receives the grounded scene ("blue square ->
               box") plus the instruction, and decides which canonical
               object/target the user means, or that the request is
               infeasible. This is the language-understanding step
               (paraphrases, synonyms, "the blue one", "the red zone").
4. VALIDATE  - The LLM's answer is normalised back onto canonical names
               (so "square" or "red_circle" are still understood), then
               checked: both must be present in the current scene, and
               the user's own wording must not contradict them (e.g.
               "green zone" can never become red_area).
5. ASSEMBLE  - The fixed SEARCH -> APPROACH -> REACH -> GRASP ->
               MOVE_TO -> PLACE sequence is built in Python from the
               validated names, so the plan is always well-formed.

plan_from_instruction() never raises: any failure (network, bad JSON,
unknown name) becomes {"feasible": False, "actions": [STOP + reason]}.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI
from PIL import Image

# Works both when imported as a package ("from task_3.planner import
# ...", e.g. from Task 5) and when run directly from inside task_3/.
try:
    from task_3.scene_builder import build_scene
except ImportError:
    from scene_builder import build_scene


# ===========================================================================
# LLM PROVIDER
# ===========================================================================
# The client is created lazily, so importing this module never crashes
# just because an API key is missing (useful for offline tests).

PROVIDER = {
    # --- Qwen (Alibaba Cloud, Singapore region, free quota) ---
    "api_key_env": "DASHSCOPE_API_KEY",
    "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "model": "qwen3-vl-flash",
    # --- OpenAI ---
    # "api_key_env": "OPENAI_API_KEY", "base_url": None, "model": "gpt-5-mini",
    # --- Local Ollama ---
    # "api_key_env": None, "base_url": "http://localhost:11434/v1", "model": "qwen3:4b",
}
MODEL = PROVIDER["model"]

_client = None


def _get_client():
    global _client
    if _client is None:
        env = PROVIDER["api_key_env"]
        api_key = os.environ.get(env) if env else "ollama"
        if not api_key:
            raise RuntimeError(f"Environment variable {env} is not set")
        _client = OpenAI(
            api_key=api_key,
            base_url=PROVIDER["base_url"],
            timeout=60,
            max_retries=2,
        )
    return _client


# ===========================================================================
# THE ROBOT'S FIXED VOCABULARY (interface constants)
# ===========================================================================
# Mirrors the bodies defined once in scene.xml and the names Task 4's
# skill library looks up. Colours are the rgba values in scene.xml:
#   box_obj  rgba 0 0 1   -> blue      stone         rgba .5 .5 .5 -> gray
#   cylinder rgba 0 1 0   -> green     target_area   rgba 1 0 0    -> red

ALLOWED_SKILLS = ["SEARCH", "APPROACH", "REACH", "GRASP", "MOVE_TO", "PLACE", "VERIFY", "STOP"]

ENTITIES = {
    "box": {
        "kind": "object",
        "color": "blue",
        "looks_like": "small blue cube",
        "nouns": {"box", "cube", "block", "square", "brick", "cuboid"},
    },
    "stone": {
        "kind": "object",
        "color": "gray",
        "looks_like": "gray sphere (a stone)",
        "nouns": {"stone", "rock", "pebble", "boulder"},
    },
    "cylinder": {
        "kind": "object",
        "color": "green",
        "looks_like": "green upright cylinder",
        "nouns": {"cylinder", "can", "tube", "pillar", "column", "tin", "canister"},
    },
    "red_area": {
        "kind": "target",
        "color": "red",
        "looks_like": "flat red circular region marked on the table",
        "nouns": {"area", "zone", "region", "target", "goal", "marker", "spot",
                  "circle", "disc", "disk", "patch", "mark", "pad", "place"},
    },
}

CANONICAL_OBJECTS = {n for n, e in ENTITIES.items() if e["kind"] == "object"}
CANONICAL_TARGETS = {n for n, e in ENTITIES.items() if e["kind"] == "target"}

# Colour words -> colour family. Includes colours that exist nowhere in
# the scene (yellow, purple, ...) so that "the yellow area" is recognised
# as a colour that CONFLICTS with red_area, rather than silently ignored.
COLOR_FAMILY = {
    "blue": "blue", "navy": "blue", "cyan": "blue", "teal": "blue", "turquoise": "blue",
    "green": "green", "lime": "green", "olive": "green",
    "gray": "gray", "grey": "gray", "silver": "gray", "grayish": "gray", "greyish": "gray",
    "red": "red", "pink": "red", "crimson": "red", "scarlet": "red", "maroon": "red",
    "reddish": "red", "magenta": "red",
    "yellow": "yellow", "gold": "yellow", "orange": "orange", "purple": "purple",
    "violet": "purple", "white": "white", "black": "black", "brown": "brown",
}


def _words(text) -> list:
    return re.findall(r"[a-z]+", str(text or "").lower().replace("_", " "))


def _colors_in(text) -> set:
    return {COLOR_FAMILY[w] for w in _words(text) if w in COLOR_FAMILY}


# ===========================================================================
# STEP 2: GROUND THE PERCEIVED SCENE
# ===========================================================================

def ground_scene(scene_info: dict) -> list:
    """
    Bind every entry Task 2 reported to a canonical name (or None).

    Returns a list of {"description", "reported_as", "canonical"}.

    Binding is by colour, because colour is what the VLM reports
    reliably. A red entry is always the target region even if Task 2
    listed it under "objects". If an entry has no usable colour, a
    distinctive shape noun is used as a fallback (cube -> box, etc.).
    """

    grounded = []

    for reported_as, key in (("object", "objects"), ("target_region", "target_regions")):
        for entry in scene_info.get(key, []) or []:
            description = str(entry).strip()
            if not description:
                continue

            colors = _colors_in(description)
            candidates = {n for n, e in ENTITIES.items() if e["color"] in colors}

            if not candidates and not colors:
                words = set(_words(description))
                kind = "object" if reported_as == "object" else "target"
                candidates = {
                    n for n, e in ENTITIES.items()
                    if e["kind"] == kind and words & e["nouns"]
                }
                # Only one target region exists, so an uncoloured region is it.
                if not candidates and kind == "target" and len(CANONICAL_TARGETS) == 1:
                    candidates = set(CANONICAL_TARGETS)

            canonical = next(iter(candidates)) if len(candidates) == 1 else None
            grounded.append({
                "description": description,
                "reported_as": reported_as,
                "canonical": canonical,
            })

    return grounded


def _present(grounded: list) -> set:
    return {g["canonical"] for g in grounded if g["canonical"]}


# ===========================================================================
# STEP 3: LLM RESOLUTION
# ===========================================================================

def _entity_table() -> str:
    lines = []
    for name, e in ENTITIES.items():
        kind = "object " if e["kind"] == "object" else "target "
        lines.append(f"  {name:<9} ({kind}) = the {e['looks_like']}")
    return "\n".join(lines)


RESOLUTION_PROMPT = f"""
You are the language-understanding module of a tabletop robot arm.
Your only job: read the user's instruction and decide which ONE object
the robot should move and to which ONE target region.

The robot knows exactly these things, by these canonical names:
{_entity_table()}

Each turn you get the CURRENT SCENE as seen by the camera. Every entry
has already been matched to a canonical name for you, e.g.
    "blue square" -> box
The camera's shape words are unreliable (a cube may be called a square,
a cylinder a circle), so trust the canonical name after the arrow, not
the shape word. An entry marked "-> none" is not something the robot
can act on.

How users refer to things (all of these are fine):
  - by name or synonym: "box", "cube", "block", "rock", "stone", "can"
  - by colour only: "the blue one", "move blue to red"
  - by the camera's wording: "blue square", "green circle"
  - the target: "red area", "red zone", "red circle", "red spot",
    "red marker", "the target", "the goal" all mean red_area.

Mark the instruction INFEASIBLE (feasible = false) when:
  - the object it asks for is not in the current scene
    (e.g. a banana, a laptop), or is a colour no present object has;
  - the destination is not a present target region (e.g. a differently
    coloured zone, "the table", "the moon");
  - there is no destination at all (e.g. "pick up the stone");
  - it is not a move-an-object-to-a-region task (e.g. "delete the box",
    "fly the drone", "move the robot somewhere");
  - it is genuinely ambiguous which single object is meant.
Otherwise it is FEASIBLE. Do not reject a request just because the
user's wording differs from the camera's wording or the canonical name.

Respond with ONLY this JSON object, nothing else:
{{
  "feasible": true or false,
  "object": "<{' | '.join(sorted(CANONICAL_OBJECTS))} | null>",
  "target": "<{' | '.join(sorted(CANONICAL_TARGETS))} | null>",
  "object_phrase": "<the words in the instruction naming the object, or null>",
  "target_phrase": "<the words in the instruction naming the destination, or null>",
  "reason": "<one short sentence>"
}}

EXAMPLES (scene for all: objects "blue square" -> box, "green circle"
-> cylinder, "gray sphere" -> stone; target regions "red circle" -> red_area)

Instruction: move blue to red
{{"feasible": true, "object": "box", "target": "red_area", "object_phrase": "blue", "target_phrase": "red", "reason": "blue object is the box; red region is red_area"}}

Instruction: put the green thing on the red spot
{{"feasible": true, "object": "cylinder", "target": "red_area", "object_phrase": "the green thing", "target_phrase": "the red spot", "reason": "green object is the cylinder"}}

Instruction: slide the blue block onto the purple mat
{{"feasible": false, "object": "box", "target": null, "object_phrase": "the blue block", "target_phrase": "the purple mat", "reason": "there is no purple target region, only the red one"}}

Instruction: lift the rock
{{"feasible": false, "object": "stone", "target": null, "object_phrase": "the rock", "target_phrase": null, "reason": "no destination was given"}}

Instruction: carry the pyramid to the red area
{{"feasible": false, "object": null, "target": "red_area", "object_phrase": "the pyramid", "target_phrase": "the red area", "reason": "no pyramid is in the scene"}}
""".strip()


def _build_scene_text(instruction: str, grounded: list) -> str:
    def block(reported_as):
        rows = [
            f'  - "{g["description"]}" -> {g["canonical"] or "none"}'
            for g in grounded if g["reported_as"] == reported_as
        ]
        return "\n".join(rows) if rows else "  (nothing detected)"

    return (
        "CURRENT SCENE\n"
        f"Objects:\n{block('object')}\n"
        f"Target regions:\n{block('target_region')}\n\n"
        f"Instruction: {instruction}"
    )


def _extract_json(raw: str) -> Optional[dict]:
    """Pull the answer object out of the model's text, tolerating
    <think> blocks, markdown fences and chatter around the JSON."""
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "feasible" in obj:
            return obj
    return None


def _call_resolution_llm(scene_text: str, retry_note: str = "") -> tuple:
    """One LLM call. Returns (parsed_dict_or_None, error_string_or_None)."""
    messages = [
        {"role": "system", "content": RESOLUTION_PROMPT},
        {"role": "user", "content": scene_text + retry_note},
    ]
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
        )
        raw = response.choices[0].message.content
    except Exception as error:  # network, auth, quota, ...
        return None, f"LLM call failed: {error}"

    parsed = _extract_json(raw)
    if parsed is None:
        return None, f"LLM reply was not valid JSON: {str(raw)[:200]!r}"
    return parsed, None


# ===========================================================================
# STEP 4: NORMALISE + VALIDATE THE LLM'S ANSWER
# ===========================================================================

def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def canonicalise(value, kind: str, grounded: list) -> Optional[str]:
    """
    Map whatever the LLM wrote for an object/target onto a canonical
    name. Accepts the canonical name itself, the camera's wording
    ("blue square", "square", "red_circle"), a synonym ("rock") or a
    colour ("blue"). Returns None if it is unknown or ambiguous.
    """
    if value is None:
        return None
    text = " ".join(_words(value))
    if not text or text in {"null", "none"}:
        return None

    valid = CANONICAL_OBJECTS if kind == "object" else CANONICAL_TARGETS
    snake = text.replace(" ", "_")
    if snake in valid:
        return snake

    # 1. Matches (part of) a perceived scene entry -> that entry's binding.
    hits = {
        g["canonical"] for g in grounded
        if g["canonical"] in valid
        and (text == g["description"].lower()
             or set(text.split()) <= set(_words(g["description"])))
    }
    if len(hits) == 1:
        return hits.pop()

    # 2. Colour or synonym of exactly one entity of this kind.
    words, colors = set(text.split()), _colors_in(text)
    hits = {
        n for n in valid
        if ENTITIES[n]["color"] in colors or words & ENTITIES[n]["nouns"]
    }
    return hits.pop() if len(hits) == 1 else None


def _phrase_conflict(phrase, canonical: str, instruction: str) -> Optional[str]:
    """
    Deterministic guard against the LLM mapping the user's words onto
    the wrong thing. Only checks phrases that really occur in the
    instruction. Returns a reason string on conflict, else None.
    """
    if not phrase or str(phrase).lower() not in instruction.lower():
        return None

    entity = ENTITIES[canonical]
    colors = _colors_in(phrase)
    if colors and entity["color"] not in colors:
        return (f"'{phrase}' is {'/'.join(sorted(colors))}, but the only matching "
                f"{'target region' if entity['kind'] == 'target' else 'object'} "
                f"({canonical}) is {entity['color']}")

    if entity["kind"] == "object":
        words = set(_words(phrase))
        for other in CANONICAL_OBJECTS - {canonical}:
            if words & ENTITIES[other]["nouns"] and not words & entity["nouns"]:
                return f"'{phrase}' names the {other}, not the {canonical}"
    return None


def _describe_scene(grounded: list) -> str:
    if not grounded:
        return "nothing was detected"
    return ", ".join(f"{g['description']} -> {g['canonical'] or 'none'}" for g in grounded)


# ===========================================================================
# STEP 5: PLAN ASSEMBLY
# ===========================================================================

def _infeasible(reason: str) -> dict:
    return {
        "feasible": False,
        "actions": [{"skill": "STOP", "reason": reason}],
    }


def _build_move_and_place_plan(object_name: str, target_name: str) -> dict:
    """The fixed move-object-to-target skill sequence."""
    return {
        "feasible": True,
        "actions": [
            {"skill": "SEARCH", "target": object_name},
            {"skill": "APPROACH", "target": object_name},
            {"skill": "REACH", "target": object_name},
            {"skill": "GRASP", "target": object_name},
            {"skill": "MOVE_TO", "target": target_name},
            {"skill": "PLACE", "object": object_name, "target": target_name},
        ],
    }


def validate_action_plan(plan):
    """
    Structural validation: valid top-level shape, only ALLOWED_SKILLS,
    only canonical object/target names, PLACE carrying both fields.
    Returns (ok, error_message_or_None).
    """
    if not isinstance(plan, dict):
        return False, "Planner output is not a dictionary"
    if "feasible" not in plan:
        return False, "Planner output is missing 'feasible'"

    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        return False, "Planner actions are not a non-empty list"

    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            return False, f"Action {index} is not a dictionary"

        skill = action.get("skill")
        if skill not in ALLOWED_SKILLS:
            return False, f"Action {index} uses invalid skill: {skill}"

        if skill in {"SEARCH", "APPROACH", "REACH", "GRASP"}:
            if action.get("target") not in CANONICAL_OBJECTS:
                return False, f"Action {index} has invalid object target: {action.get('target')}"
        elif skill == "MOVE_TO":
            if action.get("target") not in CANONICAL_TARGETS:
                return False, f"Action {index} has invalid destination: {action.get('target')}"
        elif skill == "PLACE":
            if action.get("object") not in CANONICAL_OBJECTS:
                return False, f"Action {index} PLACE has invalid object: {action.get('object')}"
            if action.get("target") not in CANONICAL_TARGETS:
                return False, f"Action {index} PLACE has invalid target: {action.get('target')}"
        elif skill == "VERIFY":
            if action.get("verify_type") not in {"GRASP", "PLACE"}:
                return False, f"Action {index} VERIFY has invalid verify_type: {action.get('verify_type')}"
            if action.get("object") not in CANONICAL_OBJECTS:
                return False, f"Action {index} VERIFY has invalid object: {action.get('object')}"
        elif skill == "STOP":
            if not action.get("reason"):
                return False, f"Action {index} STOP is missing a reason"

    return True, None


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

def _capture_live_test_image(camera_name: str = "overhead_cam"):
    """Render a frame straight from scene.xml (standalone testing only)."""
    import mujoco

    project_root = Path(__file__).resolve().parent.parent
    model = mujoco.MjModel.from_xml_path(str(project_root / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera_name)
    rgb = renderer.render()
    renderer.close()
    return Image.fromarray(rgb)


def plan_from_instruction(instruction: str, image=None, scene_info=None, debug=False) -> dict:
    """
    Resolve the user's instruction against the current scene and return
    a validated action-plan dict. Never raises.

    Scene information, in priority order:
        1. scene_info, if the caller already ran Task 2 this frame;
        2. image, run through Task 2's build_scene();
        3. neither: render a fresh frame from scene.xml (standalone use).
    """
    instruction = (instruction or "").strip()
    if not instruction:
        return _infeasible("Empty instruction")

    # ---- 1. PERCEIVE ------------------------------------------------------
    if scene_info is None:
        try:
            if image is None:
                image = _capture_live_test_image()
            scene_info = build_scene(image=image)
        except Exception as error:
            return _infeasible(f"Scene perception failed: {error}")

    # ---- 2. GROUND --------------------------------------------------------
    grounded = ground_scene(scene_info)
    present = _present(grounded)
    if debug:
        print(f"[planner] grounded scene: {_describe_scene(grounded)}")

    # ---- 3. RESOLVE (LLM) -------------------------------------------------
    scene_text = _build_scene_text(instruction, grounded)
    resolution, error = _call_resolution_llm(scene_text)
    if resolution is None:
        resolution, error = _call_resolution_llm(
            scene_text,
            retry_note=("\n\nYour previous reply could not be used. Reply with "
                        "ONLY the JSON object described in the instructions."),
        )
    if resolution is None:
        return _infeasible(f"Planner could not get a usable answer from the LLM ({error})")
    if debug:
        print(f"[planner] LLM resolution: {resolution}")

    if not _to_bool(resolution.get("feasible")):
        return _infeasible(resolution.get("reason") or "Instruction judged infeasible by the planner")

    # ---- 4. NORMALISE + VALIDATE -------------------------------------------
    object_name = canonicalise(resolution.get("object"), "object", grounded)
    target_name = canonicalise(resolution.get("target"), "target", grounded)

    if object_name is None:
        return _infeasible(
            f"Could not match the requested object ({resolution.get('object')!r}) "
            f"to anything the robot knows (scene: {_describe_scene(grounded)})"
        )
    if target_name is None:
        return _infeasible(
            f"Could not match the requested destination ({resolution.get('target')!r}) "
            f"to a known target region (scene: {_describe_scene(grounded)})"
        )

    for name in (object_name, target_name):
        if name not in present:
            return _infeasible(
                f"'{name}' is not visible in the current scene "
                f"(scene: {_describe_scene(grounded)})"
            )

    for phrase, name in ((resolution.get("object_phrase"), object_name),
                         (resolution.get("target_phrase"), target_name)):
        conflict = _phrase_conflict(phrase, name, instruction)
        if conflict:
            return _infeasible(conflict)

    # ---- 5. ASSEMBLE ------------------------------------------------------
    plan = _build_move_and_place_plan(object_name, target_name)
    ok, error = validate_action_plan(plan)
    if not ok:
        return _infeasible(f"Internal error assembling plan: {error}")
    return plan


if __name__ == "__main__":
    print("Capturing scene once for this session...")
    session_image = _capture_live_test_image()
    session_scene = build_scene(image=session_image)
    print(f"Grounded scene: {_describe_scene(ground_scene(session_scene))}")

    while True:
        text = input("\nType an instruction (or 'quit'): ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        plan = plan_from_instruction(text, scene_info=session_scene, debug=True)
        print(json.dumps(plan, indent=2))