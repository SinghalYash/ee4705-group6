"""
Task 3.iv test set: natural-language instructions for the planner.

Each case runs against a FIXED scene description (what Task 2 would
report), so the evaluation measures the Task 3 planner alone and does
not depend on MuJoCo, live perception (Task 2) or execution (Task 4).

Two groups of cases
-------------------
core   - what the planner is designed to handle: single-object
         move-to-region requests (standard phrasing, paraphrases,
         colour-only / camera wording) and clearly invalid requests.
stress - deliberately hard cases that probe known limitations: spatial
         references, multi-object requests, negation, ambiguity,
         unsupported placements, speech-recognition noise, non-English,
         and perception errors. Many of these are EXPECTED TO FAIL; they
         are included so the evaluation reports weaknesses, not just
         successes.

The expected answer for every case is the CORRECT behaviour for this
robot and scene (ground truth), not what the planner happens to output.

Fields per case
---------------
id          short unique id
group       "core" or "stress"
category    what the case is testing
instruction what the user says
scene       key into SCENES below
expected    list of (object, target) moves the plan should contain, in
            any order; an empty list means the request must be rejected
            (feasible = False)
note        why the expected answer is what it is / why the case is hard
"""

# ---------------------------------------------------------------------------
# Scenes, in the format task_3.scene_builder.build_scene() returns.
# Physical layout (scene.xml): robot base at (0, 0);
#   stone (grey sphere)     at (0.28, -0.18)
#   box   (blue cube)       at (0.32, -0.08)
#   cylinder (green)        at (0.30,  0.08)
#   red target area         at (0.25,  0.20)
# ---------------------------------------------------------------------------

SCENES = {
    # Wording Task 2 actually produces for scene.xml from the overhead camera.
    "default": {
        "objects": ["blue square", "green circle", "gray sphere"],
        "target_regions": ["red circle"],
    },
    # Same scene, different (but correct) perception wording.
    "alt_wording": {
        "objects": ["blue cube", "green cylinder", "grey stone"],
        "target_regions": ["red area"],
    },
    # Task 2 filed the red region under "objects" instead of "target_regions".
    "target_as_object": {
        "objects": ["blue square", "green circle", "gray sphere", "red circle"],
        "target_regions": [],
    },
    # Task 2 reported the same cube twice.
    "duplicate_detection": {
        "objects": ["blue square", "blue cube", "green circle", "gray sphere"],
        "target_regions": ["red circle"],
    },
    # The cylinder is genuinely not visible this frame.
    "no_cylinder": {
        "objects": ["blue square", "gray sphere"],
        "target_regions": ["red circle"],
    },
    # Genuinely no target region visible.
    "no_target": {
        "objects": ["blue square", "green circle", "gray sphere"],
        "target_regions": [],
    },
    # Nothing detected at all.
    "empty": {
        "objects": [],
        "target_regions": [],
    },
    # PERCEPTION ERROR: the grey stone is physically there but Task 2
    # called it white.
    "stone_colour_misread": {
        "objects": ["blue square", "green circle", "white sphere"],
        "target_regions": ["red circle"],
    },
    # PERCEPTION ERROR: the semi-transparent red region over the blue
    # floor was reported as purple.
    "target_colour_misread": {
        "objects": ["blue square", "green circle", "gray sphere"],
        "target_regions": ["purple circle"],
    },
}

GROUPS = ["core", "stress"]

CATEGORY_ORDER = [
    # core, valid
    "standard", "paraphrase", "colour_wording", "alt_scene",
    # core, invalid
    "unknown_object", "bad_destination", "no_destination", "not_manipulation", "not_in_scene",
    # stress
    "spatial_reference", "multi_object", "negation", "ambiguous_reference",
    "unsupported_placement", "distractor", "speech_noise", "non_english", "perception_error",
]

R = "red_area"


def _case(case_id, group, category, instruction, expected, scene="default", note=""):
    return {"id": case_id, "group": group, "category": category,
            "instruction": instruction, "scene": scene,
            "expected": [tuple(m) for m in expected], "note": note}


def core_ok(case_id, category, instruction, obj, scene="default", note=""):
    return _case(case_id, "core", category, instruction, [(obj, R)], scene, note)


def core_bad(case_id, category, instruction, scene="default", note=""):
    return _case(case_id, "core", category, instruction, [], scene, note)


def stress(case_id, category, instruction, expected, scene="default", note=""):
    return _case(case_id, "stress", category, instruction, expected, scene, note)


TEST_CASES = [
    # =======================================================================
    # CORE - valid requests
    # =======================================================================
    core_ok("S01", "standard", "Pick up the stone and place it in the red area.", "stone"),
    core_ok("S02", "standard", "Move the box to the red area.", "box"),
    core_ok("S03", "standard", "Take the cylinder and place it in the red area.", "cylinder"),
    core_ok("S04", "standard", "Grab the stone and put it in the red zone.", "stone"),
    core_ok("S05", "standard", "Place the box into the red area.", "box"),
    core_ok("S06", "standard", "Put the cylinder in the red area.", "cylinder"),
    core_ok("S07", "standard", "Move the stone to the red area.", "stone"),

    core_ok("P01", "paraphrase", "Could you move that rock to the red zone?", "stone"),
    core_ok("P02", "paraphrase", "Please pick up the nearby stone and place it on the red marker.", "stone"),
    core_ok("P03", "paraphrase", "Grab that cylinder and drop it in the red spot.", "cylinder"),
    core_ok("P04", "paraphrase", "Take the box over to the target area.", "box"),
    core_ok("P05", "paraphrase", "Find the stone and bring it to the red region.", "stone"),
    core_ok("P06", "paraphrase", "I'd like the cube moved onto the red target, please.", "box"),
    core_ok("P07", "paraphrase", "Can you carry the block over to the red zone?", "box"),
    core_ok("P08", "paraphrase", "Relocate the pebble to the target region.", "stone"),
    core_ok("P09", "paraphrase", "Would you mind setting the green can down in the red area?", "cylinder"),
    core_ok("P10", "paraphrase", "Transfer the box onto the red spot.", "box"),
    core_ok("P11", "paraphrase", "robot bring the rock to the goal", "stone",
            note="no punctuation, 'goal' as the target word"),
    core_ok("P12", "paraphrase", "The cylinder needs to go in the red area.", "cylinder",
            note="declarative rather than imperative"),

    core_ok("C01", "colour_wording", "move blue to red", "box"),
    core_ok("C02", "colour_wording", "move box to red circle", "box"),
    core_ok("C03", "colour_wording", "move blue square to red area", "box"),
    core_ok("C04", "colour_wording", "Put the green one in the red circle.", "cylinder"),
    core_ok("C05", "colour_wording", "Place the grey ball in the red area.", "stone"),
    core_ok("C06", "colour_wording", "move green to the target", "cylinder"),
    core_ok("C07", "colour_wording", "Put the blue thing on the red marker.", "box"),
    core_ok("C08", "colour_wording", "move the gray sphere to red", "stone"),
    core_ok("C09", "colour_wording", "Move the sphere to the target.", "stone",
            note="the stone is a sphere in scene.xml and Task 2 reports 'gray sphere'"),

    core_ok("A01", "alt_scene", "Move the box to the red area.", "box", scene="alt_wording"),
    core_ok("A02", "alt_scene", "Put the rock in the red zone.", "stone", scene="alt_wording"),
    core_ok("A03", "alt_scene", "Take the green cylinder to the target.", "cylinder", scene="alt_wording"),
    core_ok("A04", "alt_scene", "move blue to red", "box", scene="alt_wording"),
    core_ok("A05", "alt_scene", "Move the box to the red area.", "box", scene="target_as_object",
            note="Task 2 listed the red region as an object"),
    core_ok("A06", "alt_scene", "Move the box to the red area.", "box", scene="duplicate_detection",
            note="Task 2 reported the cube twice"),

    # =======================================================================
    # CORE - invalid requests (must be rejected)
    # =======================================================================
    core_bad("U01", "unknown_object", "Pick up the banana and place it in the red area."),
    core_bad("U02", "unknown_object", "Pick up the laptop and put it in the red zone."),
    core_bad("U03", "unknown_object", "Grab the chair and place it in the red area."),
    core_bad("U04", "unknown_object", "Move the pyramid to the target."),
    core_bad("U05", "unknown_object", "Move the apple to the red area."),
    core_bad("U06", "unknown_object", "Put the red ball in the red area.",
             note="no red object exists; the only red thing is the target region"),

    core_bad("D01", "bad_destination", "Grab the stone and place it in the green zone."),
    core_bad("D02", "bad_destination", "Put the cylinder in the yellow area."),
    core_bad("D03", "bad_destination", "Put the stone on the moon."),
    core_bad("D04", "bad_destination", "Move the box to the blue area."),
    core_bad("D05", "bad_destination", "Put the box on the table.",
             note="the table is not a target region"),
    core_bad("D06", "bad_destination", "Place the cylinder on the robot's base."),

    core_bad("N01", "no_destination", "Pick up the cylinder."),
    core_bad("N02", "no_destination", "Grab the rock."),
    core_bad("N03", "no_destination", "Lift the blue box."),
    core_bad("N04", "no_destination", "Hold the cylinder."),

    core_bad("M01", "not_manipulation", "Move the robot to Mars."),
    core_bad("M02", "not_manipulation", "Fly the drone to the red area."),
    core_bad("M03", "not_manipulation", "Delete the box."),
    core_bad("M04", "not_manipulation", "What colour is the box?"),
    core_bad("M05", "not_manipulation", "Dance."),
    core_bad("M06", "not_manipulation", "Paint the box red."),
    core_bad("M07", "not_manipulation", "Move the red area to the box.",
             note="the target region is painted on the table and cannot be moved"),

    core_bad("V01", "not_in_scene", "Move the cylinder to the red area.", scene="no_cylinder"),
    core_bad("V02", "not_in_scene", "Move the box to the red area.", scene="no_target"),
    core_bad("V03", "not_in_scene", "Put the green one in the red zone.", scene="no_cylinder"),
    core_bad("V04", "not_in_scene", "Move the stone to the red area.", scene="empty"),

    # =======================================================================
    # STRESS - known limitations (several are expected to fail)
    # =======================================================================
    stress("X01", "spatial_reference", "Move the object closest to the robot base to the red area.",
           [("cylinder", R)],
           note="cylinder is nearest (0.31 m vs 0.33 m); the scene description has no positions"),
    stress("X02", "spatial_reference", "Move the object furthest from the red area into it.",
           [("stone", R)],
           note="stone is furthest (0.38 m); needs position information the planner does not get"),
    stress("X03", "spatial_reference", "Move the object between the stone and the cylinder to the red area.",
           [("box", R)],
           note="box lies between them along y; needs spatial layout"),

    stress("X04", "multi_object", "Move the box and the stone to the red area.",
           [("box", R), ("stone", R)],
           note="robot can do this as two sequential moves; planner only supports one object"),
    stress("X05", "multi_object", "Put everything in the red area.",
           [("box", R), ("stone", R), ("cylinder", R)]),
    stress("X06", "multi_object", "Move the cylinder to the red area, then the box.",
           [("cylinder", R), ("box", R)]),

    stress("X07", "negation", "Don't move the box, move the stone to the red area.",
           [("stone", R)]),
    stress("X08", "negation", "Do not move the box to the red area.",
           [], note="negated command: nothing should be planned"),
    stress("X09", "negation", "Move every object except the cylinder to the red area.",
           [("box", R), ("stone", R)]),

    stress("X10", "ambiguous_reference", "Move the round object to the red area.",
           [], note="both the stone (sphere) and cylinder (round from above) fit; should not guess"),
    stress("X11", "ambiguous_reference", "Move it to the red area.",
           [], note="no referent for 'it'"),
    stress("X12", "ambiguous_reference", "Move the green box to the red area.",
           [], note="contradictory: the box is blue, the green object is a cylinder"),

    stress("X13", "unsupported_placement", "Move the box next to the red area.",
           [], note="PLACE puts objects inside the region; 'next to' cannot be executed"),
    stress("X14", "unsupported_placement", "Move the stone away from the red area.",
           [], note="opposite of the only supported placement"),
    stress("X15", "unsupported_placement", "Place the box in the red area, then stack the stone on top of it.",
           [], note="stacking is not a supported skill; a partial plan is not correct"),
    stress("X16", "unsupported_placement", "Move the cylinder to the red area and then bring it back.",
           [], note="no skill for returning an object to its start pose"),

    stress("X17", "distractor", "The stone is too heavy, so move the cylinder to the red area instead.",
           [("cylinder", R)], note="first object mentioned is not the one to move"),
    stress("X18", "distractor", "Leave the blue box where it is and put the grey stone in the red zone.",
           [("stone", R)]),

    stress("X19", "speech_noise", "move the books to the read area",
           [("box", R)], note="speech-to-text errors: box -> books, red -> read"),
    stress("X20", "speech_noise", "put the rack in the red zone",
           [("stone", R)], note="rock -> rack"),
    stress("X21", "speech_noise", "move the cilinder too the red aria",
           [("cylinder", R)], note="misspellings"),
    stress("X22", "speech_noise", "move the blue bocks to the red area",
           [("box", R)]),

    stress("X23", "non_english", "Bewege den blauen Würfel in den roten Bereich.",
           [("box", R)], note="German"),
    stress("X24", "non_english", "Pon la piedra en el área roja.",
           [("stone", R)], note="Spanish"),
    stress("X25", "non_english", "把绿色的圆柱放到红色区域。",
           [("cylinder", R)], note="Chinese"),

    stress("X26", "perception_error", "Move the stone to the red area.",
           [("stone", R)], scene="stone_colour_misread",
           note="stone is physically present but Task 2 called it white; colour grounding cannot match it"),
    stress("X27", "perception_error", "Move the box to the red area.",
           [("box", R)], scene="target_colour_misread",
           note="red region reported as purple; colour grounding cannot match it"),
]
