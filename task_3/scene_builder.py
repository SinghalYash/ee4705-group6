"""
Task 2 + Task 3 integration point.
 
Replaces the old hardcoded SCENE_MOCK in planner.py. Instead of always
assuming stone/box/cylinder/red_area exist, this grounds each candidate
name against the CURRENT camera image using Task 2's ground_object(), and
only reports back what vision actually confirmed is present this frame.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from task_2.perception import understand_scene
 
 
def build_scene(image) -> dict:
    """
    Ground every candidate object/region against the current camera image.
 
    Returns:
        {"objects": [names confirmed present],
          "target_regions": [names confirmed present],}
    """
    built_scene = understand_scene(image,"What objects and target areas are visible?")

    scene_info = {"objects": [f"{obj.color} {obj.label}".strip() if obj.color
    else obj.label for obj in built_scene.objects],
    "target_regions": built_scene.target_regions,
}

    return scene_info