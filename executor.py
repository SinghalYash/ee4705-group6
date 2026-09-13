"""
Mock executor for Task 3 demonstration.

Since Task 2 (VLM grounding) isn't built yet, target joint poses are
hand-tuned to roughly match each named object/region's position in
scene.xml, instead of being computed from real vision + inverse kinematics.
This lets you demonstrate the full language -> plan -> movement loop now.

Once Task 2 is ready: replace MOCK_POSES lookups with real grounding output
(e.g. a bounding box -> a visual-servoing controller, as in Task 4), rather
than these hardcoded joint angles.
"""

import time
import mujoco
import mujoco.viewer

MODEL_PATH = "scene.xml"

# Hand-tuned joint targets (radians) for reaching near each named location,
# based on scene.xml's object positions. NOT computed via inverse kinematics.
MOCK_POSES = {
    "stone":    {"act_shoulder_pan": -0.6, "act_shoulder_lift": -0.3, "act_elbow": 0.7, "act_wrist_pitch": 0.0},
    "box":      {"act_shoulder_pan": -0.2, "act_shoulder_lift": -0.3, "act_elbow": 0.7, "act_wrist_pitch": 0.0},
    "cylinder": {"act_shoulder_pan": 0.3,  "act_shoulder_lift": -0.3, "act_elbow": 0.7, "act_wrist_pitch": 0.0},
    "red_area": {"act_shoulder_pan": 0.9,  "act_shoulder_lift": -0.2, "act_elbow": 0.5, "act_wrist_pitch": 0.0},
    "home":     {"act_shoulder_pan": 0.0,  "act_shoulder_lift": 0.0,  "act_elbow": 0.0, "act_wrist_pitch": 0.0},
}


def set_ctrl(model, data, actuator_name, value):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    data.ctrl[aid] = value


def execute_plan(plan: dict, settle_steps: int = 1200):
    """Step through a validated plan from planner.py, moving the arm in MuJoCo."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if not plan.get("feasible", False):
            reason = "unspecified"
            if plan.get("actions"):
                reason = plan["actions"][0].get("reason", reason)
            print(f"Plan marked infeasible: {reason}")
            for _ in range(300):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
            return

        for action in plan["actions"]:
            skill = action.get("skill")
            target = action.get("target") or action.get("object")
            print(f"Executing: {skill} {target or ''}")

            pose = None
            if skill in ("SEARCH", "APPROACH", "REACH", "MOVE_TO") and target in MOCK_POSES:
                pose = MOCK_POSES[target]
            elif skill == "PLACE" and action.get("target") in MOCK_POSES:
                pose = MOCK_POSES[action["target"]]
            elif skill == "GRASP":
                # No real grasp verification yet -- that's Task 4. Just close the gripper.
                set_ctrl(model, data, "act_finger_left", 0.02)
                set_ctrl(model, data, "act_finger_right", 0.02)
            elif skill == "STOP":
                pose = MOCK_POSES["home"]

            if pose:
                for name, val in pose.items():
                    set_ctrl(model, data, name, val)

            for _ in range(settle_steps):
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)

        print("Plan execution complete (movement only -- no visual verification yet).")


if __name__ == "__main__":
    import json
    from planner import plan_from_instruction

    instruction = input("Type an instruction: ").strip()
    plan = plan_from_instruction(instruction)
    print(json.dumps(plan, indent=2))
    execute_plan(plan)
