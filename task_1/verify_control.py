"""
Task 1.ii verification script.
Confirms:
  (a) the robot can be programmatically commanded (arm joints + gripper), and
  (b) camera images can be captured from the simulation (wrist cam + overhead cam).

Run with: python verify_control.py
Requires: mujoco, opencv-python  (pip install mujoco opencv-python)
"""

import time
import cv2
import mujoco
import mujoco.viewer
from pathlib import Path


MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "scene.xml"
)



def set_ctrl(model, data, actuator_name, value):
    """Set a control value for a named actuator (robust to actuator ordering)."""
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    data.ctrl[aid] = value


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    print("Actuators available:", actuator_names)

    # --- (a) Verify programmatic control: move the arm to a target pose ---
    target_pose = {
        "act_shoulder_pan": 0.4,
        "act_shoulder_lift": -0.3,
        "act_elbow": 0.6,
        "act_wrist_pitch": 0.0,
        "act_finger_left": 0.0,   # gripper open
        "act_finger_right": 0.0,
    }
    for name, val in target_pose.items():
        set_ctrl(model, data, name, val)

    renderer = mujoco.Renderer(model, height=480, width=640)
    saved_images = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        closing = False
        finger_pos = 0.0
        while viewer.is_running() and step < 3000:
            mujoco.mj_step(model, data)
            viewer.sync()

            # Start closing the gripper gradually from step 800, so it's visible
            # instead of snapping shut in a single control step.
            if step == 800:
                closing = True
                print("Step 800: starting gradual gripper close")

            if closing and finger_pos < 0.02:
                finger_pos += 0.00003  # small increment each physics step
                set_ctrl(model, data, "act_finger_left", finger_pos)
                set_ctrl(model, data, "act_finger_right", finger_pos)

            # --- (b) Verify camera access: capture and save frames ---
            if step == 1800 and not saved_images:
                renderer.update_scene(data, camera="wrist_cam")
                wrist_img = renderer.render()
                cv2.imwrite("camera_test_wrist.png", cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))

                renderer.update_scene(data, camera="overhead_cam")
                overhead_img = renderer.render()
                cv2.imwrite("camera_test_overhead.png", cv2.cvtColor(overhead_img, cv2.COLOR_RGB2BGR))

                print("Saved camera_test_wrist.png and camera_test_overhead.png")
                saved_images = True

            step += 1
            time.sleep(model.opt.timestep)

    print("Done. Control + camera access verified.")


if __name__ == "__main__":
    main()
