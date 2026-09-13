import time
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"

GRIPPER_OPEN = 0.025


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    left_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "act_finger_left",
    )

    right_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "act_finger_right",
    )

    data.ctrl[left_id] = GRIPPER_OPEN
    data.ctrl[right_id] = GRIPPER_OPEN

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            data.ctrl[left_id] = GRIPPER_OPEN
            data.ctrl[right_id] = GRIPPER_OPEN

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()