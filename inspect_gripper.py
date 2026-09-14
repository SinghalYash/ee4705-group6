import time
import mujoco
import mujoco.viewer


MODEL_PATH = "scene.xml"

GRIPPER_OPEN = 0.025


def main():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    finger_names = [
        "act_finger_1",
        "act_finger_2",
        "act_finger_3",
    ]

    finger_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        for name in finger_names
    ]

    # Open all three fingers.
    for aid in finger_ids:
        data.ctrl[aid] = GRIPPER_OPEN

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while viewer.is_running():

            for aid in finger_ids:
                data.ctrl[aid] = GRIPPER_OPEN

            mujoco.mj_step(
                model,
                data,
            )

            viewer.sync()

            time.sleep(
                model.opt.timestep
            )


if __name__ == "__main__":
    main()