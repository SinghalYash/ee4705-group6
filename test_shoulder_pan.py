import time
import mujoco
import mujoco.viewer


model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

pan_actuator = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_ACTUATOR,
    "act_shoulder_pan",
)

pan_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "shoulder_pan",
)

pan_qpos = model.jnt_qposadr[pan_joint]

# Command -30 degrees (-0.5236 radians)
target = -0.5236


with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(3000):

        # Command shoulder-pan actuator
        data.ctrl[pan_actuator] = target

        # Advance physics
        mujoco.mj_step(model, data)
        viewer.sync()

        # Print diagnostic information every 500 steps
        if step % 500 == 0:
            print(
                "\nstep:", step,
                "target:", target,
                "actual:", data.qpos[pan_qpos],
                "ctrl:", data.ctrl[pan_actuator],
                "force:", data.actuator_force[pan_actuator],
                "contacts:", data.ncon,
            )

            for i in range(data.ncon):
                contact = data.contact[i]

                geom1 = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom1,
                )

                geom2 = mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    contact.geom2,
                )

                print(
                    "   CONTACT:",
                    geom1,
                    "<->",
                    geom2,
                )

        time.sleep(model.opt.timestep)

    # Keep viewer open after test
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)