import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path



MODEL_PATH = str(
    Path(__file__).resolve().parent.parent
    / "scene.xml"
)
TESTS = [
    ("shoulder_pan",  "act_shoulder_pan",  -0.40),
    ("shoulder_lift", "act_shoulder_lift",  0.30),
    ("elbow",         "act_elbow",          0.50),
    ("wrist_pitch",   "act_wrist_pitch",   -0.30),
]


def get_joint_qpos_id(model, joint_name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    return model.jnt_qposadr[jid]


def get_actuator_id(model, actuator_name):
    return mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )


def print_contacts(model, data):

    seen = set()

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

        pair = (geom1, geom2)

        if pair not in seen:
            print(
                "   CONTACT:",
                geom1,
                "<->",
                geom2,
            )
            seen.add(pair)


def main():

    model = mujoco.MjModel.from_xml_path(
        MODEL_PATH
    )

    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        for (
            joint_name,
            actuator_name,
            target,
        ) in TESTS:

            print("\n")
            print("=" * 50)
            print("TESTING:", joint_name)
            print("TARGET:", target, "rad")
            print("=" * 50)

            # Start each joint test from a clean state.
            mujoco.mj_resetData(
                model,
                data,
            )

            all_actuator_ids = {
                actuator_name: get_actuator_id(
                    model,
                    actuator_name,
                )
                for _, actuator_name, _ in TESTS
            }

            actuator_id = get_actuator_id(
                model,
                actuator_name,
            )

            qpos_id = get_joint_qpos_id(
                model,
                joint_name,
            )

            for step in range(2500):

                # Hold all arm joints at zero unless this is
                # the joint currently being tested.
                for _, other_actuator_name, _ in TESTS:
                    other_id = all_actuator_ids[
                        other_actuator_name
                    ]
                    data.ctrl[other_id] = 0.0

                # Apply target to joint under test.
                data.ctrl[actuator_id] = target

                mujoco.mj_step(
                    model,
                    data,
                )

                viewer.sync()

                if step % 500 == 0:

                    actual = data.qpos[
                        qpos_id
                    ]

                    print(
                        "step:", step,
                        "target:", target,
                        "actual:", actual,
                        "error:",
                        target - actual,
                        "force:",
                        data.actuator_force[
                            actuator_id
                        ],
                        "contacts:",
                        data.ncon,
                    )

                    print_contacts(
                        model,
                        data,
                    )

                time.sleep(
                    model.opt.timestep
                )

            actual = data.qpos[
                qpos_id
            ]

            print(
                "\nFINAL:",
                joint_name,
            )

            print(
                "Target:",
                target,
            )

            print(
                "Actual:",
                actual,
            )

            print(
                "Error:",
                target - actual,
            )

            # Pause briefly before next joint.
            time.sleep(1.0)

        print(
            "\nAll joint tests completed."
        )

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()