# EE4705 Project 1.3 – Group 6: Vision-Language-Action Robot in MuJoCo

A simulated robot arm that takes a natural-language instruction (typed or spoken), looks at the scene through a camera, plans a sequence of skills, and picks up and places an object in the target area, checking the result as it goes.

```
Instruction ──► Task 2: Vision (Qwen-VL) ──► Task 3: Planner (LLM) ──► Task 4: Execution ──► Feedback
 (typed/spoken)   scene description,          JSON action plan          closed-loop skills,
                  object grounding            (SEARCH … PLACE)          grasp/place checks
```

**Scene:** a 4-joint arm with a 3-pronged gripper, three objects (grey stone sphere, blue cube, green cylinder) and a red circular target area, defined in `scene.xml`. An overhead camera and a wrist camera provide images.

Group report: https://docs.google.com/document/d/1y9sP_LNV4IOBd3ccxVQy_5yrv92IFGVlBOTDjH5UXy4/edit?usp=sharing

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Setup](#2-setup)
3. [Running the full system (Task 5)](#3-running-the-full-system-task-5)
4. [Task 1 – Simulation and environment](#4-task-1--simulation-and-environment)
5. [Task 2 – Scene understanding and grounding](#5-task-2--scene-understanding-and-grounding)
6. [Task 3 – Language-to-action planner](#6-task-3--language-to-action-planner)
7. [Task 4 – Skill execution and closed-loop feedback](#7-task-4--skill-execution-and-closed-loop-feedback)
8. [Results files](#8-results-files)
9. [Troubleshooting](#9-troubleshooting)
10. [Legacy scripts and known issues](#10-legacy-scripts-and-known-issues)
11. [Contributions](#11-contributions)

---

## 1. Repository layout

| Path | What it is |
|---|---|
| `scene.xml` | MuJoCo world: arm, gripper, objects, target area, cameras |
| `task_1/` | Simulation set-up and verification scripts |
| `task_2/` | Vision: scene understanding and object grounding with Qwen-VL |
| `task_3/` | Planner: instruction + scene → JSON action plan; speech input |
| `task_4/` | Robot skills (SEARCH, APPROACH, REACH, GRASP, MOVE_TO, PLACE) and plan executor |
| `task 5/` | Full pipeline and end-to-end evaluation (note the **space** in the folder name) |
| `*/results/`, `*/evaluation/` | Saved evaluation results used in the report |

---

## 2. Setup

### 2.1 Requirements

- **Python 3.10 to 3.13.** 3.10 is the minimum because the code uses `str | None` type hints. On 3.14, `pyaudio` has no ready-made Windows installer yet.
- **A DashScope (Alibaba Cloud Model Studio) API key.** Tasks 2 and 3 call `qwen3-vl-flash` through the OpenAI-compatible endpoint (Singapore region). The free quota is enough for everything in this repo.
- **Internet access.** Needed for the DashScope API, for Google Web Speech (spoken input only), and for downloading Hugging Face models (local-model experiments in Section 5.3 only).
- **A screen for the MuJoCo viewer window.** The evaluation scripts run without a viewer.
- **A microphone** (optional), only for spoken instructions.

### 2.2 Install

From the project root:

```bash
python -m venv mj-env
# Windows (PowerShell):
mj-env\Scripts\Activate.ps1
# macOS / Linux:
source mj-env/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs everything the main system needs:

| Package | Used for |
|---|---|
| `mujoco` | Simulation, rendering and viewer (also installs `glfw`, used by the viewer) |
| `numpy` | Arrays, inverse kinematics, camera geometry |
| `pillow` | Camera images passed to the vision model |
| `openai` | Client for the DashScope OpenAI-compatible API (Tasks 2 and 3) |
| `opencv-python` | Saving camera images in `task_1/verify_control.py` |
| `SpeechRecognition` | Speech-to-text for spoken instructions |

**Optional: microphone support (`pyaudio`).** This is only needed to *speak* instructions; typed instructions work without it. Install it as a separate step, because on macOS and Linux it needs the PortAudio system library first:

| OS | Command |
|---|---|
| Windows | `pip install pyaudio` |
| macOS | `brew install portaudio` then `pip install pyaudio` |
| Ubuntu / Debian | `sudo apt install portaudio19-dev` then `pip install pyaudio` |

**Optional: local vision models (Task 2 experiments only).** See Section 5.3. The main system does not need these packages.

### 2.3 Environment variables (set these in every new terminal)

Every command in this README is **run from the project root** (the folder containing `scene.xml`). Two variables must be set:

- **API key:** Tasks 2, 3 and 5 need it. Some modules read it as soon as they are imported.
- **`PYTHONPATH`:** set to the project root, so the `task_2`/`task_3`/`task_4` packages can be found. Without it, some scripts stop with `ModuleNotFoundError: No module named 'task_4'`.

**Windows (PowerShell)**
```powershell
$env:DASHSCOPE_API_KEY = "sk-your-key"
$env:PYTHONPATH = (Get-Location).Path
```

**macOS / Linux**
```bash
export DASHSCOPE_API_KEY="sk-your-key"
export PYTHONPATH="$(pwd)"
```

> **macOS only:** scripts that open the MuJoCo viewer must be started with `mjpython` instead of `python` (e.g. `mjpython "task 5/task_5_pipeline.py"`). `mjpython` is installed with the `mujoco` package.

---

## 3. Running the full system (Task 5)

### 3.1 Interactive demo

```bash
python "task 5/task_5_pipeline.py"
```

**What it does for each instruction:**
1. Captures the overhead camera image.
2. Task 3 plans an action sequence from that image (using Task 2's scene description).
3. Task 4 executes the plan in the MuJoCo viewer. Task 2 perception is called during SEARCH to locate the object from the camera.
4. Prints a final report: success, completed actions, failure reason and placement error.

**Using it:**
- Type an instruction and press Enter, e.g. `move the blue cube to the red area`, `put the rock in the red zone`, `move blue to red`.
- Or press **Enter on an empty line** to speak the instruction. You get 3 attempts before it falls back to typing.
- Invalid requests are rejected with a reason and do not move the robot, e.g. `pick up the banana`, `put the stone in the green zone`.
- Type `quit`, `exit` or `q` to stop.

**Example session:**
```
Type an instruction, or press ENTER to speak it: move blue to red
Capturing scene for Task 3 planning...
Generating Task 3 plan...
Task 3 plan:
{'feasible': True, 'actions': [{'skill': 'SEARCH', 'target': 'box'}, ... {'skill': 'PLACE', 'object': 'box', 'target': 'red_area'}]}
Executing Task 5 pipeline...
...
TASK 5 FINAL RESULT
Success: True
```

### 3.2 End-to-end evaluation

```bash
python "task 5/task_5_evaluation.py"
```

**What it does:** runs 24 randomised trials (seed 4705) with the viewer off. The trials vary the target object, object positions and instruction wording. MuJoCo ground truth is used only to randomise scenes and to score results; the controller itself works from Task 2's visual estimates.

**Output:** `task 5/results/task_5_final_results.csv` (one row per trial) and `task 5/results/task_5_summary.csv` (all metrics).

**Settings:** `N_TRIALS`, `RANDOM_SEED` and `SHOW_VIEWER`, near the top of the file.

**Cost:** it makes live API calls for every trial, so expect it to take several minutes.

---

## 4. Task 1 – Simulation and environment

| Command | What it does |
|---|---|
| `python task_1/verify_control.py` | Opens the viewer, moves the arm and closes the gripper, then saves `camera_test_wrist.png` and `camera_test_overhead.png` to the current folder. Confirms control and camera access. |
| `python task_1/test_arm_joints.py` | Drives each arm joint in turn in the viewer, to check joint limits and directions. |
| `python task_1/development/inspect_gripper.py` | Development tool: inspect the gripper model (see [known issues](#10-legacy-scripts-and-known-issues)). |
| `python task_1/development/grip_box_test.py` | Development tool: scripted box grasp used while designing the 3-pronged gripper (see [known issues](#10-legacy-scripts-and-known-issues)). |

`verify_control.py` needs `opencv-python`. Run it once early on: the overhead image it saves is also used by the Task 2 demo below.

---

## 5. Task 2 – Scene understanding and grounding

Task 2 sends camera images to `qwen3-vl-flash` via `task_2/qwen_backend.py`.

### 5.1 Using the perception interface

```python
from PIL import Image
from task_2.perception import understand_scene, ground_object

image = Image.open("camera_test_overhead.png")

scene = understand_scene(image, "What objects and target areas are visible?")
print(scene.status, [f"{o.color} {o.label}" for o in scene.objects], scene.target_regions)

result = ground_object(None, None, image, "blue cube")   # model/processor unused by the cloud backend
print(result.status, result.target.bbox if result.target else None)
```

Both functions return a `PerceptionResult` (defined in `task_2/perception_interface.py`). `ground_object` returns a pixel bounding box, or `status="not_found"` when the target is not visible.

### 5.2 Quick scene-description test

`task_2/test_scene_understanding.py` describes a saved camera image. First edit the image path on line 5 to point to your `camera_test_overhead.png` (it is currently an absolute Windows path). Then run:

```bash
python task_2/test_scene_understanding.py
```

### 5.3 Grounding evaluation (20 trials)

| Step | Command | Output |
|---|---|---|
| 1. Generate the dataset | `python task_2/evaluation/generate_dataset.py` | 20 rendered images in `task_2/evaluation/images/` + `ground_truth.csv` |
| 2a. Run Qwen2.5-VL-3B (local) | `python task_2/evaluation/evaluate_grounding.py` | `evaluation_results.csv` |
| 3a. Score it | `python task_2/evaluation/score_grounding.py` | `final_results.csv` + Target Grounding Accuracy printed by view and target |
| 2b. Run Qwen3-VL-8B (local) | `python task_2/evaluation/evaluate_qwen3.py` | `qwen3_results.csv` |
| 3b. Score it | `python task_2/evaluation/score_qwen3.py` | Accuracy printed by view and target |

Steps 2a and 2b run the models **locally** through Hugging Face, so they need extra packages.

1. **PyTorch:** install `torch` and `torchvision` first, choosing the build that matches your GPU/CUDA version at https://pytorch.org/get-started/locally/.
2. **The rest:**
   ```bash
   pip install "transformers>=4.57" accelerate qwen-vl-utils
   ```
   - `transformers` 4.57 or newer is required for Qwen3-VL.
   - `accelerate` is needed for `device_map="auto"`.
   - `qwen-vl-utils` is needed by the Qwen2.5-VL script and also installs `av`.

**Hardware:** the first run downloads the model weights from Hugging Face (public, no token needed). As a rough guide, allow about 8 GB of GPU memory for Qwen2.5-VL-3B and about 17 GB for Qwen3-VL-8B. On a CPU they run, but very slowly.

The scoring scripts (steps 3a and 3b) only read the CSVs, so the saved results can be re-scored without a GPU.

**Other Task 2 scripts:**
- `task_2/tests/`: checks of the individual local-model components.
- `task_2/evaluation/test_segmentation.py` and `task_2/experiments/wrist_camera/test_wrist_views.py`: camera and segmentation experiments (for the second, see [known issues](#10-legacy-scripts-and-known-issues)).

---

## 6. Task 3 – Language-to-action planner

`task_3/planner.py` turns an instruction and the current scene into a validated JSON plan in five stages:
1. Task 2 describes the scene.
2. Each description is matched to a canonical name (`box`, `stone`, `cylinder`, `red_area`) by its colour.
3. The LLM decides which object and target the user means, or that the request is infeasible.
4. Code checks the answer against the scene and the user's words.
5. Code builds the fixed six-step plan.

The planner never raises an exception. Any failure (network, bad reply, unknown object) returns `{"feasible": false, "actions": [{"skill": "STOP", "reason": ...}]}`.

### 6.1 Try the planner on its own

```bash
python task_3/planner.py
```

**What it does:** renders the scene from `scene.xml` once, prints how each detected item was matched to a canonical name, then waits for instructions. For each one, it prints the model's answer and the final plan. Type `quit` to stop.

**From Python:**
```python
from task_3.planner import plan_from_instruction

plan = plan_from_instruction("move that rock to the red zone")      # renders + perceives the scene itself
plan = plan_from_instruction("move blue to red", image=pil_image)   # use your own camera frame
plan = plan_from_instruction("move blue to red",
                             scene_info={"objects": ["blue square", "green circle", "gray sphere"],
                                         "target_regions": ["red circle"]})  # skip perception
```

**Switching the language model:** edit the `PROVIDER` block at the top of `task_3/planner.py`. OpenAI (`gpt-5-mini`) and a local Ollama model (`qwen3:4b`) are already there as comments.

### 6.2 Spoken input

`task_3/voice_input.py` records from the microphone and transcribes with Google Web Speech (internet access is needed, but no key). It then runs a correction pass that fixes domain words the recogniser often mishears, e.g. "read" → "red" and "rack" → "rock". It is used automatically by the Task 5 pipeline when you press Enter on an empty line. To test it alone:

```bash
python task_3/voice_input.py
```

### 6.3 Planner evaluation (Task 3.iv)

The evaluation runs the planner in isolation, so no MuJoCo, no live perception and no execution are involved:
- **Test set:** 88 instructions (61 core + 27 stress), in `task_3/evaluation/planner_test_cases.py`.
- **Fixed scenes:** every case gives the planner a fixed scene description instead of a live camera frame.

```bash
python task_3/evaluation/evaluate_planner.py                   # all 88 cases, 1 trial
python task_3/evaluation/evaluate_planner.py --trials 3        # repeat each case 3 times
python task_3/evaluation/evaluate_planner.py --group core      # or --group stress
python task_3/evaluation/evaluate_planner.py --category negation --category multi_object
```

**Output:** prints Action Planning Accuracy (overall, by set and by category) and every failed case with its reason. It also saves `task_3/evaluation/results/planner_eval_runs.csv` and `planner_eval_summary.json`.

### 6.4 Before/after comparison with an earlier planner version

```bash
python task_3/evaluation/compare_planner_versions.py
python task_3/evaluation/compare_planner_versions.py --price-in 0.05 --price-out 0.4   # also estimate cost (USD per 1M tokens)
```

**What it does:** runs the archived earlier planner (`task_3/evaluation/baselines/planner_v4.py`) and the current planner on the same 35 instructions.

**Output:** accuracy, LLM calls, input/output tokens and latency per instruction, saved to `task_3/evaluation/results/version_comparison_runs.csv` and `version_comparison_summary.json`. About 70 API calls per run.

---

## 7. Task 4 – Skill execution and closed-loop feedback

`task_4/robot_skills.py` implements the skills using inverse kinematics. GRASP and PLACE verify their own outcome: the gripper contacts after a grasp, and the object position after a place. Failed grasps are retried. `task_4/task_4_executor.py` runs a Task 3 plan skill by skill and stops safely on failure.

> **Important:** in the final code, APPROACH only moves to an object whose position has been estimated by Task 2 perception plus the depth camera (`robot.perceived_target_position`). Only the Task 5 scripts connect Task 2 perception to the robot. Run on their own, the scripts below therefore stop at APPROACH with `No RGB-D position available for <object>`. To see the skills run end to end, use `task 5/task_5_pipeline.py` or `task 5/task_5_evaluation.py`.

| Command | What it does | Settings in the file |
|---|---|---|
| `python task_4/test_task_4_executor.py` | Executes a hard-coded Task-3-style plan (box → red area) in the viewer | `TEST_PLAN` |
| `python task_4/test_robot_skills.py` | Steps through each skill in the viewer, pausing for Enter between stages | `FORCE_FIRST_GRASP_FAILURE = True` shows the grasp-retry recovery |
| `python task_4/test_search_recovery.py` | Demonstrates SEARCH when the target is not initially visible | `FORCE_INITIAL_TARGET_HIDDEN`, `REVEAL_TARGET_AFTER_VIEWPOINT` |
| `python task_4/task_4_evaluation.py` | Manipulation evaluation over 12 trial configurations; reports grasp/place success and placement error | `SMOKE_TEST_ONLY = True` runs one trial only; `SHOW_VIEWER`, `REALTIME` |

**Output:** `task_4_evaluation.py` writes `task_4/results/task_4_trial_results.csv`. The saved `task_4_trial_final_results.csv` in that folder holds the results reported in the report.

---

## 8. Results files

| File | Produced by |
|---|---|
| `task_2/evaluation/ground_truth.csv`, `images/` | `generate_dataset.py` |
| `task_2/evaluation/evaluation_results.csv`, `final_results.csv` | `evaluate_grounding.py`, `score_grounding.py` |
| `task_2/evaluation/qwen3_results.csv` | `evaluate_qwen3.py` |
| `task_3/evaluation/results/planner_eval_*` | `evaluate_planner.py` |
| `task_3/evaluation/results/version_comparison_*` | `compare_planner_versions.py` |
| `task_4/results/task_4_trial_results.csv` (saved copy for the report: `task_4_trial_final_results.csv`) | `task_4_evaluation.py` |
| `task 5/results/task_5_final_results.csv`, `task_5_summary.csv` | `task_5_evaluation.py` |

Re-running a script overwrites its results files.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'task_4'` (or `task_2`, `task_3`) | Set `PYTHONPATH` to the project root (Section 2.3) and run from the project root |
| `KeyError: 'DASHSCOPE_API_KEY'` | Set the API key in the current terminal (Section 2.3) |
| Plan is always `STOP` with "LLM call failed" | Check the API key, internet connection and remaining DashScope quota; the reason text shows the underlying error |
| `can't open file ... task` | The Task 5 folder name contains a space; keep the quotes: `python "task 5/task_5_pipeline.py"` |
| Viewer does not open on macOS | Use `mjpython` instead of `python` |
| `Could not find PyAudio` when pressing Enter to speak | Install `pyaudio` (Section 2.2), or type the instruction instead |
| `pyaudio` fails to build on macOS / Linux | Install PortAudio first (`brew install portaudio` or `sudo apt install portaudio19-dev`) |
| Task 4 scripts stop with `No RGB-D position available` | Expected when run on their own; see the note in Section 7 and use the Task 5 scripts |
| Headless Linux server, rendering errors | `export MUJOCO_GL=egl` (or `osmesa`) before running |
| Planner rejects a request you think is valid | Run `python task_3/planner.py`: it prints how the scene was matched and what the model answered, which shows whether perception or language understanding caused the rejection |

---

## 10. Legacy scripts and known issues

These scripts are kept for the record of how the project developed. They are **not** part of the final system:

| Script | Status |
|---|---|
| `task_3/voice_main.py` | Early Task 3 demo; depends on a removed mock executor and no longer runs. Use `task 5/task_5_pipeline.py` instead. |
| `task_3/test_planner_archive.py` | First 20-instruction planner test; replaced by `task_3/evaluation/evaluate_planner.py`. |
| `task_4/test_planner_to_task_4.py` | Early Task 3 → Task 4 integration test. It imports `planner` directly; to run it, change line 27 to `from task_3.planner import plan_from_instruction`. |
| `test_mujoco.py` | Initial MuJoCo check with the stock `humanoid.xml` (run from the project root). |

**Scripts whose `scene.xml` path is one folder too shallow.** These were moved into subfolders after they were written, so they look for `scene.xml` in the wrong place and stop with `Error opening file`. The fix is one extra `.parent`:

| Script | Change |
|---|---|
| `task_1/development/inspect_gripper.py` | `Path(__file__).resolve().parent.parent` → `Path(__file__).resolve().parent.parent.parent` |
| `task_1/development/grip_box_test.py` | same as above |
| `task_2/experiments/wrist_camera/test_wrist_views.py` | `.parent.parent.parent` → `.parent.parent.parent.parent` |

## 11. Contributions

| Task | Owner | Main files |
|---|---|---|
| 1 – Simulation set-up | All | `scene.xml`, `task_1/` |
| 2 – Scene understanding and grounding | Muhammad Irfan Bin Abdullah | `task_2/` |
| 3 – Language-to-action planner | Singhal Yash | `task_3/` |
| 4 – Execution and closed-loop feedback | Chan Ping-Shuen Savannah | `task_4/` |
| 5 – Integration and evaluation | All | `task 5/` |
