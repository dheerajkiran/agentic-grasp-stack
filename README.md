# Agentic Grasp Stack

A simulated Franka Panda arm executes free-form natural-language instructions
via a team of coordinating agents (Planner, Perception, Grounding, Execution,
Verifier) communicating over ROS2 topics, with a closed observe → act →
verify → replan loop — rather than one monolithic model or a scripted demo.

**Status: Phase 3 (in progress) of 8.** Phase 2 (3 slices) done. Built
incrementally, each slice verified before the next. ROS2 exists for real (see
Phase 2 slice 3), but only inside a UTM VM — no rclpy on the macOS dev
machine, by design.

## Phase 1 scope (done)

- PyBullet scene: table + Franka Panda arm (with gripper) + 4 labeled colored
  cubes (`red`, `green`, `blue`, `yellow`).
- `GraspEnv.reset(randomize=True/False, seed=...)` — a fixed canonical cube
  layout for repeatable runs, or a randomized non-overlapping layout.
- A scripted (hardcoded) pick-place motion: pick one named cube, place it at a
  hardcoded target XY, using PyBullet's inverse kinematics — a smoke test that
  the arm + gripper + IK + physics pipeline works end-to-end before any
  perception/agents are added.

## Phase 2, slice 1: perception (done)

Replaces direct ground-truth access (`env.get_object_states()`) with an actual
open-vocabulary vision model — Grounding DINO (`IDEA-Research/grounding-dino-tiny`
via `transformers`) detects cubes by color name in a rendered overhead image,
still standalone (no ROS2, no agent wrapping yet). Run it:

```bash
uv run scripts/run_perception_demo.py
uv run scripts/run_perception_demo.py --box-threshold 0.25 --text-threshold 0.2
uv run scripts/run_perception_demo.py --headless --output outputs/check.png
```

Saves an annotated PNG (boxes + labels + scores) to `outputs/` and prints a
per-cube ground-truth-pixel vs. nearest-detection comparison — the visual
check is the actual validation, per the project's requirement to confirm
detections by eye before building anything else on top of them. Verified
working: 4/4 cubes detected and matched within a few pixels of ground truth.

## Phase 2, slice 2: 2D → 3D pose estimation (done)

Adds the missing inverse step: given a detected pixel, estimate the object's
3D world position (`perception/geometry.py`'s `pixel_to_world()`, the inverse
of `world_to_pixel()`) by intersecting the camera ray with the known table
plane — valid because every object here is a cube of known height on a table
of known height, not general monocular depth estimation. `run_perception_demo.py`
now prints estimated vs. ground-truth position and XY error per cube. Verified
working: 4/4 cubes, XY error 0.6–2.6mm (mean 1.7mm), well within tolerance for
a 4cm cube. Finding this out surfaced a real Phase 1 bug — the table's
AABB-reported surface height didn't match where physics actually resolved
contact (~13.6mm off), so cubes spawned slightly embedded in the table;
`GraspEnv.connect()` now self-calibrates the true surface height from a
settled cube's actual resting position instead of trusting the raw AABB.

While building slice 3, also found and fixed a second real bug: Grounding
DINO's post-processing filters by score but doesn't suppress overlapping
duplicate boxes — 2 of the "detections" earlier were near-duplicate/garbled-
label boxes sitting on top of real cubes (invisible in slice 1/2's output
since ground-truth matching implicitly picked one best box per object).
`detector.py`'s `detect()` now applies IoU-based non-max suppression, so raw
detections match real object count (verified: 6 → 4 on the canonical scene).

## Phase 2, slice 3: perception agent + ROS2 publishing (done)

Closes out Phase 2: a real agent entrypoint with **no ground-truth
involvement** (contrast with slices 1-2's demo script, which deliberately
uses ground truth to validate accuracy), plus an actual ROS2 topic on the
other end.

**Architecture:** the Mac has no CUDA and — by design — no ROS2 either; ROS2
lives only in a UTM VM. So this splits in two:
- **Mac side** (`scripts/run_perception_agent.py`, `perception/agent.py`):
  runs the same render → detect → estimate-pose pipeline as before, but reads
  nothing from `env.get_object_states()` — `run_perception_cycle()` builds its
  result purely from the detector + geometry, and writes it to
  `outputs/latest_detections.json`. Built and tested here like every other
  slice: `uv run scripts/run_perception_agent.py`, verified writing correct,
  clean (post-NMS) JSON.
- **VM side** (`ros2_ws/src/agentic_grasp_stack_perception/`): a minimal
  `ament_python` ROS2 package — `detection_publisher` node watches the JSON
  file (mtime-based change detection) and republishes it verbatim as a
  `std_msgs/String` on `/perception/detections` (QoS: RELIABLE +
  TRANSIENT_LOCAL, so a late subscriber still gets the last known detections
  immediately). JSON-as-string rather than a custom `.msg` type deliberately,
  to avoid a second `rosidl` interfaces package this slice — noted as
  deferred polish, not forgotten.

**Important caveat:** the VM-side package is written from rclpy/Humble API
knowledge only — **no ROS2 environment exists on the Mac this was built on**,
so it could not be run, built, or even import-checked here (only
syntax-checked with `py_compile`). First real test happens in your VM. If
something doesn't work there, that's expected-possible, not a sign something
else is wrong — treat it as the first real test it is.

**VM runbook** (replace `<vm-user>`/`<VM_IP>` with yours; assumes ROS2
Humble and a copy of this repo at `~/Agentic-grasp-stack` on the VM):

```bash
# one-time, on the VM:
mkdir -p ~/ros2_ws/src ~/Agentic-grasp-stack/outputs

# from the Mac, copy the package over:
scp -r ros2_ws/src/agentic_grasp_stack_perception <vm-user>@<VM_IP>:~/ros2_ws/src/

# each time you produce fresh detections on the Mac:
uv run scripts/run_perception_agent.py
scp outputs/latest_detections.json <vm-user>@<VM_IP>:~/Agentic-grasp-stack/outputs/latest_detections.json

# on the VM, build once (and again after any package edits):
source /opt/ros/humble/setup.bash
cd ~/ros2_ws && colcon build --packages-select agentic_grasp_stack_perception
source install/setup.bash

# on the VM, run it:
ros2 run agentic_grasp_stack_perception detection_publisher

# in a second VM terminal (source install/setup.bash there too):
ros2 topic echo /perception/detections
ros2 topic hz /perception/detections
```

Expect the node to log `watching ... publishing to /perception/detections` on
startup, then `published N detections (file changed)` each time a fresh JSON
file lands — and `ros2 topic echo` should show the last message immediately
even if you start it after the last publish, thanks to TRANSIENT_LOCAL.

## Phase 3: Grounding agent (in progress)

Resolves a free-form referring expression ("the red block", "the block
closest to the robot") to one specific detection from
`outputs/latest_detections.json`, via a structured Claude API call
(`client.messages.parse(..., output_format=GroundingResult)` — a Pydantic
model with `matched_index`, `matched_label`, `reasoning`, `confidence`), not
CLIP — chosen so it can handle spatial/relational phrases, not just
color/shape matching, and because the brief already earmarks Claude API
budget for this agent. Uses `claude-opus-5`.

```bash
uv run scripts/run_grounding_agent.py --expression "the red block"
uv run scripts/run_grounding_agent.py --expression "the block closest to the robot" \
    --json-input outputs/latest_detections.json
```

**Requires `ANTHROPIC_API_KEY`** (or an `ant auth login` profile) — this
makes a real, billed API call. Without it the script fails with a clear
`Could not resolve authentication method` error (confirmed — the request
itself builds and sends correctly, it only fails at the auth step).

**Verification split, same shape as Phase 2 slice 3:** the parsing/validation
logic (`tests/test_grounding_agent.py`) is built and tested with a mocked
Claude client — zero network, zero cost, verified passing. The actual live
API call has **not** been confirmed to produce a correct result — that needs
an API key, which doesn't exist on this dev machine. Not done until run for
real and checked.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync
uv run scripts/run_pick_place_demo.py
```

Flags:

```bash
uv run scripts/run_pick_place_demo.py --headless                 # no GUI window
uv run scripts/run_pick_place_demo.py --randomize --seed 7        # randomized layout
uv run scripts/run_pick_place_demo.py --pick blue_cube --place-x 0.4 --place-y -0.2
```

On success the script prints `PICK-PLACE SUCCESS: ...` and exits 0; on
failure it prints `PICK-PLACE FAILED: ...` with the actual vs. expected
position and exits 1.

Run the headless regression tests:

```bash
uv run pytest tests/ -v
```

## Apple Silicon troubleshooting

The official `pybullet` PyPI package has no prebuilt macOS arm64 wheel and
tries to build Bullet's C++ core from source. On recent macOS SDKs
(confirmed on macOS 26 "Tahoe") that source build fails outright — its
vendored zlib redefines the `fdopen` macro in a way that collides with the
SDK's `stdio.h` declaration (`error: expected identifier or '('` in
`zutil.c`). This repo therefore depends on
[`pybullet-arm64`](https://pypi.org/project/pybullet-arm64/), a community
fork that ships prebuilt arm64 wheels — it's a drop-in replacement
(`import pybullet` still works, same API) so no code here depends on the
package name. If `uv sync` ever fails trying to build the real `pybullet`
from source, check that `pyproject.toml` still lists `pybullet-arm64` and not
plain `pybullet`.

## Project layout

```
src/agentic_grasp_stack/
├── config/
│   ├── scene_config.py        # scene/robot/camera constants
│   ├── perception_config.py   # Grounding DINO model id + detection thresholds
│   └── grounding_config.py    # Claude model id + max_tokens for the Grounding agent
├── sim/
│   ├── env.py     # GraspEnv: connect/reset/close, ground-truth + camera observations
│   ├── robot.py   # PandaRobot: IK-driven Cartesian motion, gripper control
│   ├── scene.py   # static plane/table loading
│   ├── objects.py # cube spawning, non-overlapping random placement, labels
│   └── motion.py  # Phase-1-only scripted pick-place state machine
├── perception/
│   ├── detector.py    # Grounding DINO prompt-building + inference + NMS
│   ├── geometry.py    # world<->pixel camera projection (both directions)
│   ├── visualize.py   # box drawing
│   └── agent.py       # ground-truth-free detect+pose cycle, JSON hand-off writer
└── grounding/
    └── agent.py    # resolve_reference(): structured Claude call -> matched detection
scripts/
├── run_pick_place_demo.py     # Phase 1 CLI verification entrypoint
├── run_perception_demo.py     # Phase 2 CLI: detection+pose vs. ground truth (validation)
├── run_perception_agent.py    # Phase 2 CLI: the real agent, writes JSON, no ground truth
└── run_grounding_agent.py     # Phase 3 CLI: resolve an expression against that JSON
tests/
├── test_env_smoke.py            # Phase 1 headless regression guard
├── test_perception_geometry.py  # Phase 2 camera-geometry round-trip test
├── test_perception_agent.py     # Phase 2 agent output schema test
└── test_grounding_agent.py      # Phase 3 parsing/validation test (mocked Claude client)
ros2_ws/src/agentic_grasp_stack_perception/   # ROS2 package -- runs in the VM, not the Mac
├── package.xml, setup.py, setup.cfg, resource/...
└── agentic_grasp_stack_perception/detection_publisher_node.py
```

`env.py` and `robot.py` are the reusable core: a future ROS2 node wraps
`GraspEnv` unchanged, and a future Execution agent calls
`robot.move_to_pose()` / `robot.set_gripper()` directly instead of the
hardcoded state machine in `motion.py`. The Planner agent (Phase 5) will
reuse `grounding/agent.py`'s Claude-calling pattern (lazy client singleton +
`messages.parse` structured output).
