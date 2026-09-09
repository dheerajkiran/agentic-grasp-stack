# Agentic Grasp Stack

A simulated Franka Panda arm executes free-form natural-language instructions
via a team of coordinating agents (Planner, Perception, Grounding, Execution,
Verifier) communicating over ROS2 topics, with a closed observe → act →
verify → replan loop — rather than one monolithic model or a scripted demo.

**Status: Phase 2 (in progress) of 8.** Still no ROS2, agent framework, or LLM
calls — Phase 2 is being built incrementally, slice by slice, each one
verified before the next. No component here talks to another over ROS2 yet.

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

ROS2 publishing (wrapping this as an actual Perception *agent* node) is the
next and last Phase 2 slice, not part of this one.

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
│   └── perception_config.py   # Grounding DINO model id + detection thresholds
├── sim/
│   ├── env.py     # GraspEnv: connect/reset/close, ground-truth + camera observations
│   ├── robot.py   # PandaRobot: IK-driven Cartesian motion, gripper control
│   ├── scene.py   # static plane/table loading
│   ├── objects.py # cube spawning, non-overlapping random placement, labels
│   └── motion.py  # Phase-1-only scripted pick-place state machine
└── perception/
    ├── detector.py    # Grounding DINO prompt-building + inference
    ├── geometry.py    # world<->pixel camera projection (both directions)
    └── visualize.py   # box drawing
scripts/
├── run_pick_place_demo.py    # Phase 1 CLI verification entrypoint
└── run_perception_demo.py    # Phase 2 CLI verification entrypoint (detection + pose)
tests/
├── test_env_smoke.py            # Phase 1 headless regression guard
└── test_perception_geometry.py  # Phase 2 camera-geometry round-trip test
```

`env.py` and `robot.py` are the reusable core: a future ROS2 node wraps
`GraspEnv` unchanged, and a future Execution agent calls
`robot.move_to_pose()` / `robot.set_gripper()` directly instead of the
hardcoded state machine in `motion.py`. No ROS2, agent, or LLM code exists
yet — those land in later phases as sibling packages.
