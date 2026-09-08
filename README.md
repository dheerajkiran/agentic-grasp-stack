# Agentic Grasp Stack

A simulated Franka Panda arm executes free-form natural-language instructions
via a team of coordinating agents (Planner, Perception, Grounding, Execution,
Verifier) communicating over ROS2 topics, with a closed observe → act →
verify → replan loop — rather than one monolithic model or a scripted demo.

**Status: Phase 1 of 8.** This phase has no agents, no ROS2, and no LLM calls
yet — it's the PyBullet physics/motion foundation everything else builds on.

## Phase 1 scope

- PyBullet scene: table + Franka Panda arm (with gripper) + 4 labeled colored
  cubes (`red`, `green`, `blue`, `yellow`).
- `GraspEnv.reset(randomize=True/False, seed=...)` — a fixed canonical cube
  layout for repeatable runs, or a randomized non-overlapping layout.
- A scripted (hardcoded) pick-place motion: pick one named cube, place it at a
  hardcoded target XY, using PyBullet's inverse kinematics — a smoke test that
  the arm + gripper + IK + physics pipeline works end-to-end before any
  perception/agents are added.

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
├── config/scene_config.py   # scene/robot/camera constants
└── sim/
    ├── env.py     # GraspEnv: connect/reset/close, ground-truth + camera observations
    ├── robot.py   # PandaRobot: IK-driven Cartesian motion, gripper control
    ├── scene.py   # static plane/table loading
    ├── objects.py # cube spawning, non-overlapping random placement, labels
    └── motion.py  # Phase-1-only scripted pick-place state machine
scripts/run_pick_place_demo.py   # CLI verification entrypoint
tests/test_env_smoke.py          # headless regression guard
```

`env.py` and `robot.py` are the reusable core: a future ROS2 node wraps
`GraspEnv` unchanged, and a future Execution agent calls
`robot.move_to_pose()` / `robot.set_gripper()` directly instead of the
hardcoded state machine in `motion.py`. No ROS2, agent, or LLM code exists
yet — those land in later phases as sibling packages.
