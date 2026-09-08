"""Phase 1 verification: scripted pick-place smoke test.

Usage:
    uv run scripts/run_pick_place_demo.py
    uv run scripts/run_pick_place_demo.py --randomize --seed 7
    uv run scripts/run_pick_place_demo.py --headless --pick blue_cube --place-x 0.4 --place-y -0.2
"""

import argparse
import sys
import time

from agentic_grasp_stack.config.scene_config import (
    CUBE_HALF_EXTENT,
    DEFAULT_PLACE_TARGET,
    OBJECT_COLORS,
)
from agentic_grasp_stack.sim.env import GraspEnv
from agentic_grasp_stack.sim.motion import run_pick_place

SUCCESS_XY_TOLERANCE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without the PyBullet GUI")
    parser.add_argument(
        "--randomize", action="store_true", help="randomize cube layout instead of the fixed one"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for --randomize")
    parser.add_argument("--pick", default="red_cube", choices=sorted(OBJECT_COLORS))
    parser.add_argument("--place-x", type=float, default=DEFAULT_PLACE_TARGET[0])
    parser.add_argument("--place-y", type=float, default=DEFAULT_PLACE_TARGET[1])
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="seconds to keep the GUI window open after the result is printed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    place_xy = (args.place_x, args.place_y)

    env = GraspEnv(gui=not args.headless)
    env.connect()
    try:
        env.reset(randomize=args.randomize, seed=args.seed)
        run_pick_place(env, args.pick, place_xy)

        final_position, _ = env.cubes.get_pose(args.pick)
        fx, fy, fz = final_position
        target_x, target_y = place_xy
        xy_error = ((fx - target_x) ** 2 + (fy - target_y) ** 2) ** 0.5
        on_table = fz > env.surface_z
        success = xy_error <= SUCCESS_XY_TOLERANCE and on_table

        target_z = env.surface_z + CUBE_HALF_EXTENT
        if success:
            print(f"PICK-PLACE SUCCESS: {args.pick} placed at ({fx:.3f}, {fy:.3f}, {fz:.3f})")
        else:
            print(
                f"PICK-PLACE FAILED: {args.pick} at ({fx:.3f}, {fy:.3f}, {fz:.3f}), "
                f"expected near ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})"
            )

        if not args.headless and args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
    finally:
        env.close()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
