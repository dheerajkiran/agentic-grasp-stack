"""Phase 2 slice 3: the Perception agent's real entrypoint -- render, detect,
estimate 3D poses, and write the result as a JSON hand-off file (no ground
truth involved; contrast with run_perception_demo.py's visual validation).

This JSON is what gets scp'd to the ROS2 VM for
ros2_ws/src/agentic_grasp_stack_perception to publish.

Usage:
    uv run scripts/run_perception_agent.py
    uv run scripts/run_perception_agent.py --headless --json-output outputs/latest_detections.json
    uv run scripts/run_perception_agent.py --image-output outputs/agent_check.png
"""

import argparse
import sys

from agentic_grasp_stack.config.perception_config import PERCEPTION_CONFIG
from agentic_grasp_stack.perception.agent import run_perception_cycle, write_detections_json
from agentic_grasp_stack.perception.visualize import draw_detections
from agentic_grasp_stack.sim.env import GraspEnv

DEFAULT_JSON_OUTPUT = "outputs/latest_detections.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without the PyBullet GUI")
    parser.add_argument(
        "--randomize", action="store_true", help="randomize cube layout instead of the fixed one"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for --randomize")
    parser.add_argument("--box-threshold", type=float, default=PERCEPTION_CONFIG.box_threshold)
    parser.add_argument("--text-threshold", type=float, default=PERCEPTION_CONFIG.text_threshold)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument(
        "--image-output",
        default=None,
        help="if set, also save an annotated PNG here (skipped by default)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = GraspEnv(gui=not args.headless)
    env.connect()
    try:
        env.reset(randomize=args.randomize, seed=args.seed)
        image = env.get_rgb_image()

        result = run_perception_cycle(
            image,
            env.surface_z,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        json_path = write_detections_json(result, args.json_output)

        if args.image_output:
            boxes = [
                [d["box_px"]["x0"], d["box_px"]["y0"], d["box_px"]["x1"], d["box_px"]["y1"]]
                for d in result["detections"]
            ]
            scores = [d["score"] for d in result["detections"]]
            labels = [d["label"] for d in result["detections"]]
            annotated = draw_detections(
                image, {"boxes": boxes, "scores": scores, "labels": labels}
            )
            annotated.save(args.image_output)

        print(
            f"PERCEPTION AGENT: {len(result['detections'])} objects detected in "
            f"{result['inference_seconds']:.2f}s on {result['device']}, "
            f"JSON written to {json_path.resolve()}"
        )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
