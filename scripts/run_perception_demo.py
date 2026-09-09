"""Phase 2 slice 1: standalone Grounding DINO detection + visual validation.

No ROS2, no agents -- runs detection on a rendered scene image and saves an
annotated PNG so results can be checked by eye, per the project's requirement
to validate detections visually before wiring anything else to them.

Usage:
    uv run scripts/run_perception_demo.py
    uv run scripts/run_perception_demo.py --box-threshold 0.25 --text-threshold 0.2
    uv run scripts/run_perception_demo.py --headless --output outputs/check.png
"""

import argparse
import math
import sys
from pathlib import Path

from agentic_grasp_stack.config.perception_config import DEFAULT_OUTPUT_PATH, PERCEPTION_CONFIG
from agentic_grasp_stack.config.scene_config import CAMERA_CONFIG
from agentic_grasp_stack.perception.detector import build_prompt, detect
from agentic_grasp_stack.perception.visualize import draw_detections, world_to_pixel
from agentic_grasp_stack.sim.env import GraspEnv

MATCH_PIXEL_RADIUS = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="run without the PyBullet GUI")
    parser.add_argument(
        "--randomize", action="store_true", help="randomize cube layout instead of the fixed one"
    )
    parser.add_argument("--seed", type=int, default=None, help="seed for --randomize")
    parser.add_argument("--box-threshold", type=float, default=PERCEPTION_CONFIG.box_threshold)
    parser.add_argument("--text-threshold", type=float, default=PERCEPTION_CONFIG.text_threshold)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def box_center(box: list[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + x1) / 2, (y0 + y1) / 2


def nearest_detection(
    ground_truth_pixel: tuple[int, int], detections: dict
) -> tuple[int, float] | None:
    best_index, best_distance = None, math.inf
    for i, box in enumerate(detections["boxes"]):
        cx, cy = box_center(box)
        distance = math.hypot(cx - ground_truth_pixel[0], cy - ground_truth_pixel[1])
        if distance < best_distance:
            best_index, best_distance = i, distance
    if best_index is None or best_distance > MATCH_PIXEL_RADIUS:
        return None
    return best_index, best_distance


def main() -> int:
    args = parse_args()

    env = GraspEnv(gui=not args.headless)
    env.connect()
    try:
        object_states = env.reset(randomize=args.randomize, seed=args.seed)
        image = env.get_rgb_image()

        prompt = build_prompt(list(object_states.keys()))
        print(f"Prompt: {prompt!r}")

        detections = detect(image, prompt, args.box_threshold, args.text_threshold)
        print(f"Ran on device={detections['device']} in {detections['seconds']:.2f}s")
        print(f"Raw detections: {len(detections['boxes'])}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = draw_detections(image, detections)
        annotated.save(output_path)

        matched_count = 0
        print("\nGround truth vs. nearest detection:")
        for name, state in object_states.items():
            gt_pixel = world_to_pixel(state["position"], CAMERA_CONFIG)
            match = nearest_detection(gt_pixel, detections)
            if match is not None:
                index, distance = match
                label = detections["labels"][index]
                score = detections["scores"][index]
                matched_count += 1
                print(
                    f"  {name}: ground-truth pixel ~{gt_pixel} -> matched '{label}' "
                    f"({score:.2f}) at {distance:.0f}px away"
                )
            else:
                print(f"  {name}: ground-truth pixel ~{gt_pixel} -> no detection matched")

        print(
            f"\nPERCEPTION CHECK COMPLETE: {matched_count}/{len(object_states)} cubes detected, "
            f"image saved to {output_path.resolve()} -- inspect visually."
        )
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
