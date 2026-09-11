"""Phase 3: Grounding agent CLI -- resolve a referring expression to one of
the Perception agent's detections (outputs/latest_detections.json by
default, written by scripts/run_perception_agent.py).

Requires ANTHROPIC_API_KEY (or `ant auth login`) -- this makes a real,
billed Claude API call.

Usage:
    uv run scripts/run_grounding_agent.py --expression "the red block"
    uv run scripts/run_grounding_agent.py --expression "the block closest to the robot" \\
        --json-input outputs/latest_detections.json
"""

import argparse
import json
import sys
from pathlib import Path

from agentic_grasp_stack.grounding.agent import resolve_reference

DEFAULT_JSON_INPUT = "outputs/latest_detections.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", required=True, help="referring expression to resolve")
    parser.add_argument("--json-input", default=DEFAULT_JSON_INPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    detections_path = Path(args.json_input)
    if not detections_path.exists():
        print(
            f"No detections file at {detections_path} -- run "
            f"scripts/run_perception_agent.py first."
        )
        return 1

    payload = json.loads(detections_path.read_text())
    detections = payload["detections"]
    if not detections:
        print("No detections in the input file.")
        return 1

    result = resolve_reference(args.expression, detections)

    if result.matched_index is None:
        print(
            f"GROUNDING: no confident match for {args.expression!r} "
            f"({result.confidence}) -- {result.reasoning}"
        )
        return 1

    matched = detections[result.matched_index]
    pos = matched["position"]
    print(
        f"GROUNDING: {args.expression!r} -> '{matched['label']}' at "
        f"({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}) "
        f"[{result.confidence} confidence] -- {result.reasoning}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
