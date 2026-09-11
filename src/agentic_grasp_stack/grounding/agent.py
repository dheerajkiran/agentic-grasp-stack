"""Grounding agent: resolves a natural-language referring expression (e.g.
"the red block", "the one closest to the robot") to one of the Perception
agent's detections, via a structured Claude API call.

Uses client.messages.parse(..., output_format=GroundingResult) rather than
manual JSON-schema + string parsing -- the current recommended structured-
output path (see the claude-api skill).
"""

from typing import Literal

import anthropic
from pydantic import BaseModel

from agentic_grasp_stack.config.grounding_config import GROUNDING_CONFIG

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class GroundingResult(BaseModel):
    matched_index: int | None
    matched_label: str | None
    reasoning: str
    confidence: Literal["high", "medium", "low"]


SYSTEM_PROMPT = """You are the Grounding agent in a robotic manipulation pipeline. \
Given a natural-language referring expression and a list of objects currently \
detected on a tabletop, identify which single object (if any) the expression \
refers to.

Coordinate convention: x is distance from the robot's base along the table \
(larger x = farther from the robot); y is left/right (larger y = further to \
the robot's left). All positions are in meters.

If the expression is ambiguous, refers to something not in the list, or \
matches more than one object equally well, set matched_index and \
matched_label to null and explain why in "reasoning". Never invent an \
object that is not in the provided list, and never guess an index outside \
the list's range."""


def resolve_reference(expression: str, detections: list[dict]) -> GroundingResult:
    """Resolve `expression` to one of `detections` (the "detections" list as
    returned by perception.agent.run_perception_cycle()). `matched_index`
    indexes into `detections`; None means no confident match.
    """
    candidates = [
        {"index": i, "label": d["label"], "position": d["position"]}
        for i, d in enumerate(detections)
    ]

    client = _get_client()
    response = client.messages.parse(
        model=GROUNDING_CONFIG.model_id,
        max_tokens=GROUNDING_CONFIG.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f'Referring expression: "{expression}"\n\n'
                    f"Detected objects: {candidates}"
                ),
            }
        ],
        output_format=GroundingResult,
    )
    result = response.parsed_output

    if result.matched_index is not None and not (0 <= result.matched_index < len(detections)):
        # Guard against a hallucinated out-of-range index rather than letting
        # a caller crash indexing into `detections` with it.
        return GroundingResult(
            matched_index=None,
            matched_label=None,
            reasoning=(
                f"Model returned out-of-range index {result.matched_index} "
                f"for {len(detections)} detections; treating as no match."
            ),
            confidence="low",
        )
    return result
