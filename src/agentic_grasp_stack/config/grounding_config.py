"""Grounding agent constants: which Claude model resolves referring
expressions to detections.
"""

from dataclasses import dataclass

GROUNDING_MODEL_ID = "claude-opus-5"


@dataclass(frozen=True)
class GroundingConfig:
    model_id: str = GROUNDING_MODEL_ID
    max_tokens: int = 1024


GROUNDING_CONFIG = GroundingConfig()
