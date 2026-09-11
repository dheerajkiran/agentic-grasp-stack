"""Parsing/validation tests for grounding/agent.py -- monkeypatches the
Claude client so these run with zero network access or API cost, mirroring
test_perception_agent.py's detect() monkeypatch pattern. The live API call
itself is NOT exercised here (no ANTHROPIC_API_KEY on this machine) --
scripts/run_grounding_agent.py is the one to run for that."""

from types import SimpleNamespace

from agentic_grasp_stack.grounding import agent as agent_module
from agentic_grasp_stack.grounding.agent import GroundingResult

DETECTIONS = [
    {
        "label": "red cube",
        "score": 0.9,
        "position": {"x": 0.45, "y": -0.15, "z": 0.02},
        "box_px": {"x0": 100.0, "y0": 100.0, "x1": 140.0, "y1": 140.0},
    },
    {
        "label": "blue cube",
        "score": 0.85,
        "position": {"x": 0.6, "y": 0.15, "z": 0.02},
        "box_px": {"x0": 300.0, "y0": 300.0, "x1": 340.0, "y1": 340.0},
    },
]


class _FakeMessages:
    def __init__(self, result: GroundingResult):
        self._result = result

    def parse(self, **kwargs):
        return SimpleNamespace(parsed_output=self._result)


class _FakeClient:
    def __init__(self, result: GroundingResult):
        self.messages = _FakeMessages(result)


def test_resolve_reference_returns_matched_detection(monkeypatch):
    canned = GroundingResult(
        matched_index=0,
        matched_label="red cube",
        reasoning="Only red object in the scene.",
        confidence="high",
    )
    monkeypatch.setattr(agent_module, "_get_client", lambda: _FakeClient(canned))

    result = agent_module.resolve_reference("the red block", DETECTIONS)

    assert result.matched_index == 0
    assert result.matched_label == "red cube"
    assert result.confidence == "high"


def test_resolve_reference_rejects_out_of_range_index(monkeypatch):
    hallucinated = GroundingResult(
        matched_index=5,
        matched_label="ghost cube",
        reasoning="hallucinated",
        confidence="high",
    )
    monkeypatch.setattr(agent_module, "_get_client", lambda: _FakeClient(hallucinated))

    result = agent_module.resolve_reference("the green block", DETECTIONS)

    assert result.matched_index is None
    assert result.matched_label is None
    assert result.confidence == "low"
    assert "out-of-range" in result.reasoning


def test_resolve_reference_passes_through_no_match(monkeypatch):
    no_match = GroundingResult(
        matched_index=None,
        matched_label=None,
        reasoning="No green object in the scene.",
        confidence="high",
    )
    monkeypatch.setattr(agent_module, "_get_client", lambda: _FakeClient(no_match))

    result = agent_module.resolve_reference("the green block", DETECTIONS)

    assert result.matched_index is None
    assert result.reasoning == "No green object in the scene."
