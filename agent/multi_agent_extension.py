"""Multi-agent extension for Smart Hospital orchestration.

Provides a lightweight coordinator that combines specialized policies:
- triage specialist (critical queue handling)
- capacity specialist (allocation efficiency)
- transfer specialist (city network reassignment)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


Action = int


@dataclass
class AgentVote:
    action: Action
    confidence: float
    rationale: str


class TriageSpecialist:
    """Prioritizes safety and critical patients."""

    def vote(self, readable: Dict[str, Any]) -> AgentVote:
        critical = int(readable.get("critical_waiting", 0))
        waiting = int(readable.get("waiting", 0))
        doctors = int(readable.get("available_doctors", 0))
        beds = int(readable.get("available_beds", 0))

        if critical > 0 and (doctors <= 0 or beds <= 0):
            return AgentVote(2, 0.95, "Critical queue blocked by resources")
        if critical > 0 and doctors > 0 and beds > 0:
            return AgentVote(1, 0.90, "Allocate to critical patients")
        if waiting > 10:
            return AgentVote(2, 0.70, "High pressure queue")
        return AgentVote(0, 0.40, "No triage emergency")


class CapacitySpecialist:
    """Optimizes local throughput and avoids idle capacity."""

    def vote(self, readable: Dict[str, Any]) -> AgentVote:
        waiting = int(readable.get("waiting", 0))
        doctors = int(readable.get("available_doctors", 0))
        beds = int(readable.get("available_beds", 0))

        if waiting > 0 and doctors > 0 and beds > 0:
            return AgentVote(1, 0.92, "Use available local resources")
        if waiting > 8 and (doctors <= 0 or beds <= 0):
            return AgentVote(3, 0.72, "Defer low priority under overload")
        return AgentVote(0, 0.35, "Capacity stable")


class TransferSpecialist:
    """Manages reassignment to city hospitals."""

    def vote(self, readable: Dict[str, Any]) -> AgentVote:
        waiting = int(readable.get("waiting", 0))
        beds = int(readable.get("available_beds", 0))
        city_beds = int(readable.get("city_available_beds", 0))

        if waiting > 0 and beds <= 0 and city_beds > 0:
            return AgentVote(4, 0.88, "Transfer to city network")
        return AgentVote(0, 0.30, "No transfer needed")


class MultiAgentCoordinator:
    """Combine specialist votes with weighted confidence.

    Action IDs:
      0 WAIT, 1 ALLOCATE, 2 ESCALATE, 3 DEFER, 4 REASSIGN
    """

    def __init__(self) -> None:
        self.triage = TriageSpecialist()
        self.capacity = CapacitySpecialist()
        self.transfer = TransferSpecialist()

    def select_action(self, readable: Dict[str, Any]) -> Tuple[Action, Dict[str, Any]]:
        votes = [
            self.triage.vote(readable),
            self.capacity.vote(readable),
            self.transfer.vote(readable),
        ]

        # Weighted action score aggregation.
        scores: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for v in votes:
            scores[v.action] += float(v.confidence)

        chosen = max(scores.items(), key=lambda kv: kv[1])[0]
        rationale = [v.rationale for v in votes if v.action == chosen]

        meta = {
            "scores": scores,
            "votes": [v.__dict__ for v in votes],
            "rationale": rationale[0] if rationale else "Consensus default",
            "policy_source": "multi_agent",
        }
        return chosen, meta
