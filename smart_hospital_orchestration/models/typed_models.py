"""Pydantic typed models for inference pipeline contracts."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ObservationModel(BaseModel):
    """Typed observation snapshot for agent decision making."""

    task: str = Field(..., description="Task name: easy/medium/hard")
    step: int = Field(..., ge=0)
    waiting: int = Field(..., ge=0)
    admitted: int = Field(..., ge=0)
    critical_waiting: int = Field(..., ge=0)
    available_doctors: int = Field(..., ge=0)
    available_beds: int = Field(..., ge=0)
    raw: Dict[str, Any] = Field(default_factory=dict)


class ActionModel(BaseModel):
    """Typed action payload selected by baseline/LLM policy."""

    action: int = Field(..., ge=0, le=4)
    rationale: Optional[str] = Field(default=None)


class RewardModel(BaseModel):
    """Typed reward value emitted at each step."""

    value: float
    normalized_0_to_1: float = Field(..., ge=0.0, le=1.0)
