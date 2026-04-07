from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    MOVE = "move"
    PICK = "pick"
    PLACE = "place"
    SCAN = "scan"
    DOCK = "dock"
    WAIT = "wait"


class RobotAction(BaseModel):
    action_type: ActionType
    target: str | None = Field(default=None, description="Location id or object id")
    note: str | None = None


class RewardSignal(BaseModel):
    total: float = Field(ge=0.0, le=1.0)
    progress: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    penalties: float = Field(ge=-1.0, le=0.0)
    task_score: float = Field(ge=0.0, le=1.0)


class RobotObservation(BaseModel):
    task_id: str
    task_description: str
    robot_location: str
    holding_object: str | None
    battery_level: int = Field(ge=0, le=100)
    object_locations: dict[str, str]
    completed_subgoals: list[str]
    remaining_subgoals: list[str]
    valid_actions: list[str]
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    latest_event: str


class EnvState(BaseModel):
    episode_id: str
    task_id: str
    difficulty: str
    step_count: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    terminated: bool
    truncated: bool
    robot_location: str
    holding_object: str | None
    battery_level: int = Field(ge=0, le=100)
    object_locations: dict[str, str]
    completed_subgoals: list[str]
    reward: RewardSignal
    action_history: list[RobotAction]
    score: float = Field(ge=0.0, le=1.0)


class StepResult(BaseModel):
    observation: RobotObservation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward")
    @classmethod
    def clamp_reward(cls, value: float) -> float:
        return max(0.0, min(1.0, value))
