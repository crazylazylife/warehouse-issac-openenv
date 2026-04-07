from __future__ import annotations

import uuid
from typing import Any

from .isaac_bridge import BridgeEvent, IsaacSimBridge
from .models import ActionType, EnvState, RewardSignal, RobotAction, RobotObservation, StepResult
from .tasks import TASK_SPECS, TaskSpec, ordered_subgoal_progress


class WarehouseRobotEnv:
    """
    OpenEnv-style real-world robotics environment.

    API:
    - reset(task_id: str | None = None) -> RobotObservation
    - step(action: RobotAction | dict[str, Any]) -> StepResult
    - state() -> EnvState
    """

    def __init__(self, task_id: str = "easy_pick_and_stage") -> None:
        if task_id not in TASK_SPECS:
            raise ValueError(f"Unknown task_id={task_id}. Available: {list(TASK_SPECS)}")
        self._bridge = IsaacSimBridge()
        self._task: TaskSpec = TASK_SPECS[task_id]
        self._episode_id = ""
        self._step_count = 0
        self._robot_location = "start"
        self._holding_object: str | None = None
        self._battery_level = 100
        self._object_locations: dict[str, str] = {}
        self._completed_subgoals: list[str] = []
        self._action_history: list[RobotAction] = []
        self._invalid_actions = 0
        self._repeat_penalties = 0
        self._safety_violations = 0
        self._terminated = False
        self._truncated = False
        self._latest_event = ""
        self._reward = RewardSignal(
            total=0.0,
            progress=0.0,
            safety=1.0,
            efficiency=1.0,
            penalties=0.0,
            task_score=0.0,
        )

    def list_tasks(self) -> list[str]:
        return list(TASK_SPECS.keys())

    def close(self) -> None:
        """No-op close for compatibility with OpenEnv client lifecycle."""
        return None

    def reset(self, task_id: str | None = None) -> RobotObservation:
        if task_id is not None:
            if task_id not in TASK_SPECS:
                raise ValueError(f"Unknown task_id={task_id}. Available: {list(TASK_SPECS)}")
            self._task = TASK_SPECS[task_id]

        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._robot_location = "start"
        self._holding_object = None
        self._battery_level = 100
        self._object_locations = dict(self._task.initial_object_locations)
        self._completed_subgoals = []
        self._action_history = []
        self._invalid_actions = 0
        self._repeat_penalties = 0
        self._safety_violations = 0
        self._terminated = False
        self._truncated = False
        self._latest_event = "Episode reset."
        self._reward = RewardSignal(
            total=0.0,
            progress=0.0,
            safety=1.0,
            efficiency=1.0,
            penalties=0.0,
            task_score=0.0,
        )
        return self._build_observation()

    def state(self) -> EnvState:
        return EnvState(
            episode_id=self._episode_id,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            terminated=self._terminated,
            truncated=self._truncated,
            robot_location=self._robot_location,
            holding_object=self._holding_object,
            battery_level=self._battery_level,
            object_locations=dict(self._object_locations),
            completed_subgoals=list(self._completed_subgoals),
            reward=self._reward,
            action_history=list(self._action_history),
            score=self.grade_episode(),
        )

    def step(self, action: RobotAction | dict[str, Any]) -> StepResult:
        if self._terminated or self._truncated:
            raise RuntimeError("Episode already completed. Call reset() before step().")

        robot_action = self._coerce_action(action)
        self._step_count += 1
        prev_progress = ordered_subgoal_progress(self._task, self._completed_subgoals)
        self._action_history.append(robot_action)

        (
            self._robot_location,
            self._holding_object,
            self._object_locations,
            self._battery_level,
            event,
        ) = self._bridge.apply(
            robot_action,
            task_id=self._task.task_id,
            robot_location=self._robot_location,
            holding_object=self._holding_object,
            object_locations=dict(self._object_locations),
            battery_level=self._battery_level,
        )

        self._update_subgoals(robot_action, event)
        self._latest_event = event.message
        if event.invalid_action:
            self._invalid_actions += 1
        if event.safety_violation:
            self._safety_violations += 1

        repeated = self._is_repeated_action()
        if repeated:
            self._repeat_penalties += 1

        success = self._is_task_complete()
        if success:
            self._terminated = True

        if event.safety_violation:
            self._terminated = True

        if self._step_count >= self._task.max_steps and not self._terminated:
            self._truncated = True

        self._reward = self._compute_reward(
            prev_progress=prev_progress,
            invalid_action=event.invalid_action,
            repeated_action=repeated,
            safety_violation=event.safety_violation,
            success=success,
        )

        obs = self._build_observation()
        info = {
            "episode_id": self._episode_id,
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "reward_signal": self._reward.model_dump(),
            "score": self.grade_episode(),
            "invalid_actions": self._invalid_actions,
            "repeat_penalties": self._repeat_penalties,
            "safety_violations": self._safety_violations,
        }

        return StepResult(
            observation=obs,
            reward=self._reward.total,
            done=self._terminated or self._truncated,
            info=info,
        )

    def grade_episode(self) -> float:
        eps = 0.01

        def strict_unit_interval(value: float) -> float:
            return max(eps, min(1.0 - eps, value))

        progress = ordered_subgoal_progress(self._task, self._completed_subgoals)
        if self._task.task_id == "easy_pick_and_stage":
            base = progress
            if "placed_tote_red" in self._completed_subgoals:
                base = 1.0
            penalty = min(0.4, 0.08 * self._invalid_actions)
            return strict_unit_interval(base - penalty)

        if self._task.task_id == "medium_qc_and_pack":
            has_scan = "scanned_crate_blue_at_qc" in self._completed_subgoals
            has_place = "placed_crate_blue" in self._completed_subgoals
            base = progress
            if has_place and has_scan:
                base = 1.0
            elif has_place and not has_scan:
                base = 0.6
            penalty = min(0.5, 0.07 * self._invalid_actions)
            return strict_unit_interval(base - penalty)

        if self._task.task_id == "hard_fragile_pack_and_dock":
            has_checkpoint = "passed_checkpoint_1" in self._completed_subgoals
            has_place = "placed_fragile_box" in self._completed_subgoals
            has_dock = "docked_with_charge" in self._completed_subgoals
            base = progress
            if has_checkpoint and has_place and has_dock:
                base = 1.0
            penalty = min(0.6, 0.06 * self._invalid_actions + 0.1 * self._repeat_penalties)
            return strict_unit_interval(base - penalty)

        return strict_unit_interval(progress)

    def _coerce_action(self, action: RobotAction | dict[str, Any]) -> RobotAction:
        if isinstance(action, RobotAction):
            return action
        if isinstance(action, dict):
            return RobotAction.model_validate(action)
        raise TypeError("action must be RobotAction or dict")

    def _is_repeated_action(self) -> bool:
        if len(self._action_history) < 3:
            return False
        a, b, c = self._action_history[-1], self._action_history[-2], self._action_history[-3]
        return a.model_dump() == b.model_dump() == c.model_dump()

    def _is_task_complete(self) -> bool:
        return all(subgoal in self._completed_subgoals for subgoal in self._task.subgoals)

    def _add_subgoal(self, subgoal: str) -> None:
        if subgoal in self._task.subgoals and subgoal not in self._completed_subgoals:
            self._completed_subgoals.append(subgoal)

    def _update_subgoals(self, action: RobotAction, event: BridgeEvent) -> None:
        task_id = self._task.task_id

        if task_id == "easy_pick_and_stage":
            if self._robot_location == "shelf_a":
                self._add_subgoal("at_shelf_a")
            if self._holding_object == "tote_red":
                self._add_subgoal("holding_tote_red")
            if self._robot_location == "staging_zone":
                self._add_subgoal("at_staging_zone")
            if self._object_locations.get("tote_red") == "staging_zone" and self._holding_object is None:
                self._add_subgoal("placed_tote_red")

        elif task_id == "medium_qc_and_pack":
            if self._robot_location == "shelf_b":
                self._add_subgoal("at_shelf_b")
            if self._holding_object == "crate_blue":
                self._add_subgoal("holding_crate_blue")
            if (
                action.action_type == ActionType.SCAN
                and action.target == "crate_blue"
                and self._robot_location == "qc_station"
                and not event.invalid_action
            ):
                self._add_subgoal("scanned_crate_blue_at_qc")
            if self._object_locations.get("crate_blue") == "packing_zone" and self._holding_object is None:
                self._add_subgoal("placed_crate_blue")

        elif task_id == "hard_fragile_pack_and_dock":
            if self._robot_location == "shelf_c":
                self._add_subgoal("at_shelf_c")
            if self._holding_object == "fragile_box":
                self._add_subgoal("holding_fragile_box")
            if self._robot_location == "checkpoint_1":
                self._add_subgoal("passed_checkpoint_1")
            if (
                self._object_locations.get("fragile_box") == "fragile_pack_zone"
                and self._holding_object is None
            ):
                self._add_subgoal("placed_fragile_box")
            if (
                action.action_type == ActionType.DOCK
                and self._robot_location == "dock"
                and self._battery_level >= 100
            ):
                self._add_subgoal("docked_with_charge")

    def _compute_reward(
        self,
        *,
        prev_progress: float,
        invalid_action: bool,
        repeated_action: bool,
        safety_violation: bool,
        success: bool,
    ) -> RewardSignal:
        progress = ordered_subgoal_progress(self._task, self._completed_subgoals)
        progress_delta = max(0.0, progress - prev_progress)

        task_score = self.grade_episode()
        efficiency = max(0.0, 1.0 - (self._step_count / self._task.max_steps))
        safety = 0.0 if safety_violation else 1.0

        penalties = 0.0
        if invalid_action:
            penalties -= 0.10
        if repeated_action:
            penalties -= 0.05
        if safety_violation:
            penalties -= 0.40
        if self._truncated:
            penalties -= 0.15

        dense_progress = min(1.0, 0.6 * progress + 0.4 * progress_delta)
        total = 0.50 * dense_progress + 0.25 * safety + 0.15 * efficiency + 0.10 * task_score + penalties

        if success:
            total = max(total, 0.98)

        total = max(0.0, min(1.0, total))

        return RewardSignal(
            total=total,
            progress=min(1.0, max(0.0, dense_progress)),
            safety=safety,
            efficiency=min(1.0, max(0.0, efficiency)),
            penalties=min(0.0, penalties),
            task_score=min(1.0, max(0.0, task_score)),
        )

    def _build_observation(self) -> RobotObservation:
        remaining = [s for s in self._task.subgoals if s not in self._completed_subgoals]

        return RobotObservation(
            task_id=self._task.task_id,
            task_description=self._task.description,
            robot_location=self._robot_location,
            holding_object=self._holding_object,
            battery_level=self._battery_level,
            object_locations=dict(self._object_locations),
            completed_subgoals=list(self._completed_subgoals),
            remaining_subgoals=remaining,
            valid_actions=self._valid_actions(),
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            latest_event=self._latest_event,
        )

    def _valid_actions(self) -> list[str]:
        actions = ["move:<location>", "wait"]

        for obj, loc in self._object_locations.items():
            if self._holding_object is None and loc == self._robot_location:
                actions.append(f"pick:{obj}")

        if self._holding_object is not None:
            actions.append(f"place:{self._robot_location}")

        if self._task.task_id == "medium_qc_and_pack":
            if self._robot_location == "qc_station":
                actions.append("scan:crate_blue")

        if self._robot_location == "dock":
            actions.append("dock")

        return actions
