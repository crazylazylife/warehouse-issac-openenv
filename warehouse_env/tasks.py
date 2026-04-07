from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: str
    description: str
    max_steps: int
    subgoals: tuple[str, ...]
    initial_object_locations: dict[str, str]


TASK_SPECS: dict[str, TaskSpec] = {
    "easy_pick_and_stage": TaskSpec(
        task_id="easy_pick_and_stage",
        difficulty="easy",
        description=(
            "Pick tote_red from shelf_a and place it in staging_zone for outbound handling."
        ),
        max_steps=20,
        subgoals=(
            "at_shelf_a",
            "holding_tote_red",
            "at_staging_zone",
            "placed_tote_red",
        ),
        initial_object_locations={"tote_red": "shelf_a"},
    ),
    "medium_qc_and_pack": TaskSpec(
        task_id="medium_qc_and_pack",
        difficulty="medium",
        description=(
            "Pick crate_blue from shelf_b, scan it at qc_station, then place it at packing_zone."
        ),
        max_steps=30,
        subgoals=(
            "at_shelf_b",
            "holding_crate_blue",
            "scanned_crate_blue_at_qc",
            "placed_crate_blue",
        ),
        initial_object_locations={"crate_blue": "shelf_b"},
    ),
    "hard_fragile_pack_and_dock": TaskSpec(
        task_id="hard_fragile_pack_and_dock",
        difficulty="hard",
        description=(
            "Pick fragile_box from shelf_c, route through checkpoint_1, place at fragile_pack_zone, and dock to recharge."
        ),
        max_steps=40,
        subgoals=(
            "at_shelf_c",
            "holding_fragile_box",
            "passed_checkpoint_1",
            "placed_fragile_box",
            "docked_with_charge",
        ),
        initial_object_locations={"fragile_box": "shelf_c"},
    ),
}


def ordered_subgoal_progress(spec: TaskSpec, completed: list[str]) -> float:
    total = len(spec.subgoals)
    if total == 0:
        return 1.0
    matched = 0
    for subgoal in spec.subgoals:
        if subgoal in completed:
            matched += 1
    return matched / total
