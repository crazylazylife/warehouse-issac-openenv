from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from warehouse_env.environment import WarehouseRobotEnv
from warehouse_env.models import RobotAction

SYSTEM_PROMPT = """You are controlling a warehouse robot.
Return ONLY strict JSON with keys action_type and target.
Valid action_type values: move, pick, place, scan, dock, wait.
Rules:
- Use move to go between locations.
- pick/scan require an object id target.
- place should use the current robot location as target.
- dock requires robot at location dock.
- If uncertain, choose wait.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic OpenAI baseline on all tasks")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default="outputs/evals/baseline_scores.json",
        help="Output JSON path for baseline scores",
    )
    parser.add_argument(
        "--mode",
        choices=["openai", "scripted"],
        default="openai",
        help="Use OpenAI policy (required for submission) or deterministic scripted fallback",
    )
    return parser.parse_args()


def _scripted_action(task_id: str, obs: dict[str, Any]) -> RobotAction:
    loc = obs["robot_location"]
    holding = obs["holding_object"]

    if task_id == "easy_pick_and_stage":
        if loc != "shelf_a" and holding is None:
            return RobotAction(action_type="move", target="shelf_a")
        if holding is None:
            return RobotAction(action_type="pick", target="tote_red")
        if loc != "staging_zone":
            return RobotAction(action_type="move", target="staging_zone")
        return RobotAction(action_type="place", target="staging_zone")

    if task_id == "medium_qc_and_pack":
        if loc != "shelf_b" and holding is None and obs["object_locations"].get("crate_blue") == "shelf_b":
            return RobotAction(action_type="move", target="shelf_b")
        if holding is None and obs["object_locations"].get("crate_blue") == "shelf_b":
            return RobotAction(action_type="pick", target="crate_blue")
        if "scanned_crate_blue_at_qc" not in obs["completed_subgoals"]:
            if loc != "qc_station":
                return RobotAction(action_type="move", target="qc_station")
            return RobotAction(action_type="scan", target="crate_blue")
        if loc != "packing_zone":
            return RobotAction(action_type="move", target="packing_zone")
        return RobotAction(action_type="place", target="packing_zone")

    if task_id == "hard_fragile_pack_and_dock":
        if loc != "shelf_c" and holding is None and obs["object_locations"].get("fragile_box") == "shelf_c":
            return RobotAction(action_type="move", target="shelf_c")
        if holding is None and obs["object_locations"].get("fragile_box") == "shelf_c":
            return RobotAction(action_type="pick", target="fragile_box")
        if "passed_checkpoint_1" not in obs["completed_subgoals"]:
            return RobotAction(action_type="move", target="checkpoint_1")
        if "placed_fragile_box" not in obs["completed_subgoals"]:
            if loc != "fragile_pack_zone":
                return RobotAction(action_type="move", target="fragile_pack_zone")
            return RobotAction(action_type="place", target="fragile_pack_zone")
        if loc != "dock":
            return RobotAction(action_type="move", target="dock")
        return RobotAction(action_type="dock", target=None)

    return RobotAction(action_type="wait", target=None)


def choose_action_openai(
    client: Any,
    *,
    model: str,
    seed: int,
    task_id: str,
    obs: dict[str, Any],
    state: dict[str, Any],
) -> RobotAction:
    user_prompt = {
        "task_id": task_id,
        "task_description": obs["task_description"],
        "observation": obs,
        "state": {
            "step_count": state["step_count"],
            "max_steps": state["max_steps"],
            "score": state["score"],
        },
        "instruction": "Return the best next action as strict JSON.",
    }

    try:
        response = client.responses.create(
            model=model,
            temperature=0,
            top_p=1,
            seed=seed,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
        )
        text = response.output_text.strip()
        payload = json.loads(text)
        return RobotAction.model_validate(payload)
    except Exception:
        return RobotAction(action_type="wait", target=None)


def run_single_task(env: WarehouseRobotEnv, task_id: str, args: argparse.Namespace, client: Any | None) -> dict[str, Any]:
    obs = env.reset(task_id=task_id)
    trajectory: list[dict[str, Any]] = []
    last_info: dict[str, Any] = {}

    for _ in range(min(args.max_steps, env.state().max_steps)):
        if args.mode == "scripted":
            action = _scripted_action(task_id, obs.model_dump())
        else:
            if client is None:
                raise RuntimeError("OpenAI client not initialized")
            action = choose_action_openai(
                client,
                model=args.model,
                seed=args.seed,
                task_id=task_id,
                obs=obs.model_dump(),
                state=env.state().model_dump(),
            )

        result = env.step(action)
        last_info = result.info
        trajectory.append(
            {
                "action": action.model_dump(),
                "reward": result.reward,
                "done": result.done,
                "score": result.info["score"],
            }
        )
        obs = result.observation
        if result.done:
            break

    final_state = env.state()
    return {
        "task_id": task_id,
        "difficulty": final_state.difficulty,
        "steps": final_state.step_count,
        "terminated": final_state.terminated,
        "truncated": final_state.truncated,
        "score": final_state.score,
        "completed_subgoals": final_state.completed_subgoals,
        "invalid_actions": int(last_info.get("invalid_actions", 0)),
        "trajectory": trajectory,
    }


def main() -> None:
    args = parse_args()

    if args.mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --mode openai")

    client = None
    if args.mode == "openai":
        from openai import OpenAI

        client = OpenAI()

    env = WarehouseRobotEnv()
    tasks = env.list_tasks()
    evaluations = [run_single_task(env, task_id, args, client) for task_id in tasks]
    average = sum(item["score"] for item in evaluations) / len(evaluations)

    report = {
        "model": args.model if args.mode == "openai" else "scripted-baseline",
        "mode": args.mode,
        "seed": args.seed,
        "num_tasks": len(evaluations),
        "average_score": round(average, 4),
        "tasks": evaluations,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"average_score": report["average_score"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
