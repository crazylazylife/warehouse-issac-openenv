"""
Submission inference runner for warehouse_isaac_openenv.

Required env vars:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN

Optional:
- MAX_STEPS_PER_TASK (default: task max_steps)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from warehouse_env.environment import WarehouseRobotEnv
from warehouse_env.models import RobotAction


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "warehouse_isaac_openenv"
SUCCESS_SCORE_THRESHOLD = 0.80


def _require_env() -> None:
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def choose_action(client: OpenAI, obs: dict[str, Any], state: dict[str, Any]) -> RobotAction:
    system = (
        "You control a warehouse robot. Return strict JSON only with keys: "
        "action_type and target. action_type must be one of move,pick,place,scan,dock,wait."
    )
    user_payload = {
        "task_id": obs["task_id"],
        "task_description": obs["task_description"],
        "robot_location": obs["robot_location"],
        "holding_object": obs["holding_object"],
        "object_locations": obs["object_locations"],
        "completed_subgoals": obs["completed_subgoals"],
        "remaining_subgoals": obs["remaining_subgoals"],
        "valid_actions": obs["valid_actions"],
        "step_count": state["step_count"],
        "max_steps": state["max_steps"],
    }

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0,
            max_tokens=120,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        payload = json.loads(text)
        return RobotAction.model_validate(payload)
    except Exception:
        return RobotAction(action_type="wait", target=None)


def run_task(client: OpenAI, task_id: str, max_steps_override: int | None) -> dict[str, Any]:
    env = WarehouseRobotEnv(task_id=task_id)
    rewards: list[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME or "")

    try:
        obs = env.reset(task_id)
        state = env.state()
        limit = max_steps_override if max_steps_override is not None else state.max_steps

        for step in range(1, limit + 1):
            action = choose_action(client, obs.model_dump(), state.model_dump())
            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            error: str | None = None
            try:
                result = env.step(action)
            except Exception as exc:
                result = None
                error = str(exc)

            if result is None:
                log_step(step=step, action=action_str, reward=0.0, done=True, error=error)
                steps_taken = step
                break

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            obs = result.observation
            state = env.state()
            if done:
                break

        score = float(env.state().score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "steps": steps_taken,
        "score": round(score, 4),
        "success": success,
        "rewards": [round(r, 2) for r in rewards],
    }


def main() -> None:
    _require_env()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = WarehouseRobotEnv()
    tasks = env.list_tasks()
    max_steps_override = os.getenv("MAX_STEPS_PER_TASK")
    parsed_max_steps = int(max_steps_override) if max_steps_override else None

    task_results = [run_task(client, task, parsed_max_steps) for task in tasks]
    avg = sum(t["score"] for t in task_results) / len(task_results)

    report = {
        "env": BENCHMARK,
        "model": MODEL_NAME,
        "num_tasks": len(task_results),
        "average_score": round(avg, 4),
        "tasks": task_results,
    }

    out = ROOT / "outputs" / "evals" / "inference_scores.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
