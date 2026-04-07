from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from warehouse_env.environment import WarehouseRobotEnv
from warehouse_env.models import EnvState, RobotAction, RobotObservation, StepResult


def _create_fallback_app() -> FastAPI:
    app = FastAPI(title="warehouse_isaac_openenv", version="0.1.0")
    env = WarehouseRobotEnv()
    env.reset()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/reset", response_model=RobotObservation)
    def reset(payload: dict[str, Any] | None = None) -> RobotObservation:
        task_id = None
        if payload:
            task_id = payload.get("task_id")
        return env.reset(task_id=task_id)

    @app.post("/step", response_model=StepResult)
    def step(action: RobotAction) -> StepResult:
        return env.step(action)

    @app.get("/state", response_model=EnvState)
    def state() -> EnvState:
        return env.state()

    @app.get("/tasks")
    def tasks() -> dict[str, list[str]]:
        return {"tasks": env.list_tasks()}

    return app


# Prefer native OpenEnv server wrapper when available.
try:
    from openenv.core.env_server.http_server import create_app  # type: ignore

    app = create_app(
        WarehouseRobotEnv,
        RobotAction,
        RobotObservation,
        env_name="warehouse_isaac_openenv",
    )
except Exception:
    app = _create_fallback_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
