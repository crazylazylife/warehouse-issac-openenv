# Warehouse Isaac OpenEnv

A real-world warehouse fulfillment simulation environment designed for agent training with the OpenEnv API.

This project implements:
- Typed Pydantic `Action`, `Observation`, `Reward`, and `State` models
- Full `step()` / `reset()` / `state()` environment API
- Three deterministic graded tasks (`easy` -> `medium` -> `hard`)
- Dense trajectory reward shaping (partial progress + penalties)
- Submission `inference.py` using OpenAI Client with required env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`)
- Docker + Hugging Face Spaces deployment scaffolding

## Real-World Task Domain

The environment simulates robotic warehouse fulfillment operations:
- Picking items from shelves
- Performing quality-control scans
- Routing through checkpoints
- Placing items at packaging/staging zones
- Docking and recharging

These are real operational tasks in logistics and e-commerce warehouses.

## Task Set and Graders

1. `easy_pick_and_stage` (easy)
- Objective: move to `shelf_a`, pick `tote_red`, place at `staging_zone`
- Grader: completion of ordered subgoals with penalty for invalid actions

2. `medium_qc_and_pack` (medium)
- Objective: pick `crate_blue`, scan at `qc_station`, place at `packing_zone`
- Grader: requires scan before full score; placing without scan is capped

3. `hard_fragile_pack_and_dock` (hard)
- Objective: pick `fragile_box`, pass `checkpoint_1`, place at `fragile_pack_zone`, dock and recharge
- Grader: full score requires checkpoint + place + successful dock

Each grader returns deterministic score in `[0.0, 1.0]`.

## Action Space

`RobotAction`:
- `action_type`: one of `move`, `pick`, `place`, `scan`, `dock`, `wait`
- `target`: location or object ID depending on action
- `note`: optional free-form text

## Observation Space

`RobotObservation` includes:
- `task_id`, `task_description`
- `robot_location`, `holding_object`, `battery_level`
- `object_locations`
- `completed_subgoals`, `remaining_subgoals`
- `valid_actions`
- `step_count`, `max_steps`
- `latest_event`

## Reward Function

Dense reward is computed every step and includes:
- Progress signal: completion ratio + incremental subgoal delta
- Safety signal: battery depletion / safety events
- Efficiency signal: remaining step budget
- Penalties: invalid actions, repeated loops, timeout, unsafe termination

Returned as typed `RewardSignal`:
- `total`, `progress`, `safety`, `efficiency`, `penalties`, `task_score`

## Project Structure

```text
warehouse_isaac_openenv/
├── app.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── warehouse_env/
│   ├── __init__.py
│   ├── environment.py
│   ├── isaac_bridge.py
│   ├── models.py
│   └── tasks.py
├── scripts/
│   └── run_baseline.py
├── server/
│   ├── app.py
│   └── Dockerfile
└── tests/
    └── test_env.py
```

## Setup

```bash
cd warehouse_isaac_openenv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Fallback API endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`

## Baseline Inference (OpenAI)

Submission script (required name/location): `inference.py` in repo root.

Required env vars:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="<your_hf_token>"
```

Run:

```bash
python inference.py
```

This prints structured logs in the required format:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

and writes:
- `outputs/evals/inference_scores.json`

Alternative local baseline helper:

```bash
export OPENAI_API_KEY=<your_key>
python scripts/run_baseline.py --mode openai --model gpt-4.1-mini --seed 7
```

Output:
- `outputs/evals/baseline_scores.json`

Deterministic fallback baseline (no API call):

```bash
python scripts/run_baseline.py --mode scripted
```

## Validation

If `openenv` CLI is installed:

```bash
openenv validate .
```

Pre-submission validator script:

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://<your-space>.hf.space .
```

## Docker

Build and run:

```bash
docker build -t warehouse-isaac-openenv:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 warehouse-isaac-openenv:latest
```

Validator compatibility:
- A root `Dockerfile` is included for automated `docker build <repo_dir>` checks.

## Hugging Face Spaces Deployment

1. Create a new Space with **Docker** SDK.
2. Push this repository contents to the Space.
3. Space will build from `server/Dockerfile`.
4. Set `OPENAI_API_KEY` in Space Secrets if running baseline there.

## Isaac Sim Integration Notes

This scaffold includes `IsaacSimBridge` as the integration seam.
Current implementation is deterministic for reproducibility and hackathon speed.
To use full Isaac Sim physics, replace `IsaacSimBridge.apply(...)` with calls into your Isaac scene and robot controller while keeping model/API outputs unchanged.
