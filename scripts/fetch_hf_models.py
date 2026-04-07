from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


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


def main() -> None:
    load_dotenv(ENV_PATH)

    base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError("HF_TOKEN is missing. Set it in .env or environment first.")

    client = OpenAI(base_url=base_url, api_key=token)
    models = client.models.list()

    ids = sorted({m.id for m in models.data if getattr(m, "id", None)})
    output = {
        "api_base_url": base_url,
        "count": len(ids),
        "models": ids,
    }

    out = ROOT / "outputs" / "evals" / "hf_models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Fetched {len(ids)} models. Saved to {out}")
    for mid in ids[:30]:
        print(mid)


if __name__ == "__main__":
    main()
