#!/usr/bin/env bash
set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

PING_URL="${PING_URL%/}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  echo "Error: repo dir not found"
  exit 1
fi

echo "Step 1/3: Ping HF Space /reset"
HTTP_CODE=$(curl -s -o /tmp/validate_body.out -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || echo 000)
if [ "$HTTP_CODE" != "200" ]; then
  echo "FAILED: expected 200, got $HTTP_CODE"
  exit 1
fi
echo "PASSED"

echo "Step 2/3: Docker build"
if [ -f "$REPO_DIR/Dockerfile" ]; then
  docker build "$REPO_DIR" >/tmp/docker_build.log 2>&1 || {
    tail -40 /tmp/docker_build.log
    exit 1
  }
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  docker build "$REPO_DIR/server" >/tmp/docker_build.log 2>&1 || {
    tail -40 /tmp/docker_build.log
    exit 1
  }
else
  echo "FAILED: Dockerfile missing"
  exit 1
fi
echo "PASSED"

echo "Step 3/3: openenv validate"
openenv validate "$REPO_DIR" || exit 1
echo "PASSED"

echo "All checks passed."
