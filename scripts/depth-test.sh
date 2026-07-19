#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
UV_BIN=${UV_BIN:-$(command -v uv 2>/dev/null || true)}

if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [ -z "$UV_BIN" ]; then
  echo "uv was not found. Install uv or set UV_BIN." >&2
  exit 1
fi

if [ -z "${UV_CACHE_DIR:-}" ]; then
  UV_CACHE_DIR="$PROJECT_DIR/depth_pipeline/.uv-cache"
  export UV_CACHE_DIR
fi
PYTHONPATH="$PROJECT_DIR/depth_pipeline/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

exec "$UV_BIN" run --project "$PROJECT_DIR/depth_pipeline" --extra dev --frozen pytest -q "$PROJECT_DIR/depth_pipeline/tests"
