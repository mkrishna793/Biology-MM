#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend/biology-mm-studio"

export PYTHONPATH="$REPO_ROOT/python:$BACKEND_DIR${PYTHONPATH:+:$PYTHONPATH}"

DEV_MODE=0
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"

for arg in "$@"; do
  case "$arg" in
    --dev)
      DEV_MODE=1
      ;;
    -h|--help)
      echo "Usage: $0 [--dev]"
      echo "  (default)  Starts FastAPI/Uvicorn backend on http://$HOST:$PORT"
      echo "  --dev      Starts Uvicorn backend + Vite dev server"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

# Resolve or bootstrap Python virtualenv & Uvicorn
UVICORN_BIN="$(command -v uvicorn || true)"
if [[ -z "$UVICORN_BIN" ]]; then
  if [[ -f "$REPO_ROOT/.venv/bin/uvicorn" ]]; then
    UVICORN_BIN="$REPO_ROOT/.venv/bin/uvicorn"
  elif [[ -f "$SCRIPT_DIR/.venv/bin/uvicorn" ]]; then
    UVICORN_BIN="$SCRIPT_DIR/.venv/bin/uvicorn"
  else
    echo "[Biology-MM Studio] Initializing virtualenv and installing dependencies..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip
    "$SCRIPT_DIR/.venv/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
    "$SCRIPT_DIR/.venv/bin/pip" install -e "$REPO_ROOT/python"
    UVICORN_BIN="$SCRIPT_DIR/.venv/bin/uvicorn"
  fi
fi

PIDS=()
cleanup() {
  local exit_code=$?
  trap - INT TERM EXIT
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit "$exit_code"
}
trap cleanup INT TERM EXIT

if [[ "$DEV_MODE" -eq 1 ]]; then
  echo "[Biology-MM Studio] Starting development mode (Uvicorn :$PORT + Vite)..."
  "$UVICORN_BIN" trium_studio.app:app --host "$HOST" --port "$PORT" &
  PIDS+=("$!")

  if command -v pnpm &>/dev/null; then
    (
      cd "$FRONTEND_DIR"
      exec pnpm run dev
    ) &
    PIDS+=("$!")
  fi

  wait
else
  echo "[Biology-MM Studio] Starting server at http://$HOST:$PORT ..."
  "$UVICORN_BIN" trium_studio.app:app --host "$HOST" --port "$PORT" &
  PIDS+=("$!")
  wait "${PIDS[0]}"
fi
