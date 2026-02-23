#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — launch Crystal Annotator (macOS)
#
#   bash run.sh
#
# First run installs Flask into a lightweight venv.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "==> Creating virtual environment…"
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# install Flask if not present
python -c "import flask" 2>/dev/null || {
  echo "==> Installing Flask…"
  pip install --quiet flask
}

# install OpenCV + numpy for auto droplet detection (optional but recommended)
python -c "import cv2" 2>/dev/null || {
  echo "==> Installing opencv-python and numpy…"
  pip install --quiet opencv-python numpy
}

echo ""
echo "  ┌──────────────────────────────────────────┐"
echo "  │  Crystal Annotator                       │"
echo "  │  Open:  http://localhost:5050             │"
echo "  │  Stop:  Ctrl-C                           │"
echo "  └──────────────────────────────────────────┘"
echo ""

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${PYTHONPATH:-}"
python app.py
