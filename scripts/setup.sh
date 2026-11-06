#!/usr/bin/env bash
# Reproducible environment setup (CPU). Creates .venv, installs pinned dependencies,
# fetches MiniWoB++ task HTML, and validates the installation.
#
#   bash scripts/setup.sh            # CPU torch
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup.sh   # GPU torch
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if command -v uv >/dev/null 2>&1; then
  uv venv .venv --python "$PY"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP="uv pip"
else
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP="pip"
fi
# torch first, from the CPU (or given) index only, so no CUDA wheels are pulled in
$PIP install --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}" "torch==2.15.0"
# everything else pinned to the versions this project was developed and tested with
grep -vE '^(torch|browsergym|-e |agentrs)' requirements.lock.txt > /tmp/agentrs_reqs.txt
$PIP install -r /tmp/agentrs_reqs.txt
# BrowserGym pins playwright==1.44 (no Python 3.13 wheels for its greenlet pin); it works with
# the newer playwright installed above, so install it without its dependency pins.
$PIP install --no-deps browsergym-core==0.14.3 browsergym-miniwob==0.14.3
$PIP install --no-deps -e .
# MiniWoB++ task HTML (MIT licensed) for the BrowserGym adapter
if [ ! -d third_party/miniwob-plusplus ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus third_party/miniwob-plusplus
fi
# A Chromium for Playwright: either `playwright install chromium`, or point AGENTRS_CHROMIUM
# at an existing chrome / chrome-headless-shell binary.
python -m agentrs setup-check
