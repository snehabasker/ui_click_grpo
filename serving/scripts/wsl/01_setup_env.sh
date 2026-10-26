#!/bin/bash
# Phase 1 environment inside WSL2 Ubuntu. Needs no sudo. Idempotent.
#   - uv (user-level installer) -> Python 3.12 (the system Python is 3.14, which the engines do not support yet)
#   - one venv per engine plus a separate venv for the load client, so engine installs cannot change the client
#   - a clone of this repo on the Linux filesystem (code + manifests + trace.jsonl); the git-ignored images and the merged
#     checkpoint are copied in by 02_copy_artifacts.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
uv python install 3.12

mkdir -p "$HOME/venvs" "$HOME/work"
for env in vllm sglang bench; do
  [ -d "$HOME/venvs/$env" ] || uv venv "$HOME/venvs/$env" --python 3.12
done

if [ ! -d "$HOME/work/ui_click_grpo" ]; then
  git clone https://github.com/snehabasker/ui_click_grpo.git "$HOME/work/ui_click_grpo"
fi
git -C "$HOME/work/ui_click_grpo" pull --ff-only || true
git -C "$HOME/work/ui_click_grpo" rev-parse HEAD

# client-side tooling (aiohttp, numpy, pillow, pytest)
uv pip install --python "$HOME/venvs/bench/bin/python" -r "$HOME/work/ui_click_grpo/serving/requirements.txt"
"$HOME/venvs/bench/bin/python" --version
echo "SETUP_OK"
