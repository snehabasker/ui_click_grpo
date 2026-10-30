#!/usr/bin/env bash
# the symlink tests of make_trtllm_view.py, in Linux (Windows accounts usually cannot create symlinks); uses the clone and the vllm venv's pytest
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"; sed 's/\r$//' "$W/sync_clone.sh" > /tmp/sync_clone.sh; bash /tmp/sync_clone.sh | tail -1
cd ~/work/ui_click_grpo && "$HOME/venvs/vllm/bin/python" -m pytest serving/tests/test_trtllm_view.py -q -p no:cacheprovider 2>&1 | tail -3
