#!/bin/bash
# untracked files in the Linux clone that could block a pull (read-only)
git -C "$HOME/work/ui_click_grpo" status --short | head -20
echo "--- untracked outside results/raw:"
git -C "$HOME/work/ui_click_grpo" ls-files --others --exclude-standard | grep -v "^serving/results/raw/" | head -10
