#!/usr/bin/env bash
# Phase 5 smoke: start trtllm-serve (PyTorch backend) on the merged checkpoint, wait for readiness, send one text request and one trace session
# (images) with tag explore-trtllm. Prints the log tail if the server dies or never becomes ready. Not a result.
W="/mnt/c/Users/Sneha Basker/ui-click-rl/serving/scripts/wsl"
for s in sync_clone trtllm_env; do sed 's/\r$//' "$W/$s.sh" > /tmp/$s.sh; done
bash /tmp/sync_clone.sh | tail -1
sed 's/\r$//' "$W/serve_trtllm.sh" > /tmp/serve_trtllm.sh
mkdir -p ~/work/logs; LOG=~/work/logs/trtllm_serve.log; PORT=${PORT:-8003}
# TensorRT-LLM's transformers (4.57.3) cannot read the transformers-5 config of the merged checkpoint: serve a view of it (same weights)
python3 ~/work/ui_click_grpo/serving/scripts/make_trtllm_view.py --src ~/work/ui_click_grpo/serving/models/merged-clean --dst ~/work/merged-clean-trt || exit 1
MODEL_DIR=$HOME/work/merged-clean-trt PORT=$PORT setsid bash /tmp/serve_trtllm.sh > "$LOG" 2>&1 &
SP=$!
t0=$(date +%s); ready=0
while [ $(( $(date +%s) - t0 )) -lt "${READY_TIMEOUT:-900}" ]; do
  if ! kill -0 $SP 2>/dev/null; then echo "SERVER_EXITED after $(( $(date +%s) - t0 ))s"; break; fi
  if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; echo "READY after $(( $(date +%s) - t0 ))s"; break; fi
  sleep 3
done
if [ $ready = 1 ]; then
  curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"merged-clean","messages":[{"role":"user","content":"Say hi"}],"max_tokens":8,"temperature":0}' | cut -c1-400; echo
  cd ~/work/ui_click_grpo && "$HOME/venvs/bench/bin/python" -u serving/scripts/loadtest.py --base-url http://127.0.0.1:$PORT --model "$(curl -s http://127.0.0.1:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')" \
    --trace serving/traces/default --concurrency 1 --repeats 1 --max-sessions 2 --engine trtllm --tag explore-trtllm --model-dir serving/models/merged-clean 2>&1 | tail -3 | cut -c1-300
else
  echo "---- server log tail:"; grep -v -i "warn" "$LOG" | tail -25 | cut -c1-260
fi
pkill -P $SP 2>/dev/null; kill $SP 2>/dev/null; pkill -f "[t]rtllm-serve" 2>/dev/null; sleep 3; echo done
