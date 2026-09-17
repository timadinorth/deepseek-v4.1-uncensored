#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH=${1:?edited checkpoint path required}
CONTEXT_LENGTH=${2:-262144}
TOTAL_TOKENS=${3:-1048576}
SPECULATIVE_MODE=${4:-dspark}
CHUNKED_PREFILL_SIZE=${5:-2048}
speculative_args=()
case "$SPECULATIVE_MODE" in
  plain) ;;
  dspark) speculative_args=(--speculative-algorithm DSPARK --speculative-dspark-block-size 5) ;;
  *) echo 'Expected speculative mode plain or dspark' >&2; exit 2 ;;
esac
# Container-native GPU.ai replacement. Anonymous THP is available; shmem THP is not.
# Do not change host-wide sysfs settings on the shared provider machine.
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=private
export SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN=1
export SGLANG_DEFAULT_THINKING=true
export XDG_CACHE_HOME=/workspace/cache
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name deepseek-v41-eval \
  --tp-size 4 --ep-size 4 --trust-remote-code \
  --context-length "$CONTEXT_LENGTH" --max-total-tokens "$TOTAL_TOKENS" --max-running-requests 4 \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --mem-fraction-static 0.8 --cuda-graph-max-bs-decode 4 \
  "${speculative_args[@]}" \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 \
  --host 127.0.0.1 --port 30000 --enable-metrics
