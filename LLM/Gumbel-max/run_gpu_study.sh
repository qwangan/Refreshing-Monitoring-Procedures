#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PATHS="$ROOT/study_paths"
RESULTS="$REPO_ROOT/results/llm"

: "${PYTHON:=python3}"
: "${HF_HOME:=$ROOT/.hf-cache-opt13b}"
: "${CHUNK_PATHS:=100}"
export HF_HOME TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT"
export MPLCONFIGDIR="$ROOT/.mplconfig"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
  echo "A CUDA GPU is required for the OPT-1.3B generation stage." >&2
  exit 2
fi

"$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/opt-1.3b",
    revision="3f5c25d0bc631cb57ac65913f76e22c2dfb61d62",
)
PY

until "$PYTHON" "$ROOT/generate_fresh_opt13b.py" \
  --profile study --output-dir "$PATHS" --batch-size 10 --num-threads 8 \
  --device cuda --local-files-only --validate-only; do
  "$PYTHON" "$ROOT/generate_fresh_opt13b.py" \
    --profile study --output-dir "$PATHS" --batch-size 10 --num-threads 8 \
    --device cuda --local-files-only --max-new-paths "$CHUNK_PATHS"
done

"$PYTHON" "$ROOT/analyze_opt13b_paths.py" \
  --input-dir "$PATHS" --output-dir "$RESULTS/primary" \
  --general-fdr-target 0.10

"$PYTHON" "$ROOT/analyze_eprocess_comparison.py" \
  --input-dir "$PATHS" --output-dir "$RESULTS/processes" \
  --general-fdr-target 0.10

"$PYTHON" "$ROOT/analyze_fixed_lambda_benchmarks.py" \
  --opt-input-dir "$PATHS" --output-dir "$RESULTS/fixed"

"$PYTHON" "$ROOT/make_paper_outputs.py"

echo "Section 5.2 outputs written to $RESULTS/paper_outputs"
