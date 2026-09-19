#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
PATHS="$ROOT/study_paths"
RESULTS="$REPO_ROOT/results/llm/tournament"

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

until "$PYTHON" "$ROOT/generate_tournament_opt13b.py" \
  --profile study --output-dir "$PATHS" --batch-size 10 --num-threads 8 \
  --device cuda --local-files-only --validate-only; do
  "$PYTHON" "$ROOT/generate_tournament_opt13b.py" \
    --profile study --output-dir "$PATHS" --batch-size 10 --num-threads 8 \
    --device cuda --local-files-only --max-new-paths "$CHUNK_PATHS"
done

"$PYTHON" "$ROOT/analyze_tournament_paths.py" \
  --input-dir "$PATHS" --output-dir "$RESULTS/study"

if [[ -f "$RESULTS/efron_case/case.json" && -f "$RESULTS/efron_case/case_arrays.npz" ]]; then
  "$PYTHON" "$ROOT/run_efron_case.py" \
    --output-dir "$RESULTS/efron_case" --local-files-only --rebuild-derived
else
  "$PYTHON" "$ROOT/run_efron_case.py" \
    --output-dir "$RESULTS/efron_case" --device cuda --local-files-only
fi

"$PYTHON" "$ROOT/make_paper_outputs.py"

echo "Tournament outputs written to $RESULTS"
