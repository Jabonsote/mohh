#!/usr/bin/env bash
# Ablaciones de prompt (mismo modelo, semilla, instancias y presupuesto).
#
# Cada variante queda en results/prompts/<nombre>/ y compara una decision del
# prompt. La tabla de resultados se arma con:
#   python scripts/analyze_llamea_runs.py 'results/prompts/**/exp-*'
#
# Uso:
#   ./scripts/llamea_prompt_ablations.sh
#   BUDGET=50 MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/llamea_prompt_ablations.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CATEGORY=prompts
export BUDGET="${BUDGET:-30}"
export PYTHON="${PYTHON:-python}"
export NO_DOWNLOAD="${NO_DOWNLOAD:-1}"
export TRAIN_INSTANCES="${TRAIN_INSTANCES:-8}"
export SOLVER="${SOLVER:-MOEAD}"

tag="$(basename "${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"

echo "==> example (referencia)"
NAME="${tag}_example_b${BUDGET}" ./scripts/llamea_search.sh

echo "==> noexample"
NAME="${tag}_noexample_b${BUDGET}" EXTRA="--no-example" ./scripts/llamea_search.sh

echo "==> skeleton"
NAME="${tag}_skeleton_b${BUDGET}" EXTRA="--skeleton-example" ./scripts/llamea_search.sh

echo "==> refine-only"
NAME="${tag}_refineonly_b${BUDGET}" EXTRA="--refine-only" ./scripts/llamea_search.sh

echo "==> temperature 0.6"
NAME="${tag}_temp06_b${BUDGET}" TEMPERATURE=0.6 ./scripts/llamea_search.sh
