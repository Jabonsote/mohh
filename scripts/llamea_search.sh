#!/usr/bin/env bash
# Lanza una busqueda LLaMEA con valores por defecto sensatos.
#
# Todo queda bajo results/<categoria>/<nombre>/ (ver results/README.md):
#   results/searches/  busquedas completas        (default)
#   results/tests/     pruebas de humo cortas
#   results/prompts/   ablaciones de prompts
#   results/examples/  ejemplos y operadores de referencia
#
# Uso local (servidor en el puerto 8001):
#   ./scripts/llamea_search.sh
#   BUDGET=20 CATEGORY=tests NAME=smoke ./scripts/llamea_search.sh
#
# Uso en el cluster (vLLM en otro nodo):
#   BASE_URL=http://gpu-nodo:8000/v1 NO_DOWNLOAD=1 WAIT_SERVER=1 \
#     CATEGORY=searches NAME=qwen7b_moead_b100 ./scripts/llamea_search.sh
#
# Uso con modelos ya descargados en Ollama (sin bajar nada):
#   MODEL=qwen2.5-coder:7b-instruct BASE_URL=http://127.0.0.1:11434/v1 \
#     NO_DOWNLOAD=1 BUDGET=30 NAME=qwen25coder7b_moead_b30 ./scripts/llamea_search.sh
#
# Variables:
#   MODEL (Qwen/Qwen2.5-0.5B-Instruct)  modelo servido
#   BASE_URL (http://localhost:8001/v1) endpoint OpenAI-compatible
#   BUDGET (1000)                       generaciones
#   SOLVER (MOEAD)                       MOEAD | NSGAII | SMSEMOA
#   TRAIN_INSTANCES (8)                  instancias de train (o "all")
#   MO_GENERATIONS (30)                  generaciones del MOEA interno
#   MO_POPULATION_SIZE (30)              poblacion del MOEA interno
#   TEMPERATURE (0.2)                    temperatura del LLM
#   TOP_P / TOP_K                        muestreo nucleus / top-k (vacío = servidor)
#   SAMPLING_REPETITION_PENALTY          repetition_penalty de vLLM (vacío = servidor)
#   THINKING (auto)                      auto | on | off (enable_thinking del template;
#                                        on necesita más MAX_TOKENS)
#   OPERATOR_MEMORY (0)                  mecanismos recientes recordados en el prompt
#                                        (0 = off; 12 recomendado con modelos grandes)
#   MAX_TOKENS (vacío)                   máximo de tokens de salida por petición
#   REPEAT_PENALTY (0.5)                 presión de diversidad: castigo al
#                                        ranking por repetir código (1.0 = off)
#   PATIENCE (0)                         parada temprana tras N evaluaciones sin
#                                        mejorar el mejor HV (0 = off)
#   PARENT_SELECTION (tournament)        random | roulette | tournament
#   NICHING (none)                       none | sharing | clearing | novelty |
#                                        map_elites (presión de diversidad)
#   EVAL_REPEATS (11)                    repeticiones de la evaluación final
#                                        (11 = comparación pareada vs clásicos)
#   STALL_TIMEOUT (0)                    segundos sin generación nueva antes de
#                                        abortar (watchdog; 0 = off)
#   CATEGORY (searches)                  subcarpeta de results/
#   NAME                                 nombre del experimento (default: auto)
#   PYTHON                               interprete (default: python)
#   NO_DOWNLOAD / WAIT_SERVER / EVAL_TEST / REFINE_ONLY / COMMA / MUTATION
#   NO_SAVE_FRONTS                       no guardar los frentes por candidato
#   EXTRA                                argumentos extra para mohh.llamea.run
#                                        (p. ej. EXTRA=--stationary-frame) 
#
# Por defecto el HV se normaliza con el marco de la evaluacion GE (frente
# consolidado del panel de referencia y de todos los candidatos evaluados), asi
# que los numeros son comparables con la tabla de la busqueda gramatical.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
BASE_URL="${BASE_URL:-http://localhost:8001/v1}"
BUDGET="${BUDGET:-1000}"
SOLVER="${SOLVER:-MOEAD}"
TRAIN_INSTANCES="${TRAIN_INSTANCES:-8}"
MO_GENERATIONS="${MO_GENERATIONS:-30}"
MO_POPULATION_SIZE="${MO_POPULATION_SIZE:-30}"
TEMPERATURE="${TEMPERATURE:-0.2}"
CATEGORY="${CATEGORY:-searches}"
PYTHON="${PYTHON:-python}"

short_model="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')"
NAME="${NAME:-${short_model}_${SOLVER,,}_b${BUDGET}}"

args=(
  --model "$MODEL"
  --base-url "$BASE_URL"
  --budget "$BUDGET"
  --solver "$SOLVER"
  --train-instances "$TRAIN_INSTANCES"
  --mo-generations "$MO_GENERATIONS"
  --mo-population-size "$MO_POPULATION_SIZE"
  --temperature "$TEMPERATURE"
  --repeat-penalty "${REPEAT_PENALTY:-0.5}"
  --patience "${PATIENCE:-0}"
  --parent-selection "${PARENT_SELECTION:-tournament}"
  --niching "${NICHING:-none}"
  --eval-repeats "${EVAL_REPEATS:-11}"
  --stall-timeout "${STALL_TIMEOUT:-0}"
  --results-dir "results/$CATEGORY"
  --experiment-name "$NAME"
)

[[ "${NO_DOWNLOAD:-0}" == "1" ]] && args+=(--no-download)
[[ "${WAIT_SERVER:-0}" == "1" ]] && args+=(--wait-server)
[[ "${EVAL_TEST:-0}" == "1" ]] && args+=(--eval-test)
[[ "${REFINE_ONLY:-0}" == "1" ]] && args+=(--refine-only)
[[ "${COMMA:-0}" == "1" ]] && args+=(--comma)
[[ "${MUTATION:-0}" == "1" ]] && args+=(--mutation)
[[ "${NO_SAVE_FRONTS:-0}" == "1" ]] && args+=(--no-save-fronts)
[[ "${QUIET:-0}" == "1" ]] && args+=(--quiet)
[[ -n "${BOUNDS_FILE:-}" ]] && args+=(--bounds-file "$BOUNDS_FILE")
[[ -n "${MAX_TOKENS:-}" ]] && args+=(--max-tokens "$MAX_TOKENS")
[[ -n "${TOP_P:-}" ]] && args+=(--top-p "$TOP_P")
[[ -n "${TOP_K:-}" ]] && args+=(--top-k "$TOP_K")
[[ -n "${SAMPLING_REPETITION_PENALTY:-}" ]] && \
  args+=(--sampling-repetition-penalty "$SAMPLING_REPETITION_PENALTY")
[[ -n "${THINKING:-}" ]] && args+=(--thinking "$THINKING")
[[ -n "${OPERATOR_MEMORY:-}" ]] && args+=(--operator-memory "$OPERATOR_MEMORY")

# shellcheck disable=SC2206
[[ -n "${EXTRA:-}" ]] && args+=(${EXTRA})

echo "==> results/$CATEGORY/$NAME"
echo "==> $PYTHON -m mohh.llamea.run ${args[*]}"
exec "$PYTHON" -m mohh.llamea.run "${args[@]}"
