#!/bin/bash
#
# Encadena servidor vLLM + busqueda LLaMEA en ixachi.
#
# Envia el job del servidor (particion gpu) y, dependiente de su arranque,
# el job de busqueda (particion ixachi) que espera al servidor con
# --wait-server. El nodo del servidor se autodetecta por JobID.
#
# Uso:
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash slurm/llamea-full-chain.sh
#   MODEL=Qwen/Qwen2.5-Coder-7B-Instruct BUDGET=200 EVAL_TEST=1 \
#     EXPERIMENT_NAME=qwen7bcoder_moead_b200 bash slurm/llamea-full-chain.sh
#
# Opciones por entorno: MODEL, BUDGET, TRAIN_INSTANCES, TEST_INSTANCES,
# EVAL_TEST=1, SOLVER, MO_GENERATIONS, MO_POPULATION_SIZE, N_PARENTS,
# N_OFFSPRING, TEMPERATURE, EXPERIMENT_NAME, MAX_WORKERS, COMMA=1,
# REFINE_ONLY=1, CATEGORY, NO_NORMALIZE=1, SAVE_BOUNDS, BOUNDS_FILE.
# Se exportan al job de busqueda tal cual (--export=ALL).
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# 0. Pesos en la raiz persistente: se bajan AQUI (login, con internet) una sola
#    vez; los nodos de computo suelen no tener salida a internet. Si el modelo
#    ya esta verificado, esta linea no hace nada.
if python scripts/mohh_models.py verify "$MODEL" >/dev/null 2>&1; then
    echo "Modelo ya disponible en la raiz persistente: $MODEL"
else
    echo "Descargando $MODEL en la raiz persistente (login con internet) ..."
    python scripts/mohh_models.py download "$MODEL"
fi

echo "Enviando servidor vLLM para $MODEL ..."
SERVER_JOB="$(MODEL="$MODEL" sbatch --parsable --export=ALL slurm/llamea-vllm-server.slrm)"
echo "  job vLLM: $SERVER_JOB"

echo "Enviando busqueda LLaMEA (dependiente de $SERVER_JOB) ..."
GEN_JOB="$(MODEL="$MODEL" LLAMEA_SERVER_JOBID="$SERVER_JOB" \
    sbatch --parsable --export=ALL --dependency=after:"$SERVER_JOB" slurm/llamea-generate.slrm)"
echo "  job LLaMEA: $GEN_JOB"

echo
echo "Monitoreo:"
echo "  squeue --user=\$USER"
echo "  tail -f llamea-vllm-${SERVER_JOB}.out"
echo "  tail -f llamea-generate-${GEN_JOB}.out"
echo
echo "Para cancelar todo: scancel $SERVER_JOB $GEN_JOB"
