#!/bin/bash
#
# UN SOLO COMANDO para correr una busqueda LLaMEA completa en el cluster.
#
#   bash slurm/llamea-all.sh
#
# Que hace:
#   1. (login, con internet) descarga/verifica el modelo en la raiz persistente
#      si falta; los nodos de computo no lo pueden bajar.
#   2. Envia UN solo job a la particion gpu que sirve el modelo con vLLM y corre
#      la busqueda contra 127.0.0.1 (ver slurm/llamea-single.slrm).
#
# Opciones por entorno (se exportan al job con --export=ALL):
#   PROFILE (big-quant)    perfil GPU/modelo (ver slurm/llamea-gpu-profile.sh):
#                          big-quant | h100-fp16 | a100-fp16 | a10-fp16 | smoke
#   MODEL                  override del modelo (id de HuggingFace)
#   BUDGET (1000)          generaciones de la busqueda
#   EVAL_TEST=1            evaluar el mejor operador en el conjunto de test
#   EVAL_REPEATS (1)       repeticiones de esa evaluacion final
#   COMPARE=1              al terminar, comparacion pareada contra los clasicos
#                          en el mismo job (COMPARE_SPLIT=test, COMPARE_REPEATS=11)
#   SOLVER, TRAIN_INSTANCES, TEST_INSTANCES, MO_GENERATIONS, MO_POPULATION_SIZE,
#   TEMPERATURE, MAX_TOKENS, SEED, MUTATION=1, COMMA=1, REFINE_ONLY=1,
#   REPEAT_PENALTY, PATIENCE, STALL_TIMEOUT, PARENT_SELECTION, NICHING,
#   EXPERIMENT_NAME, CATEGORY, CONSTRAINT, MAX_MODEL_LEN, GPU_MEM_UTIL.
#
# Ejemplos:
#   bash slurm/llamea-all.sh
#   PROFILE=h100-fp16 BUDGET=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
#   BUDGET=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh   # todo el pipeline
#   PROFILE=smoke BUDGET=20 EVAL_TEST=1 bash slurm/llamea-all.sh   # prueba
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

source slurm/llamea-gpu-profile.sh
echo "Perfil ${PROFILE}: modelo=$MODEL  constraint=$CONSTRAINT"

# 1. Modelo en la raiz persistente, en el LOGIN (que si tiene internet).
#    Para esto solo hace falta el env de cliente (extra "llamea").
if command -v module >/dev/null 2>&1; then
    module load python/miniforge
    eval "$(conda shell.bash hook)"
    conda activate "$PWD/.conda-env"
fi

if python scripts/mohh_models.py verify "$MODEL" >/dev/null 2>&1; then
    echo "Modelo ya disponible en la raiz persistente: $MODEL"
else
    echo "Descargando $MODEL en la raiz persistente (login con internet) ..."
    python scripts/mohh_models.py download "$MODEL"
fi

# 2. Enviar el job unico (servidor + busqueda). --export=ALL propaga todas las
#    variables de la busqueda (BUDGET, EVAL_TEST, PROFILE, ...) al job.
echo "Enviando el job unico (vLLM + busqueda) a la particion gpu ..."
JOB="$(MODEL="$MODEL" PROFILE="$PROFILE" CONSTRAINT="$CONSTRAINT" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    sbatch --parsable --export=ALL --constraint="$CONSTRAINT" \
    slurm/llamea-single.slrm)"
echo "  job: $JOB"

echo
echo "Monitoreo:"
echo "  squeue --user=\$USER"
echo "  tail -f llamea-single-${JOB}.out"
echo "  cat llamea-single-${JOB}.err"
echo "Resultados en: results/${CATEGORY:-searches}/${EXPERIMENT_NAME:-LLaMEA_single}/"
