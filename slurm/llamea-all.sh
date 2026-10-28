#!/bin/bash
#
# UN SOLO COMANDO para correr una busqueda LLaMEA completa en el cluster.
#
#   bash slurm/llamea-all.sh
#
# No hay que pedir un job para el modelo y otro para la busqueda: este comando
# hace todo lo que necesita el login y deja UN solo job en la cola GPU. El job
# sirve el modelo con vLLM en 127.0.0.1 y corre la busqueda contra ese servidor
# (ver slurm/llamea-single.slrm). Si el modelo no esta descargado, lo baja aqui
# (el login tiene internet); los nodos de computo no pueden bajarlo.
#
# Que hace, en orden:
#   1. Prepara el ambiente .conda-env/ si es la primera vez (modulo + conda).
#   2. Verifica/descarga el modelo del perfil en la raiz persistente.
#   3. Envia UN job GPU (vLLM + LLaMEA + evaluacion test + comparacion).
#   4. Imprime el JobID y como monitorear.
#
# Opciones por entorno (se exportan al job con --export=ALL):
#   PROFILE (big-quant)    perfil GPU/modelo (ver slurm/llamea-gpu-profile.sh):
#                          big-quant | coder30b-awq | coder30b-fp8 |
#                          coder30b-awq-a10 | glm47-flash-awq |
#                          coder-next-awq | coder8b |
#                          h100-fp16 | a100-fp16 | a10-quant | smoke
#                          `PROFILE=list bash slurm/llamea-all.sh` imprime la tabla.
#   MODEL                  override del modelo (id de HuggingFace)
#   GENERATIONS (1000)     iteraciones reales de LLaMEA (presupuesto =
#                          GENERATIONS*NOFFSPRING + NPARENTS)
#   NPARENTS (5)           tamano de la poblacion de padres
#   NOFFSPRING (2)         hijos/evaluaciones por generacion
#   BUDGET                 override explicito del presupuesto en evaluaciones
#                          (si se define, ignora GENERATIONS)
#   COMBINE (1)            1 = incluir el combine conductual propio; 0 = solo
#                          refine/redesign (comportamiento LLaMEA original)
#   BEHAVIORAL_PARENTS (1) 0 = elegir el padre focal solo por fitness
#   NOVELTY_WEIGHT (0.5)   peso de novedad en el torneo conductual
#   TARGET_PENALTY (1.0)   castigo de ranking si el hijo no alcanza la familia
#                          objetivo (1.0 = solo feedback; 0.0 = no puede ganar)
#   REQUIRE_TARGET_FAMILY (0) 1 = rechazar el hijo antes de gastar el MOEA
#   EVAL_TEST=1            evaluar el mejor operador en el conjunto de test
#   EVAL_REPEATS (11)      repeticiones de esa evaluacion final (11 = comparacion
#                          pareada contra los clasicos; 1 = solo una muestra)
#   COMPARE=1              al terminar, comparacion pareada contra los clasicos
#                          en el mismo job (COMPARE_SPLIT=test, COMPARE_REPEATS=11,
#                          COMPARE_FAIL_POLICY=zero; incluye clasicos de 1 y 2 hijos)
#   COMPARE_INSTANCES      instancias de la comparacion; default TEST_INSTANCES
#                          (usar 'all' para las 20 y que sea pareada con EVAL_TEST)
#   COMPARE_GE_PROTOCOL=1  comparacion con el protocolo GE completo (3 solvers x
#                          2 mutaciones x 300/105); es caro, para tabla final
#   COMPARE_SOLVERS        solvers de la comparacion (default SOLVER)
#   COMPARE_MUTATION=1 / COMPARE_MUTATION_BOTH=1
#   COMPARE_MO_GENERATIONS / COMPARE_MO_POPULATION_SIZE
#   SOLVER, TRAIN_INSTANCES, TEST_INSTANCES, MO_GENERATIONS, MO_POPULATION_SIZE,
#   TEMPERATURE, MAX_TOKENS, THINKING (auto|on|off), SEED, MUTATION=1, COMMA=1,
#   REFINE_ONLY=1,
#   REPEAT_PENALTY, DIVERSITY_FEEDBACK=0, PATIENCE, STALL_TIMEOUT,
#   PARENT_SELECTION, NICHING, EXPERIMENT_NAME (unico por defecto:
#   LLaMEA_single_<fecha>_<pid>), CATEGORY, CONSTRAINT, MAX_MODEL_LEN,
#   GPU_MEM_UTIL, CPUS (16), FITNESS_WORKERS (CPUS-4; 1 = secuencial),
#   MEM (64G), TIME (limite SLURM),
#   WAIT=1 (esperar y mostrar el final del log antes de salir).
#
# Ejemplos:
#   bash slurm/llamea-all.sh
#   PROFILE=h100-fp16 GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
#   GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh  # pipeline
#   PROFILE=smoke BUDGET=20 EVAL_TEST=1 bash slurm/llamea-all.sh      # prueba
#   WAIT=1 BUDGET=50 PROFILE=smoke bash slurm/llamea-all.sh           # esperar
#   COMBINE=0 GENERATIONS=1000 bash slurm/llamea-all.sh  # ablacion 1+1 original
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_DIR="$PWD"

# Un job no puede enviar otro job: este comando es para el nodo de login.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: este comando envia un job; ejecutalo en el nodo de login," >&2
    echo "       no dentro de un job de SLURM (SLURM_JOB_ID=$SLURM_JOB_ID)." >&2
    exit 1
fi

source slurm/llamea-gpu-profile.sh
if [ "${PROFILE:-}" = "list" ]; then
    # El perfil `list` ya imprimio la tabla y salio con `exit 0`.
    exit 0
fi
echo "Perfil ${PROFILE}: modelo=$MODEL  constraint=$CONSTRAINT"

# Un nombre por corrida. Antes el default era siempre LLaMEA_single y dos
# corridas escribian en la misma carpeta: run.json se sobrescribia y
# llm_requests.jsonl se appendaba (21 peticiones viejas + 1 nueva = 22).
EXPERIMENT_NAME="${EXPERIMENT_NAME:-LLaMEA_single_$(date +%Y%m%d_%H%M%S)_$$}"
export EXPERIMENT_NAME
echo "Experimento: ${EXPERIMENT_NAME}"

# 16 CPUs es el default recomendado. El MOEA interno ya no corre secuencial:
# FITNESS_WORKERS evalua en paralelo las instancias de cada candidato (como el
# pool de GE). Se dejan 4 nucleos para vLLM/tokenizacion.
CPUS="${CPUS:-16}"
MEM="${MEM:-64G}"
if [ "${FITNESS_WORKERS:-}" = "" ]; then
    if [ "$CPUS" -gt 4 ] 2>/dev/null; then
        FITNESS_WORKERS=$((CPUS - 4))
    else
        FITNESS_WORKERS=1
    fi
fi
export FITNESS_WORKERS

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN: no se prepara el ambiente, no se descarga el modelo y no se envia nada."
    echo "  EXPERIMENT_NAME=$EXPERIMENT_NAME FITNESS_WORKERS=$FITNESS_WORKERS"
    echo "  sbatch --parsable --export=ALL --constraint=\"$CONSTRAINT\" --cpus-per-task=$CPUS --mem=$MEM slurm/llamea-single.slrm"
    exit 0
fi

# 1. Ambiente. En el login, `module` existe. Si ya hay .conda-env con mohh y
#    vLLM, no se reinstala nada; si falta, se crea en este mismo comando.
mkdir -p logs results
if command -v module >/dev/null 2>&1; then
    module load python/miniforge || {
        echo "ERROR: no se pudo cargar python/miniforge." >&2
        exit 1
    }
    eval "$(conda shell.bash hook)"
    # No importar `vllm` en el login: puede intentar cargar CUDA y fallar sin
    # GPU. `find_spec` comprueba que el paquete esta instalado sin ejecutarlo.
    if [ ! -x "$PROJECT_DIR/.conda-env/bin/python" ] \
        || ! "$PROJECT_DIR/.conda-env/bin/python" -c \
            "import importlib.util as u; import mohh, llamea; assert u.find_spec('vllm')" \
            >/dev/null 2>&1; then
        echo "Preparando .conda-env/ (primera vez; puede tardar varios minutos) ..."
        WITH_VLLM=1 bash slurm/ixachi-setup-env.sh
    fi
    conda activate "$PROJECT_DIR/.conda-env"
else
    if [ -x "$PROJECT_DIR/.conda-env/bin/python" ]; then
        # Entorno ya creado y activable sin `module` (p. ej. sesion ya cargada).
        # shellcheck disable=SC1091
        source "$PROJECT_DIR/.conda-env/bin/activate"
    else
        echo "ERROR: no hay 'module' ni .conda-env/." >&2
        echo "       Corre este comando en el nodo de login de ixachi." >&2
        exit 1
    fi
fi

if ! python -c "import mohh.llamea" >/dev/null 2>&1; then
    echo "ERROR: el ambiente no tiene mohh.llamea." >&2
    echo "       Recrea con: WITH_VLLM=1 bash slurm/ixachi-setup-env.sh" >&2
    exit 1
fi

# Raiz persistente: el job hereda exactamente la misma ruta. Si no se define
# MOHH_MODELS_DIR, models_root() usa $SCRATCH o $LUSTRE/$USER (no $HOME).
if [ -z "${MOHH_MODELS_DIR:-}" ] && [ -n "${SCRATCH:-}" ] && [ -d "${SCRATCH}" ]; then
    # Misma prioridad que `models_root()`: SCRATCH antes que LUSTRE.
    export MOHH_MODELS_DIR="${SCRATCH}/models"
elif [ -z "${MOHH_MODELS_DIR:-}" ] && [ -n "${LUSTRE:-}" ] && [ -d "${LUSTRE}/${USER:-}" ]; then
    export MOHH_MODELS_DIR="${LUSTRE}/${USER}/mohh-models"
fi
if [ -n "${MOHH_MODELS_DIR:-}" ]; then
    export MOHH_MODELS_DIR
else
    unset MOHH_MODELS_DIR
fi
echo "Raiz de modelos: $(python scripts/mohh_models.py root | head -1)"

# 2. Modelo: verificar/descargar en el LOGIN. El job GPU solo lee la ruta
#    local; si faltara, falla con un mensaje claro en vez de intentar bajar
#    pesos en un nodo sin internet.
if python scripts/mohh_models.py verify "$MODEL" >/dev/null 2>&1; then
    echo "Modelo ya disponible: $MODEL"
else
    echo "Descargando $MODEL en la raiz persistente (login con internet) ..."
    if ! python scripts/mohh_models.py download "$MODEL"; then
        echo "ERROR: no se pudo descargar '$MODEL'." >&2
        echo "       Revisa la conexion del login, HF_TOKEN/terminos del modelo" >&2
        echo "       y la raiz persistente ($(python scripts/mohh_models.py root | head -1))." >&2
        exit 1
    fi
    python scripts/mohh_models.py verify "$MODEL" >/dev/null
fi

# 3. Un unico job GPU. --export=ALL propaga PROFILE, BUDGET, EVAL_TEST, etc.
SBATCH_ARGS=(--parsable --export=ALL --constraint="$CONSTRAINT"
             --cpus-per-task="$CPUS" --mem="$MEM")
if [ -n "${TIME:-}" ]; then
    SBATCH_ARGS+=(--time="$TIME")
fi
if [ -n "${JOB_NAME:-}" ]; then
    SBATCH_ARGS+=(--job-name="$JOB_NAME")
fi

echo "Enviando el job unico (vLLM + busqueda) a la particion gpu ..."
JOB="$(MODEL="$MODEL" PROFILE="$PROFILE" CONSTRAINT="$CONSTRAINT" \
    GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MOHH_MODELS_DIR="${MOHH_MODELS_DIR:-}" \
    sbatch "${SBATCH_ARGS[@]}" slurm/llamea-single.slrm)"
echo "  job: $JOB"
echo "  log: logs/llamea-single-${JOB}.out"
echo "  err: logs/llamea-single-${JOB}.err"

if [ "${WAIT:-0}" = "1" ] && command -v squeue >/dev/null 2>&1; then
    echo "Esperando a que termine el job $JOB ..."
    while squeue -h -j "$JOB" 2>/dev/null | grep -q .; do
        sleep 30
    done
    echo "--- ultimas lineas de logs/llamea-single-${JOB}.out ---"
    tail -n 80 "logs/llamea-single-${JOB}.out" 2>/dev/null || true
    echo "--- ultimas lineas de logs/llamea-single-${JOB}.err ---"
    tail -n 40 "logs/llamea-single-${JOB}.err" 2>/dev/null || true
fi

echo
echo "Monitoreo:"
echo "  squeue --user=\$USER"
echo "  tail -f logs/llamea-single-${JOB}.out"
echo "  cat logs/llamea-single-${JOB}.err"
echo "  tail -f results/${CATEGORY:-searches}/${EXPERIMENT_NAME}/progress.txt"
echo "  tail -f results/${CATEGORY:-searches}/latest/progress.txt"
echo "Resultados en: results/${CATEGORY:-searches}/${EXPERIMENT_NAME}/"
