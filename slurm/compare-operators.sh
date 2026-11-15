#!/bin/bash
#
# Comparacion pareada de operadores LLaMEA contra los clasicos, en paralelo.
#
# La evaluacion es CPU-bound (MOEA por instancia): no necesita GPU ni vLLM.
# Se usa la particion CPU `ixachi` con todos los nucleos del job
# (FITNESS_WORKERS = CPUS - 1) para que `compare_operators.py` evalue las
# tareas (rep x protocolo x operador x instancia) en paralelo.
#
# Uso (login):
#     OPERATORS="llamea=ruta/al/mejor.py ox=classic:ox pbx=classic:pbx" \
#       bash slurm/compare-operators.sh
#
# Opciones por entorno (se exportan al job con --export=ALL):
#   OPERATORS          lista "nombre=fuente" (separada por espacios)
#   GROUPS             "nombre=op1,op2" (repetible): agrupa operadores
#   CONTRASTS          "grupoA:grupoB" (repetible): contraste entre grupos
#   SPLIT (test)       train | test
#   INSTANCES (all)    numero de instancias o 'all'
#   REPEATS (31)       repeticiones independientes (N_EXPERIMENTS de GE)
#   GE_PROTOCOL (1)    1 = protocolo GE completo (3 solvers, ambas mutaciones,
#                      300/105); 0 = protocolo corto (MOEAD, 30/30)
#   MO_GENERATIONS     override explicito de generaciones MOEA
#   MO_POPULATION_SIZE override explicito de poblacion MOEA
#   SOLVER             override del solver (default MOEAD / los 3 con GE)
#   SEED (42)          semilla base
#   NAME               nombre del experimento de salida
#   CPUS (32)          CPUs del job
#   MEM (64G)          memoria
#   TIME               limite SLURM
#
# Ejemplo rapido (protocolo corto, 3 reps):
#   OPERATORS="llamea=.../llamea_best_operator.py ox=classic:ox" \
#   REPEATS=3 GE_PROTOCOL=0 INSTANCES=8 \
#     bash slurm/compare-operators.sh
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: ejecuta este comando en el login, no dentro de un job." >&2
    exit 1
fi

if [ -z "${OPERATORS:-}" ]; then
    echo "ERROR: define OPERATORS (lista 'nombre=fuente')." >&2
    exit 1
fi

export SPLIT="${SPLIT:-test}"
export INSTANCES="${INSTANCES:-all}"
export REPEATS="${REPEATS:-31}"
# Grupos and contrasts: "name=op1,op2" and "groupA:groupB" (repeatable).
export GROUPS="${GROUPS:-}"
export CONTRASTS="${CONTRASTS:-}"
export GE_PROTOCOL="${GE_PROTOCOL:-1}"
export SEED="${SEED:-42}"
export NAME="${NAME:-llamea_vs_clasicos}"
export CPUS="${CPUS:-32}"
export MEM="${MEM:-64G}"

# Aprovechar todos los nucleos: FITNESS_WORKERS es el paralelismo de
# compare_operators.py; se deja 1 hilo para el proceso principal.
export FITNESS_WORKERS=$((CPUS - 1))

SBATCH_ARGS=(
    --parsable --export=ALL
    -p ixachi -c "$CPUS" --mem="$MEM"
)
if [ -n "${TIME:-}" ]; then
    SBATCH_ARGS+=(--time="$TIME")
fi

# El job (heredoc con delimitor entre comillas: sin expansion en el login) lee
# todo del entorno exportado arriba; asi no se interpola nada por error.
SLURM_SCRIPT="$(mktemp /tmp/compare-operators.XXXXXX.slrm)"
cat > "$SLURM_SCRIPT" <<'SLURM'
#!/bin/bash
#SBATCH -J compare-llamea
#SBATCH -p ixachi
#SBATCH -o logs/compare-llamea-%j.out
#SBATCH -e logs/compare-llamea-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/miniforge 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "$SLURM_SUBMIT_DIR/.conda-env"

export FITNESS_WORKERS
echo "FITNESS_WORKERS=$FITNESS_WORKERS  SPLIT=$SPLIT INSTANCES=$INSTANCES REPEATS=$REPEATS GE_PROTOCOL=$GE_PROTOCOL"

ARGS=(
    --split "$SPLIT"
    --instances "$INSTANCES"
    --repeats "$REPEATS"
    --seed "$SEED"
    --name "$NAME"
)
if [ "$GE_PROTOCOL" = "1" ]; then
    ARGS+=(--ge-protocol)
fi
if [ -n "${MO_GENERATIONS:-}" ]; then
    ARGS+=(--mo-generations "$MO_GENERATIONS")
fi
if [ -n "${MO_POPULATION_SIZE:-}" ]; then
    ARGS+=(--mo-population-size "$MO_POPULATION_SIZE")
fi
if [ -n "${SOLVER:-}" ]; then
    ARGS+=(--solver "$SOLVER")
fi
# Cada grupo/contraste se pasa como un argumento propio.
for group in $GROUPS; do
    ARGS+=(--group "$group")
done
for contrast in $CONTRASTS; do
    ARGS+=(--contrast "$contrast")
done
# OPERATORS: cada token "nombre=fuente" se pasa como --operator.
for spec in $OPERATORS; do
    ARGS+=(--operator "$spec")
done

python scripts/compare_operators.py "${ARGS[@]}"
SLURM

mkdir -p logs
JOB="$(sbatch "${SBATCH_ARGS[@]}" "$SLURM_SCRIPT")"
rm -f "$SLURM_SCRIPT"
echo "  job: $JOB"
echo "  log: logs/compare-llamea-${JOB}.out"
echo "  err: logs/compare-llamea-${JOB}.err"
echo "Monitoreo:"
echo "  squeue --user=\$USER"
echo "  tail -f logs/compare-llamea-${JOB}.out"
echo "Resultados en: results/evaluations/${NAME}/"
