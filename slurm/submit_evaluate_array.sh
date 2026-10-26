#!/bin/bash
#
# Lanza mohh-evaluate en un job array (una tarea por instancia de test) y
# encadena la generación de figuras con afterok.
#
# Uso:
#   EXPERIMENT=/LUSTRE/$USER/proyectos/SDD/results/experiments_larger/Experiment_001 \
#   DATASET_PATH=/LUSTRE/$USER/proyectos/SDD/results/dataset_larger/mqap \
#   bash slurm/submit_evaluate_array.sh
#
set -euo pipefail

mkdir -p logs

EXPERIMENT="${EXPERIMENT:?define EXPERIMENT=...}"
DATASET_PATH="${DATASET_PATH:?define DATASET_PATH=...}"
SOLVERS="${SOLVERS:-MOEAD,NSGAII,SMSEMOA}"
N_EXPERIMENTS="${N_EXPERIMENTS:-31}"
MAX_CONCURRENT="${MAX_CONCURRENT:-12}"

if [ ! -d "$DATASET_PATH/test" ]; then
    echo "error: no existe $DATASET_PATH/test" >&2
    exit 1
fi
if [ ! -d "$EXPERIMENT" ]; then
    echo "error: no existe $EXPERIMENT" >&2
    exit 1
fi

N=$(find "$DATASET_PATH/test" -maxdepth 1 -name '*.txt' | wc -l)
if [ "$N" -eq 0 ]; then
    echo "error: no hay instancias .txt en $DATASET_PATH/test" >&2
    exit 1
fi

SOLVER_LIST="${SOLVERS//,/:}"
EXPORT="ALL,EXPERIMENT=$EXPERIMENT,DATASET_PATH=$DATASET_PATH,SOLVER_LIST=$SOLVER_LIST,N_EXPERIMENTS=$N_EXPERIMENTS"
if [ -n "${CONDA_ENV:-}" ]; then
    EXPORT="$EXPORT,CONDA_ENV=$CONDA_ENV"
fi

echo "Experimento : $EXPERIMENT"
echo "Dataset     : $DATASET_PATH ($N test)"
echo "Array       : 0-$((N - 1))%$MAX_CONCURRENT"
echo "Réplicas    : $N_EXPERIMENTS por combinación"

JOB=$(sbatch --parsable \
    --array="0-$((N - 1))%$MAX_CONCURRENT" \
    --export="$EXPORT" \
    slurm/mohh-evaluate-array.slrm)
echo "Array job   : $JOB"

FIG=$(sbatch --parsable \
    --dependency="afterok:$JOB" \
    --export="$EXPORT" \
    slurm/mohh-evaluate-figures.slrm)
echo "Figuras job : $FIG"
