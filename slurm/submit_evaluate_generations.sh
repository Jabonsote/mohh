#!/bin/bash
#
# Lanza la evaluación INTERCALADA: un job por generación del experimento.
#
# Uso:
#   EXPERIMENT=/LUSTRE/$USER/proyectos/SDD/results/experiments_larger/Experiment_001 \
#   DATASET_PATH=/LUSTRE/$USER/proyectos/SDD/results/dataset_larger/mqap \
#   bash slurm/submit_evaluate_generations.sh
#
set -euo pipefail

EXPERIMENT="${EXPERIMENT:?define EXPERIMENT=...}"
DATASET_PATH="${DATASET_PATH:?define DATASET_PATH=...}"
SOLVERS="${SOLVERS:-MOEAD,NSGAII,SMSEMOA}"
N_EXPERIMENTS="${N_EXPERIMENTS:-31}"

GEN_ROOT=$(find "$EXPERIMENT" -type d -path '*/generation_results/generations' | head -n 1)
if [ -z "$GEN_ROOT" ]; then
    echo "error: no encuentro generation_results/generations en $EXPERIMENT" >&2
    exit 1
fi
N=$(find "$GEN_ROOT" -maxdepth 1 -type d -name 'generation_*' | wc -l)
if [ "$N" -eq 0 ]; then
    echo "error: no hay generaciones en $GEN_ROOT" >&2
    exit 1
fi

EXPORT="ALL,EXPERIMENT=$EXPERIMENT,DATASET_PATH=$DATASET_PATH,SOLVERS=$SOLVERS,N_EXPERIMENTS=$N_EXPERIMENTS"
if [ -n "${CONDA_ENV:-}" ]; then
    EXPORT="$EXPORT,CONDA_ENV=$CONDA_ENV"
fi

echo "Experimento : $EXPERIMENT"
echo "Dataset     : $DATASET_PATH"
echo "Generaciones: $N (0..$((N-1)))"
echo "Réplicas    : $N_EXPERIMENTS por combinación"

JOB=$(sbatch --parsable \
    --array="0-$((N - 1))" \
    --export="$EXPORT" \
    slurm/mohh-evaluate-generations-array.slrm)
echo "Array eval  : $JOB"

FIG=$(sbatch --parsable \
    --dependency="afterok:$JOB" \
    --array="0-$((N - 1))" \
    --export="$EXPORT" \
    slurm/mohh-evaluate-generations-figures.slrm)
echo "Array figs  : $FIG"
