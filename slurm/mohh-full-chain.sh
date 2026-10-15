#!/bin/bash
#
# Cadena completa generacion -> evaluacion con dependencia SLURM.
#
# Uso (desde el nodo de login, con el repo en /LUSTRE/<miusuario>/MOOHH-R):
#
#   bash slurm/mohh-full-chain.sh
#
# Envia mohh-generate-big (110 cores exclusivos en graphos1) y deja
# mohh-evaluate (32 cores, 31 experimentos) encolado con
# --dependency=afterok, asi la evaluacion arranca sola cuando la
# generacion termina bien. Cada etapa usa sus propios cores,
# por eso son dos jobs y no uno.
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

GEN_JID="$(sbatch --parsable slurm/mohh-generate-big.slrm)"
echo "Generacion: job $GEN_JID"

EVAL_JID="$(sbatch --parsable --dependency=afterok:"$GEN_JID" slurm/mohh-evaluate.slrm)"
echo "Evaluacion: job $EVAL_JID (tras $GEN_JID)"

squeue --user="$USER"
