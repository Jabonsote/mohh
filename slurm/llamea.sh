#!/bin/bash
#
# Alias corto del comando unico de LLaMEA en el cluster:
#
#   bash slurm/llamea.sh
#
# Hace exactamente lo mismo que `bash slurm/llamea-all.sh`: prepara el
# ambiente si falta, verifica/descarga el modelo en el login y envia UN solo
# job GPU que sirve vLLM y corre la busqueda. No hay jobs separados.
#
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/llamea-all.sh" "$@"
