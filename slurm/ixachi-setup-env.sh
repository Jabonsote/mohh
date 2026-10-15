#!/bin/bash
#
# Ambiente MOHH en el cluster ixachi.
#
# Uso (UNA sola vez, desde el nodo de login con internet, NO con sbatch):
#
#   cd /LUSTRE/<miusuario>/MOOHH-R
#   bash slurm/ixachi-setup-env.sh
#
# Lo que hace:
#   1. Carga el modulo de Python (python/miniforge).
#   2. Crea un venv en .venv-ixachi/ (dentro de LUSTRE, visible en los nodos).
#   3. Instala el paquete mohh en modo editable + verifica los entry points.
#
# Despues de esto, los scripts mohh-*.slrm ya pueden enviarse con sbatch.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Proyecto: $PROJECT_DIR"

# 1. Modulo de Python del cluster (si existe; si no, se usa el python3 del login).
if command -v module >/dev/null 2>&1; then
    module load python/miniforge || echo "AVISO: no se pudo cargar python/miniforge, sigo con python3 del sistema."
fi

command -v python3 >/dev/null 2>&1 || { echo "ERROR: no hay python3 disponible."; exit 1; }
python3 --version

# 2. Venv dedicado al cluster (no se reutiliza .venv/ local: suele apuntar
#    a rutas de otra maquina y rompe los symlinks en los nodos).
if [ ! -d ".venv-ixachi" ]; then
    python3 -m venv .venv-ixachi
fi

# shellcheck disable=SC1091
source .venv-ixachi/bin/activate
pip install -U pip
pip install -e .

# 3. Verificacion (equivale al "Hola mundo" de la metodologia: confirma que
#    el ambiente quedo funcional antes de gastar cola).
python -c "import mohh; print('mohh OK')"
mohh-generate -h >/dev/null && echo "mohh-generate OK"
mohh-evaluate -h >/dev/null && echo "mohh-evaluate OK"

echo "Ambiente listo. Envia trabajos con: sbatch slurm/mohh-generate.slrm"
