#!/bin/bash
#
# Ambiente MOHH en el cluster ixachi (via Modules + conda env).
#
# Uso (UNA sola vez, desde el nodo de login con internet, NO con sbatch):
#
#   cd /LUSTRE/<miusuario>/MOOHH-R
#   bash slurm/ixachi-setup-env.sh
#
# Lo que hace:
#   1. Carga el modulo python/miniforge (conda/forge con python 3.12).
#   2. Crea un conda env con prefijo local .conda-env/ (dentro de LUSTRE,
#      visible desde login y nodos de computo; no depende de $HOME ni de
#      rutas de otra maquina como un .venv local).
#   3. Instala el paquete mohh en modo editable + verifica los entry points.
#
# Despues de esto, los scripts mohh-*.slrm ya pueden enviarse con sbatch.
# Ver comandos en la seccion 3.1 (Modules) de la metodologia del cluster.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Proyecto: $PROJECT_DIR"

# 1. Software de desarrollo del cluster via Modules.
#    Por defecto ningun modulo esta cargado: hay que cargarlo explicito.
if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' no disponible; corre este script en el login de ixachi."
    exit 1
fi
module load python/miniforge || { echo "ERROR: no se pudo cargar python/miniforge."; exit 1; }
module list

# Habilita 'conda activate' en scripts no interactivos.
eval "$(conda shell.bash hook)"

# 2. Conda env con prefijo dentro del proyecto (LUSTRE).
ENV_PREFIX="$PROJECT_DIR/.conda-env"
if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.12
fi

conda activate "$ENV_PREFIX"
pip install -U pip
pip install -e .

# 3. Verificacion (equivale al "Hola mundo" de la metodologia: confirma que
#    el ambiente quedo funcional antes de gastar cola).
python -c "import mohh; print('mohh OK')"
mohh-generate -h >/dev/null && echo "mohh-generate OK"
mohh-evaluate -h >/dev/null && echo "mohh-evaluate OK"

echo "Ambiente listo. Envia trabajos con: sbatch slurm/mohh-generate.slrm"
