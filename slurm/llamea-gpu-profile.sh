#!/bin/bash
# Perfiles de GPU/modelo para LLaMEA (ixachi: H100 80 GB, A100 40 GB, A10 24 GB).
#
# Se "sourcea" desde los .slrm / wrappers:
#     source slurm/llamea-gpu-profile.sh
#
# Define (respetando lo que ya este en el entorno):
#     MODEL, CONSTRAINT, MAX_MODEL_LEN, GPU_MEM_UTIL
#
# PROFILE (default big-quant):
#   big-quant   32B AWQ (~19-21 GB): el mayor numero de parametros comodo en
#               A100 40 GB y H100 80 GB. Es el default.
#   h100-fp16   32B en fp16 (~65 GB), solo H100 80 GB (enviar con
#               `sbatch --constraint=H100 ...`).
#   a100-fp16   14B en fp16 (~29 GB) en A100 40 GB.
#   a10-quant   14B AWQ (~10 GB) / 7B fp16 en A10 24 GB (`a10-fp16` es alias).
#   smoke       1.5B para pruebas de humo.
#
# Overrides: MODEL=..., CONSTRAINT=..., MAX_MODEL_LEN=..., GPU_MEM_UTIL=...
# Ejemplos:
#   PROFILE=h100-fp16 bash slurm/llamea-all.sh
#   MODEL=Qwen/Qwen2.5-Coder-14B-Instruct PROFILE=a100-fp16 bash slurm/llamea-all.sh

: "${PROFILE:=big-quant}"

case "$PROFILE" in
    h100-fp16)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        ;;
    a100-fp16)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-A100|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        ;;
    a10-fp16|a10-quant)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct-AWQ}"
        CONSTRAINT="${CONSTRAINT:-A10}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        ;;
    smoke)
        MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-A100|A10|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        ;;
    big-quant|*)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
        CONSTRAINT="${CONSTRAINT:-A100|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
        ;;
esac

export MODEL CONSTRAINT MAX_MODEL_LEN GPU_MEM_UTIL PROFILE
