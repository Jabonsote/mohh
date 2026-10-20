#!/bin/bash
# Perfiles de GPU/modelo para LLaMEA (ixachi: H100 80 GB, A100 40 GB, A10 24 GB).
#
# Se "sourcea" desde los .slrm / wrappers:
#     source slurm/llamea-gpu-profile.sh
#
# Define (respetando lo que ya este en el entorno):
#     MODEL, CONSTRAINT, MAX_MODEL_LEN, GPU_MEM_UTIL, VLLM_ARGS
#
# `VLLM_ARGS` son argumentos del perfil (cuantizacion, KV cache); se suman a
# `EXTRA_VLLM_ARGS`, que es el override del usuario. Para ver el catalogo:
#
#     bash -c 'source slurm/llamea-gpu-profile.sh'   # con PROFILE=list
#     PROFILE=list bash slurm/llamea-all.sh
#
# Evidencia de la eleccion (HumanEval / HumanEval+ / BigCodeBench / LCB):
#   - Qwen2.5-Coder-32B-Instruct  HE 92.7  HE+ 87.2  BCB 49.6/27.0  LCB 31.4
#   - Qwen3-Coder-30B-A3B-Instruct (3.3B activos) ~HE 92.1 HE+ 89.0, 256k ctx
#   - Qwen2.5-Coder-14B-Instruct  HE 89.6  HE+ 87.2  BCB 48.4/22.2  LCB 23.4
#   - Seed-Coder-8B-Instruct      HE 84.8  MBPP 85.2 BCB 53.3/26.4  LCB 24.7
#   - DeepSeek-Coder-V2-Lite       HE 81.1  HE+ 75.6  BCB 36.8/16.2  LCB 16.3
# Fuentes y matices en modelos.md (seccion 1).
#
# PROFILE (default big-quant):
#   big-quant        32B AWQ (~19-21 GB). Default historico y fallback probado:
#                    cabe en A100 40 GB y H100 80 GB con contexto largo.
#   coder30b-awq     Qwen3-Coder-30B-A3B AWQ (~18 GB). Recomendado en A100/H100:
#                    30.5B totales pero solo 3.3B activos -> mucho mas rapido que
#                    un denso 32B, con HumanEval+ algo mejor. Cuantizacion de
#                    comunidad (QuantTrio): validar con `mohh_models.py verify`.
#   coder30b-fp8     Qwen3-Coder-30B-A3B FP8 oficial (~31 GB). Recomendado en
#                    H100 (FP8 nativo sm90); deja KV cache para 32k.
#   coder30b-awq-a10 El mismo 30B AWQ en A10 24 GB, contexto 8k.
#   h100-fp16        32B fp16 (~65 GB), solo H100 80 GB: maxima calidad del
#                    modelo ya validado, sin cuantizar.
#   a100-fp16        14B fp16 (~29 GB) en A100 40 GB.
#   a10-quant        14B AWQ (~10 GB) / 7B fp16 en A10 24 GB.
#   coder8b          Seed-Coder-8B (MIT, ~16 GB fp16) en A10 24 GB: mejor
#                    BigCodeBench entre los ~8B y deja KV cache holgado.
#   smoke            1.5B para pruebas de humo.
#
# Overrides: MODEL=..., CONSTRAINT=..., MAX_MODEL_LEN=..., GPU_MEM_UTIL=...,
#            VLLM_ARGS=..., EXTRA_VLLM_ARGS=...
# Ejemplos:
#   PROFILE=coder30b-awq BUDGET=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
#   PROFILE=coder30b-fp8 BUDGET=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
#   PROFILE=h100-fp16 bash slurm/llamea-all.sh
#   MODEL=Qwen/Qwen2.5-Coder-14B-Instruct PROFILE=a100-fp16 bash slurm/llamea-all.sh

: "${PROFILE:=big-quant}"

if [ "$PROFILE" = "list" ]; then
    cat <<'EOF'
PROFILE           MODELO                                      VRAM    GPU
big-quant         Qwen/Qwen2.5-Coder-32B-Instruct-AWQ         ~20 GB  A100/H100
coder30b-awq      QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ  ~18 GB  A100/H100
coder30b-fp8      Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8       ~31 GB  H100
coder30b-awq-a10  QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ  ~18 GB  A10
h100-fp16         Qwen/Qwen2.5-Coder-32B-Instruct             ~65 GB  H100
a100-fp16         Qwen/Qwen2.5-Coder-14B-Instruct             ~29 GB  A100/H100
a10-quant         Qwen/Qwen2.5-Coder-14B-Instruct-AWQ         ~10 GB  A10
coder8b           ByteDance-Seed/Seed-Coder-8B-Instruct       ~16 GB  A10
smoke             Qwen/Qwen2.5-1.5B-Instruct                   ~3 GB  cualquiera

Recomendado: coder30b-awq en A100/H100, coder30b-fp8 en H100,
coder8b o a10-quant en A10. big-quant sigue siendo el default probado.
EOF
    exit 0
fi

case "$PROFILE" in
    coder30b-fp8)
        MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
        CONSTRAINT="${CONSTRAINT:-H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:---kv-cache-dtype fp8}"
        ;;
    coder30b-awq)
        MODEL="${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}"
        CONSTRAINT="${CONSTRAINT:-A100|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:---quantization awq_marlin}"
        ;;
    coder30b-awq-a10)
        MODEL="${MODEL:-QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ}"
        CONSTRAINT="${CONSTRAINT:-A10}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:---quantization awq_marlin}"
        ;;
    coder8b)
        MODEL="${MODEL:-ByteDance-Seed/Seed-Coder-8B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-A10}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:-}"
        ;;
    h100-fp16)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:-}"
        ;;
    a100-fp16)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-A100|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:-}"
        ;;
    a10-fp16|a10-quant)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct-AWQ}"
        CONSTRAINT="${CONSTRAINT:-A10}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        # Auto-detect: el AWQ oficial de Qwen ya elige awq_marlin en vLLM.
        VLLM_ARGS="${VLLM_ARGS:-}"
        ;;
    smoke)
        MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
        CONSTRAINT="${CONSTRAINT:-A100|A10|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
        VLLM_ARGS="${VLLM_ARGS:-}"
        ;;
    big-quant|*)
        MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}"
        CONSTRAINT="${CONSTRAINT:-A100|H100}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
        # Default probado: se deja el autodetect de vLLM (sin flags extra).
        VLLM_ARGS="${VLLM_ARGS:-}"
        ;;
esac

export MODEL CONSTRAINT MAX_MODEL_LEN GPU_MEM_UTIL VLLM_ARGS PROFILE
