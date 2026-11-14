#!/bin/bash
# Normalizacion de CUDA_VISIBLE_DEVICES para ixachi (GPU asignada en un
# indice fisico no visible y nodos A100 con MIG).
#
# Uso (despues de `conda activate`):
#     source slurm/llamea-cuda-env.sh
#     normalize_cuda_visible_devices
#
# Slurm puede exportar CUDA_VISIBLE_DEVICES=<indice> (p. ej. 1). Si el cgroup
# solo expone esa GPU, el ordinal visible es 0 y vLLM 0.32 falla al mapear el
# indice fisico (NVMLError_InvalidArgument). En nodos A100 con MIG, un indice
# entero no es valido y torch ve 0 GPUs; ademas, el slice asignado puede estar
# ocupado. Esta funcion enumera los dispositivos MIG y elige uno con memoria
# libre suficiente (>= GPU_MEM_UTIL * total, con margen) para vLLM.

normalize_cuda_visible_devices() {
    local util="${GPU_MEM_UTIL:-0.90}"
    local count free total threshold best_dev="" best_free=0
    local candidate result seen="" uniq=() summary=""

    # Candidatos: valor heredado, indices 0/1 y UUIDs MIG de nvidia-smi.
    local candidates=()
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && candidates+=("$CUDA_VISIBLE_DEVICES")
    candidates+=("0" "1")
    while IFS= read -r line; do
        if [[ "$line" =~ UUID:\ (MIG-[A-Za-z0-9-]+) ]]; then
            candidates+=("${BASH_REMATCH[1]}")
        fi
    done < <(nvidia-smi -L 2>/dev/null || true)

    # Deduplicar preservando orden.
    for candidate in "${candidates[@]}"; do
        if [[ " $seen " != *" $candidate "* ]]; then
            seen+=" $candidate"
            uniq+=("$candidate")
        fi
    done

    # Primera pasada: elegir el dispositivo con mas memoria libre que cumpla
    # el umbral de vLLM (free >= util * total).
    for candidate in "${uniq[@]}"; do
        result="$(CUDA_VISIBLE_DEVICES="$candidate" python -c '
import torch, sys
n = torch.cuda.device_count()
if n == 0:
    print("0 0 0")
    sys.exit(0)
free, total = torch.cuda.mem_get_info()
print(n, free / 1024**3, total / 1024**3)
' 2>/dev/null || echo "0 0 0")"
        read -r count free total <<<"$result"
        count="${count:-0}"
        free="${free:-0}"
        total="${total:-0}"
        threshold="$(awk "BEGIN{print $util * $total + 0.25}" 2>/dev/null || echo 0)"
        if [ "$count" -ge 1 ] 2>/dev/null; then
            summary+=" $candidate:${free}G"
            if awk "BEGIN{exit !($free >= $threshold)}" 2>/dev/null; then
                if awk "BEGIN{exit !($free > $best_free)}" 2>/dev/null; then
                    best_free="$free"
                    best_dev="$candidate"
                fi
            fi
        else
            summary+=" $candidate:0G"
        fi
    done
    echo "GPU candidatas:$summary" >&2

    # Si ninguna cumple el umbral, tomar la de mayor memoria libre (best effort)
    # para que el error de vLLM sea claro en vez de fallar en silencio.
    if [ -z "$best_dev" ]; then
        best_free=0
        for candidate in "${uniq[@]}"; do
            result="$(CUDA_VISIBLE_DEVICES="$candidate" python -c '
import torch, sys
n = torch.cuda.device_count()
if n == 0:
    print("0 0")
    sys.exit(0)
free, _ = torch.cuda.mem_get_info()
print(n, free / 1024**3)
' 2>/dev/null || echo "0 0")"
            read -r count free <<<"$result"
            count="${count:-0}"
            free="${free:-0}"
            if [ "$count" -ge 1 ] 2>/dev/null && awk "BEGIN{exit !($free > $best_free)}" 2>/dev/null; then
                best_free="$free"
                best_dev="$candidate"
            fi
        done
    fi

    if [ -z "$best_dev" ]; then
        echo "ERROR: no hay GPU/MIG visible (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>})." >&2
        nvidia-smi -L >&2 || true
        return 1
    fi

    export CUDA_VISIBLE_DEVICES="$best_dev"
    echo "CUDA_VISIBLE_DEVICES=$best_dev (libres: ${best_free} GB)" >&2
    return 0
}
