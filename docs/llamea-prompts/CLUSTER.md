# LLaMEA cluster

Checkout en LUSTRE. El login tiene internet; los nodos de cómputo no.

```bash
cd /LUSTRE/$USER/MOOHH-R
WITH_VLLM=1 bash slurm/ixachi-setup-env.sh       # una vez
PROFILE=coder30b-awq GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
```

Prueba:

```bash
PROFILE=smoke BUDGET=20 EVAL_TEST=1 TEST_INSTANCES=2 EVAL_REPEATS=1 \
  WAIT=1 bash slurm/llamea-all.sh
```

Perfiles:

```bash
PROFILE=list bash slurm/llamea-all.sh
```

GPU libres por nodo:

```bash
for n in $(sinfo -p gpu -h -N -o "%N"); do
  echo "== $n"
  scontrol show node "$n" | grep -E "State=|Gres=|AllocTRES="
done
```

`Gres` es el total del nodo; `AllocTRES` lo ocupado. `MIXED` puede tener GPU
libre; `IDLE` está libre; `DOWN` no se puede usar.

A100/H100: `coder30b-awq`, `glm47-flash-awq`, `big-quant`.
H100: `coder30b-fp8`, `coder-next-awq`.
A10: `coder30b-awq-a10`, `a10-quant`, `coder8b`.
Pasa `PROFILE` explícito.

| Variable | Uso |
|---|---|
| `GENERATIONS` | iteraciones; `BUDGET` la sobrescribe |
| `EVAL_TEST=1` | evaluación final en test |
| `COMPARE=1` | comparación pareada contra clásicos; caro |
| `EXPERIMENT_NAME` | nombre de la carpeta |
| `SOLVER` | `MOEAD`, `NSGAII`, `SMSEMOA` |
| `TRAIN_INSTANCES` / `TEST_INSTANCES` | `all` o un número |
| `CPUS` / `MEM` / `TIME` | recursos SLURM |
| `WAIT=1` / `DRY_RUN=1` | esperar al job / no enviarlo |

`SIMPLE=1` es el modo actual: `BUDGET=GENERATIONS`, 1+1 y fitness de GE (rango
medio por población, 1 = mejor). `COMPARE=1` añade la comparación en el mismo
job.

Resultados: `results/searches/<EXPERIMENT_NAME>/`. Ahí están `progress.txt`,
`llamea_best_operator.py`, `candidates.jsonl` y `llm_requests.jsonl`.
`COMPARE=1` escribe en `results/evaluations/`.

```bash
tail -f logs/llamea-single-*.out
tail -f results/searches/<EXPERIMENT_NAME>/progress.txt
```

Modelo ausente: `python scripts/mohh_models.py download <modelo>`.
OOM: baja `MAX_MODEL_LEN` o `GPU_MEM_UTIL`.
El wrapper se ejecuta en el login, no dentro de SLURM.
