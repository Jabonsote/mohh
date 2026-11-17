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

A100/H100: `coder30b-awq`, `glm47-flash-awq`, `big-quant`, `gptoss-20b`.
H100: `coder30b-fp8`, `coder-next-awq`.
A10: `coder30b-awq-a10`, `qwen3-8b-awq`, `a10-quant`, `coder8b`, `gptoss-20b`.
Pasa `PROFILE` explícito.

### Modelos de razonamiento: `MAX_TOKENS`

`gptoss-20b`, `glm47-flash-awq` y `glm47-flash-awq-a10` razonan antes de
responder, y el pensamiento se come `max_tokens`. Con `MAX_TOKENS=2048` el
31,8 % de las respuestas se cortaron **antes de escribir el codigo**
(`results/searches/glm-gptoss`): ese presupuesto perdido es la causa de
casi todos los fallos de parseo.

El modo pensamiento **no** se desactiva; lo que hace falta es presupuesto:

```bash
PROFILE=gptoss-20b GENERATIONS=1000 bash slurm/llamea-all.sh   # MAX_TOKENS=8192
```

Los perfiles de razonamiento ya traen `MAX_TOKENS=8192`. Para subirlos:
`MAX_TOKENS=16384`. Si la respuesta se corta, el resumen lo dice:
`search_summary.json` -> `failures.by_reason["respuesta truncada: ..."]` y
`tokens.truncated_responses`.

Si vLLM muere al arrancar con `NVMLError_InvalidArgument` o `No CUDA
runtime is found`, revisa el mapeo de GPU dentro del job:

```bash
srun --jobid=<JOBID> --overlap bash -lc '
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}";
  nvidia-smi -L;
  .conda-env/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
'
```

En `graphos1`, Slurm puede asignar la GPU 1 y exportar
`CUDA_VISIBLE_DEVICES=1`; si el cgroup solo expone esa GPU, el ordinal visible
es `0` y vLLM 0.32 falla al mapear el índice físico. `llamea-single.slrm` y
`llamea-vllm-server.slrm` ya lo detectan: si `torch` no ve ninguna GPU con el
valor heredado, prueban `CUDA_VISIBLE_DEVICES=0` antes de arrancar vLLM.

La A100 de `graphos1` está particionada en MIG (slices de ~20 GB). El job
puede caer en un slice ocupado (~6 GB libres) y vLLM falla por falta de
memoria. La normalización (`slurm/llamea-cuda-env.sh`) enumera los slices MIG
y elige el de mayor memoria libre que cumpla `GPU_MEM_UTIL * total`; si ninguno
cumple, elige el mejor disponible y deja que vLLM reporte el error.

Para forzar A100 (aunque solo esté libre una de sus dos GPU):

```bash
CONSTRAINT=A100 PROFILE=qwen3-8b-awq GENERATIONS=1000 EVAL_TEST=1 \
  bash slurm/llamea-all.sh
```

Si ambas A100 están ocupadas, el job queda en `PD` y arranca cuando se libere
cualquiera; no hay que reenviarlo.

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
job. En `SIMPLE=1`, `EVAL_TEST=1` no llega a ejecutar la evaluación final:
`run_simple` retorna antes; el job solo entrena.

Resultados: `results/searches/<EXPERIMENT_NAME>/`. Ahí están `progress.txt`,
`llamea_best_operator.py`, `candidates.jsonl` y `llm_requests.jsonl`.

El operador encontrado debe ser **reproducible**: la búsqueda rechaza los
operadores que crean su propio generador sin semilla (`np.random.default_rng()`),
porque el arnés siembra `np.random.seed(...)` y ese generador lo ignora. Para
comparar un operador antiguo que use esa API, se siembra antes:

```bash
python scripts/seed_operator.py \
  results/searches/<EXPERIMENT_NAME>/llamea_best_operator.py \
  results/searches/<EXPERIMENT_NAME>/llamea_best_operator_seeded.py
```

El script sustituye el generador por el RNG global (`np.random.*`), verifica
que el operador sigue siendo válido y avisa si hay algo que no pueda
convertir solo. `compare_operators.py` carga el operador sin sembrar (para
poder compararlo) pero avisa por stderr.
`COMPARE=1` escribe en `results/evaluations/`.

```bash
tail -f logs/llamea-single-*.out
tail -f results/searches/<EXPERIMENT_NAME>/progress.txt
```

Modelo ausente: `python scripts/mohh_models.py download <modelo>`.
OOM: baja `MAX_MODEL_LEN` o `GPU_MEM_UTIL`.
El wrapper se ejecuta en el login, no dentro de SLURM.

### Activar el env en el login (y anadir paquetes)

El env de conda esta en `<repo>/.conda-env` y NO trae `bin/activate` (eso es
de virtualenv). La receta es la misma que usan los jobs:

```bash
cd /LUSTRE/$USER/mohh
module load python/miniforge
eval "$(conda shell.bash hook)"
conda activate "$PWD/.conda-env"

# anadir paquetes SIN reinstalar el proyecto (el backend es hatchling y un
# `pip install -e .` con un pip viejo falla; solo hacen falta los nuevos):
pip install "crystalbleu>=0.2" "nltk>=3.9"
python -c "import crystalbleu, nltk; print('ok')"
```

Sin `crystalbleu`, el CrystalBLEU del resumen de la busqueda sale null (el
hilo lo reporta en `llamea.log` con un WARNING).

## Comparar GE vs LLaMEA (prueba estadistica)

`scripts/compare_operators.py` corre el mismo arnés para los tres tipos de
operador y contrasta sus HV de forma pareada (misma semilla, misma población
inicial, mismo marco), como la evaluación gramatical:

- **GE**: los operadores que la GE escribió, `<experimento>/<MODELO>_operators.txt`
  (`Best:`, `Middle:`, `Worst:` con el fenotipo de la gramática).
- **LLaMEA**: el `.py` con `def crossover(x, y)` que exporta la búsqueda
  (`llamea_best_operator.py`).
- **Clásicos**: el panel de `mohh.llamea.reference` (`ox`, `pbx`, `pmx`, `cx`,
  `erx`).

```bash
OPERATORS="ox=classic:ox pmx=classic:pmx"
GE_EXP=/LUSTRE/$USER/ge/Experiment_001
LLAM=/LUSTRE/$USER/mohh/results/searches/<EXP>/llamea_best_operator.py

FITNESS_WORKERS=16 python scripts/compare_operators.py \
  --operator "$OPERATORS" \
  --operator "llamea=$LLAM" \
  --ge-operators "$GE_EXP" \
  --group "base=ox,pmx" --group "ge=MOEAD_Best,MOEAD_Middle,MOEAD_Worst" \
  --group "llamea=llamea" \
  --contrast "base:ge" --contrast "llamea:base" \
  --split test --instances all --repeats 31 --ge-protocol \
  --name ge_vs_llamea
```

Los grupos (`--group`) y sus contrastes (`--contrast`) son modulares: cada
contraste es un test de Wilcoxon pareado (con Holm sobre la familia) entre dos
grupos, usando el marco consolidado de los dos.

Se guarda todo para no repetir el MOEA: `fronts.npz` (frentes crudos),
`failures.json` (motivo de cada fallo), `hv_por_instancia/<instancia>.csv`
(filas = repetición, columnas = operador, como el CSV del original),
`hv_por_instancia.csv`, `resumen_operadores.csv`, `contrastes.csv`,
`fronts/<instancia>/*.pof` (mediana y consolidado, mismo formato que el
original) y `evaluation.json`.

Para reanalizar sin volver a correr el MOEA:

```bash
python scripts/compare_operators.py --operator ... --load-fronts --name ge_vs_llamea
```

Con `--ge-protocol` se replica el protocolo GE: 3 solvers, ambas mutaciones,
300 generaciones, población 105 y 31 repeticiones (`N_EXPERIMENTS` del
original). Con menos objetivos, baja `--mo-population-size`: pymoo necesita una
población mínima válida para el número de objetivos.
