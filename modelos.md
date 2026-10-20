# modelos.md — Modelos locales para LLaMEA + vLLM

El modelo **es local**: lo sirve un servidor de inferencia con API
OpenAI-compatible (vLLM en el cluster; `scripts/local_openai_server.py` para
pruebas) y LLaMEA se conecta por HTTP. En el cluster los pesos viven en una
**raíz persistente** (LUSTRE/`$SCRATCH`, nunca `$HOME`) gestionada por
`scripts/mohh_models.py`; la descarga se hace en el login (los nodos de cómputo
no tienen internet).

```bash
python scripts/mohh_models.py root       # dónde viven los pesos
python scripts/mohh_models.py list       # qué hay descargado
python scripts/mohh_models.py verify <model_id>   # integridad (sin red)
python scripts/mohh_models.py download <model_id> # login (con red)
```

`scripts/ensure_model.py` sigue disponible para el cache clásico de
HuggingFace (`~/.cache/huggingface/hub/models--<org>--<modelo>`, respeta
`HF_HOME` y `HUGGINGFACE_HUB_CACHE`).

---

## 1. Modelos: evidencia, VRAM y elección

### 1.1 Qué dice la evidencia de código

La tarea del bucle es **sintetizar una función Python corta** (`crossover(x, y)`)
que cumpla un contrato estricto y use numpy. Los benchmarks relevantes son los
de generación de funciones (HumanEval, HumanEval+, MBPP, BigCodeBench,
LiveCodeBench), no los agentivos (SWE-bench), que miden otra cosa.

| Modelo | Params (activos) | Licencia | HE | HE+ | MBPP | BCB full/hard | LCB | Fuente |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-Coder-32B-Instruct` | 32B | Apache-2.0 | 92.7 | 87.2 | 90.2 | 49.6 / 27.0 | 31.4 | Qwen2.5-Coder report, Table 16 |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | 14B | Apache-2.0 | 89.6 | 87.2 | 86.2 | 48.4 / 22.2 | 23.4 | idem |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | Apache-2.0 | 88.4 | 84.1 | 83.5 | 41.0 / 18.2 | 18.2 | idem |
| `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | 16B (2.4B) | DeepSeek License | 81.1 | 75.6 | 82.8 | 36.8 / 16.2 | 16.3 | idem |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | 30.5B (3.3B) | Apache-2.0 | 92.1† | 89.0† | — | — | ~40† | medición de terceros; card oficial con figura |
| `ByteDance-Seed/Seed-Coder-8B-Instruct` | 8B | MIT | 84.8 | — | 85.2 | 53.3 / 26.4 | 24.7 | Seed-Coder card (harness propio) |

† El `HumanEval 92.1 / HumanEval+ 89.0` de Qwen3-Coder-30B-A3B es un *anchor*
medido por terceros sobre el modelo base (no una tabla oficial en texto); el
`LiveCodeBench ~40` viene de otro harness. Tómalo como indicativo, no como una
fila comparable al pie de la letra con las de Qwen2.5-Coder. La card oficial de
Qwen3-Coder publica la comparación como imagen y el modelo es **non-thinking**
(no genera ` thinking`), lo que simplifica el parseo.

Lectura:

- **32B vs 14B**: la diferencia en HumanEval es 3 puntos (92.7 vs 89.6), pero
  el 32B cuesta ~2.3× más cómputo por token. Para 1000 generaciones, el
  throughput importa tanto como el benchmark.
- **Qwen3-Coder-30B-A3B** es el punto dulce: calidad de 32B en HumanEval+ (89.0
  vs 87.2), solo **3.3B parámetros activos** (MoE) y 256k de contexto. En vLLM
  sobre H100 se han medido ~1200 tok/s a concurrencia 32 (benchmark de terceros,
  Massed Compute), frente a un denso 32B. Eso reduce el tiempo de una corrida
  de 1000 generaciones de horas a bastante menos.
- **Seed-Coder-8B** es el mejor ≤8B en BigCodeBench (53.3/26.4) y tiene licencia
  MIT; cabe en A10 24 GB en fp16 (~16 GB) dejando KV cache.
- **DeepSeek-Coder-V2-Lite** ya no compite en calidad (HE 81.1) y su licencia es
  custom; se conserva como alternativa si ya está descargado.
- **Devstral-Small-2507** (24B, Apache-2.0) y **GLM-4-32B-0414** (32B, MIT) son
  buenos modelos, pero su evidencia fuerte es agentiva (SWE-bench, tool use),
  no la síntesis de una función aislada. No se justifica el costo extra.
- **gpt-oss-120b** (117B/5.1B activos, Apache-2.0) cabe en H100 y es fuerte en
  código, pero es un modelo de razonamiento: el CoT consume tokens y el
  presupuesto de 1000 generaciones se vuelve mucho más caro. Queda como
  experimento, no como default.
- **Qwen3-Coder-Next** (80B/3B activos, Apache-2.0) es la evolución natural,
  pero su FP8 oficial ocupa ~80 GB y no cabe en una H100 con KV cache; haría
  falta AWQ/MXFP4 (menos evidencia) o `--tensor-parallel-size 2`. No hay perfil
  todavía; es candidato para cuando el cluster tenga 2 GPUs por job.

Por qué `-Coder` y no solo "más grande": en las corridas locales el fallo típico
del 0.5B fue **escribir código degenerado pero válido** (`return x`,
`np.array(x, copy=False)`, truncamientos), no proponer malas ideas. Un modelo
afinado en código del mismo tamaño escribe mejor código; por eso
`Qwen2.5-Coder-7B-Instruct` es la primera opción antes que un genérico más
grande.

### 1.2 Cuantización: 4-bit es seguro, 2-3 bits no

La evidencia sobre cuantización de modelos de código es consistente:

- Un estudio sobre modelos de hasta 34B encuentra que **4-bit reduce ~70 % la
  memoria sin degradación significativa** en generación de código; 3 y 2 bits sí
  pierden calidad ([arXiv:2503.07103](https://arxiv.org/abs/2503.07103)).
- En modelos de razonamiento, **W8A8 y W4A16 son lossless**; por debajo de 4
  bits la caída es grande y los modelos chicos sufren más
  ([arXiv:2504.04823](https://arxiv.org/abs/2504.04823)).
- Un estudio de robustez incluso reporta que los modelos cuantizados a 4-bit
  aguantan mejor perturbaciones adversariales, sin perder capacidad
  ([arXiv:2506.22776](https://arxiv.org/abs/2506.22776)).

Conclusión práctica: **AWQ W4A16** es la opción correcta para A100 40 GB/A10
24 GB; **FP8** es preferible en H100 (más preciso que 4-bit y entra de sobra);
**fp16/bf16** queda para la corrida de referencia sin cuantizar. No uses 3-bit
ni 2-bit para este bucle.

### 1.3 VRAM de referencia

VRAM de pesos, sin KV cache; el KV cache y las activaciones dependen de
`--max-model-len` y del tamaño del prompt.

| Modelo | fp16/bf16 | 4-bit AWQ | FP8 | Nota |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | ~3 GB | — | — | smoke |
| Qwen2.5-Coder-7B-Instruct | ~15 GB | ~5 GB | ~8 GB | A10 con contexto corto |
| Seed-Coder-8B-Instruct | ~16 GB | ~6 GB | ~9 GB | A10 24 GB |
| Qwen2.5-Coder-14B-Instruct | ~29 GB | ~10 GB | ~15 GB | A100 40 GB / A10 AWQ |
| Qwen2.5-Coder-32B-Instruct | ~65 GB | ~19-21 GB | ~33 GB | H100 fp16; AWQ en A100 |
| Qwen3-Coder-30B-A3B-Instruct | ~61 GB | ~18 GB | ~31 GB | MoE 3.3B activos |
| DeepSeek-Coder-V2-Lite | ~31 GB | ~9 GB | ~16 GB | 16B MoE (2.4B activos) |

Notas:

- vLLM reserva memoria para el KV cache. Para prompts largos (el bucle evolutivo
  manda código + feedback de varias generaciones) vigilar
  `--gpu-memory-utilization` (default 0.9). Si hay OOM: bajarlo
  (`GPU_MEM_UTIL=0.8` en el `.slrm`) o acortar `--max-model-len`
  (`MAX_MODEL_LEN=8192`).
- **Elección por GPU**: `slurm/llamea-gpu-profile.sh` elige el modelo y, en
  algunos perfiles, argumentos de vLLM (`coder30b-awq` fuerza
  `--quantization awq_marlin`; `coder30b-fp8` usa `--kv-cache-dtype fp8`):

  | GPU | VRAM | `PROFILE` | Modelo | Por qué |
  |---|---|---|---|---|
  | H100 | 80 GB | `coder30b-fp8` | Qwen3-Coder-30B-A3B FP8 (~31 GB) | mejor calidad/velocidad, FP8 oficial, 32k contexto |
  | H100 | 80 GB | `h100-fp16` | Qwen2.5-Coder-32B fp16 (~65 GB) | referencia sin cuantizar, ya validada |
  | A100 | 40 GB | `coder30b-awq` | Qwen3-Coder-30B-A3B AWQ (~18 GB) | 30B con 3.3B activos; cuantización de comunidad |
  | A100 | 40 GB | `big-quant` (default) | Qwen2.5-Coder-32B AWQ (~20 GB) | fallback probado, cuantización oficial |
  | A10 | 24 GB | `coder8b` | Seed-Coder-8B fp16 (~16 GB) | mejor ≤8B en BigCodeBench, KV holgado |
  | A10 | 24 GB | `a10-quant` | Qwen2.5-Coder-14B AWQ (~10 GB) | más parámetros, ya validado |
  | cualquiera | — | `smoke` | Qwen2.5-1.5B | sanity check |

  `big-quant` sigue siendo el **default** para no invalidar las corridas ya
  hechas. Para una corrida nueva en A100/H100, `coder30b-awq` es la apuesta
  basada en evidencia; en H100, `coder30b-fp8`. En A10, `coder8b` deja más KV
  cache que `coder30b-awq-a10` (30B AWQ en 24 GB va justo).
  `coder30b-awq` usa un AWQ de comunidad (`QuantTrio`); verifica el manifiesto
  antes de confiar en él y, si algo falla, vuelve a `big-quant`.
- **En esta máquina de pruebas (RTX 3060, 12 GB)**: usar `Qwen2.5-0.5B/1.5B`.
- Los modelos `-Instruct` tienen chat template: el prompt de LLaMEA va como
  mensaje de usuario y la respuesta se parsea por bloques ```` ```python ````.
- `scripts/mohh_models.py download` consulta el tamaño remoto y **falla antes de
  descargar** si la raíz persistente no tiene cuota (margen del 15 %). Para
  reproducibilidad exacta, fija la revisión:
  `python scripts/mohh_models.py download <modelo> --revision <sha>`.

### 1.4 Cómo elegir con evidencia de *esta* tarea

Los benchmarks de código no predicen el anclaje (repetir el mismo operador) ni
la tasa de operadores válidos en LLaMEA. Para decidir entre dos perfiles, corre
el mismo presupuesto y semilla con cada uno y compara:

```bash
# Dos corridas cortas, mismo seed y budget, en la misma GPU.
PROFILE=coder30b-awq BUDGET=50 SEED=42 EXPERIMENT_NAME=bakeoff_30b \
  bash slurm/llamea-all.sh

# Cuando terminen, tabla comparativa (validez, distintos, HV, tokens, latencia).
python scripts/analyze_llamea_runs.py 'results/searches/bakeoff_*' \
  --summary --csv results/bakeoff.csv
```

La tabla `--summary` es la evidencia que importa: `validos`, `distintos`,
`mejor HV`, `tokens/eval` y `s/llamada`. Un modelo con mejor HumanEval pero que
repite el 90 % de las veces pierde frente a uno más chico y diverso.

---

## 2. Servir el modelo

### A) vLLM (cluster, partición GPU) — recomendado

```bash
# Servidor persistente (opcional; el default es el job único de la sección B
# del README: `bash slurm/llamea-all.sh`).
sbatch slurm/llamea-vllm-server.slrm                 # perfil big-quant (32B AWQ)
PROFILE=coder30b-fp8 sbatch --constraint=H100 slurm/llamea-vllm-server.slrm
PROFILE=coder30b-awq sbatch --constraint=A100 slurm/llamea-vllm-server.slrm
```

Variables de entorno del `.slrm` (todas opcionales):
`PROFILE` (`big-quant`; `PROFILE=list` imprime el catálogo), `MODEL`,
`CONSTRAINT`, `PORT` (8000), `GPU_MEM_UTIL` (0.9), `MAX_MODEL_LEN`,
`TENSOR_PARALLEL`, `VLLM_ARGS` (del perfil, p. ej. `--quantization awq_marlin`
o `--kv-cache-dtype fp8`) y `EXTRA_VLLM_ARGS` (override del usuario). El script elige
el modelo con `slurm/llamea-gpu-profile.sh`, verifica los pesos en la raíz
persistente (`python scripts/mohh_models.py root`) y deja el modelo cargado
hasta el time limit (30 días).

### B) Servidor local de desarrollo (sin vLLM)

```bash
pip install -e ".[local-server]"
python scripts/local_openai_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8001
```

Implementa la misma interfaz (`/v1/models`, `/v1/chat/completions`) con
`transformers`; sirve para probar el pipeline completo en una GPU pequeña. Es
más lento que vLLM y no tiene batching; úsalo solo para pruebas.

### C) Modelos ya descargados en Ollama (`:11434`)

Ollama expone la misma API OpenAI-compatible, así que sirve modelos **ya
descargados** sin bajar nada nuevo ni montar vLLM:

```bash
ollama list                      # inventario local (GGUF cuantizados)
ollama ps                        # qué está cargado y cuánto va a GPU/CPU

MODEL=qwen2.5-coder:7b-instruct BASE_URL=http://127.0.0.1:11434/v1 \
  NO_DOWNLOAD=1 BUDGET=30 NAME=qwen25coder7b_moead_b30 \
  ./scripts/llamea_search.sh
```

Notas:

- Ollama reparte el modelo entre GPU y CPU según la VRAM libre. Si otro proceso
  ocupa la GPU (p. ej. un `llama-server` propio) el modelo puede caer a ~86 %
  CPU y bajar a ~7 tokens/s; con la GPU libre va entero a GPU.
- El contexto por defecto suele ser 4096 tokens: suficiente para el prompt de
  LLaMEA (~1.5-2k), pero no lo subas sin mirar la VRAM.
- El campo `usage` de Ollama trae `prompt_tokens`/`completion_tokens`, así que la
  contabilidad de tokens del pipeline funciona igual.
- Los modelos de razonamiento puro (p. ej. que llenan `reasoning_content`) no
  sirven para este bucle: se gastan el presupuesto de tokens razonando y dejan
  el `content` vacío, sin bloque de código que parsear.
- `qwen2.5-coder:7b-instruct` es el único del inventario que se comporta como
  modelo de búsqueda: 21/30 operadores válidos y 24 códigos distintos en 30
  generaciones, con un mejor HV de train de **0.583228** en el marco GE frente a
  0.174–0.225 de la línea base clásica medida en el mismo marco. Los detalles
  (y por qué el texto del feedback cambia el resultado) están en
  [`docs/llamea-prompts/pruebas-locales.md`](docs/llamea-prompts/pruebas-locales.md).

> El HV que reporta el bucle usa por defecto el **marco de la evaluación GE**
> (frente consolidado del panel de referencia y de todos los candidatos ya
> evaluados, normalizado con nadir `[1.1]*K`), así que los números son
> comparables con la tabla de la búsqueda gramatical. `--stationary-frame`
> recupera el marco fijo del panel y `--no-normalize` el HV crudo.

---

## 3. Conectar LLaMEA al modelo

```bash
python -m mohh.llamea.run \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://<nodo>:8000/v1 \
  --wait-server \
  --budget 50 --n-parents 1 --n-offspring 1 \
  --solver MOEAD --train-instances 8 \
  --mo-generations 300 --mo-population-size 105 \
  --eval-test --test-instances 8
```

- `--wait-server`: espera a que el servidor responda (útil en SLURM).
- `--eval-test`: evalúa el mejor operador en `datasets/mqap/test` al terminar.
- `--no-download`: no descarga el modelo si falta (nodos de cómputo).

En SLURM: `slurm/llamea-generate.slrm` hace esto mismo; se le pasa
`LLAMEA_SERVER_HOST=<nodo>` o `LLAMEA_SERVER_JOBID=<jobid>` y variables como
`BUDGET`, `TRAIN_INSTANCES`, `TEST_INSTANCES`, `EVAL_TEST=1`, `SOLVER`,
`MO_GENERATIONS`, `MO_POPULATION_SIZE`, `TEMPERATURE`, `N_PARENTS`,
`N_OFFSPRING`, `EXPERIMENT_NAME`.

---

## 4. Calidad del modelo / de la búsqueda

- `temperature=0.8` por default en `VLLM_LLM` (balance exploración/explotación);
  subir para más diversidad, bajar a 0.2-0.4 para operadores más conservadores
  (`--temperature` o `TEMPERATURE`; `scripts/llamea_search.sh` usa 0.2 para
  modelos chicos locales). El job del cluster usa **0.7 + top_p 0.8 + top_k 20 +
  repetition_penalty 1.05**, los valores que recomienda Qwen para sus modelos
  Coder. `top_k` y `repetition_penalty` van en `extra_body` (extensiones de
  vLLM); `--sampling-repetition-penalty` es distinto de `--repeat-penalty`.
- `--operator-memory N` recuerda en el prompt los últimos `N` mecanismos
  generados y pide no repetirlos. Ataca el anclaje observado (108 copias del
  mismo operador en `b200`) antes de gastar una evaluación; el job del cluster
  usa `N=12`. `0` lo desactiva.
- `budget` = número de llamadas al LLM / generaciones (default 1000 en el CLI y
  en el job; corridas serias 100-1000; con caché se pueden lanzar 1000 sin pagar
  reevaluaciones).
- `n_parents`/`n_offspring`: `1/1` = un algoritmo por generación (por defecto);
  subir para búsqueda poblacional.
- Modelos muy chicos (≤1.5B) generan código inválido con frecuencia: el fitness
  lo rechaza con feedback, pero la búsqueda aprovecha poco. En el cluster usar
  7B+.
- **Evidencia local (0.5B)**: en 9 corridas, 61/128 operadores fueron válidos
  (HV>0) y 13 distintos; a `temperature=0.2-0.25` el modelo reproduce el ejemplo
  OX del prompt (válido), a 0.6 casi todo sale inválido, y varios "válidos" son
  degenerados (permutación aleatoria). Detalle en
  `docs/llamea-prompts/pruebas-locales.md`. Para inspeccionar una corrida:
  `python scripts/analyze_llamea_runs.py` (recorre `results/**/exp-*`) o
  `python scripts/analyze_llamea_runs.py 'results/searches/**/exp-*' --list-codes`.
- **`-Coder` a 0.5B tampoco ayuda**: `Qwen2.5-Coder-0.5B-Instruct` da el mismo
  mejor HV que el generalista (el OX del ejemplo, copiado literal) y gasta ~17 %
  más tokens (respuestas más largas). El salto de calidad está en el tamaño
  (≥7B), no en el apellido del modelo a igual tamaño.
- El bucle tolera código malo sin caerse: `NoCodeException` en lugar de
  `IndexError`, score 0 con feedback y un *timeout* por llamada/instancia
  (`_call_with_timeout`). Aun así, conviene no bajar de 7B en el cluster.
- El prompt se adapta al solver elegido (`--solver`) y aporta conocimiento de
  dominio (familias de operadores OX/PMX/PBX/ERX/CX y qué preservan), en vez de
  listar los primitivos de la gramática; ver `docs/llamea-prompts/`.
- La búsqueda alterna dos operadores, `refine` (explotar) y `redesign`
  (explorar); `--refine-only` usa solo el refinamiento.

### Comando listo: 100 generaciones

Servidor local de desarrollo (misma API que vLLM):

```bash
python scripts/local_openai_server.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --port 8001
```

Búsqueda (guarda todo en `results/searches/<nombre>/`):

```bash
./scripts/llamea_search.sh
# equivale a:
python -m mohh.llamea.run \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --base-url http://localhost:8001/v1 \
    --budget 100 --solver MOEAD --train-instances 8 \
    --mo-generations 30 --mo-population-size 30 --temperature 0.2 \
    --results-dir results/searches --experiment-name qwen05b_moead_b100 \
    --eval-test --save-bounds
```

En el cluster (**un solo comando**, recomendado: descarga en el login y envía un
job que sirve el 32B AWQ y corre la búsqueda):

```bash
PROFILE=big-quant BUDGET=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
```

Alternativa con un servidor ya cargado (el cliente solo necesita el ID):

```bash
BASE_URL=http://gpu-nodo:8000/v1 NO_DOWNLOAD=1 WAIT_SERVER=1 EVAL_TEST=1 \
CATEGORY=searches NAME=qwen32b_moead_b100 \
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ ./scripts/llamea_search.sh
```

Encadenado servidor + búsqueda en dos jobs (reutiliza el servidor):

```bash
PROFILE=h100-fp16 BUDGET=200 EVAL_TEST=1 \
  EXPERIMENT_NAME=qwen32b_moead_b200 bash slurm/llamea-full-chain.sh
```

Cada corrida deja `run.json` (configuración + comando exacto),
`llamea_best_operator.py`, `llamea_logs/exp-*/report.md`, `evolution.csv` y
`evolution.png`. Para comparar corridas: `python scripts/analyze_llamea_runs.py`.

### Corridas largas (1000 generaciones)

```bash
BUDGET=1000 NAME=qwen05b_moead_b1000 EVAL_TEST=1 EXTRA="--save-bounds" \
  ./scripts/llamea_search.sh
```

Con el servidor local y 0.5B cada generación tarda ~2-3 s (casi todo es la
llamada al LLM) y consume ~1.9k tokens: ~50 min y ~1.9M tokens por corrida de
1000. El caché por código evita repetir el MOEA interno cuando el LLM devuelve
el mismo operador, pero **no** ahorra la llamada al LLM (el modelo sigue
generando). Con 7B el tiempo por generación sube ~5-10x.

Para no gastar en vano:

- `EXTRA="--no-example"` explora más (sin ancla) pero con 0.5B la tasa de
  validez baja a ~80 %.
- `MAX_TOKENS=512` acota la salida (el default del servidor suele ser 512).
- Si el mejor HV no cambia en decenas de generaciones, la corrida está
  estancada: revisa `report.md` (sección de tokens) y `generation_summary.jsonl`
  antes de lanzar otra.

---

## 5. Recetas de descarga

El cluster usa una **raíz persistente** (LUSTRE/`$SCRATCH`, nunca `$HOME`) y el
CLI `scripts/mohh_models.py`, que registra hashes y un manifiesto. La descarga
se hace en el **login** (tiene internet); los nodos de cómputo no.

```bash
python scripts/mohh_models.py root      # mostrar la raíz persistente
python scripts/mohh_models.py list      # modelos descargados + tamaño

# Default del cluster (32B AWQ, cabe en A10/A100/H100)
python scripts/mohh_models.py download Qwen/Qwen2.5-Coder-32B-Instruct-AWQ

# H100: 32B sin cuantizar (mayor calidad, ~65 GB)
python scripts/mohh_models.py download Qwen/Qwen2.5-Coder-32B-Instruct

# A100 40 GB: 14B fp16
python scripts/mohh_models.py download Qwen/Qwen2.5-Coder-14B-Instruct

# Verificar / smoke test (sin descargar de nuevo)
python scripts/mohh_models.py verify Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
python scripts/mohh_models.py smoke  Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
```

El camino corto es `bash slurm/llamea-all.sh`: verifica/descarga el modelo del
perfil en el login y envía el job único que lo sirve y corre la búsqueda.

Descargar a un directorio propio (útil si la raíz persistente es chica; luego
pasar esa ruta como `--model`):

```bash
python scripts/ensure_model.py Qwen/Qwen2.5-7B-Instruct \
    --local-dir /LUSTRE/<miusuario>/models/Qwen2.5-7B-Instruct
```
