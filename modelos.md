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

## 1. VRAM de referencia (fp16, sin KV cache extra)

| Modelo | Parámetros | VRAM fp16 | Caso de uso |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | ~1 GB | **Pruebas locales / smoke test** |
| Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | ~3.3 GB | Pruebas locales de código si hay ≥4 GB libres (no cabe junto a otro modelo que ocupe la GPU) |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | Pruebas rápidas (cabe en 12 GB) |
| **Qwen/Qwen2.5-Coder-7B-Instruct** | 7B | ~15 GB | **Recomendado en 24 GB**: escribe código correcto |
| Qwen/Qwen2.5-7B-Instruct | 7B | ~15 GB | **Default en 24 GB**: buen balance |
| Qwen/Qwen2.5-Coder-14B-Instruct | 14B | ~29 GB | A100 40 GB / H100 |
| **Qwen/Qwen2.5-Coder-32B-Instruct-AWQ** | 32B | ~19-21 GB | **Default del cluster**: cabe en A10 24 GB, A100 40 GB y H100 80 GB |
| **Qwen/Qwen2.5-Coder-32B-Instruct** | 32B | ~65 GB | **H100 80 GB**: el de más calidad |
| Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | 32B | ~18 GB | Alternativa cuantizada (GPTQ) |
| Meta-Llama-3.1-8B-Instruct | 8B | ~16 GB | 24 GB |
| Meta-Llama-3.1-70B-Instruct | 70B | ~140 GB | 50 GB: solo cuantizado agresivo o `--tensor-parallel-size 2` |
| deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct | 16B MoE (2.4B act.) | ~31 GB (AWQ ~9 GB) | 24 GB: alternativa fuerte en código |

Por qué `-Coder` y no solo "más grande": en las corridas locales el fallo típico
del 0.5B fue **escribir código degenerado pero válido** (`return x`,
`np.array(x, copy=False)`, truncamientos), no proponer malas ideas. Un modelo
afinado en código del mismo tamaño escribe mejor código; por eso
`Qwen2.5-Coder-7B-Instruct` es la primera opción antes que un genérico más
grande.

Notas:

- vLLM reserva memoria para el KV cache. Para prompts largos (el bucle evolutivo
  manda código + feedback de varias generaciones) vigilar
  `--gpu-memory-utilization` (default 0.9). Si hay OOM: bajarlo
  (`GPU_MEM_UTIL=0.8` en el `.slrm`) o acortar `--max-model-len`
  (`MAX_MODEL_LEN=8192`).
- **Elección por VRAM disponible**: el cluster tiene tres GPUs y
  `slurm/llamea-gpu-profile.sh` elige el modelo más grande que cabe en cada una:

  | GPU | VRAM | `PROFILE` | Modelo |
  |---|---|---|---|
  | A100 | 40 GB | `big-quant` (default) | `Qwen2.5-Coder-32B-Instruct-AWQ` (~19-21 GB) |
  | H100 | 80 GB | `h100-fp16` | `Qwen2.5-Coder-32B-Instruct` fp16 (~65 GB) |
  | A100 | 40 GB | `a100-fp16` | `Qwen2.5-Coder-14B-Instruct` fp16 (~29 GB) |
  | A10 | 24 GB | `a10-quant` | `Qwen2.5-Coder-14B-Instruct-AWQ` (~10 GB) |

  `big-quant` es el default porque es el **mayor número de parámetros** que
  cabe con contexto largo en las GPUs grandes (A100 40 GB y H100 80 GB). En una
  A10 (24 GB) usar `PROFILE=a10-quant`; el 32B AWQ no deja KV cache suficiente
  para 16k de contexto. Si el job cae en una H100, `PROFILE=h100-fp16` da la
  versión sin cuantizar.
- **En esta máquina de pruebas (RTX 3060, 12 GB)**: usar `Qwen2.5-0.5B/1.5B`.
- Los modelos `-Instruct` tienen chat template: el prompt de LLaMEA va como
  mensaje de usuario y la respuesta se parsea por bloques ```` ```python ````.

---

## 2. Servir el modelo

### A) vLLM (cluster, partición GPU) — recomendado

```bash
# Servidor persistente (opcional; el default es el job único de la sección B
# del README: `bash slurm/llamea-all.sh`).
sbatch slurm/llamea-vllm-server.slrm                 # perfil big-quant (32B AWQ)
PROFILE=h100-fp16 sbatch --constraint=H100 slurm/llamea-vllm-server.slrm
```

Variables de entorno del `.slrm` (todas opcionales):
`PROFILE` (`big-quant`), `MODEL`, `CONSTRAINT`, `PORT` (8000), `GPU_MEM_UTIL`
(0.9), `MAX_MODEL_LEN`, `TENSOR_PARALLEL`, `EXTRA_VLLM_ARGS`. El script elige
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
  (`--temperature` o `TEMPERATURE`; `scripts/llamea_search.sh` usa 0.2).
- `budget` = número de llamadas al LLM / generaciones (default 100; corridas
  serias 100-200; con caché se pueden lanzar 1000 sin pagar reevaluaciones).
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
