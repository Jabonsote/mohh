# cambios.md — Integración LLaMEA + vLLM (buscar operadores con LLM)

Objetivo: usar **LLaMEA** (LLM como operador de variación evolutivo) para el
**mismo problema** que MOHH-R: diseñar operadores de variación (crossover) para
MOEAs sobre mQAP, con los **mismos conjuntos** `datasets/mqap/train` y
`datasets/mqap/test`, evaluados por **hipervolumen**.

Puntos clave del diseño:

- **No se toca el núcleo de evolución gramatical.** Toda la integración vive
  en `src/mohh/llamea/`; la búsqueda GE (`mohh-generate`, `HH_Operator`,
  `HyperHeuristic`, gramáticas) queda intacta y con su comportamiento original.
- Se **reutiliza la lógica actual** de MOHH-R para evaluar: el mismo `MOSolver`
  interno (MOEA/D, NSGA-II o SMS-EMOA) y la misma métrica de hipervolumen.
  Lo único que cambia es el origen del operador: en vez de decodificar una
  gramática, el LLM escribe una función `crossover(x, y)`.
- El modelo es **local**: vLLM lo deja cargado en GPU y LLaMEA se conecta por
  API OpenAI-compatible. Antes de correr se **verifica/descarga** el modelo.

---

## 1. Qué se agregó

### Paquete `src/mohh/llamea/`

| Archivo | Contenido |
|---|---|
| `operator.py` | `LLMOperator`: un `Crossover` de pymoo que evalúa el código del LLM. Espeja la semántica de `HH_Operator` (mismo shape de entrada/salida y segundo descendiente `child[::-1]`), pero recibe un namespace explícito. **Es el único puente con el MOEA.** |
| `fitness.py` | `OperatorEvaluator` (fitness picklable) + `build_evaluator()` + `compile_operator()` + `init_params()`. Reutiliza `MOSolver`, `Instance` y `compute_hypervolume` de MOHH-R. Robusto ante código del LLM: namespace de ejecución con `np`/`numpy` y nombres de `typing`; captura cualquier excepción por instancia (score 0 + feedback); `_call_with_timeout` (SIGALRM) acota validación (10 s) e instancia (120 s) contra `while True`; `__getstate__` lo hace picklable para `--max-workers > 1`. |
| `statistics.py` | Estadística pareada compartida por la evaluación final y `compare_operators.py`: diferencias pareadas, Wilcoxon de dos colas, IC95, `dz`, diferencia mínima detectable y ajuste de Holm. |
| `vllm.py` | `VLLM_LLM` (conector OpenAI-compatible → servidor vLLM local), `ensure_model()` (verifica en cache HF y descarga si falta), `model_is_available()`, `wait_for_server()` (espera a que el servidor arranque en SLURM). `extract_algorithm_code` exige `def crossover` y lanza `NoCodeException` si falta (evita un `IndexError` interno de LLaMEA). |
| `prompts.py` | `ROLE_PROMPT`, `make_task_prompt(solver)` (contrato `crossover(x, y)` para mQAP + conocimiento de dominio de familias de operadores, adaptado al MOEA interno: MOEA/D, NSGA-II o SMS-EMOA), `EXAMPLE_PROMPT` (OX), `OUTPUT_FORMAT_PROMPT` y `make_operators()` (`refine`/`redesign`). Diseño basado en evidencia: ver `docs/llamea-prompts/`. |
| `export_prompts.py` | Exporta los prompts finales a Markdown (`python -m mohh.llamea.export_prompts docs/llamea-prompts/prompt-final.md`). |
| `parsing.py` | Parseo/reparación de respuestas: quita `<think>`, extrae el bloque de código, cierra brackets truncados, elimina `if __name__ == '__main__'`. |
| `run.py` | CLI `python -m mohh.llamea.run` / `mohh-llamea` (1 algoritmo por generación por defecto; opción poblacional; verificación final en test). |

### Scripts

| Script | Descripción |
|---|---|
| `slurm/llamea-all.sh` / `llamea.sh` | **Comando único**: prepara el ambiente, descarga el modelo en el login y envía un solo job GPU con vLLM + búsqueda. |
| `slurm/llamea-single.slrm` | Job único: sirve el modelo en `127.0.0.1`, corre LLaMEA, evaluación test y comparación opcional; falla rápido si faltan los pesos. |
| `slurm/llamea-vllm-server.slrm` | Servidor persistente (avanzado): carga el modelo en GPU y lo sirve por `/v1`. |
| `slurm/llamea-generate.slrm` | Búsqueda contra un servidor persistente. Acepta `LLAMEA_SERVER_HOST` o `LLAMEA_SERVER_JOBID`. |
| `slurm/llamea-full-chain.sh` | Cadena de dos jobs (avanzado). Por defecto redirige al job único; `ALLOW_TWO_JOBS=1` la fuerza. |
| `scripts/ensure_model.py` | Verifica/descarga el modelo en el cache de HuggingFace (`--check` solo verifica). |
| `scripts/local_openai_server.py` | Servidor OpenAI-compatible mínimo (transformers + FastAPI) para **pruebas locales sin vLLM** (misma interfaz `/v1/models` y `/v1/chat/completions`). |
| `scripts/analyze_llamea_runs.py` | Resume los directorios `exp-*` que deja LLaMEA: validez, mecanismos conductuales, errores, HV, tokens y latencia (`--summary`, `--csv`, `--list-codes`). |
| `scripts/analyze_diversity.py` | Mapa conductual, novedad, familias y cobertura MAP-Elites de una corrida (`diversity.json` o `diversity.jsonl`), con `--plot`/`--csv`. |

### `tests/test_llamea.py`

35 pruebas (+1 de conexión viva, *opt-in*): adaptador de operadores, que el
núcleo GE **no** cambió, compilación y validación de código (incluye `np`/`typing`
sin importar y rechazo de bucles infinitos), fitness (acepta/rechaza y captura de
fallos por instancia), picklabilidad del fitness, conector vLLM (sin servidor),
contrato del extractor (`NoCodeException`), parseo/reparación, diseño de prompts
(contrato, conocimiento de dominio, no basado en gramática, operadores
refine/redesign) y un bucle LLaMEA **de extremo a extremo con un LLM simulado**
(sin red). La prueba de conexión viva se activa con `LLAMEA_SERVER_URL`.

### `pyproject.toml`

Extras opcionales para instalar de forma reproducible:

- `[llamea]` → `llamea==1.4.0`, `openai`, `huggingface_hub` (búsqueda con LLM).
- `[vllm]` → `vllm` (servidor en GPU, pesado).
- `[local-server]` → `transformers`, `fastapi`, `uvicorn` (servidor de pruebas).
- `[dev]` → `pytest`.
- Entry point nuevo: `mohh-llamea`.

### `docs/llamea-prompts/` (diseño de prompts)

Carpeta con el diseño de prompts **basado en evidencia** (LLaMEA, FunSearch,
EoH, ReEvo, ...) y las citas: `README.md`, `diseno-de-prompts.md`,
`prompt-final.md` (texto exacto de los prompts), `referencias.md` (BibTeX),
`pruebas-locales.md` (corridas reales con un modelo local chico: resultados,
fallas típicas y arreglos de robustez) y `pdfs/` con los papers descargados.
`prompt-final.md` se regenera con
`python -m mohh.llamea.export_prompts docs/llamea-prompts/prompt-final.md`.

---

## 2. Cómo se conecta con la evolución gramatical (sin modificarla)

La búsqueda GE construye el operador así:

```
load_operator("operator_template", "HH_Operator", operator=fenotipo, ...)
  → MOSolver.crossover = HH_Operator(fenotipo)   # evalúa una expresión de la gramática
```

La búsqueda LLaMEA hace exactamente lo mismo salvo el último paso:

```
solver.crossover = LLMOperator(namespace)        # evalúa crossover(x, y) definido por el LLM
solver.load_operator("mutation", "NullMutation") # idéntico a GE
solver.load_instance(inst); solver.start_model(model_name)
```

Es decir: **mismo solver, mismo problema, misma métrica y mismos conjuntos**. El
cambio es puramente local a `mohh.llamea` (inyección del operador). Esto también
implica que un operador descubierto por LLaMEA se puede benchmarkear después con
`mohh-evaluate` (mismos frentes `.pof` y CSVs), igual que los de GE.

`build_evaluator()` inicializa `Params` con los **defaults de MOHH-R**
(`mohh.generation.params.params_dict`: `DATASET=mqap`, `PROBLEM_NAME=QAP`, ...),
así que `instance_paths(train=True/False)` lista exactamente los mismos
`datasets/mqap/{train,test}`.

### Alineación con MOHH-R: mismos problemas, mismos algoritmos, misma lógica

| Aspecto | MOHH-R (GE) | LLaMEA |
|---|---|---|
| Problema | mQAP (permutaciones, `f_k`) | **igual** |
| Datos | `datasets/mqap/train` (búsqueda), `test` (eval) | **iguales** (`--train-instances` / `--eval-test`) |
| Algoritmos | `MOEAD`, `NSGAII`, `SMSEMOA` (`--models`) | **los mismos** (`--solver`), y el prompt se adapta al elegido |
| Métrica | hipervolumen medio | **igual** (`compute_hypervolume`) |
| Operador | expresión de gramática (`HH_Operator`) | `crossover(x, y)` (`LLMOperator`) |
| Primitivos | `reverse`, `order_based`, ... (`naturals.bnf`) | **no** se usan en el prompt; el LLM escribe numpy libre (los primitivos siguen disponibles en el namespace por compatibilidad) |

El prompt (`make_task_prompt(solver)`) nombra y describe el MOEA interno concreto
(por eso con `--solver NSGAII` el prompt habla de NSGA-II, no de MOEA/D) y aporta
conocimiento de dominio sobre familias de operadores de permutación (orden,
posición, adyacencia, ciclos). **No** está derivado de la gramática: es un diseño
basado en evidencia, documentado en `docs/llamea-prompts/`. El operador puede
devolver una permutación o la tupla `(child1, child2)`.

Igual que en GE (`--models MOEAD,NSGAII,SMSEMOA` corre una búsqueda por modelo),
aquí se corre **una búsqueda por `--solver`**; para cubrir varios algoritmos,
lanzar una búsqueda por cada uno.

---

## 3. Instalación (replicable)

### Cluster (ixachi)

```bash
cd /LUSTRE/<miusuario>/MOOHH-R
bash slurm/ixachi-setup-env.sh                 # mohh + extra [llamea]
WITH_VLLM=1 bash slurm/ixachi-setup-env.sh     # además vLLM (pesado)
```

El script usa `pip install -e ".[llamea]"` (no `--no-deps`), de modo que
`llamea==1.4.0` y sus dependencias quedan resueltas por pip. Al terminar verifica
`import mohh`, `mohh-generate -h`, `mohh-evaluate -h` y `mohh-llamea -h`.

### Local (esta máquina, sin tocar el entorno global)

```bash
uv venv /tmp/opencode/llamea-venv
uv pip install --python /tmp/opencode/llamea-venv/bin/python -e ".[llamea,local-server]" pytest
```

> En esta máquina el `torch` del entorno conda base está roto por un conflicto de
> `libnvJitLink`; por eso las pruebas locales se hacen en un venv aparte con un
> `torch` compatible (`/tmp/opencode/llamea-venv`).

---

## 4. Flujo de uso

### A) Prueba local (sin GPU grande ni vLLM)

```bash
# 1. Descargar un modelo pequeño (una vez)
python scripts/ensure_model.py Qwen/Qwen2.5-0.5B-Instruct

# 2. Servidor local OpenAI-compatible (en otra terminal)
python scripts/local_openai_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8001

# 3. Búsqueda LLaMEA (1 algoritmo por generación)
python -m mohh.llamea.run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --base-url http://127.0.0.1:8001/v1 \
  --budget 20 --train-instances 8 --mo-generations 30 --mo-population-size 30 \
  --eval-test --test-instances 8
```

### B) Cluster: servidor vLLM + búsqueda (dos jobs) — histórico

> **Ya no es el camino recomendado.** Usa `bash slurm/llamea-all.sh`: prepara
> el ambiente si falta, descarga el modelo en el login y envía **un solo job**
> con vLLM + búsqueda. Esta sección queda como referencia para reutilizar un
> servidor ya cargado (`ALLOW_TWO_JOBS=1`).

```bash
# 1. Servidor con el modelo cargado en GPU (partición gpu)
MODEL=Qwen/Qwen2.5-7B-Instruct sbatch slurm/llamea-vllm-server.slrm
squeue --user=$USER          # anotar el JobID / nodo

# 2. Búsqueda (partición ixachi), autodetectando el nodo del servidor
LLAMEA_SERVER_JOBID=<jobid-vllm> MODEL=Qwen/Qwen2.5-7B-Instruct \
EVAL_TEST=1 BUDGET=50 sbatch slurm/llamea-generate.slrm
```

O ambos jobs de una vez (el segundo espera al primero):

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct bash slurm/llamea-full-chain.sh
```

Salidas en `results/<EXPERIMENT_NAME>/`:
`llamea_best_operator.py` (código del mejor operador) y
`llamea_best_info.json` (modelo, parámetros, fitness de train y de test).

### C) Poblacional (opcional)

Por defecto es **un algoritmo por generación** (`(1+1)-EA`,
`--n-parents 1 --n-offspring 1`). Para más por generación:

```bash
python -m mohh.llamea.run --n-parents 4 --n-offspring 4 --budget 50 ...
# o en SLURM: N_PARENTS=4 N_OFFSPRING=4 sbatch slurm/llamea-generate.slrm
```

---

## 5. Decisiones de diseño

- **Prompt diseñado por evidencia, no por gramática**: el LLM genera la función
  `crossover(x, y)` directa. El prompt (rol experto + tarea precisa + familias de
  operadores + "pensar y luego codificar") retoma LLaMEA/FunSearch/EoH/ReEvo; ver
  `docs/llamea-prompts/`. La búsqueda GE (`mohh-generate`) queda intacta como vía
  alternativa.
- **Exploración/explotación**: LLaMEA recibe dos operadores, `refine` y
  `redesign` (inspirados en EoH y en el "refine or redesign" de LLaMEA);
  `--refine-only` desactiva el rediseño.
- **Prompt adaptado al algoritmo**: `make_task_prompt(solver)` describe el MOEA
  interno que se usará (`MOEAD`/`NSGAII`/`SMSEMOA`), igual que la búsqueda GE
  usa el modelo en el que se evaluará el operador.
- **Mismo fitness que GE**: hipervolumen medio sobre instancias mQAP. Durante la
  búsqueda se usan **train**; `--eval-test` verifica **test** al final (nunca
  durante la búsqueda, para medir generalización).
- **Un algoritmo por generación por defecto** (`(1+1)-EA`, elitismo); flags
  `--n-parents/--n-offspring` habilitan poblacional.
- **Modelo siempre verificado**: `ensure_model()` corre antes de cada búsqueda;
  los `.slrm` lo ejecutan explícitamente (`--check` en nodos sin internet).
- **vLLM para no pagar carga por request**: el modelo queda cargado en GPU y el
  bucle evolutivo solo manda HTTP a `/v1`.
- **Fitness picklable** (`OperatorEvaluator`, no un closure): permite
  `--max-workers > 1` y re-inicializa `Params` de forma idempotente en cada
  proceso.

---

## 6. Pruebas realizadas

- **Suite completa**: `97 passed, 1 skipped` (35 de LLaMEA + 32 de SLURM +
  30 de normalización/ranking/reportes) en el venv de desarrollo. El *skip* es la
  prueba de conexión viva (opt-in).
- **Los 3 algoritmos con el mismo operador**: un operador que usa los primitivos
  (`order_based(fill_first_occurring, preserve_segments, x, y)`) se evaluó con
  `MOEAD`, `NSGAII` y `SMSEMOA` sin errores (misma lógica interna que GE).
- **9 corridas locales** con `Qwen/Qwen2.5-0.5B-Instruct` servido por
  `scripts/local_openai_server.py` (misma interfaz que vLLM): 128 operadores
  generados, 61 válidos (HV>0), 13 válidos distintos, con evaluación train y
  test. Incluye `--refine-only`, temperaturas 0.2/0.25/0.6 y una corrida
  paralela (`--n-offspring 2 --max-workers 2`). Detalle completo (tablas,
  ejemplos de operadores y verificación independiente en test) en
  `docs/llamea-prompts/pruebas-locales.md`.
- **Conexión viva**: `pytest tests/test_llamea.py -k live` con
  `LLAMEA_SERVER_URL` apuntando al servidor local pasa (genera un operador válido
  en ≤5 intentos).
- **Instalación**: `uv pip install -e ".[llamea]"` resuelve sin conflictos.

Estas corridas destaparon y corrigieron 5 fragilidades reales (todas fuera del
núcleo de MOHH-R): nubes de `IndexError` de LLaMEA con respuestas sin
`def crossover`, caída del fitness si el operador fallaba en una instancia,
`List`/`np` sin importar, cuelgues por `while True` y fallo de *pickle* tras
construir instancias. Ver la tabla en `docs/llamea-prompts/pruebas-locales.md`.

---

## 7. Resultados, hipervolumen justo y reportes

Todos los artefactos viven bajo `results/` (nada en la raíz). La estructura y
las convenciones están en `results/README.md`:

```
results/{searches,tests,prompts,examples,legacy}/{experimento}/...
```

### 7.1 Hipervolumen justo y comparable con la evaluación GE

El fitness no usa `max(frente)*1.1` por frente. Replica la evaluación
gramatical (`src/mohh/evaluation/experiment.py`, función
`normalize_values`):

1. **Marco de normalización por instancia**: frente no dominado consolidado de
   todo lo evaluado en esa instancia. En la evaluación GE son *todas* las
   configuraciones (operadores × repeticiones); aquí son los operadores de
   referencia (OX, PBX, PMX, CX, ERX; `mohh/llamea/reference.py`) **más todos
   los candidatos ya evaluados** (marco `ge`, por defecto). Si un candidato
   amplía la envolvente, los HV de los candidatos anteriores se recalculan a
   partir de los frentes cacheados (solo hipervolumen, sin repetir el MOEA).
2. **Normalización** `(f - vmin) / (vmax - vmin)` e **hipervolumen** contra el
   nadir fijo `[1.1] * K`, idénticos al original.
3. **Instancias sin escala** (`vmax == vmin` en algún objetivo): el marco no
   define escala y el HV es indefinido. El original las invalida (NaN); aquí se
   excluyen de la media y se registran en el `feedback` y en
   `llamea_best_info.json` (`degenerate_instances`).

Por qué el marco importa (medido con las funciones originales sobre las 8
instancias de train y el mejor operador guardado de
`results/searches/qwen25coder7b_moead_b30_feedback`,
`Qwen2.5-Coder-7B-Instruct`, 30 generaciones):

| Operador | Marco = solo panel | Marco GE (envolvente de todo) |
|---|---:|---:|
| OX | 0.426177 | 0.174201 |
| PBX | 0.504689 | 0.224239 |
| PMX | 0.390263 | 0.225029 |
| CX | 0.148689 | 0.091615 |
| ERX | 0.254144 | 0.102664 |
| mejor candidato | 4.433042 | **0.583228** |

Con el marco fijo el HV del candidato no tiene cota (4.43 frente a 0.43-0.50 del
panel) y no es comparable con la tabla GE. Con el marco GE, el mejor candidato
(0.583) supera a toda la línea base clásica medida en el mismo marco (0.17-0.23),
que es exactamente la comparación que se quiere publicar.

El marco es **por corrida** (incluye al mejor candidato de esa corrida), así que
el HV de un operador baja cuando otro mejor entra en la envolvente: la
comparación válida es entre filas de una misma tabla, no entre corridas. A 30
generaciones el 7B se ancla en el OX del ejemplo (mejor HV 0.396-0.425, PBX por
delante); con **100 generaciones** (`results/searches/qwen25coder7b_moead_b100_ge`)
sale del anclaje y encuentra un OBX con reparación de adyacencia que gana a toda
la línea base en el mismo marco, en train (0.511 frente a PBX 0.423) y en test
(0.215 frente a 0.209). Los números y el detalle están en
[`docs/llamea-prompts/pruebas-locales.md`](docs/llamea-prompts/pruebas-locales.md).

`--stationary-frame` recupera el marco fijo solo-panel; `--no-normalize` el HV
crudo; `--bounds-file` permite reutilizar límites externos (las esquinas de la
caja guardada semillan el marco GE) y `--save-bounds` guardarlos.

### 7.2 Ranking y feedback

`OperatorEvaluator` mantiene un archivo (`HVArchive`) de los HV evaluados y el
`feedback` que ve el LLM empieza con

```
Ranking: this crossover is #R of N evaluated so far by mean hypervolume
(1 = best; higher is better; HV ties are broken by recombination novelty).
Mean hypervolume (normalized with the consolidated front of the reference
operators and every candidate evaluated so far, exactly like the MOHH-R grammar
pipeline): ...
Best so far: ...
Reference operators in the same frame (higher is better): you are X% BELOW the
best classic PBX = 0.423080 [PBX ..., OX ..., PMX ..., CX ..., ERX ...]. Your
goal is to beat PBX.
Recombination novelty: 0.35 (0 = copies a parent, 1 = child unlike both parents)
Per-instance HV: [...]
```

El ranking es la señal que el prompt pide mejorar. Con `--max-workers > 1` cada
proceso lleva su vista parcial (documentado en el código).

**Mejora (feedback contra la línea base).** Antes el feedback solo comparaba al
candidato con otros candidatos: el modelo no sabía que OX/PBX existen ni si iba
perdiendo contra ellos, y por eso se anclaba en el OX del ejemplo sin saber que
PBX lo superaba. Ahora el feedback incluye el HV de los cinco clásicos medido en
el **mismo marco** y cuánto falta para superar al mejor (`reference_scores()`, sin
MOEA extra: es solo hipervolumen sobre frentes ya calculados). El archivo de
ranking sigue conteniendo solo candidatos (no se sembró con los clásicos), para no
cambiar la semántica de `HVArchive`.

### 7.2.1 Chequeo de degeneración y desempate por novedad

La validación “es una permutación” es demasiado débil: en la corrida de 1000
generaciones sin ejemplo, 22 de los operadores válidos distintos resultaron ser
`return np.array(x, copy=False)` (devuelven el primer padre), todos con el mismo
HV exacto. Para cortar ese modo de fallo:

- `recombination_novelty(crossover)` aplica el operador a 8 pares aleatorios
  (n ∈ {11, 16, 24, 50}) y devuelve `(ok, novedad, motivo)`. `novedad` es la
  distancia de Hamming normalizada media al padre más cercano.
- Si en todos los intentos devuelve un padre (o una vista: `np.shares_memory`
  con `x`/`y`), el candidato puntúa 0 **antes** de correr el MOEA interno y el
  feedback explica el motivo.
- `HVArchive.add(value, novelty, key=...)` desempata por novedad: dos operadores
  con el mismo HV se ordenan por cuánto recombinan. En el marco GE el archivo
  guarda la clave del código y consulta el HV actual al calcular el ranking, de
  modo que al ampliarse el marco (`archive.update`) todos los candidatos se
  reordenan con el valor refrescado.
- El `task_prompt` lo anuncia ("a cheap check … it scores 0 immediately") para
  que el modelo no pierda intentos en `return x` disfrazado.

### 7.2.2 Feedback de validación con diagnóstico

Antes, cualquier operador que no superara la validación recibía siempre el mismo
texto (`crossover must return a valid permutation …`). Un modelo capaz — pero
con un error aritmético concreto — no tenía forma de saber *qué* estaba mal y
repetía el mismo fallo: `Qwen2.5-Coder-7B-Instruct` (vía Ollama) gastó las 30
generaciones en un OX/PBX con un off-by-one en el relleno del segmento, 27 de
ellas con ese mismo mensaje genérico y **0/30 operadores válidos**
(`results/searches/qwen25coder7b_moead_b30`).

`validate_operator(crossover)` sustituye a la comprobación booleana y devuelve
`(ok, motivo)`, donde `motivo` es el feedback que se le pasa al LLM. Distingue y
describe:

- **excepción del código generado**, con la línea exacta que falló
  (`OPERATOR_FILENAME` se registra en `linecache` para que el traceback pueda
  mostrar la fuente):
  `crossover(x, y) raised AssertionError for n=11: ; the failing line is line 37: assert np.array_equal(np.sort(child1), np.arange(n))`.
- **longitud** incorrecta (`expected length 11, got length 10`).
- **huecos sin rellenar**: `1 value(s) outside range(n) (-1) - a -1 usually
  means a slot left unfilled by the copy loop; 8 missing value(s) (0, 1, 3, …)`.
- **duplicados** y valores fuera de rango, excluyendo los fuera de rango del
  recuento de duplicados para no mezclar dos diagnósticos.
- **mutación in-place de los padres** (`crossover modified its inputs for n=11:
  x and y are read-only…`), comparando `x`/`y` antes y después de la llamada.
- **vista de un padre** (`it is a view of a parent; the child must be a NEW
  array`): el algoritmo escribe en los descendientes, así que una vista
  corrompería la población. Esto hace que `return reverse(x)` (un primitivo de
  MOHH-R que devuelve `x[::-1]`) se rechace con explicación.
- **tupla de tamaño incorrecto** (`returned a tuple of 3 elements`).

También se mejoran los errores de compilación (`Code did not compile:
SyntaxError on line 1: expected ':'`).

### 7.3 Logging por generación y reportes

Se sustituye el logger de LLaMEA por `ResultsExperimentLogger`
(`mohh/llamea/reporting.py`), que escribe en
`results/<categoría>/<experimento>/llamea_logs/exp-*/`:

- `log.jsonl` y `conversationlog.jsonl`: todo lo que ya registraba LLaMEA.
- `generation_summary.jsonl`: una línea por generación (mejor/medio HV,
  válidos, errores, nombre del mejor, **tokens** acumulados y por generación) y
  una línea por generación en consola.
- `generations.csv`: la misma información por generación, lista para graficar o
  analizar en una hoja de cálculo.
- `evolution.csv`, `report.md` y `evolution.png`: reporte final; `report.md`
  incluye una sección con el consumo de tokens.
- `tokens.json` (en la raíz del experimento): acumulado, histórico por petición
  y `temperature`/`max_tokens` usados.

`run.json` guarda la configuración exacta y el comando reproducible.

Los tokens salen del campo `usage` que devuelve el servidor; `VLLM_LLM` los
acumula (`usage_snapshot()`) y, si el servidor no los reporta, los estima por
longitud (~4 caracteres/token). `--max-tokens` fija el máximo de salida por
petición (por defecto, el del servidor).

### 7.4 Caché de evaluaciones

En corridas largas el LLM repite el mismo código muchas veces (con 0.5B, OX 99
de 100 generaciones). `OperatorEvaluator` guarda el resultado por código
normalizado y reutiliza `score` y `per_instance` sin repetir el MOEA interno.
La repetición **sí** se registra en `HVArchive` (el ranking la cuenta) y el
feedback lo indica con `(reused cached evaluation)`. El caché es por proceso y
por corrida; no se comparte entre workers.

### 7.5 Scripts nuevos

| Script | Para qué |
|---|---|
| `scripts/llamea_search.sh` | Lanza una búsqueda con defaults (`budget=100`, 8 instancias, MOEA/D, temp 0.2) y la guarda en `results/<categoría>/`. |
| `scripts/compare_operators.py` | Comparación pareada de operadores con N repeticiones y el mismo marco (`results/evaluations/`). Ver 7.6. |
| `scripts/llamea_examples.py` | Evalúa OX/PBX/PMX/CX/ERX y una permutación aleatoria como línea base (`results/examples/reference_operators/`). |
| `scripts/llamea_prompt_ablations.sh` | Corre las variantes de prompt (ejemplo / sin ejemplo / esqueleto / refine-only / temp 0.6) en `results/prompts/`. |
| `scripts/analyze_llamea_runs.py` | Resume cualquier corrida bajo `results/` (válidos, distintos, errores). |

Line base medida (8 instancias, MOEA/D, 30/30, HV normalizado): PBX 0.505,
OX 0.426, PMX 0.390, ERX 0.254, aleatorio 0.235, CX 0.149.

### 7.6 Comparación justa entre operadores

La tabla de una corrida **no** es una comparación entre operadores: el marco GE
de una corrida incluye a sus propios candidatos, usa una sola semilla por
instancia y el "mejor por marco final" se elige con información de todos los
candidatos. `scripts/compare_operators.py` implementa la comparación válida:

- **Pareada**: el par (instancia, repetición) usa la misma semilla en todos los
  operadores, así que todos parten de la misma población inicial y se comparan
  diferencias, no medias independientes.
- **Un solo marco por instancia**: el frente no dominado consolidado de *todas*
  las configuraciones (todos los operadores × todas las repeticiones),
  normalizado `(f - vmin) / (vmax - vmin)` con nadir `[1.1] * K`, igual que
  `experiment.py`; las instancias sin escala se excluyen y se reportan.
- **Cabeza a cabeza**: cada pareja se renormaliza además con el marco de *esas
  dos* configuraciones (un "experimento GE de dos configuraciones"), porque un
  tercer operador muy bueno puede aplastar a los demás a HV ≈ 0 en el marco
  compartido.
- **Clásicos de dos hijos**: `classic2:ox` devuelve `(ox(x, y), ox(y, x))`, el
  mismo contrato que un operador que devuelve una tupla. Comparar un operador de
  dos hijos contra clásicos de un solo hijo (que reciben `child[::-1]` como
  segundo descendiente) no es justo.
- **Estadística**: N repeticiones (11 por defecto), victorias/derrotas,
  diferencia media, `dz`, IC95 de la diferencia y Wilcoxon pareado. Las
  comparaciones de cada tabla se ajustan con Holm (control de FWER), no con
  Bonferroni manual; con 11 pares el p mínimo alcanzable es 0.0010.
- **Fallos**: un operador que lanza una excepción en una instancia se cuenta
  como HV 0 (`--fail-policy zero`, default) y se reporta. Descartar el par haría
  que un operador roto pareciera mejor; `exclude` y `abort` quedan disponibles.
- **Protocolo GE**: `--solver` acepta una lista y `--mutation-both` evalúa con y
  sin `Swap_Mutation`; `--ge-protocol` activa el protocolo completo (3 solvers ×
  2 mutaciones × 300/105), compartiendo un solo marco consolidado. El default
  corto (MOEAD, 30/30, sin mutación) replica el bucle de búsqueda.
- **Reutilización**: `--save-fronts` / `--load-fronts` guardan y recargan los
  frentes, así se agregan operadores a una comparación ya corrida sin repetir el
  MOEA.

Resultado y lectura en `docs/llamea-prompts/comparacion-justa.md`.

### 7.7 `--mutation` ya no es un no-op

El núcleo GE tiene `with_mutation` comentado (`multiobjective.py:35`) y el arnés
cargaba siempre `solver.load_operator("mutation", "NullMutation")`: `--mutation`
no cambiaba nada (se vio al correr una ablación con y sin la bandera: frentes
idénticos bit a bit). Ahora `--mutation` inyecta `Swap_Mutation` (la mutación de
permutación que usa `experiment.py`, con `MOEAD_MUTATION_PROBABILITY = 0.05`) por
atributo del solver, igual que se inyecta el cruce del LLM; el núcleo sigue
intacto. Cubierto por `test_mutation_flag_is_not_a_noop`.

---

## 8. Notas y limitaciones

- vLLM es pesado y necesita GPU; en el cluster va en la partición `gpu`. Para
  pruebas locales está `scripts/local_openai_server.py` (más lento, misma API).
- El hipervolumen de LLaMEA usa la **misma normalización** que la generación GE
  (`vmin/vmax` del frente consolidado y nadir fijo `1.1`), así que los HV son
  comparables dentro del mismo conjunto de instancias. Entre solvers distintos
  las escalas siguen difiriendo; comparar siempre operador vs. operador con el
  mismo `--solver`.
- El LLM puede devolver código inválido (rechazado con fitness 0 y feedback);
  con modelos pequeños esto es frecuente — usar 7B+ en el cluster.
- **Válido ≠ bueno**: un operador que devuelve una permutación válida puede ser
  degenerado (p. ej. `np.random.permutation(n)`); el contrato lo acepta y el
  hipervolumen decide su calidad.
- **Código que no termina**: `_call_with_timeout` (SIGALRM) corta la validación
  (10 s) y cada instancia (`--instance-timeout`, 300 s por defecto). Solo
  funciona en el hilo principal de POSIX; si no aplica, el código se ejecuta sin
  límite (usa `--max-workers > 1`, donde joblib también puede aplicar
  `eval_timeout`).
- `--max-workers > 1` funciona (fitness picklable), pero cada worker reconstruye
  instancias; para presupuestos chicos dejarlo en 1. LLaMEA además capa
  `max_workers` a `n_offspring`, así que el paralelismo real requiere
  `--n-offspring > 1`.

---

## 9. Job único, modelos grandes y presión de diversidad (2026-10)

### 9.1 Un solo comando y un solo job

La cadena anterior (job vLLM + job de búsqueda + coordinación por nodo) se
reemplaza, como camino recomendado, por **un único job GPU** que:

1. verifica el modelo en la raíz persistente;
2. arranca vLLM en `127.0.0.1` en segundo plano;
3. corre `mohh.llamea.run` contra ese servidor (con `--wait-server`);
4. apaga el servidor y el sampler de GPU en el `trap EXIT`.

`slurm/llamea-all.sh` (alias `slurm/llamea.sh`) es el **único comando** desde
el login: en la primera ejecución crea `.conda-env` con vLLM si falta,
verifica/descarga el modelo (el login tiene internet; los nodos de cómputo no)
y envía `slurm/llamea-single.slrm`. El servidor y la búsqueda comparten nodo, así
que no hay que esperar a que aparezca un job ajeno ni adivinar el host.

El job GPU **no descarga**: si faltan los pesos falla de inmediato con el
comando a ejecutar en el login, en lugar de gastar una asignación de GPU en un
nodo sin internet. La cadena de dos jobs (`llamea-full-chain.sh`) queda solo
para reutilizar un servidor ya cargado; por defecto redirige al job único salvo
`ALLOW_TWO_JOBS=1`.

Dimensionamiento del job único: **16 CPUs, 64 GB y una GPU**. El cuello es la
inferencia; el MOEA interno corre secuencial y no escala con 24 núcleos, así
que pedir más solo alarga la cola. Se dejan 4 hilos OpenMP para
vLLM/tokenización y un hilo BLAS (`OPENBLAS`/`MKL`) para la búsqueda. `CPUS` y
`MEM` son overrides del wrapper.

vLLM recibe `--max-num-seqs 4`, `--enable-prefix-caching` y desactiva el log
de peticiones. En vLLM 0.32 el flag es `--no-enable-log-requests`;
`--disable-log-requests` fue eliminado. El `.slrm` consulta `api_server --help`
y usa el que exista, para no romper versiones antiguas. LLaMEA envía una
petición cada vez; el prefijo largo del prompt se reutiliza entre generaciones
y los prompts completos ya se guardan en `prompts/index.jsonl`, así que
desactivar el log de peticiones no pierde evidencia.

### 9.2 Perfiles de GPU con modelos grandes

`slurm/llamea-gpu-profile.sh` elige el modelo según la GPU del cluster
(H100 80 GB, A100 40 GB, A10 24 GB):

| `PROFILE` | Modelo | VRAM aprox. | GPU |
|---|---|---|---|
| `big-quant` (default) | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | 19–21 GB | A100/H100 |
| `h100-fp16` | `Qwen/Qwen2.5-Coder-32B-Instruct` | 65 GB | H100 |
| `a100-fp16` | `Qwen/Qwen2.5-Coder-14B-Instruct` | 29 GB | A100/H100 |
| `a10-quant` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | 10 GB | A10 |
| `smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | 3 GB | cualquiera (humo) |

El default deja de ser un modelo de juguete: es el mayor número de parámetros
con contexto largo en A100 40 GB/H100 80 GB (32B AWQ). En A10 24 GB se usa
`a10-quant` (14B AWQ), porque el 32B AWQ no deja KV cache para 16k. `MODEL`,
`CONSTRAINT`, `MAX_MODEL_LEN` y `GPU_MEM_UTIL` se pueden sobreescribir por
entorno.

### 9.3 Parada temprana y diversidad

Se exponen mecanismos de LLaMEA 1.4 que atacan el anclaje observado (el modelo
repetía el mismo operador 108 veces en `b200`):

| Bandera | Efecto | Default |
|---|---|---|
| `--patience N` | corta tras `N` evaluaciones sin mejorar el mejor HV | `0` (off) |
| `--parent-selection` | `random`, `roulette`, `tournament` | `tournament` |
| `--adaptive-mutation` | mutación adaptativa de LLaMEA | on |
| `--feature-guided-mutation` | guía por características del AST | off |
| `--niching` | `sharing`, `clearing`, `novelty`, `map_elites` | `none` |
| `--novelty-k`, `--novelty-archive-size` | archivo de novedad | `5`, LLaMEA |

`EarlyStop` se lanza desde el evaluador y `run.py` cierra con el mejor operador
guardado; el checkpoint y `best_so_far.py` ya están en disco.

### 9.3.1 Diversidad conductual (2026-10)

El hash de código no mide diversidad: dos reescrituras del mismo OX son hashes
distintos y el mismo mecanismo. `src/mohh/llamea/diversity.py` mide **lo que
hace** el operador sobre pares de padres fijos (n = 11, 24, 50, 100) y resume 14
características en `[0,1]`: recombinción, distancia/sesgo a cada padre,
preservación de posiciones, orden (Kendall), adyacencia, unión de aristas,
rachas y cambios de padre.

| Mecanismo | Uso | Evidencia |
|---|---|---|
| Novedad conductual | distancia al operador ya evaluado más cercano | novelty search (Lehman & Stanley) |
| Archivo MAP-Elites | rejilla recombinción × estructura; mejor HV por celda | Mouret & Clune (2015) |
| Familias | `order`, `position`, `cycle`, `adjacency`, `exploratory`, `copy`, `mixed`; se pide la menos representada | diagnóstico explícito |
| Reflexión en feedback | familia, novedad, duplicado conductual y dirección poco explorada | ReEvo / EoH |
| Hooks de LLaMEA | `distance_metric` y `behavior_descriptor` conductuales; `--niching novelty/clearing/map_elites` dejan de mirar el texto | LLaMEA 1.4 |

Artefactos: `diversity.json` (resumen + fingerprints), `diversity.jsonl`
(traza incremental), columnas `behavior_*` en `generation_summary.jsonl` y
`individuals.jsonl`, y `scripts/analyze_diversity.py` para el mapa conductual.
`--no-diversity-feedback` desactiva solo el texto extra del prompt; la medición
y el archivo siguen activos. Ver `tests/test_llamea_diversity.py`.

### 9.4 Evaluación final repetida

`--eval-repeats R` repite la evaluación en test con semillas distintas y
reporta media, desviación, mínimo y máximo. El default es `R = 11`, como
`N_EXPERIMENTS` de la evaluación GE. El marco se fija **siempre** al panel de
referencia (`stationary`): un marco que incluye al propio candidato lo favorece
y no es un metro fijo. En cada repetición se evalúan también los clásicos del
panel con la misma población inicial; `llamea_best_info.json` guarda
`test_reference_hv_repeats` y `test_comparison` (Wilcoxon pareado, IC95, `dz` y
Holm), y `report.md` los muestra. La comparación con el marco GE consolidado de
todas las configuraciones sigue siendo `scripts/compare_operators.py`
(`COMPARE=1` en el job único).

### 9.5 Documentación del cluster

`cluster_ixachi/` es documentación local (no material del proyecto) y queda en
`.gitignore`.

### 9.6 Pruebas

`tests/test_slurm.py` valida los perfiles (default 32B), la sintaxis y el
contrato del job único (vLLM + búsqueda + `127.0.0.1`), el comando único y el
paso de las nuevas banderas. `tests/test_llamea.py` cubre `EarlyStop`,
`_llamea_kwargs` y el combine conductual de punta a punta;
`tests/test_llamea_diversity.py` cubre fingerprint, familias, archivo, pickling,
selección conductual y validación de la familia objetivo. Suite completa:
`211 passed, 1 skipped`.

---

## 10. Alternativas de modelo basadas en evidencia (2026-10)

### 10.1 Catálogo de perfiles

Se agregan perfiles para modelos con mejor evidencia en generación de código
que el 32B AWQ, sin retirar los existentes:

| `PROFILE` | Modelo | VRAM | GPU | Papel |
|---|---|---:|---|---|
| `coder30b-awq` | Qwen3-Coder-30B-A3B AWQ | ~18 GB | A100/H100 | recomendado A100/H100 |
| `coder30b-fp8` | Qwen3-Coder-30B-A3B FP8 | ~31 GB | H100 | recomendado H100 |
| `coder30b-awq-a10` | Qwen3-Coder-30B-A3B AWQ | ~18 GB | A10 | 30B en 24 GB, 8k |
| `coder8b` | Seed-Coder-8B-Instruct | ~16 GB | A10 | mejor ≤8B en BigCodeBench |
| `big-quant` (default) | Qwen2.5-Coder-32B AWQ | ~20 GB | A100/H100 | fallback probado |

Cada perfil define `VLLM_ARGS` (`--quantization awq_marlin`,
`--kv-cache-dtype fp8`), que se suma al override `EXTRA_VLLM_ARGS`.
`PROFILE=list` imprime el catálogo. La justificación (HumanEval/HumanEval+/
BigCodeBench/LCB, licencias y estudios de cuantización) está en `modelos.md` §1.

Por qué Qwen3-Coder-30B-A3B: rinde como un 32B en HumanEval+ (89.0 vs 87.2)
pero solo activa 3.3B parámetros, así que una corrida de 1000 generaciones es
mucho más rápida; el 32B denso sigue disponible como referencia sin cuantizar
(`h100-fp16`). El AWQ de 30B es de comunidad (`QuantTrio`): se verifica con
`mohh_models.py verify` y, si vLLM no lo carga, se vuelve a `big-quant`.

### 10.2 Memoria de mecanismos (anti-anclaje)

`--operator-memory N` inyecta en el prompt los últimos `N` `# Description:` /
`# Design idea:` generados y pide explícitamente no repetirlos. Es complementario
a `--repeat-penalty`: este último castiga en el ranking después de evaluar; la
memoria evita gastar la generación. El job del cluster usa `N=12`.

### 10.3 Muestreo recomendado por el modelo

El job único pasa `top_p=0.8`, `top_k=20` y `repetition_penalty=1.05`
(recomendación de Qwen para sus modelos Coder) y sube `temperature` de 0.2 a
0.7, porque 0.2 favorecía la copia del ejemplo. `top_k` y `repetition_penalty`
viajan en `extra_body` (extensiones de vLLM); `--sampling-repetition-penalty`
es distinto de `--repeat-penalty`.

### 10.4 Evidencia en la tarea real

`scripts/analyze_llamea_runs.py --summary [--csv PATH]` produce una fila por
corrida con validez, operadores distintos, mejor HV, tokens por evaluación y
segundos por petición. Es la tabla que decide entre perfiles: los benchmarks de
código no miden el anclaje ni la tasa de operadores válidos del bucle.

### 10.5 Preflight de disco

`mohh_models.download_model` consulta el tamaño remoto (`HfApi.model_info`) y
falla antes de descargar si la raíz persistente no tiene espacio (margen 15 %).
Evita llenar LUSTRE a mitad de un 30B/80B y deja un mensaje accionable
(`remove`, `MOHH_MODELS_DIR`, `SCRATCH`). Si la API no responde, avisa y sigue.

### 10.6 Pruebas

Se agregan pruebas de los perfiles nuevos, de `VLLM_ARGS` por perfil, del
catálogo `PROFILE=list`, de la inyección/acotado de la memoria, del preflight
de disco y de las nuevas banderas del job único.

---

## 11. Combine conductual propio (extensión, no LLaMEA original)

LLaMEA 1.4 ya permite `Operator(number_of_parents=2)`, pero elige los padres
por fitness. Activarlo sin más produce una recombinación semántica de dos
individuos parecidos, no una exploración conductual. Por eso se agrega una
extensión en `src/mohh/llamea/behavioral.py` y `TargetedCombineOperator` en
`prompts.py`:

1. **Padre focal conductual.** `BehavioralLLaMEA._select_parents` usa un torneo
   sobre `HV * (1 + novelty_weight * novelty)`, no solo HV.
2. **Compañero por distancia.** El segundo padre maximiza la distancia entre
   fingerprints; si existe un operador de la familia objetivo, se prefiere ese
   para aportar su mecanismo. El fitness solo rompe empates.
3. **Validación del hijo.** Antes de evaluar se adjunta `target_family` a la
   `Solution`; el evaluador mide el fingerprint y registra `target_family_hit`,
   `target_family_observed` y el feedback PASS/MISS. La familia objetivo es la
   menos representada en `BehaviorTracker`.
4. **Presión opcional.** `--target-penalty` multiplica solo el ranking (el HV
   crudo no cambia); `--require-target-family` rechaza el hijo antes del MOEA
   interno. Por defecto la validación es feedback-only (`target_penalty=1.0`).

La implementación no toca `src/mohh/{generation,evaluation,core}` ni el paquete
`llamea` instalado. El job único usa `--n-parents 5 --n-offspring 2` y deriva
`BUDGET = 1000*2 + 5 = 2005` para que `GENERATIONS=1000` sean 1000 iteraciones.
`COMBINE=0` deja la ablación refine/redesign original; `BEHAVIORAL_PARENTS=0`
aísla la validación de la familia objetivo.

---

## 12. Trazabilidad de prompt/respuesta y modo pensamiento (2026-10)

### 12.1 Registro por petición

`VLLM_LLM` escribe `results/<categoria>/<nombre>/llm_requests.jsonl`, una línea
por petición con:

- el prompt **exacto enviado** (incluida la memoria de mecanismos), no solo el
  que LLaMEA conserva en `conversationlog.jsonl`;
- los parámetros efectivos de muestreo (`temperature`, `max_tokens`, `top_p`,
  `top_k`, `repetition_penalty`, `thinking` y `extra_body`);
- la respuesta cruda, la respuesta limpia que evalúa LLaMEA y
  `reasoning_content` cuando el servidor lo devuelve;
- `usage` (prompt/completion/total, razonamiento, tokens cacheados),
  `finish_reason`, id de respuesta, modelo, latencia y error si falló.

`tokens.json` y `metrics.json` guardan la ruta y el historial por petición.
`prompts/gen-NNNN.md` ahora muestra el prompt enviado, la respuesta limpia, el
razonamiento y la respuesta cruda; `prompts/index.jsonl` añade modelo,
parámetros, `finish_reason`, conteos de razonamiento y tokens cacheados.

### 12.2 Modo pensamiento

`--thinking auto|on|off` (o `THINKING=...`) controla
`chat_template_kwargs.enable_thinking` en servidores que lo soportan (Qwen3 y
similares). `auto` no toca el template para no romper Ollama/llama.cpp. Con
`on`, subir `MAX_TOKENS`: el razonamiento consume presupuesto y, si el modelo
deja `content` vacío, se usa `reasoning_content` solo como fallback de parseo.
El parser reconoce `<think>`, `<thinking>` y `<reasoning>`.

### 12.3 Cota del HV en el marco fijo

En el marco `stationary` (panel fijo) un candidato puede superar el mejor punto
del panel. Sin recorte, sus coordenadas normalizadas negativas inflaban el HV
por encima del máximo teórico `1.1**K` (se observaron valores ~5.9 y ~103 en
instancias de 5 objetivos). `clip_to_reference_box` recorta a `[0, 1.1]` antes
de la integral; la mejora se cuenta como el volumen completo hasta el nadir y
los puntos peores que el nadir aportan 0. En el marco GE el candidato ya está
incluido, así que el cambio no altera los valores normales.

### 12.4 Modelo grande para H100

Se agrega `PROFILE=coder-next-awq`:
`qtum/Qwen3-Coder-Next-AWQ` (80B/3B activos, W4A16 compressed-tensors, ~42 GB).
Reporta LiveCodeBench v6 58.9 y EvalPlus 86.6, y deja ~30 GB de KV en una H100
80 GB. No cabe en A100 40 GB. La cuantización sube la perplejidad de wikitext-2
de 7.74 a 12.53: es un candidato a validar en la tarea, no el nuevo default.

---

## 13. GLM cuantizado: local y cluster (2026-10)

### 13.1 GPUs disponibles

El cluster (`cluster_ixachi/docs/README.md`) tiene:

| Partición | GPU | VRAM | Uso |
|---|---|---:|---|
| `gpu` | 1× H100 | 80 GB | modelo grande, fp16/FP8/AWQ |
| `gpu` | 1× A100 | 40 GB | 30B AWQ o 14B fp16 |
| `gpu` | 2× A10 | 24 GB | modelos ≤14B AWQ u 8B fp16 |

La máquina local de pruebas es una **RTX 3060 de 12 GB**. Por eso se eligen dos
modelos distintos, no el mismo quant en dos máquinas.

### 13.2 Pequeño local: `GLM-Z1-9B-0414` Q4_K_M

- Repo GGUF: `unsloth/GLM-Z1-9B-0414-GGUF`, archivo
  `GLM-Z1-9B-0414-Q4_K_M.gguf`, **6.17 GB**, licencia MIT, 32k de contexto.
- Es un 9B denso de razonamiento entrenado con RL en matemáticas, código y
  lógica. La card oficial lo describe como líder entre los open-source de su
  tamaño, pero no publica una tabla numérica en texto; se usa para *smoke tests*
  y para validar el pipeline de razonamiento, no como evidencia del cluster.
- Compatible con llama.cpp (`llama-server -m ... -c 8192 -ngl 99`) y con Ollama
  (`ollama pull sllm/glm-z1-9b`, ~8.6 GB). El Q4_K_M deja más VRAM para KV.
- Con 12 GB entra completo en GPU; el contexto recomendado es 8192, no 32k.

### 13.3 Grande cluster: `GLM-4.7-Flash-AWQ`

- Modelo base: `zai-org/GLM-4.7-Flash` (30B totales / 3B activos, MIT,
  `glm4_moe_lite`). AWQ: `QuantTrio/GLM-4.7-Flash-AWQ`, **~19.8 GB**, W4A16.
- Benchmarks oficiales: AIME25 91.6, GPQA 75.2, LiveCodeBench v6 64.0,
  SWE-bench Verified 59.2, τ²-Bench 79.5. Es la mejor combinación de
  razonamiento/matemáticas/código que cabe en **una A100 40 GB** o **una H100
  80 GB**.
- Perfil nuevo `glm47-flash-awq` (`CONSTRAINT=A100|H100`, 16k, 0.92 de VRAM).
  Añade `--trust-remote-code --reasoning-parser glm45` y exporta las variables
  que pide la card de QuantTrio (`VLLM_USE_DEEP_GEMM=0`,
  `VLLM_USE_FLASHINFER_MOE_FP16=1`, `VLLM_USE_FLASHINFER_SAMPLER=0`).
- Requiere vLLM main/nightly: la card del AWQ se probó con
  `vllm==0.14.0rc2`. Si el `.conda-env` del cluster es viejo, actualizarlo en el
  login antes de lanzar; el job GPU no tiene internet.
- Para llama.cpp existe el mismo modelo en GGUF
  (`unsloth/GLM-4.7-Flash-GGUF`, Q4_K_M ~18.3 GB), pero en el cluster se usa
  vLLM+AWQ. En A10 24 GB no se recomienda: los ~20 GB de pesos dejan poco KV.
- Alternativas descartadas para una sola GPU: `GLM-4.7` completo
  (358B/32B, ~717 GB bf16) y `GLM-5.3-Flash` (~328 GB bf16).

### 13.4 Pruebas

`tests/test_slurm.py` verifica que el perfil GLM existe, que el catálogo lo
imprime y que `VLLM_ARGS` incluye `--reasoning-parser glm45`. El Q4_K_M local se
puede validar antes de una búsqueda con `llama-server` y una petición corta
(la descarga son ~6.2 GB); en esta sesión no se ejecutó esa descarga.

### 13.5 Raíz persistente cuando `$LUSTRE` no está exportado

En ixachi el directorio real es `/LUSTRE/<usuario>`, pero el login no siempre
exporta la variable `$LUSTRE` (el job mostró `LUSTRE=(sin definir)`). Eso hacía
que `models_root()` cayera en `~/mohh-models`, violando la regla de no usar
`$HOME`. Ahora, si `MOHH_MODELS_DIR`, `$SCRATCH` y `$LUSTRE` no están definidos,
se usa `/LUSTRE/<usuario>/mohh-models` cuando esa ruta existe; `$HOME` queda
solo como último recurso en una máquina personal.

También se corrigió `MODEL_ROOT="$(python scripts/mohh_models.py root)"`: el
comando imprime la ruta y dos líneas de diagnóstico, así que `HF_HOME` quedaba
con una ruta multilínea. Ahora los `.slrm` usan `root | head -1`.

### 13.6 vLLM 0.32 y `--disable-log-requests`

El job `15301` murió al arrancar con:

```text
api_server.py: error: unrecognized arguments: --disable-log-requests
```

vLLM 0.32 eliminó ese flag. El `.slrm` ahora consulta `api_server --help` y usa
`--no-enable-log-requests` si existe; solo cae en `--disable-log-requests` para
instalaciones antiguas. El cliente seguía esperando el servidor hasta el
timeout de 30 minutos porque el proceso vLLM ya había terminado; por eso la GPU
seguía en 0 MiB.
