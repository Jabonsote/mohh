# Pruebas locales con modelos pequeños

Registro de las corridas reales hechas en una estación de trabajo (RTX 3060,
12 GB) para validar de extremo a extremo la integración LLaMEA de MOHH-R
**antes** de tocar el cluster. Sirven como evidencia de que el bucle funciona
(conexión, generación, validación, evaluación train/test) y como catálogo de las
fallas típicas de un modelo pequeño.

## Montaje

- Modelo local: `Qwen/Qwen2.5-0.5B-Instruct` (ya en caché de HuggingFace).
- Servidor OpenAI-compatible de desarrollo: `scripts/local_openai_server.py`
  en el puerto 8001 (el 8000 estaba ocupado por otro proceso del sistema).
- Evaluación: `MOSolver` interno (`MOEAD`, `NSGAII`, `SMSEMOA`) sobre 4
  instancias de train y 4 de test del split de MOHH-R, 20 generaciones y
  población 20 por instancia.
- Presupuesto: 10–12 generaciones de LLaMEA (llamadas al LLM) por corrida.

```bash
python scripts/local_openai_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8001
python -m mohh.llamea.run --model Qwen/Qwen2.5-0.5B-Instruct \
    --base-url http://127.0.0.1:8001/v1 --solver MOEAD --budget 12 \
    --train-instances 4 --test-instances 4 --mo-generations 20 \
    --mo-population-size 20 --eval-test --experiment-name local_moead
```

## Resultados por corrida

| Corrida | Solver | Temp. | Aciertos (HV>0) | Mejor HV train | HV test |
|---|---|---|---:|---:|---:|
| `search_moead` | MOEA/D | 0.2 | 12/12 | 2.14e31 | 1.26e41 |
| `search_nsgaii` | NSGA-II | 0.2 | 0/12 | 0 | 0 |
| `search_smsemoa` | SMS-EMOA | 0.2 | 10/10 | 1.72e32 | 1.78e42 |
| `search_moead_t06` | MOEA/D | 0.6 | 0/10 | 0 | 0 |
| `search_moead_refine` | MOEA/D (`--refine-only`) | 0.2 | 10/10 | 2.14e31 | 1.26e41 |
| `v2_smsemoa` | SMS-EMOA | 0.25 | 3/12 | 1.61e32 | 1.12e42 |
| `v2_moead_t06` | MOEA/D | 0.6 | 5/10 | 2.14e31 | 1.26e41 |
| `v3_smsemoa` | SMS-EMOA | 0.25 | 10/12 | 1.15e32 | 1.33e42 |
| `v3_parallel` | MOEA/D (`--n-offspring 2 --max-workers 2`) | 0.2 | 11/13 | 2.14e31 | 1.26e41 |

**Agregado (9 corridas, 128 operadores evaluados):** 61 válidos (HV>0),
13 operadores válidos **distintos**.

> El HV no es comparable entre solvers distintos (cada MOEA usa su propia
> referencia de hipervolumen); sí es comparable entre corridas del mismo solver.

## Qué se observó

1. **El bucle completo funciona**: conexión HTTP al servidor, generación,
   parseo del bloque de código, validación del contrato, evaluación con el
   `MOSolver` de MOHH-R y evaluación final en el conjunto de test.
2. **La temperatura manda con modelos chicos.** A 0.2–0.25 el modelo tiende a
   reproducir el ejemplo OX del prompt (operador válido); a 0.6 casi todo sale
   inválido.
3. **`--refine-only` usa el operador por defecto de LLaMEA** (`Operator:
   mutation`) en vez de `refine`/`redesign`, como estaba previsto.
4. **`--n-offspring 2 --max-workers 2`** produce HV positivo: el fitness
   sobrevive el pickle a los procesos hijos (ver `__getstate__`).
5. **Operadores "válidos" pueden ser degenerados.** Varios operadores aceptados
   terminan en `return np.random.permutation(n)` (una permutación aleatoria): la
   validación de contrato es necesaria pero **no** suficiente para juzgar la
   calidad; para eso está el hipervolumen.
6. Errores de los modelos chicos, por frecuencia: `ValueError` genérico
   (contrato no cumplido), `NoCodeException` (respuesta sin `def crossover`) y,
   en una corrida **previa a los arreglos**, un `IndexError` interno de LLaMEA.

## Arreglos de robustez que salieron de estas pruebas

Estas corridas destaparon fragilidades reales que se corrigieron en el adaptador
(siempre fuera del núcleo de MOHH-R):

| Síntoma | Causa | Arreglo |
|---|---|---|
| `IndexError: list index out of range` | LLaMEA hace `re.findall(...)[0]` si el código no trae `def/class` | `VLLM_LLM.extract_algorithm_code` lanza `NoCodeException` si falta `def crossover` |
| La búsqueda se cae si el operador falla en una instancia | El fitness solo capturaba errores de compilación | `OperatorEvaluator` captura toda excepción por instancia y da score 0 + feedback |
| `name 'List' is not defined` | El LLM usa `List[int]` sin importar `typing` | El namespace de ejecución aporta `List/Tuple/Dict/...` |
| `np` no definido | El LLM olvida `import numpy as np` | El namespace de ejecución aporta `np`/`numpy` |
| Cuelgue con `while True` | Con `max_workers=1` joblib no aplica timeout | `_call_with_timeout` (SIGALRM) acota validación (10 s) y cada instancia (120 s) |
| El fitness no se podía picklear tras construir instancias | `Instance` no es picklable | `__getstate__` descarta la caché de instancias al picklear |

Además, la validación de contrato (`_valid_operator`) usa tamaños del dominio
mQAP (`n = 11, 24, 50`): con `n = 2` la primitiva `preserve_segments` de MOHH-R
entra en bucle infinito y no se debe tocar el núcleo.

## Ejemplos generados y verificación en test

Con el modelo pequeño, los operadores válidos son casi siempre variantes del OX
del ejemplo. Dos ejemplos reales:

**OX con comentarios y `dtype` (válido, 2.14e31 en train MOEA/D):**

```python
import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]
    used = set(x[a:b].tolist())
    fill = [v for v in y if v not in used]
    positions = list(range(b, n)) + list(range(0, a))
    for pos, val in zip(positions, fill):
        child[pos] = val
    return child
```

**Operador distinto basado en ordenar y concatenar (válido, 8.30e30):**

```python
import numpy as np

def crossover(x, y):
    n = len(x)
    sorted_x = np.sort(x)
    sorted_y = np.sort(y)
    child = np.concatenate((sorted_x[:n//2], sorted_y[n//2:]))
    used = set(sorted_x[:n//2].tolist())
    fill = [v for v in y if v not in used]
    positions = list(range(n//2, n))
    for pos, val in zip(positions, fill):
        child[pos] = val
    return child
```

Verificación independiente: se tomaron los **13 operadores válidos distintos** y
se evaluaron con el mismo harness (MOEA/D, 4 train + 4 test, 20 generaciones,
población 20):

| digest | train HV | test HV | origen |
|---|---:|---:|---|
| `914d3fb9` | 5.863e31 | 5.954e41 | SMS-EMOA |
| `b6fc03db` | 5.863e31 | 5.954e41 | SMS-EMOA |
| `0baaa540` | 5.267e31 | 2.198e41 | SMS-EMOA |
| `8da4d2ef` | 5.267e31 | 2.198e41 | SMS-EMOA |
| `de7575a0` | 5.267e31 | 2.198e41 | SMS-EMOA |
| `3a8ac00b` / `4fc294e2` / `5291969a` / `536d9d81` / `ab4d6efb` / `afc728d2` / `c0466ffa` | 2.14e31 | 1.257e41 | varias |
| `fca5ad5c` | 8.297e30 | 4.768e41 | MOEA/D |

El orden relativo train/test se mantiene por familia de operador, así que la
evaluación en test es consistente con la de train: no hay fuga de información ni
un sesgo del split.

## Analizar corridas

LLaMEA deja un directorio `exp-<fecha>-LLaMEA-<modelo>-<nombre>/` por corrida con
`log.jsonl` (fitness, feedback, error, código, operador) y `code/try-N-*.py`.
El script `scripts/analyze_llamea_runs.py` los resume:

```bash
python scripts/analyze_llamea_runs.py 'exp-*0.5B-Instruct*'
python scripts/analyze_llamea_runs.py 'exp-*0.5B-Instruct*' --list-codes
```

## Pruebas unitarias y de conexión

- Suite sin red: `python -m pytest tests/test_llamea.py -q` → **35 passed, 1 skipped**.
- Test de conexión **viva** (opt-in, contra el servidor local):

```bash
LLAMEA_SERVER_URL=http://127.0.0.1:8001/v1 \
LLAMEA_SERVER_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
python -m pytest tests/test_llamea.py -q -k live
```

Como los modelos chicos son estocásticos, el test vivo no exige que *todas* las
muestras sean válidas: exige **al menos una** en 5 intentos (mide conexión y
pipeline, no la calidad del modelo).

## Reproducir

```bash
python scripts/analyze_llamea_runs.py 'results/searches/**/exp-*'
```

Con un modelo más grande (7B+) en el cluster los mismos comandos deberían dar
más operadores válidos y distintos; la lógica de evaluación es idéntica.

---

## Corridas de 100 generaciones con HV normalizado

Después de alinear el hipervolumen con el proyecto original (panel de
referencia OX/PBX/PMX/CX/ERX → frente consolidado → nadir fijo `1.1`) se
corrieron tres búsquedas completas de **100 generaciones** con
`Qwen/Qwen2.5-0.5B-Instruct` (servidor local en `:8001`, temperatura 0.2, 8
instancias de train y 8 de test, MOEA interno 30/30). Todo está en
`results/searches/<nombre>/`, con `report.md`, `evolution.csv`,
`evolution.png`, `run.json` y el operador en `llamea_best_operator.py`.

> Los HV de esta sección están en el **marco fijo del panel** (escala abierta),
> que era el comportamiento de entonces. En el marco GE de la evaluación
> gramatical los mismos operadores dan números más chicos (ver más abajo): no
> compares estas columnas con las de las secciones nuevas.

| Corrida | MOEA | Válidos | Distintos | Mejor HV train | HV test | Mejor en |
|---|---|---:|---:|---:|---:|---:|
| `qwen05b_moead_b100` | MOEA/D | 99/100 | 1 | 0.426177 | 0.312510 | gen 0 |
| `qwen05b_nsgaii_b100` | NSGA-II | 90/100 | 17 | 0.389657 | 0.271350 | gen 1 |
| `qwen05b_smsemoa_b100` | SMS-EMOA | 99/100 | 1 | 0.316706 | 0.214423 | gen 0 |

### Línea base de operadores clásicos

Mismos instancias, semilla y presupuesto (`scripts/llamea_examples.py`):

| Operador | MOEA/D | NSGA-II | SMS-EMOA |
|---|---:|---:|---:|
| CX | 0.1487 | **0.6076** | **0.6883** |
| PMX | 0.3903 | 0.5473 | 0.6143 |
| PBX | **0.5047** | 0.4463 | 0.4503 |
| OX | 0.4262 | 0.3659 | 0.3167 |
| ERX | 0.2541 | 0.3213 | 0.3156 |
| aleatorio | 0.2351 | 0.3418 | 0.3100 |

### Qué muestran estas corridas

1. **La normalización es consistente**: el mejor HV de MOEA/D (0.426177) y de
   SMS-EMOA (0.316706) coincide *exactamente* con el OX de la línea base. El
   modelo reprodujo el ejemplo del prompt, no un operador nuevo.
2. **El sesgo de anclaje es real y medible**: MOEA/D y SMS-EMOA produjeron **un
   solo operador distinto** en 100 generaciones; NSGA-II produjo 17. Es la
   evidencia local que justifica la mejora #7 de
   [`mejoras-prompt.md`](mejoras-prompt.md) (sustituir el ejemplo completo por
   un esqueleto).
3. **El modelo de 0.5B no es un competidor**: en los tres MOEA los operadores
   clásicos (CX/PMX/PBX) superan al mejor operador del LLM. Sirve como prueba de
   humo del pipeline, no como evidencia de calidad.
4. **Train > test**: el mejor operador se elige sobre train y baja en test
   (p. ej. 0.426 → 0.313 en MOEA/D), como en la búsqueda GE. Confirma que el
   split se respeta.

---

## Ablaciones de prompt (30 generaciones)

Cinco variantes con el mismo modelo (`Qwen/Qwen2.5-0.5B-Instruct`), solver
(MOEA/D), semilla, 8 instancias de train y presupuesto (30 generaciones). Se
lanza con `./scripts/llamea_prompt_ablations.sh` y queda en `results/prompts/`.
La columna *tokens* es el total consumido por el LLM (solo disponible en las
corridas posteriores a la instrumentación de `usage`).

| Variante | Válidos | Distintos | Mejor HV train | Tokens totales |
|---|---:|---:|---:|---:|
| `example` (OX completo, referencia) | 30/30 | 1 | **0.426177** | - |
| `refineonly` (sin rediseño) | 30/30 | 2 | **0.426177** | - |
| `noexample` (sin ejemplo; cadena vacía) | 29/30 | 1 | **0.426177** | 58 894 |
| `llameadefault` (ejemplo por defecto de LLaMEA) | 24/30 | 6 | 0.102935 | - |
| `skeleton` (esqueleto) | 0/30 | 0 | 0.000000 | - |
| `temp06` (temperatura 0.6, con ejemplo) | 2/30 | 1 | 0.108861 | 62 468 |

### Lectura

1. **El esqueleto no funciona con 0.5B**: las 30 respuestas fueron inválidas
   (código truncado a mitad de línea y lógica que no devuelve una permutación).
   El ancla no es solo semántica: en modelos pequeños el ejemplo completo
   también hace de *plantilla sintáctica*. La ablación debe repetirse con 7B+
   antes de concluir que la mejora #7 es mala; con 0.5B solo dice que el modelo
   no puede rellenar el hueco.
2. **Quitar el ejemplo no elimina el anclaje**: con `--no-example` (cadena
   vacía) el 0.5B escribió **el mismo OX de memoria** (29/30 válidas, 1
   operador distinto, HV idéntico a la línea base OX 0.426177). El ancla también
   está en el conocimiento de dominio del `task_prompt` (lista de familias
   OX/PMX/PBX/…), donde OX aparece primero. Con 0.5B, "buscar" es recuperar el
   clásico más famoso.
3. **El ejemplo por defecto de LLaMEA confunde**: si se pasa
   `example_prompt=None`, LLaMEA inyecta su propio ejemplo (una clase de
   *random search* de otro dominio). Resultado: 24/30 válidas y 6 operadores
   distintos, todos malos. Por eso `--no-example` ahora pasa cadena vacía; el
   nombre `llameadefault` marca las corridas viejas con ese comportamiento.
4. **Subir la temperatura no ayuda**: a 0.6 la mayoría de las respuestas deja de
   ser válida (2/30) y el mejor HV cae a 0.109. Para 0.5B conviene temperatura
   baja (0.2); en 7B+ hay que repetir la ablación.
5. **`refineonly` se comporta como `example`**: el paso de rediseño no cambia
   nada porque el modelo nunca abandona el OX del ejemplo. No es evidencia
   contra el rediseño, sino otra manifestación del anclaje.

---

## Corridas de 1000 generaciones (0.5B)

Dos corridas largas con `Qwen/Qwen2.5-0.5B-Instruct`, MOEA/D, 8 instancias de
train y el mismo preprocesado (`results/searches/qwen05b_moead_b1000` y
`qwen05b_moead_noexample_b1000`).

| Corrida | Generaciones | Válidos | Distintos (texto) | Distintos (comportamiento) | Mejor HV train | HV test | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen05b_moead_b1000` (con ejemplo) | 1000 | 995 | 3 | 1 (variantes de OX) | 0.426177 (gen 0) | 0.312510 | 1 963 104 |
| `qwen05b_moead_noexample_b1000` (sin ejemplo) | 900* | 884 | 22 | 1 (`return x`) | 0.102935 (gen 3) | - | 1 713 753 |

\* interrumpida (el servidor local se cayó y el bucle quedó esperando).

### Qué muestran

1. **10× presupuesto, mismo resultado**: la corrida con ejemplo reproduce
   exactamente el resultado de la corrida de 100 generaciones (mismo mejor HV
   train 0.426177 y test 0.312510). El mejor operador es el OX del ejemplo desde
   la generación 0 y nunca se mejora.
2. **“Válido” no significa “recombina”**: de los 22 operadores distintos (texto)
   de la corrida sin ejemplo, **todos** resultaron ser
   `return np.array(x, copy=False)` — devuelven el primer padre, algunos con
   comentarios copiados de la sección de dominio del prompt. Todos obtienen el
   mismo HV (0.102935) porque, si el cruce no recombina, el MOEA produce el
   mismo frente.
3. **La validación era demasiado débil** y el ranking no lo detectaba (empate
   total → todos `#1 of N`). Esto motivó dos cambios ya implementados:
   `recombination_novelty` (chequeo barato que puntúa 0 al operador que devuelve
   un padre o una vista) y el desempate del ranking por novedad. Ver
   §7.2.1 de [`cambios.md`](../../cambios.md) y el punto 5 de
   [`areas-de-mejora.md`](areas-de-mejora.md).
4. **Costo sin retorno**: en conjunto, 1900 generaciones y ~3.7 M tokens
   produjeron **dos comportamientos** (OX e identidad). Es el argumento empírico
   para el criterio de parada por estancamiento (mejora #9 de
   [`areas-de-mejora.md`](areas-de-mejora.md)) y para subir a un modelo
   `-Coder` de 7B en el cluster antes de gastar corridas largas.

---

## Comparación de modelos a 30 generaciones

Misma configuración (MOEA/D, 8 instancias de train, temperatura 0.2, ejemplo OX,
presupuesto 30) cambiando solo el modelo. Corridas en `results/searches/`.

| Modelo | Corrida | Válidos | Distintos | Tokens | Tokens/gen |
|---|---|---:|---:|---:|---:|
| `Qwen2.5-0.5B-Instruct` (generalista) | `qwen05b_moead_b100` | 99/100 | 1 | ~196 000 | ~1 960 |
| `Qwen2.5-Coder-0.5B-Instruct` | `qwen25coder05b_moead_b30` | 27/30 | 1 | 68 633 | 2 288 |
| `Qwen2.5-0.5B-Instruct` (sin ejemplo) | `qwen05b_moead_noexample_b1000` | 29/30 | 1 | 58 894 | ~1 963 |
| `Qwen2.5-Coder-7B-Instruct`, mensaje genérico | `qwen25coder7b_moead_b30` | **0/30** | 0 | 75 893 | 2 530 |
| `Qwen2.5-Coder-7B-Instruct`, diagnóstico | `qwen25coder7b_moead_b30_feedback` | **21/30** | 24 | 95 011 | 3 166 |
| `Qwen2.5-Coder-7B-Instruct`, diagnóstico, marco GE | `qwen25coder7b_moead_b30_ge` | 20/30 | 10 | 77 490 | 2 583 |
| … misma config, réplica 2 | `qwen25coder7b_moead_b30_ge_r2` | 10/30 | 10 | 71 452 | 2 382 |
| … misma config, réplica 3 | `qwen25coder7b_moead_b30_ge_r3` | 16/30 | 7 | 78 841 | 2 628 |
| `Qwen2.5-Coder-7B-Instruct`, diagnóstico, marco GE, **100 gen** | `qwen25coder7b_moead_b100_ge` | 51/100 | 21 | 241 313 | 2 413 |

### Lectura

1. **A 0.5B, cambiar a `-Coder` no cambia nada**: mismo mejor HV (idéntico al OX
   de la línea base) y mismo HV de test hasta el último decimal. Las 27
   respuestas válidas son **el ejemplo OX copiado literalmente**, incluidos sus
   comentarios (`# preserve a segment of parent x`).
2. **Cuesta un poco más**: el Coder escribe respuestas más largas (373 tokens de
   salida por generación frente a ~205 del generalista; 2 288 tokens totales por
   generación frente a ~1 960) y produjo 3 respuestas inválidas (código
   malformado al concatenar dos mecanismos) frente a 0.
3. **Conclusión operativa**: el salto de calidad no está en el apellido `-Coder`
   a igual tamaño, sino en el tamaño (≥7B) — o en cambiar la tarea para que el
   modelo no pueda limitarse a recuperar un operador clásico de memoria. Las
   corridas serias van al cluster con `Qwen2.5-Coder-7B-Instruct`.
4. El chequeo de degeneración **no marcó nada aquí**, y es correcto: copiar el OX
   del ejemplo es una recombinación real (el problema es de exploración, no de
   degeneración).

> Nota de VRAM: primero se intentó `Qwen2.5-Coder-1.5B-Instruct`, pero no cabe en
> esta máquina (12 GB, de los que 8.5 GB usa otro modelo): el servidor murió con
> `torch.OutOfMemoryError`. El 7B sí corrió cuantizado en Ollama con la GPU libre
> (ver [`modelos.md`](../../modelos.md) §C).

### El feedback cambia el resultado (abla válida con cualquier marco)

Dos corridas idénticas salvo el mensaje de validación que recibe el modelo
(`Qwen2.5-Coder-7B-Instruct` en Ollama, 30 generaciones, MOEA/D, 8 instancias):

| Corrida | Válidos | Distintos | Mejor HV train (marco de la corrida) | Tokens |
|---|---:|---:|---:|---:|
| `qwen25coder7b_moead_b30` (mensaje genérico) | **0/30** | 0 | 0 | 75 893 |
| `qwen25coder7b_moead_b30_feedback` (diagnóstico) | **21/30** | 24 | 4.433042 (gen 19) | 95 011 |

En la primera, 27 de 30 generaciones recibieron el mismo texto
(`crossover must return a valid permutation...`) y el modelo repitió el mismo
bug (un OX/PBX con off-by-one en el relleno que dejaba un `-1`). Con el
diagnóstico, el bucle le dice el tamaño, los huecos, los duplicados y la línea
exacta del `assert` que falló, y el modelo corrige: 21/30 válidos, 24 códigos
distintos y encuentra un híbrido (PBX/OBX + adyacencia + CX) que supera al OX del
ejemplo. Los fallos restantes fueron 3 timeouts, 1 operador degenerado y 1
`NoCodeException`. Conclusión: **el texto del feedback es parte del algoritmo**;
con un modelo capaz, un mensaje genérico tira el presupuesto entero.

## El marco de hipervolumen (justo y comparable con la evaluación GE)

El marco de normalización debe ser el de la evaluación gramatical
(`src/mohh/evaluation/experiment.py`): frente no dominado consolidado de
**todo lo evaluado** en la instancia, normalización `(f - vmin) / (vmax - vmin)`
y nadir fijo `[1.1] * K`. En el bucle, "todo lo evaluado" es el panel de
referencia (OX/PBX/PMX/CX/ERX) más los candidatos ya evaluados en esa corrida.

Los mismos frentes del mejor operador del Coder 7B (corrida
`qwen25coder7b_moead_b30_feedback`) y del panel clásico, medidos con las
funciones originales:

| Operador | Marco = solo panel | Marco GE (panel + mejor candidato) |
|---|---:|---:|
| OX | 0.426177 | 0.174201 |
| PBX | 0.504689 | 0.224239 |
| PMX | 0.390263 | 0.225029 |
| CX | 0.148689 | 0.091615 |
| ERX | 0.254144 | 0.102664 |
| mejor candidato 7B | **4.433042** | **0.583228** |

Con el marco del panel el HV del candidato queda en una escala abierta (4.43
frente a 0.43–0.50 del panel) y no se puede comparar con la tabla GE. Con el
marco GE, el mejor candidato (0.583) supera a **toda** la línea base medida en el
mismo marco (0.17–0.23): esa es la afirmación publicable.

La implementación reproduce **exactamente** los números calculados con las
funciones originales (`normalize_values`, `non_dominated_sorting_vectorized`,
`compute_hypervolume`), y `--stationary-frame` reproduce bit a bit la corrida
anterior (candidato 4.433042, OX 0.426177).

### El marco es por corrida: compara dentro de la tabla, no entre tablas

El marco incluye al mejor candidato de la corrida, así que un candidato muy
bueno **baja** el HV de todos los operadores de esa tabla. Las tres corridas con
el marco GE, más la del marco fijo, cada una con su marco final:

| Operador | `..._feedback` (fijo) | `..._ge` | `..._ge_r2` | `..._ge_r3` | `..._b100_ge` |
|---|---:|---:|---:|---:|---:|
| OX | 0.174201 | 0.413137 | 0.379557 | 0.425393 | 0.361932 |
| PBX | 0.224239 | **0.496138** | **0.469917** | **0.473776** | 0.423080 |
| PMX | 0.225029 | 0.376748 | 0.368247 | 0.370305 | 0.391366 |
| CX | 0.091615 | 0.136196 | 0.154133 | 0.155818 | 0.152874 |
| ERX | 0.102664 | 0.228550 | 0.220285 | 0.243004 | 0.233634 |
| mejor candidato | **0.583228** | 0.413137 | 0.395452 | 0.425393 | **0.511142** |

- En `..._feedback` el candidato (híbrido con adyacencia) gana a todo el panel.
- En las tres corridas de 30 gen con marco GE el mejor candidato es un **clon
  conductual del OX** (OBX+PMX, OBX+ERX…), con HV indistinguible del OX. En las
  tres, **PBX queda por delante** del mejor candidato (0.470–0.496 frente a
  0.396–0.425).
- **A 100 generaciones la historia cambia**: `..._b100_ge` (51/100 válidos,
  21 operadores distintos) encuentra un OBX con reparación de adyacencia que da
  **0.511142** frente a PBX 0.423080 y OX 0.361932 **en el mismo marco**, y gana
  también en test (0.215061 frente a PBX 0.209354). Ver la sección siguiente.
- Por eso la comparación válida es **por filas dentro de una misma tabla**
  (mismo marco); los valores absolutos de corridas distintas no son comparables.

### A 100 generaciones sí supera a la línea base (`qwen25coder7b_moead_b100_ge`)

Misma configuración que las de 30 generaciones, presupuesto 100, marco GE y
`--eval-test`. El mejor operador del marco final (guardado también como
`llamea_best_operator_final_frame.py`) es un **OBX con reparación de
adyacencia**: preserva un segmento de `x` y, al rellenar con `y`, intercambia
elementos consecutivos cuando `|child[i] - child[i-1]| != 1`.

| | Train (8 inst.) | Test (8 inst.) |
|---|---:|---:|
| **Mejor candidato** | **0.511142** | **0.215061** |
| PBX | 0.423080 | 0.209354 |
| PMX | 0.391366 | 0.150330 |
| OX | 0.361932 | 0.179259 |
| ERX | 0.233634 | 0.075327 |
| CX | 0.152874 | 0.039318 |

- Gana a **todos** los clásicos en train y en test, medido en el mismo marco.
- Sin instancias sin escala ni en train ni en test.
- 51/100 válidos, 21 códigos distintos, 241 313 tokens (~2 413 por generación).
- El HV de LLaMEA en el momento de la evaluación era 0.530818; en el marco final
  (que creció con los candidatos posteriores) el mismo operador vale 0.511142.
  Es la no-estacionariedad esperada del marco GE, y por eso se reporta el número
  del marco final con la línea base medida ahí mismo.

> **Corrección posterior (comparación pareada).** Esa tabla es de **una** corrida
> y su marco incluye al propio candidato, así que no es una comparación. Con 11
> repeticiones independientes y las mismas semillas para todos los operadores
> (`scripts/compare_operators.py`), este operador queda **al nivel de PBX**
> (6/11 victorias, p = 0.52), por encima de OX solo marginalmente (p = 0.083) y
> claramente por encima de PMX/CX/ERX. La afirmación "gana a todos los clásicos"
> no se sostiene; ver [`comparacion-justa.md`](comparacion-justa.md). Lo que sí se
> sostiene es el operador de la corrida de 30 con feedback diagnóstico, que gana
> 11/11 a los diez clásicos.

### Varianza a 30 generaciones (y por qué el presupuesto importa)

Con la misma configuración y solo cambiando la semilla de muestreo del LLM, la
tasa de validez del 7B va de **0/30 a 22/30** y el mejor HV de 0.396 a 4.433. A
30 generaciones la varianza manda: el resultado de una corrida no es una
conclusión. Detalles observados:

- En `..._ge_r2`, el modelo repitió **16 veces el mismo bug** (la línea
  `child[a:b] = np.concatenate((x[a:b], y[a:b]))`, que revienta al asignar) pese a
  que el feedback le decía el error y **la línea exacta**; solo al final acertó.
- El experimento 3+3 (marco GE vs marco fijo) **no confirma** que el marco GE
  aplaste la búsqueda: con el marco fijo, dos de tres réplicas también se
  quedaron al nivel del OX (0.426 y 0.415) y solo una superó a PBX (0.546).
  La diferencia entre corridas la explica la varianza del muestreo, no el marco.
- Lo que sí cambia el resultado es el **presupuesto**: con 100 generaciones
  (`..._b100_ge`) el modelo sale del anclaje en el OX y encuentra un operador que
  gana a todos los clásicos en train y en test. A 30 generaciones no hay margen
  para corregir los bugs, explorar y explotar.

### Tablas de test (marco GE)

El test se evalúa en las mismas 8 instancias para todos los operadores, en el
mismo marco (panel + candidato):

| Operador | `..._feedback` | `..._ge` |
|---|---:|---:|
| OX | 0.133646 | 0.184297 |
| PBX | 0.190108 | 0.200442 |
| PMX | 0.140416 | 0.190143 |
| CX | 0.039858 | 0.042885 |
| ERX | 0.078610 | 0.087373 |
| mejor candidato | **0.429711** | 0.184297 |

- `..._feedback` (el híbrido con adyacencia) supera a todo el panel también en
  test (0.430 frente a 0.134–0.190).
- `..._ge` empata con OX, coherente con que su mejor candidato sea un clon.
- Cada tabla excluye una instancia sin escala: `KC18-4fl-1un.txt` en
  `..._feedback` y `KC59-2fl-1rl.txt` en `..._ge`. El HV de test de
  `..._feedback` antes se reportó como `2.75e14` (bug de normalización); el
  número corregido es **0.429711**.

### Artefacto de normalización ya corregido (`2.75e14`)

El HV de test de `..._feedback` se reportó como `2.75e14`. No era el operador:
era normalización. En la instancia `KC59-2fl-1rl` el panel colapsa
(`vmin == vmax`), la normalización dividía por cero y el HV salía del valor
crudo (~1e8 → ~2.2e15). Es como el original invalida esas instancias (NaN);
ahora, si el marco final sigue sin escala, la instancia se **excluye** de la
media y queda anotada en `degenerate_instances`/`test_degenerate_instances` del
`llamea_best_info.json`.

En `..._ge` el marco de test sí queda degenerado en esa instancia
(`test_degenerate_instances: ["KC59-2fl-1rl.txt"]`) y se excluye, que es el
comportamiento correcto. En las 8 instancias de train no hay ninguna.

> El conjunto degenerado depende del marco (panel + candidato), así que no es
> siempre la misma instancia: en `..._feedback` (test) fue `KC18-4fl-1un.txt`.
> El `llamea_best_info.json` viejo de `..._feedback` conserva el `test_fitness`
> malo (`2.75e14`) porque se generó con el bug; el valor corregido es 0.429711.

---

## Comparación pareada de operadores (11 repeticiones)

La tabla de una corrida no compara operadores (el marco incluye a sus propios
candidatos y usa una sola semilla). `scripts/compare_operators.py` corre 11
repeticiones independientes con las mismas semillas para todos los operadores, un
solo marco por instancia y una sección cabeza a cabeza; método y discusión
completos en [`comparacion-justa.md`](comparacion-justa.md). Resultados
(`results/evaluations/b100_ge_vs_clasicos_11x/`, 8 instancias de test, MOEA/D
30/30, sin mutación):

| Operador | HV (marco compartido) | vs OX | vs PBX | vs PMX | vs CX | vs ERX |
|---|---:|---:|---:|---:|---:|---:|
| `feedback_b30` | **0.252647** | 11/11, p=0.001 | 11/11, p=0.001 | 11/11, p=0.001 | 11/11, p=0.001 | 11/11, p=0.001 |
| `b100_ge` | 0.107449 | 9/11, p=0.083 | 6/11, p=0.520 | 11/11, p=0.001 | 11/11, p=0.001 | 11/11, p=0.001 |
| PBX | 0.108003 | — | — | — | — | — |
| OX | 0.094046 | — | — | — | — | — |
| PMX | 0.084704 | — | — | — | — | — |
| ERX | 0.071390 | — | — | — | — | — |
| CX | 0.054396 | — | — | — | — | — |

(Cada celda "n/11" es victorias pareadas; el p es el de Wilcoxon. Con 11 pares el
mínimo alcanzable es 0.0010.)

Lectura:

- **El operador de 100 generaciones no supera a PBX**: 6 de 11, p = 0.52. Está al
  nivel de PBX y por encima de OX solo marginalmente. La afirmación "gana a todos
  los clásicos" del reporte de esa corrida era un artefacto de comparar dentro de
  su propio marco y con una semilla.
- **El operador de 30 generaciones con feedback diagnóstico gana 11 de 11 a los
  diez clásicos** (incluidos los de dos hijos), con `dz` de 2.2 a 4.0. Es el
  resultado que se sostiene.
- Con `--mutation` la conclusión no cambia, solo el margen
  (`results/evaluations/b100_ge_vs_clasicos_11x_mut/`).

---

## Contabilidad de tokens

Cada petición al servidor OpenAI-compatible devuelve `usage`
(`prompt_tokens`, `completion_tokens`, `total_tokens`). `VLLM_LLM` los acumula
(`usage_snapshot()`, `usage_history`) y, si el servidor no los reporta, los
estima por longitud (~4 caracteres/token). Con ellos:

- `llamea_logs/exp-*/generation_summary.jsonl` registra, por generación, los
  acumulados (`tokens_prompt`, `tokens_completion`, `tokens_total`,
  `requests`) y los deltas de esa generación (`*_gen`).
- `report.md` incluye una sección **Tokens consumidos** (total, media y máximo
  por generación).
- `<experimento>/tokens.json` guarda el acumulado, el histórico por petición y
  el `max_tokens`/`temperature` usados.

Ejemplo (2 generaciones, MOEA/D, 0.5B, 512 tokens de salida por petición):
3 695 tokens totales, ~1 848 por generación (1 532–1 782 de prompt y 169–212
de salida). El prompt domina el costo, así que recortar el ejemplo ayuda tanto
en tokens como en el sesgo de copia.

