# Prompt 1 — Metodología de búsqueda: cómo funciona el ranking GE en LLaMEA

> Copia y pega el bloque de abajo en Claude. Es autocontenido: incluye el
> código real, las constantes, las rutas y los datos de la corrida. No necesita
> leer archivos del repo.

---

## PROMPT

Eres un investigador que debe redactar el apartado de **metodología de búsqueda** de un artículo sobre LLaMEA aplicado a mQAP (multi-objective Quadratic Assignment Problem). Escribe en español académico, con precisión algorítmica y sin inventar nada: usa únicamente los datos y el código que te doy aquí. El resultado debe poder ir directamente a una tesis o paper.

### 1. Contexto

- Proyecto: **MOHH-R** (`github.com/AdrianMP1/MOHH-Specialized-Operators` es la evaluación gramatical original; LLaMEA sustituye la gramática por un LLM que escribe `crossover(x, y)`).
- Problema: mQAP. Las instancias de entrenamiento son **80** (`datasets/mqap/train/`) y las de test **20** (`datasets/mqap/test/`). Cada instancia declara entre 2 y 5 objetivos (`head -1 datasets/mqap/test/KC*.txt`).
- El MOEA interno es el mismo en GE y en LLaMEA: `MOSolver` con MOEAD, **300 generaciones**, población **105**, semilla del MOEA fijada por repetición.
- La búsqueda es **1+1** (`--simple`, `--n-parents 1 --n-offspring 1`), presupuesto **1000 iteraciones**, modelo `openai/gpt-oss-20b`, temperatura 0.7, top-p 0.8, top-k 20, `max_tokens` 8192, `operator_memory` 6, `thinking=auto`, `prior_every=10`, `repeat_penalty=0.5`, `patience=0`, `eval_repeats=31`, `fitness_workers=31`, `seed=42`.
- Los 5 operadores iniciales (priors) son `OX, PMX, CX, PBX, ERX` (`src/mohh/llamea/simple_priors.py:127-133`).

### 2. Cómo se calcula el HV que ve el LLM (idéntico al de la evaluación GE)

`PopulationEvaluator` (`src/mohh/llamea/fitness.py:1813-2039`) hace, para **cada instancia**:

1. Mantiene un **marco consolidado incremental** (`_fold_ge_frame`, líneas 1860-1884) con los frente de *todo* lo evaluado hasta ese momento (priors + todos los candidatos). Cada frente nuevo se fusiona con el marco vigente mediante `_nd_merge`, equivalente a recalcular el filtro no dominado sobre todo el histórico en `O(|marco|·|nuevo|)`.
2. Saca el frente no dominado del consolidado y le aplica `comprobate`: si el frente tiene ≤2 puntos y quedan puntos, se pelan y se recalcula (`fitness.py:785-806`, `comprobate_frame`).
3. Del frente resultante toma `vmin, vmax = frame.min(axis=0), frame.max(axis=0)` (por objetivo). Si `vmax == vmin` en algún objetivo, la instancia se excluye (misma invalidez que el original con NaN).
4. Normaliza cada frente con `(f - vmin) / (vmax - vmin)` (`normalize_front`, líneas 696-709).
5. Calcula el HV con nadir **fijo** `[1.1] * K` mediante `pymoo.indicators.hv.HV(ref_point=...)` (`compute_hypervolume`, `src/mohh/core/utilities/algorithm/MO.py:58-66`). El recorte opcional `clip_to_reference_box` a `[0, 1.1]` es neutro aquí porque cada frente evaluado forma parte del marco.
6. El marco **crece con cada operador evaluado**, así que los HV se recalculan en cada evaluación (los frentes del MOEA están cacheados; no se vuelve a correr el solver). En una evaluación dada, padres y descendientes comparten marco → comparación justa.

### 3. El ranking GE (el corazón que debes explicar)

Es la función `compute_rank` de `src/mohh/generation/utilities/algorithm/HH_auxiliars.py:6-24`, literal:

```python
def compute_rank(hypervolumes):
    sorted_values = sorted(hypervolumes, reverse=True)
    ranks = {}
    for i, value in enumerate(sorted_values):
        ranks.setdefault(value, []).append(i + 1)
    average_ranks = {v: sum(r)/len(r) for v, r in ranks.items()}
    return [average_ranks[v] for v in hypervolumes]
```

Es un **ranking de competencia con empates promediados**: se ordena de mayor a menor HV, se asigna la posición `i+1`, y los valores idénticos reciben el promedio de sus posiciones. **Rango 1 = mejor, rango K = peor**, donde `K` es el número de operadores distintos en el lote.

En `PopulationEvaluator._evaluate_batch` (líneas 1925-2038):

- El lote es `padres + hijos` deduplicados por código (`unique`).
- Para **cada instancia** se calcula el HV de cada operador del lote (pasos del apartado 2) y se llama `compute_rank(hypervolumes)`.
- Cada operador acumula un rango por instancia; su **fitness es el rango medio sobre las instancias válidas**: `mean_rank = mean(ranks)`.
- Con `raw_rank=True` (lo que se usa en esta corrida, ver `search_summary.json: fitness_metric = "GE population mean rank"`, `fitness_direction = "lower is better"`) el score es directamente `mean_rank` (**menor es mejor**). Con `raw_rank=False` se invierte a `(K + 1) - mean_rank` para clientes que maximizan; el orden es idéntico.
- Un operador sin frente válido recibe `+inf` (peor posible).
- El feedback que se inyecta al LLM dice literalmente: `"GE population ranking: mean rank X over N instances. Best instance rank b, worst w, K=k. Reference point: fixed nadir [1.1]*K over the consolidated frame of everything evaluated so far."`

**Ejemplo numérico para el texto** (4 operadores en una instancia): HV = `[0.52, 0.31, 0.52, 0.10]` → ordenados `[0.52, 0.52, 0.31, 0.10]` → posiciones `[1,2,3,4]` → empate en 0.52 promedia `(1+2)/2 = 1.5` → rangos `[1.5, 3, 1.5, 4]`. Si en otra instancia el mismo operador saca rango 1, su rango medio es `(1.5+1)/2 = 1.25`.

### 4. Qué hace el bucle 1+1 y cómo se usa el ranking para aceptar

`src/mohh/llamea/simple.py:521-661`:

- Cada iteración elige padre: cada 10 iteraciones un prior; si no, 50 % `current-best` (refinar) y 50 % `new-operator` (crear desde un esqueleto, **sin** código de referencia). Selección por torneo (`--parent-selection tournament`, tamaño 3), niching none, novelty_k 5.
- El prompt (`build_prompt`, líneas 106-171) lleva: `ROLE`, `TASK`, historial de los últimos 6 operadores con su **mean rank**, la leyenda `"The score is the GE mean rank over the training instances (1 = best, lower is better; 2 = worst)"`, el `REFERENCE CODE` del padre (o un `SKELETON` si es `create`), la estrategia y el formato de salida. Si la respuesta anterior falló, se añade `PREVIOUS ATTEMPT FAILED` con el error exacto.
- El candidato se compila, se valida (mínimo: permutación válida) y se evalúa con el `PopulationEvaluator`.
- **Aceptación** (`simple.py:648-661`): `improved = (error is None and fitness finito and (fitness < best.fitness or (fitness == best.fitness and hv > best.hv + 1e-12)))`. Si mejora, se actualiza `best_so_far.py`, `llamea_best_operator.py` y `best_so_far.json`, y se incrementa el contador de mejoras.

### 5. Datos reales de la corrida `glm-gptoss2` (no inventar otros)

De `results/searches/glm-gptoss2/search_summary.json` y `run.json`:

| Magnitud | Valor real |
|---|---|
| Modelo | `openai/gpt-oss-20b` |
| Iteraciones | 1000 (866 válidas, 134 inválidas, 86.6 %) |
| Mejoras aceptadas | 25 |
| Mejor generación | **909** |
| **Mean rank final** | **1.03125** (menor es mejor) |
| HV del mejor | **0.4930605191734604** |
| Operador final | `llamea_best_operator.py` |
| Fallos | 117 filtro mínimo (ValueError), 12 truncados, 5 sin frente válido |
| Tokens | 2 554 759 totales (929 519 prompt + 1 625 240 completion; 1 278 838 reasoning) |
| Tiempo inferencia | 16 463 s |
| Diversidad de código | CrystalBLEU = 0.9895 sobre 866 códigos |
| Duplicados | 875 evaluaciones distintas, 0 duplicados |
| Prior panel | OX, PMX, CX, PBX, ERX |

El operador ganador (gen 909) es un **PMX por mapeo de valores con índice inverso**, simetrizado al 50 % (swap de padres), con swap opcional 30 % y 3-ciclo 10 %, usando `np.random.*` global (reproducible con la semilla del MOEA).

### 6. Lo que debes entregar

1. Un apartado de metodología (400-600 palabras) que explique: por qué se usa rango medio en vez de HV crudo (escala cambiante del marco), cómo se construye el marco consolidado, la fórmula del `compute_rank` con empates, el significado de `mean_rank = 1.03125` y por qué `raw_rank=True`.
2. El pseudocódigo del bucle de evaluación y aceptación.
3. El ejemplo numérico de empates del apartado 3, desarrollado.
4. Una frase honesta sobre la convergencia: 25 mejoras en 1000 iteraciones, con un estancamiento largo entre las iteraciones 208 y 673 y la mejor solución apareciendo en la 909 (los datos están en `progress.txt`).
5. No menciones OX en la comparación final; si necesitas citar los priors, di "cinco operadores clásicos de referencia".

---
