# Auditoría: arnés de comparación vs evaluación original

Fuente: `AdrianMP1/MOHH-Specialized-Operators@main`, archivos bajo
`src/mohh/evaluation/` y `src/mohh/core/`. Revisado contra
`scripts/compare_operators.py`, `src/mohh/llamea/fitness.py`,
`src/mohh/generation/algorithm/multiobjective.py` y
`src/mohh/evaluation/results_layout.py`.

## HV y punto de nadir (conclusión: sin desviación en la fórmula)

Rastreo literal de `nadir`, `ref_point` y `HV` en el repo original:

- `evaluation/experiment.py:398` — `nadir_point = np.array([1.1, 1.1])`,
  comentario `Fixed for normalized PFAs`. La línea siguiente
  (`compute_nadir_point(set(best_front))`) está **comentada**: el nadir NUNCA
  se calcula a partir de los frentes.
- `core/utilities/algorithm/MO.py:47-55` — `compute_nadir_point` existe pero
  no se usa en ningún sitio (solo la definición).
- `core/problem/instance.py:41-42,178` — `nadir_point`/`previous_nadir_point`
  son bookkeeping inicializado a ceros; tampoco intervienen en el HV.
- `MO.py:58-66` — `HV(ref_point=ref_point)` de pymoo, sin nada más.

**Ojo con el número de objetivos.** El repo de GitHub tiene
`np.array([1.1, 1.1])` (solo 2 objetivos): con pymoo eso **falla** si el frente
tiene 3/4/5 columnas (`ValueError: 1D array must have length 5 but it has
length 2`). Los 20 test instances tienen 2, 3, 4 y 5 objetivos
(`head -1 datasets/mqap/test/KC*.txt`).

La copia local `avance/MOHH original/src/mohh/evaluation/experiment.py:496-499`
—que es la que produjo los resultados de referencia `experimentos_001_002/`—
ya lo corrigió:

```python
n_obj = instance.k_objectives
nadir_point = np.array([1.1] * n_obj) # Fixed for normalized PFAs
```

Arnés: `NADIR_FACTOR = 1.1` (`fitness.py:693`) y
`compute_hypervolume(np.full(K, 1.1), normalized)` — equivalente a `[1.1] *
n_obj`. **No hay desviación respecto de la evaluación que generó los
resultados de referencia** (sí la hay respecto del código viejo de GitHub, que
ya no sirve para estos datos). Nuestro `MO.py` solo añade `RunMetrics`; la
llamada a `HV` está intacta.

La única diferencia que afecta al HV es indirecta: los **límites de
normalización** `(vmin, vmax)` salen del marco consolidado, y el marco depende
de qué operadores y cuántas repeticiones entren en la corrida (ver desviación
1). Mismo nadir y misma fórmula, distinta escala → HV no comparables entre
corridas distintas ni contra la GE publicada.

El `clip_to_reference_box` extra del arnés es no-op en el marco GE: todo punto
evaluado forma parte del marco, así que su versión normalizada cae en `[0, 1]`;
los puntos hipotéticos por encima de 1.1 darían HV 0 tanto con como sin clip.

**Frente consolidado y filtrado:** verificado numéricamente. Repliqué
`non_dominated_sorting_vectorized` + `comprobate` del original y los comparé
con `_nondominated` + `comprobate_frame` del arnés en 300 conjuntos aleatorios
(2 objetivos, con y sin duplicados): **300/300 frentes idénticos**. El marco
se construye igual: unión de todos los puntos → `unique` → NDS → pelado si
tiene ≤2 puntos → min/max por objetivo del frente resultante.

## Coincide (sin desviación)

| Elemento | Original | Arnés |
|---|---|---|
| Solvers | MOEAD, NSGAII, SMSEMOA | igual (`GE_PROTOCOL=1`, sin `SOLVER`) |
| Mutaciones | NullMutation / Swap_Mutation | igual (`--ge-protocol` → ambas) |
| Población / generaciones | 105 / 500 | igual |
| Presupuestos | 10k=10500, 30k=31500, 50k=52500 | igual (snapshots gen 100/300/500) |
| Prob. cruce | 1.0 | `LLMOperator(prob=1.0)` |
| Prob. mutación | 0.05 (`*_MUTATION_PROBABILITY`) | `_load_mutation` lee el mismo dict |
| Contrato del operador | `HH_Operator`: `child, child[::-1]` | `LLMOperator`: idéntico |
| CX/PMX clásicos | dos hijos del mismo corte | adaptadores devuelven la tupla de dos hijos |
| Nadir | `[1.1, 1.1]` fijo | `[1.1]*K` fijo |
| Marco | unión de todos los frentes → NDS → `comprobate` | `shared_frame` + `_frame_from_points` idénticos |
| Normalización | `(f-vmin)/(vmax-vmin)` | `normalize_front` idéntica |
| HV | `pymoo.HV(ref_point=nadir)` | igual (`MO.py` solo añade logging) |
| MOEAD `ref_dirs` | `n_points=105` | `n_partitions=104` → 105 direcciones (K=2, equivalente) |
| NSGAII / SMSEMOA | `pop_size`, `sampling`, operadores | igual |
| Parser instancias | cabecera con espacios | `=` → espacio (solo formato del dataset) |
| Estructura de salida | `evaluation_results/` | `results_layout.py` la reproduce |

## Desviaciones reales

### 1. Número de experimentos (relevante para comparar con la GE publicada)

- Original: `N_EXPERIMENTS = 11` (`evaluation/params.py:20`).
- Arnés: `--repeats 31` (decisión propia).

Afecta al marco de normalización (más puntos) y a la potencia estadística.
Los HV de una corrida con 31 repeticiones **no** son comparables en valor
absoluto con los del Experiment_001/002 de la GE (11). Para réplica estricta:
`REPEATS=11`. Para comparaciones internas (LLaMEA vs GE vs clásicos, todos en
la misma corrida) 31 es válido y más potente.

### 2. Semillas (relevante para comparar con la GE publicada)

- Original: `generate_incremental_seeds` usa microsegundos del reloj +.incrementos
  aleatorios, y guarda `seeds.txt` **por instancia** (`experiment.py:50-76, 273`).
- Arnés: `seeds = [args.seed + rep]`, con `--seed 42` por defecto, la misma
  serie para todas las instancias.

Ventaja: es determinista y pareado entre todos los operadores de la corrida.
Inconveniente: no coincide con las semillas de la evaluación original.

### 3. Poblaciones iniciales (relevante para comparar con la GE publicada)

- Original: `create_solutions` genera claves reales con `np.random.random`
  (RNG global) **antes** de fijar las semillas de los experimentos, y las
  guarda en `initial_solutions/{real,permutation}/<inst>/experiment_XXX.txt`
  (`instance.py:71-103`).
- Arnés: genera `real` con `np.random.default_rng(seed + rep*1000 + index)`
  por `(instancia, repetición)` y decodifica a permutación.

Misma distribución (uniforme → random-key decode), sorteos distintos. El
arnés **no** tiene opción para cargar el `initial_solutions/` de una corrida
anterior, aunque `evaluation/params.py` ya define `INITIAL_POPULATION_PATH`
(lo usa `evaluation/experiment.py`, no `compare_operators.py`). Si se quiere
replicar exactamente la GE, hay que añadir esa opción al arnés.

### 4. Detalles menores (sin efecto numérico)

- `clip_to_reference_box` recorta a `[0, 1.1]`; en el marco GE es no-op.
- Columnas CSV: los clásicos terminan en `_<solver>` en el arnés y en `_None`
  en la GE (el generador de tablas del arnés espera `_<solver>`).
- `Figures/` las genera `results_layout.write_figures` en vez de
  `make_plots.py`, con la misma lógica de boxplots y límites ±5 %.

## Conclusión

- **Comparabilidad interna** (los tres contrastes de LLaMEA/GE/clásicos entre
  sí): correcta si salen de una sola corrida con los cuatro operadores.
- **Comparabilidad externa** (contra los HV publicados de la GE): no es
  estricta por los puntos 1-3. Para hacerla estricta hacen falta: `REPEATS=11`,
  las semillas de `seeds.txt` y las poblaciones iniciales de la GE.
