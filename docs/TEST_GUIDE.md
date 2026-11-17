# Guía de tests unitarios

Cómo se prueba este repositorio, qué cubre cada test y qué falta. Las cifras
son medidas, no estimadas:

```bash
python3 -m pytest tests/ -q                                  # 358 pasan, 1 skip, ~62 s
python3 -m pytest tests/ -q --cov=mohh --cov-report=term     # cobertura: 46.4 %
```

`pytest-cov` está instalado; `coverage` como módulo suelto no, así que se usa el
plugin de pytest.

---

## 1. Reglas de la casa

| regla | por qué |
|---|---|
| Un archivo por subsistema: `tests/test_<modulo>.py` | se encuentra lo que falta mirando el módulo |
| Nombre `test_<funcion>_<escenario>` | el fallo dice qué se rompió |
| Sin red, sin GPU, sin vLLM | la suite tiene que correr en el login y en un contenedor |
| Sin MOEA largos (población y generaciones mínimas) o doble de prueba | un test de 30 s no se ejecuta |
| Semilla fija o generador con semilla | los números deben ser reproducibles |
| Sin rutas absolutas ni datos de una máquina | los datos vienen de `datasets/` y de rutas relativas |
| Sin `sleep` | el tiempo se controla con dobles, no esperando |

El doble de prueba del LLM ya existe: `mohh.llamea.vllm` acepta cualquier
cliente con la interfaz del servidor; los tests inyectan uno falso
(`tests/test_llamea_simple.py`).

---

## 2. Mapa por sección

`%` es la cobertura real de `pytest --cov=mohh`; `test` es el archivo que lo
cubre hoy.

### 2.1 `core` — base compartida (MO, indicadores, representación, parámetros)

| módulo | stmt | % | test | huecos |
|---|---|---|---|---|
| `core/utilities/algorithm/MO.py` | 43 | 86.0 | `test_llamea_nondominated`, `test_ge_evaluation` | `compute_hypervolume` con frentes degenerados |
| `core/utilities/algorithm/HH_functions.py` | 131 | 47.3 | — | primitivas de la gramática (se usan en GE) |
| `core/utilities/algorithm/sdd.py` | 153 | 25.5 | `test_indicator_directions` | descomposición de la dispersión por celda |
| `core/utilities/metrics.py` | 148 | 29.7 | — | temporizadores y volcado de métricas |
| `core/utilities/instance_utils.py` | 16 | 25.0 | — | `instance_paths(train=…)` y `DATASET_PATH` |
| `core/utilities/load_modules.py` | 15 | 20.0 | — | carga dinámica de módulos |
| `core/utilities/paths.py` | 3 | 66.7 | — | — |
| `core/utilities/print_utils.py` | 8 | 37.5 | — | — |
| `core/indicators/igd.py` | 37 | 97.3 | `test_indicators_pymoo` | — |
| `core/indicators/deltap.py` | 65 | 78.5 | — | ΔP con referencias vacías |
| `core/indicators/r2.py` | 47 | 66.0 | — | R2 con una sola referencia |
| `core/indicators/reference_sets.py` | 141 | 26.2 | `test_indicators_pymoo` | carga de conjuntos de referencia del disco |
| `core/indicators/directions.py` | 17 | 94.1 | `test_indicator_directions` | — |
| `core/representation/grammar.py` | 238 | **0.0** | — | **gramática BNF: parseo, genotipo→fenotipo, individual's inválidos** |
| `core/representation/mapper.py` | 47 | **10.6** | — | gramática → árbol |
| `core/representation/population.py` | 79 | **21.5** | — | población inicial y elitismo |
| `core/problem/instance.py` | 155 | 73.5 | — | carga del problema de afinidad de ubicación |
| `core/operators/operator_template.py` | 28 | 89.3 | `test_llamea` | — |
| `core/params.py` | 24 | 75.0 | — | singleton de parámetros |

### 2.2 `generation` — evolución gramatical

| módulo | stmt | % | test | huecos |
|---|---|---|---|---|
| `generation/algorithm/hyperheuristic.py` | 330 | **9.1** | `test_end_to_end_analysis` (parcial) | **bucle GE: 3 solvers × 2 mutaciones y población inicial por experimento** |
| `generation/algorithm/multiobjective.py` | 79 | 72.2 | — | — |
| `generation/operators/crossover.py` | 62 | **0.0** | — | cruces de la gramática |
| `generation/operators/mutation.py` | 30 | **0.0** | — | mutaciones GE |
| `generation/operators/selection.py` | 27 | **0.0** | — | ruleta y selección por torneo |
| `generation/operators/initialization.py` | 34 | **0.0** | — | generación de la población inicial |
| `generation/operators/operator.py` | 6 | **0.0** | — | contrato del operador |
| `generation/saver.py` | 150 | **18.7** | — | escritura de los operadores descubiertos y del CSV de la GE |
| `generation/utilities/algorithm/reference_fitness.py` | 135 | 37.8 | `test_llamea_reporting` (parcial) | GE contra el panel de referencia |
| `generation/utilities/algorithm/HH_auxiliars.py` | 27 | 55.6 | `test_llamea_reporting` | `compute_rank` (empates y rango medio) |
| `generation/problem/instance.py` | 2 | 100 | — | — |
| `generation/representation/population.py` | 8 | 50.0 | — | — |
| `generation/params.py` | 34 | 44.1 | — | — |
| `generation/main.py` | 16 | 25.0 | — | CLI |

### 2.3 `evaluation` — protocolo GE (el marco de comparabilidad)

| módulo | stmt | % | test | huecos |
|---|---|---|---|---|
| `evaluation/experiment.py` | 315 | **15.2** | `test_ge_evaluation` (11) | **marco consolidado, `comprobate`, normalización, nadir, frente mediano, CSV por solver** |
| `evaluation/algorithms.py` | 116 | 60.3 | `test_ge_evaluation`, `test_evaluate_sharding` | — |
| `evaluation/operators/crossover.py` | 77 | **0.0** | — | cruces clásicos, los de la comparación |
| `evaluation/operators/mutation.py` | 28 | 71.4 | — | — |
| `evaluation/problem/instance.py` | 87 | **13.8** | — | `set_initial_solutions` por experimento (la base del pareado) |
| `evaluation/params.py` | 27 | 51.9 | — | — |
| `evaluation/make_plots.py` | 117 | **8.5** | — | figuras del original |
| `evaluation/main.py` | 11 | 36.4 | — | — |

### 2.4 `llamea` — búsqueda con LLM

| módulo | stmt | % | test | huecos |
|---|---|---|---|---|
| `llamea/fitness.py` | 998 | 76.2 | `test_llamea`, `test_llamea_simple`, `test_ge_evaluation`, `test_llamea_nondominated` | ramas del evaluador de población que no se ejercitan |
| `llamea/run.py` | 535 | **17.2** | `test_llamea` (parcial) | **CLI: topes, presupuestos, `MAX_MODEL_LEN`, GE frame** |
| `llamea/reporting.py` | 472 | 69.3 | `test_llamea_reporting` (28), `test_llamea_best_global` | layout de los informes |
| `llamea/vllm.py` | 348 | 75.6 | `test_llamea_metrics` | reconexión, reintentos, contadores de tokens |
| `llamea/simple.py` | 297 | 81.8 | `test_llamea_simple` (23) | — |
| `llamea/models.py` | 232 | 62.5 | `test_llamea_models` (14) | — |
| `llamea/parsing.py` | 141 | 71.6 | `test_llamea` | formatos poco frecuentes de respuesta |
| `llamea/metrics.py` | 127 | 78.7 | `test_llamea_metrics` | — |
| `llamea/reference.py` | 101 | 84.2 | `test_llamea` | — |
| `llamea/statistics.py` | 80 | 96.2 | `test_llamea_statistics` | — |
| `llamea/operator.py` | 36 | 88.9 | `test_llamea_simple` | — |
| `llamea/ge_operators.py` | 36 | 94.4 | `test_ge_operators` | — |
| `llamea/simple_priors.py` | 17 | 88.2 | `test_llamea_simple` | — |
| `llamea/prompts.py` | 20 | 95.0 | `test_llamea` | — |
| `llamea/priors.py` | 21 | 47.6 | — | carga de los priors de GE |
| `llamea/export_prompts.py` | 29 | **0.0** | — | volcar los prompts a disco |

### 2.5 `analysis` — métricas y tablas

| módulo | stmt | % | test | huecos |
|---|---|---|---|---|
| `analysis/io_utils.py` | 317 | **17.4** | `test_end_to_end_analysis` (parcial) | **leer el layout del original (CSV por instancia, `.pof`)** |
| `analysis/hv_scale.py` | 161 | 45.3 | `test_analysis_hv_scale` (15) | escala de HV entre evaluaciones |
| `analysis/indicators_all.py` | 149 | 61.1 | `test_indicator_directions`, `test_indicators_pymoo` | — |
| `analysis/stats.py` | 146 | 30.8 | `test_indicator_directions` | Wilcoxon pareado y corrección de Holm |
| `analysis/phenotype.py` | 82 | 18.3 | — | pasar el fenotipo GE a Python |
| `analysis/complexity.py` | 68 | 22.1 | — | medidas de complejidad |
| `analysis/sdd_all.py` | 64 | 15.6 | — | descomposición de la dispersión |
| `analysis/layout.py` | 62 | 43.5 | — | rutas del layout |
| `analysis/cache.py` | 49 | 22.4 | — | caché de operadores |
| `analysis/code_diversity.py` | 48 | 93.8 | `test_code_diversity` | — |

### 2.6 Infraestructura

| módulo | stmt | % | test |
|---|---|---|---|
| `cli.py` | 185 | 21.6 | `test_evaluate_sharding` |
| `run_evaluation.py` | 65 | 67.7 | `test_evaluate_sharding` |
| `parallel.py` | 38 | 84.2 | `test_slurm` (44) |
| `main.py` | 40 | 17.5 | — |

---

## 3. Los huecos, por prioridad

### P0 — el marco de comparabilidad

Un error aquí invalida la tabla final, así que van primero.

| módulo | stmt sin cubrir | qué hay que probar |
|---|---|---|
| `evaluation/experiment.py` | 267 | `comprobate` (pela mientras el frente tenga ≤2 puntos); `vmin/vmax` **del frente no dominado consolidado**, no de todos los puntos; nadir fijo `[1.1]*K`; normalización `(f−vmin)/(vmax−vmin)`; el frente mediano es el de **HV mediana** (no `np.median`, que interpola); el CSV por solver incluye las columnas estándar |
| `generation/algorithm/hyperheuristic.py` | 300 | el bucle GE completo (3 solvers × 2 mutaciones); una población inicial **por experimento**, guardada y reutilizable; el guardado de los operadores descubiertos |
| `evaluation/problem/instance.py` | 75 | `set_initial_solutions(experiment=…, kind=…)`: que dos operadores con la misma semilla receive **exactamente** la misma población (es lo que hace pareado el experimento) |
| `core/representation/grammar.py` | 238 | gramática BNF: parseo, fenotipo, Individuo inválido, `non_dominated_sorting` |
| `analysis/io_utils.py` | 262 | leer el layout del original: `evaluation_results/<solver>/<budget>/<instancia>.csv` y los `.pof` |

### P1 — código de producción sin ningún test

| módulo | stmt | por qué importa |
|---|---|---|
| `generation/operators/crossover.py` | 62 | los cruces que compite la GE |
| `evaluation/operators/crossover.py` | 77 | los clásicos de la comparación |
| `generation/operators/{mutation,selection,initialization,operator}.py` | 97 | el motor de la GE |
| `llamea/export_prompts.py` | 29 | vuelca los prompts a disco |
| `core/representation/mapper.py` | 42 | mapeo de la gramática |
| `core/representation/population.py` | 62 | población inicial y elitismo |

### P2 — utilidades con cobertura parcial

`analysis/{phenotype,complexity,cache,sdd_all,layout}.py`,
`core/utilities/{metrics,load_modules,instance_utils}.py`,
`core/utilities/algorithm/{sdd,HH_functions}.py`,
`core/indicators/{reference_sets,deltap,r2}.py`,
`generation/{saver,params}.py`, `evaluation/{params,make_plots}.py`,
`llamea/{run,priors}.py`, `cli.py`, `main.py`.

---

## 4. Matriz función → test (lo que debe estar cubierto)

Las funciones que sostienen el protocolo. Cada una necesita al menos un test
propio; la columna de la derecha es el que la cubre hoy.

| # | función | módulo | test actual |
|---|---|---|---|
| 1 | `_nondominated` | `llamea/fitness.py` | `test_llamea_nondominated` (contra la referencia O(n²)) |
| 2 | `comprobate_frame` | `llamea/fitness.py` | falta el caso del peel |
| 3 | `normalize_front` | `llamea/fitness.py` | `test_llamea` |
| 4 | `clip_to_reference_box` | `llamea/fitness.py` | `test_ge_evaluation` (vía `clip_to_nadir`) |
| 5 | `shared_frame` | `scripts/compare_operators.py` | `test_shared_frame_ignores_failed_front_but_uses_the_rest` |
| 6 | `hv_values` | `scripts/compare_operators.py` | `test_failed_front_counts_as_zero_or_is_excluded` |
| 7 | `pairwise_design` | `scripts/compare_operators.py` | `test_group_contrasts_compares_two_groups` |
| 8 | `collect_fronts` | `scripts/compare_operators.py` | `test_compare_operators` |
| 9 | `write_ge_layout` | `scripts/compare_operators.py` | `test_ge_layout_matches_the_original_evaluation` |
| 10 | `compute_rank` | `generation/…/HH_auxiliars.py` | `test_llamea_reporting` |
| 11 | `run_experiments` | `evaluation/experiment.py` | `test_ge_evaluation` (parcial) |
| 12 | `comprobate` (original) | `evaluation/experiment.py` | — |
| 13 | `generate_incremental_seeds` | `evaluation/experiment.py` | — |
| 14 | `write_front` | `evaluation/experiment.py` | — |
| 15 | `set_initial_solutions` | `evaluation/problem/instance.py` | — |
| 16 | `compile_operator` | `llamea/fitness.py` | `test_llamea_simple` (4 tests) |
| 17 | `build_prompt` | `llamea/prompts.py` | `test_llamea_simple` (5 tests) |
| 18 | `paired_summary` / `add_holm` | `llamea/statistics.py` | `test_llamea_statistics` (8) |
| 19 | `run_parallel` | `parallel.py` | `test_slurm` (44) |
| 20 | `compute_hypervolume` | `core/utilities/algorithm/MO.py` | `test_ge_evaluation` |

---

## 5. Casos que deben tener test (invariantes del protocolo)

Estos son los que, si se rompen, invalidan una tabla sin que nadie lo note.
Cada uno va con el archivo donde debería vivir.

1. **El marco compartido sale del frente no dominado consolidado**, tras
   `comprobate`, y sus `vmin/vmax` son los de ese frente
   (`evaluation/experiment.py`).
2. **El nadir es `[1.1] * K` fijo**, no el máximo de los puntos
   (`evaluation/experiment.py`, `llamea/fitness.py`).
3. **La normalización es `(f − vmin) / (vmax − vmin)`** con los límites del
   marco de esa instancia y presupuesto (`llamea/fitness.py`).
4. **Todos los operadores comparten la población inicial de la repetición**
   (`evaluation/problem/instance.py`, `scripts/compare_operators.py`): es lo que
   hace pareada la comparación.
5. **Un fallo es HV 0**, no una repetición descartada
   (`scripts/compare_operators.py`).
6. **El frente mediano es el de HV mediana**, no `np.median`
   (`evaluation/experiment.py`).
7. **El layout de salida es el del original**: `<solver>/<budget>/<instancia>.csv`
   y `fronts/<instancia>/<config>.pof` (`scripts/compare_operators.py`).
8. **`_nondominated` da el mismo conjunto y el mismo orden** que la
   referencia O(n²) (`tests/test_llamea_nondominated.py`).
9. **La selección de instancias es ordenada antes de muestrear**, para que el
   mismo `--seed` elija las mismas en el clúster y en local
   (`scripts/compare_operators.py`, ya cubierto).

---

## 6. Qué no va en un test

- **LLM real**: siempre un doble. El doble de `mohh.llamea.vllm` devuelve
  respuestas preparadas.
- **MOEA largo**: población y generaciones mínimas, o una instancia sintética.
  El presupuesto por defecto (300×105 sobre 80 instancias) tarda minutos y no
  es un test.
- **Rutas absolutas**: todo relativo al repositorio o a `datasets/`.
- **Números que dependan de la máquina**: se comparan relaciones (igualdad,
  orden, signo), no valores absolutos, salvo los que Saleen de una fórmula
  cerrada.
- **`sleep`**: el determinismo se consigue con dobles, no esperando.

---

## 7. Cómo añadir un test

1. El archivo va en `tests/test_<modulo>.py`; si el módulo es nuevo, el
   archivo también.
2. El nombre describe función y escenario: `test_shared_frame_ignores_failed_front_but_uses_the_rest`.
3. Si toca el protocolo, el caso va también en la sección 5 de esta guía.
4. Se ejecuta `python3 -m pytest tests/ -q` y debe quedar en verde.
5. Si el módulo baja de cobertura, se actualiza la tabla de la sección 2.

## 8. Cobertura completa por módulo

```
46.4 % del paquete (8091 sentencias, 4336 sin cubrir)
```

Módulos al 0 %: `core/representation/grammar.py` (238),
`evaluation/operators/crossover.py` (77),
`generation/operators/crossover.py` (62),
`generation/operators/initialization.py` (34),
`generation/operators/mutation.py` (30),
`generation/operators/selection.py` (27),
`generation/operators/operator.py` (6),
`llamea/export_prompts.py` (29).

Por debajo del 20 %:
`evaluation/make_plots.py` (8.5),
`generation/algorithm/hyperheuristic.py` (9.1),
`core/representation/mapper.py` (10.6),
`evaluation/problem/instance.py` (13.8),
`evaluation/experiment.py` (15.2),
`analysis/sdd_all.py` (15.6),
`llamea/run.py` (17.2),
`analysis/io_utils.py` (17.4),
`main.py` (17.5),
`analysis/phenotype.py` (18.3),
`generation/saver.py` (18.7).
