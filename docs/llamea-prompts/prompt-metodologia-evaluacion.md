# Prompt 2 — Metodología completa del marco de evaluación

> Copia y pega el bloque de abajo en Claude. Es autocontenido: incluye protocolo,
> fórmulas, estructura de salida, resultados reales y comandos. No necesita leer
> archivos del repo.

---

## PROMPT

Eres un investigador que debe redactar el apartado de **metodología de evaluación completa** de un artículo sobre búsqueda automática de operadores de cruce (GE y LLaMEA) para mQAP. Escribe en español académico, con precisión algorítmica y sin inventar nada: usa solo lo que te doy aquí. El texto debe ser reproducible y defendible ante un revisor.

### 1. Diseño experimental

- Panel de operadores: se comparan **LLaMEA** (un `crossover(x,y)` escrito por el LLM), operadores **GE** (fenotipos de la gramática `naturals.bnf`, etiquetas Best/Middle/Worst de `*_operators.txt`) y clásicos exactos de la evaluación original (`CX_Cross`, `PMX_Cross`, adaptados para devolver los dos hijos del mismo corte).
- **Sin OX** en ninguna comparación que incluya LLaMEA.
- Protocolo GE completo: **3 solvers** (MOEAD, NSGAII, SMSEMOA) × **2 mutaciones** (NullMutation / Swap_Mutation, prob. 0.05) × **3 presupuestos** (10k=10 500, 30k=31 500, 50k=52 500 evaluaciones = 100/300/500 generaciones con población 105). Se activa con `GE_PROTOCOL=1` y **sin** pasar `SOLVER` (si se pasa `SOLVER=MOEAD` solo se genera `MOEAD/` y el resultado no es comparable con la GE).
- Los 3 presupuestos salen de **una sola corrida del MOEA** por `(operador, repetición, instancia, protocolo)` mediante un callback de snapshots en las generaciones 100/300/500 (idéntico al `MyCallback` del original, `evaluation/algorithms.py:23-43`). No son corridas independientes.
- Instancias: **20 de test** (`datasets/mqap/test/`), con 2, 3, 4 y 5 objetivos; **31 repeticiones** pareadas (misma semilla base 42 + réplica, misma población inicial por `(instancia, repetición)` para todos los operadores).
- Coste: con 3 operadores, 3 × 31 × 20 × 6 = 11 160 tareas (≈30 480 frentes).

### 2. Estructura de salida estándar (idéntica a `evaluation_results/` del original)

```
evaluation_results/
├── initial_solutions/{permutation,real}/<inst>/experiment_001..031.txt + seeds.txt
├── MOEAD/10k|30k|50k/
│   ├── fronts/<inst>/<config>_<NM|WM>_<solver>.pof  (+ _norm, best_front_consolidated*)
│   ├── <inst>.csv
│   └── Figures/{NoMutation,WithMutation}/<inst>_<NM|WM>.png
├── NSGAII/10k|30k|50k/    (igual)
├── SMSEMOA/10k|30k|50k/   (igual)
└── run_metrics/evaluation_<inst>.json
```

Las claves reales uniformes y sus permutaciones se generan juntas (random-key decode: `permutation = argsort(real)`), como `problem/instance.py` del original. Los `.pof` son el frente de la repetición mediana por HV, con cabecera `# N M`. Las figuras son boxplots agrupados por prefijo, con límites por instancia calculados sobre todos los solvers/presupuestos y ensanchados 5 % (`make_plots.py` del original).

### 3. Cálculo del HV y punto de nadir (el corazón metodológico)

Fuente: `AdrianMP1/MOHH-Specialized-Operators@main`, `evaluation/experiment.py:361-422` y `core/utilities/algorithm/MO.py:58-66`.

Por **instancia y presupuesto**:

1. **Consolidado**: unión de *todos* los puntos de *todos* los frentes de todas las configuraciones y repeticiones (`consolidated[label].union(...)`, línea 335). Duplicados eliminados.
2. **Frente y filtrado**: `best_front = non_dominated_sorting_vectorized(consolidado)`. Si tiene ≤2 puntos y quedan puntos en el consolidado, se aplica `comprobate`: se pelan los puntos del frente y se recalcula hasta que tenga >2 o no quede nada (líneas 368-374). Para minimización, `sign = -1` en el filtro (`MO.py:9-11`).
3. **Límites**: `vmin, vmax = min(best_front, axis=0), max(best_front, axis=0)` (por objetivo, línea 391).
4. **Normalización**: `f_norm = (f - vmin) / (vmax - vmin)` (`normalize_values`, líneas 171-173).
5. **Nadir fijo**: `nadir_point = np.array([1.1] * K)` — **no se calcula desde los frentes**; `compute_nadir_point` está comentado en el original (líneas 397-399). En la copia local que generó los resultados de referencia se generalizó a K objetivos (`experiment.py:496-499`); el GitHub hardcodea `[1.1, 1.1]`, que es **incompatible** con las instancias de 3-5 objetivos (pymoo lanza `ValueError: 1D array must have length K but it has length 2`). Nuestro arnés usa `[1.1] * K`.
6. **HV**: `compute_hypervolume(nadir, f_norm)` = `pymoo.indicators.hv.HV(ref_point=nadir)(f_norm)`. Sin otra transformación.

Verificación realizada: el pipeline completo (consolidado → no dominados → comprobate → normalización → HV) se comparó contra una réplica literal del original para **K=2, 3, 4 y 5** (12 frentes aleatorios por K): frente idéntico y HV idénticos con `np.allclose`. El NDS y `comprobate` solos dieron 300/300 frentes idénticos.

**Detalle honesto para el texto**: el repositorio original tiene dos inconsistencias internas con este dataset — el nadir 2D y `get_reference_directions("uniform", K, n_points=105)`, que lanza excepción para K=4 y K=5 (pymoo solo permite 84 o 120 direcciones en K=4). El arnés corrige ambas: nadir `[1.1]*K` y `n_partitions` más cercano con ajuste de `pop_size` (K=2→105, K=3→105, K=4→120, K=5→126). Esto **no afecta la comparabilidad interna**: todos los operadores de una corrida usan la misma población en la misma instancia.

### 4. Comparabilidad — la regla que un revisor va a exigir

- El HV **solo es comparable dentro de una misma corrida**, porque la normalización depende del marco y el marco depende de qué operadores entren. Por eso los tres contrastes (LLaMEA vs GE, GE vs clásicos, LLaMEA vs clásicos) deben salir de **una sola corrida** con los cuatro operadores y tres `--contrast`.
- Dos corridas distintas tienen marcos distintos → sus HV **no** se cruzan en valor absoluto.
- Tampoco se comparan los HV contra los CSV del `Experiment_001/002` original: difieren el marco (panel distinto), `N_EXPERIMENTS` (11 en el original vs 31 aquí), las semillas (time-based en el original vs deterministas aquí) y las poblaciones iniciales.
- Cada operador se reporta con: media, desviación, mediana, min y max de HV sobre las 31 repeticiones; estabilidad del ranking (veces 1.º); y diferencias mínimas detectables al 95 %.

### 5. Contrastes estadísticos

- **Wilcoxon pareado de dos colas** sobre las 31 repeticiones (mismo `seed`, misma población inicial).
- **Holm** para controlar la familia de comparaciones.
- Se reportan: diferencia media, IC 95 %, efecto `dz` (Cohen para muestras pareadas), `p` y `p_holm`.
- Se calcula la **diferencia mínima detectable** al 95 % para no sobreinterpretar mejoras pequeñas.
- Se escriben tres vistas: marco compartido de todos, marco propio de la pareja y contraste entre grupos.

### 6. Resultado real ya obtenido (`results/evaluations/llamea_vs_ge_best/`)

| Operador | Media HV (31 reps) | Desv. | Veces 1.º |
|---|---:|---:|---:|
| LLaMEA (gen 909) | **0.475874** | 0.015508 | **31/31** |
| GE Best | 0.276445 | 0.008557 | 0/31 |

Diferencia **+0.199429** [IC95 +0.193101, +0.205757], **+72.1 %**, `dz = 11.56`, `p = 9.313e-10` (Holm igual). Mínima diferencia detectable: 0.006328. LLaMEA gana 31/31 pares, sin empates ni fallos. Nota: esa corrida se lanzó solo con MOEAD, así que es orientativa; la versión estricta usa los 6 protocolos.

### 7. Comandos de reproducción (incluir en el apéndice)

```bash
OPERATORS="llamea=results/searches/glm-gptoss2/llamea_best_operator.py \
ge_best=ge:Best@results/Experiment_002/MOEAD_operators.txt \
cx=scripts/ge_classics/cx_ge.py pmx=scripts/ge_classics/pmx_ge.py" \
COMPARE_GROUPS="base=cx,pmx ge_best=ge_best llamea=llamea" \
CONTRASTS="llamea:ge_best ge_best:base llamea:base" \
GE_PROTOCOL=1 INSTANCES=all REPEATS=31 CPUS=52 \
NAME=comparacion_completa bash slurm/compare-operators.sh
```

`COMPARE_GROUPS` (no `GROUPS`: es una variable reservada de bash con los GIDs). `GE_PROTOCOL=1` sin `SOLVER`. Reanálisis sin repetir el MOEA: añadir `LOAD_FRONTS=1` y un `NAME` nuevo. El artefacto `operators.txt` guarda el código exacto de cada operador usado.

### 8. Lo que debes entregar

1. Apartado de metodología (600-900 palabras) con: diseño pareado, snapshots de presupuesto, construcción del marco consolidado, fórmula del HV con nadir fijo, justificación de por qué el marco se comparte y consecuencias de que no se comparta.
2. Un párrafo específico titulado "Punto de referencia y comparabilidad" que explique el nadir `[1.1]^K`, por qué no se calcula desde los frentes y por qué los HV de corridas distintas no son comparables.
3. La tabla de resultados reales del apartado 6 y su interpretación (tamaño del efecto frente al ruido).
4. Un párrafo de "Amenazas a la validez" con las tres desviaciones del apartado 3 y 4 (nadir, `n_points`, repeticiones/semillas) y por qué no invalidan las comparaciones internas.
5. El comando de reproducción y la lista de artefactos.
6. No menciones OX en ninguna parte.

---
