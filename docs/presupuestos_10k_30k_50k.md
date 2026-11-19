# Presupuestos 10k/30k/50k: snapshots de una sola corrida

## Resolución final (2026-11-18)

Después de revisar la copia local y el repositorio original de GitHub, el
usuario decidió **mantener snapshots**, es decir, el comportamiento actual del
arnés (idéntico al original). **No se cambia `solve_front_snapshots`** y no
hay que repetir `llamea_vs_ge_best`.

Lo que sigue es el historial de la discusión y la evidencia consultada.

## Requisito planteado inicialmente

> Los tres presupuestos deben ser **completamente independientes**. No vale
> tomar snapshots de una única corrida larga.

Esto aplica a todas las comparaciones de operadores (LLaMEA vs GE Best,
LLaMEA vs CX/PMX, etc.). Cada presupuesto debe corresponder a una ejecución
del MOEA **desde cero**, con su propia semilla y su propia trayectoria
estocástica.

## Lo que hace el original (verificado)

**El original NO hace corridas independientes.** Usa snapshots de una sola
corrida por experimento:

- `avance/MOHH original/src/mohh/evaluation/params.py`:
  `MO_POPULATION_SIZE=105`, `MO_GENERATIONS=500`, y
  `EVAL_BUDGETS = {"10k": 10500, "30k": 31500, "50k": 52500}`
  (= 105 × 100/300/500 generaciones).
- `algorithms.py:23-43` — `MyCallback` guarda el frente en las generaciones
  100, 300 y 500 dentro de **una** corrida.
- `experiment.py:165-215` — `_solve_experiment_task` llama a
  `solver.solve_instance()` **una sola vez** (línea 207) y devuelve
  `results` (el dict de snapshots) en la clave `"snapshots"` (línea 211).
- `experiment.py:405-432` — el bucle principal solo lee los snapshots ya
  calculados: `results_snapshots = result["snapshots"]` y
  `pareto_front = results_snapshots.get(label)`.

### Verificado también en el repositorio original de GitHub

`https://github.com/AdrianMP1/MOHH-Specialized-Operators` (branch `main`,
commit actual; archivos descargados vía `raw.githubusercontent.com`):

- `src/mohh/evaluation/params.py`: `MO_POPULATION_SIZE=105`,
  `MO_GENERATIONS=500`, `EVAL_BUDGETS = {10k: 10500, 30k: 31500, 50k: 52500}`.
- `src/mohh/evaluation/algorithms.py:23-43`: el mismo `MyCallback` con
  checkpoints en las generaciones 100/300/500.
- `src/mohh/evaluation/experiment.py:304-335`: un `for i in tqdm(...)`, y en
  cada experimento **una sola** llamada `results = solver.solve_instance()`
  (línea 321); después `pareto_front = results.get(label)` (línea 326) para
  cada etiqueta. No hay un segundo `solve_instance()` por presupuesto.

**Conclusión:** tanto la copia local como el repositorio original usan
snapshots de una sola corrida. Nuestra implementación actual
(`solve_front_snapshots` con `_Snapshots`) es fiel al original, pero
contradice el requisito del usuario de corridas independientes.

### Alternativa planteada y descartada

Se planteó pasar a corridas independientes (MO_GENERATIONS 100/300/500,
misma población inicial por repetición para mantener el diseño pareado,
semilla distinta por presupuesto, coste ~3×). El usuario decidía mantener
la población inicial por repetición si se adoptaba esa vía, pero
**finalmente descartó el cambio**: el original usa snapshots y se prefiere
mantener la fidelidad al protocolo original.

## Estado actual del arnés

- `src/mohh/llamea/fitness.py:519-577` — `solve_front_snapshots`: corre hasta
  500 generaciones y devuelve los tres frentes (snapshots).
- `scripts/compare_operators.py` — `GE_PROTOCOL=1` fija `MO_GENERATIONS=500`
  y usa etiquetas 10k/30k/50k; `fronts.npz` guarda 3 frentes por tarea.
- La corrida `llamea_vs_ge_best` ya existente usa snapshots y **no hay que
  repetirla**.

Si en el futuro se quisiera independencia, el formato de `fronts.npz` ya
distingue el presupuesto en la clave, así que no cambiaría el formato, solo
la forma de calcular cada frente.
