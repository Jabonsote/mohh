# Análisis, search y evaluación

Este repositorio incluye las herramientas de análisis que antes vivían en el
proyecto de tesis, más una forma nueva de evaluar **generación por generación**.

## Contenido

```text
analyze_gen.py              análisis por individuo/generación
analyze_search.py           análisis de la búsqueda train -> test
analyze_operators.py        análisis de los operadores descubiertos
evaluate_31.py              evaluación de 31 réplicas
evaluate_generations.py     indicadores por generación
ast_diversity.py            diversidad AST
indicator_quality.py        calidad/saturación de indicadores
export_figure_values.py     valores de las figuras -> CSV
generation_comparison.py    comparación de generaciones entre experimentos
generate_reference_sets_pop.py  genera los P* con N = población de MOEAD
reference_sets_report.py    resumen y gráficas de P*
hv_scale_from_evaluation.py escala de HV compartida entre evaluaciones
references_moea/            P* del dataset nuevo (80 train / 20 test)
references_original/        P* del dataset antiguo (6 train / 16 test)
src/mohh/analysis/          paquete de análisis
src/mohh/core/indicators/   IGD+, DeltaP, R2 y ReferenceSet
results/analysis_scripts/   scripts que trabajan sobre results/evaluations/
docs/TEST_GUIDE.md          guía de tests unitarios (cobertura y huecos)
```

## Referencias P*

```text
references_moea/train/reference_sets_pop/     # ACTUAL: N = poblacion de MOEAD
references_moea/test/reference_sets_pop/
references_moea/train/reference_sets/         # antiguo: hasta 100 puntos
references_moea/test/reference_sets/
references_moea/reference_sets_pop_report.{csv,md}
references_moea/nondominated_counts.{csv,md}
references_original/{train,test}/reference_sets/
```

Los P* que usan por defecto todos los scripts son los de `reference_sets_pop`:
su tamano es el de la poblacion REAL de MOEAD para el numero de objetivos de
la instancia (Das-Dennis con `MO_POPULATION_SIZE=105`), es decir **105** para
m=2 y m=3, **120** para m=4 y **126** para m=5. Si el ground set es mas
pequeno que eso, el P* se queda con los puntos que hay y la instancia aparece
como `truncated` en el reporte (22 de 100). Se regeneran con:

```bash
python generate_reference_sets_pop.py --dry-run     # solo imprime el N
python generate_reference_sets_pop.py               # train + test (~75 s)
python reference_sets_report.py --subdir reference_sets_pop \
    --output results/reference_sets_pop
```

La extraccion es la oficial del pipeline `set-reference` (RSEIterative, 21
corridas, 10000 iteraciones, `select=spread`), asi que es determinista: volver
a correrla produce los mismos `.pof`.

Los P* de `reference_sets/` (version antigua, hasta 100 puntos) se dejan sin
tocar para poder comparar; para usarlos hay que pasar `--references` a mano.

## Dataset

- Nuevo: `datasets/mqap` en este repositorio.
- Antiguo: `datasets/mqap_original`.
- Externo: `--dataset-path /ruta/al/dataset`.

## `analyze_gen`

```bash
python analyze_gen.py /ruta/Experiment_001 \
    --references references_moea/train/reference_sets \
    --output /ruta/Experiment_001/analysis
```

## `analyze_search`

```bash
python analyze_search.py /ruta/Experiment_001 \
    --output /ruta/Experiment_001/analysis_search \
    --train-data datasets/mqap/train \
    --test-data datasets/mqap/test
```

Las `--*-references` apuntan por defecto a `reference_sets_pop`; solo hacen
falta si se quiere otra carpeta.

Si el test no tiene un train equivalente exacto, empareja por `n` y omite los
casos sin train del mismo tamaño. Lo deja registrado en `summary.json`.

### Caja de P* fija (sin pooling)

IGD+, DELTA_P y R2 se normalizan con la caja del PROPIO P* (`ideal`, `nadir` y
`span` calculados una sola vez en `load_references`). Antes la caja se ampliaba
con los frentes medidos (`pooled_reference_sets`) y entonces cada experimento
tenia la suya: los mismos individuos daban IGD+ 0.53 aqui y 1.96 en
`analyze_gen`. Con la caja fija los tres indicadores son comparables entre
train, test y cualquier corrida que use los mismos P*, y coinciden con los de
`analyze_gen`. `summary.json` lo registra como `reference_box: pstar_only` e
`indicators_comparable_across_experiments: true`.

El HV es el unico que sigue necesitando `--hv-scale` para ser comparable
entre experimentos (ver arriba).

### Direccion de cada indicador

| indicador | direccion | una curva que sube |
|---|---|---|
| HV | **maximizar** | mejora |
| IGD+ | minimizar | empeora |
| DELTA_P | minimizar | empeora |
| R2 | minimizar | empeora |

Registro unico: `mohh.core.indicators.directions`; volcado en
`00_config/indicators_direction.md` y en `summary.json
(indicator_directions)`, y en la columna `direction` de los CSV. Todas las
figuras normalizadas usan **1 = mejor**, asi que mas arriba es mejor en todas.
HV e IGD+ se calculan con las versiones oficiales de pymoo
(`pymoo.indicators.hv.HV`, `pymoo.indicators.igd_plus.IGDPlus`); DELTA_P y R2
no tienen equivalente oficial y siguen siendo codigo del repositorio.

### Escala de HV justa

Por defecto la escala `vmin/vmax` se congela con los individuos de **ese**
experimento: el HV es comparable entre sus generaciones, pero no entre corridas
(la escala de `Experiment_001` y `Experiment_002` puede diferir en las 80
instancias). Para comparar generaciones entre experimentos, o para que el HV
cuadre con la tabla de `mohh-evaluate`, hay que usar **una sola escala**:

```bash
# 1) congelar la escala de la evaluación GE (la que escribió mohh-evaluate)
python analyze_search.py Experiment_001 --output .../analysis_search \
    --hv-from-evaluation --hv-budget 50k --save-hv-scale results/hv_scale.json

# 2) medir el segundo experimento con esa misma escala
python analyze_search.py Experiment_002 --output .../analysis_search \
    --hv-scale results/hv_scale.json
```

Las instancias con `vmax == vmin` en algún objetivo se excluyen (igual que
`mohh-evaluate` las invalida) y quedan anotadas en `summary.json`
(`hv_scale.<split>.degenerate_excluded`). `summary.json` también registra
`hv_comparable_across_experiments`.

Si train y test necesitan ficheros distintos (porque `mohh-evaluate` solo
evaluó uno de los splits), pasa `--hv-scale` varias veces; se combinan y se
rechaza cualquier instancia común con `vmin/vmax` contradictorios:

```bash
python analyze_search.py Experiment_002 --output .../analysis_search \
    --hv-scale results/hv_scale_train.json \
    --hv-scale results/hv_scale_test.json
```

### Escala de la evaluacion final (nadir 1.1)

`mohh-evaluate` normaliza con los `vmin`/`vmax` de su propio consolidado y
mide contra `nadir_ref_point = [1.1] * n_obj`. Esos limites quedan en
`evaluation_results/run_metrics/evaluation_<instancia>.json` y se pueden
congelar en un JSON compartido con:

```bash
python hv_scale_from_evaluation.py \
    --experiment experimentos_001_002/Experiment_001 \
    --experiment experimentos_001_002/Experiment_002 \
    --budget 50k --output results/baseline/hv_scale_eval_shared.json

python analyze_search.py results/baseline/Experiment_001 \
    --output results/baseline/Experiment_001/analysis_search \
    --hv-scale results/baseline/hv_scale_eval_shared.json
```

Como cada evaluacion recalcula sus propios `vmin/vmax`, dos corridas nunca
dan la misma caja; por eso la escala compartida es la **union** de las de las
dos corridas (misma caja para las dos, y sigue saliendo de la evaluacion).

La evaluacion solo corrio sobre **test**, asi que `analyze_search` completa
**train** con la misma receta (frente no dominado consolidado de la corrida y
el MISMO `ref_point` 1.1). Lo registra en `summary.json`:
`hv_scale.<split>.gap_filled_from_consolidated`. Si un nombre existe en los
dos splits con matrices distintas (`KC64-2fl-1rl` en `mqap`), la entrada de
train se recalcula en vez de reutilizar la de test.

### Dataset antiguo (baseline)

```bash
python analyze_search.py results/baseline/Experiment_001 \
    --output results/baseline/Experiment_001/analysis_search \
    --train-data datasets/mqap_original/train \
    --test-data datasets/mqap_original/test \
    --train-references references_original/train/reference_sets \
    --test-references references_original/test/reference_sets \
    --hv-scale results/baseline/hv_scale_eval_shared.json
```

(con `hv_scale_from_evaluation.py` sobre `results/baseline/Experiment_00{1,2}`
para generar la escala). Los P* del dataset antiguo se usan tal cual: no hay
ground sets para regenerarlos, y un frente verdaderamente de un punto
(`KC10-2fl-2un`) se mide con la caja `P* + frentes medidos`, anotado en
`summary.json` (`reference_box_degenerate_pooled`).

## `evaluate_31`

```bash
python evaluate_31.py /ruta/Experiment_001 \
    --splits train \
    --dataset mqap \
    --references references_moea/train/reference_sets \
    --replicates 31
```

Para test:

```bash
python evaluate_31.py /ruta/Experiment_001 \
    --splits test \
    --dataset mqap \
    --references references_moea/test/reference_sets \
    --replicates 31
```

## Evaluación intercalada: una generación por job

`mohh-evaluate` acepta ahora:

```text
--generation N          evalúa Best/Middle/Worst de generation_NNNN
--evaluation-dir DIR    salida propia; por defecto evaluation_by_generation/generation_NNNN
```

Una evaluación suelta:

```bash
mohh-evaluate /ruta/Experiment_001 \
    --dataset-path datasets/mqap \
    --generation 5 \
    --n-experiments 31
```

Salida:

```text
/ruta/Experiment_001/evaluation_by_generation/generation_0005/
```

Para lanzar **todas las generaciones** en paralelo:

```bash
EXPERIMENT=/ruta/Experiment_001 \
DATASET_PATH=$PWD/datasets/mqap \
bash slurm/submit_evaluate_generations.sh
```

Cada tarea:

- evalúa una generación;
- usa 32 núcleos (31 workers + proceso principal);
- escribe en `evaluation_by_generation/generation_NNNN/`;
- no dibuja figuras.

Las figuras se generan después con un array dependiente:

```text
slurm/mohh-evaluate-generations-array.slrm
slurm/mohh-evaluate-generations-figures.slrm
```

## Evaluación por instancia de test

Además sigue existiendo el modo ultra paralelo por instancia:

```bash
EXPERIMENT=/ruta/Experiment_001 \
DATASET_PATH=$PWD/datasets/mqap \
bash slurm/submit_evaluate_array.sh
```

Una tarea por instancia de test y figuras finales con `afterok`.
