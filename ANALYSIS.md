# Análisis, search y evaluación

Este repositorio incluye las herramientas de análisis que antes vivían en el
proyecto de tesis, más una forma nueva de evaluar **generación por generación**.

## Contenido

```text
analyze_gen.py              análisis por individuo/generación
analyze_search.py           análisis de la búsqueda train -> test
evaluate_31.py              evaluación de 31 réplicas
ast_diversity.py            diversidad AST
indicator_quality.py        calidad/saturación de indicadores
references_moea/            P* del dataset nuevo (80 train / 20 test)
references_original/        P* del dataset antiguo (6 train / 16 test)
src/mohh/analysis/          paquete de análisis
src/mohh/core/indicators/   IGD+, DeltaP, R2 y ReferenceSet
```

## Referencias P*

```text
references_moea/train/reference_sets/
references_moea/test/reference_sets/
references_moea/nondominated_counts.{csv,md}
references_original/{train,test}/reference_sets/
```

Se pueden usar con cualquier experimento compatible (train/test del dataset
larger).

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
    --test-data datasets/mqap/test \
    --train-references references_moea/train/reference_sets \
    --test-references references_moea/test/reference_sets
```

Si el test no tiene un train equivalente exacto, empareja por `n` y omite los
casos sin train del mismo tamaño. Lo deja registrado en `summary.json`.

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
