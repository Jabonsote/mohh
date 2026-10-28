# Scripts de análisis

Copia ejecutable de las herramientas de análisis, situada junto a los
resultados para poder modificarlas sin tocar la raíz del repositorio.

```text
analyze_gen.py          análisis por individuo y generación
analyze_search.py       análisis de la búsqueda train -> test
evaluate_31.py          evaluación de 31 réplicas
ast_diversity.py        diversidad AST
indicator_quality.py    calidad de indicadores
export_figure_values.py valores de las figuras -> CSV
reference_sets_report.py resumen y gráficas de P*
generation_comparison.py comportamiento por generación comparable
```

## Datos disponibles

```text
results/experiments_larger/Experiment_001/{analysis,analysis_search}
results/experiments_larger/Experiment_002/{analysis,analysis_search}
datasets/mqap/train   (80)
datasets/mqap/test    (20)
references_moea/{train,test}/reference_sets
```

## Uso

Desde `MOOHH-R/`:

```bash
python results/analysis_scripts/analyze_gen.py \
    results/experiments_larger/Experiment_001 \
    --output results/experiments_larger/Experiment_001/analysis

python results/analysis_scripts/analyze_search.py \
    results/experiments_larger/Experiment_001 \
    --output results/experiments_larger/Experiment_001/analysis_search
```

Los scripts calculan la raíz del repo desde su propia ubicación
(`results/analysis_scripts`), así que no hay que exportar `PYTHONPATH`.

## Escala de HV compartida (comparación justa)

El HV sólo se puede comparar si todos los individuos, generaciones y
experimentos se miden con los mismos `vmin/vmax` y `ref_point = 1.1`.

- `analyze_search.py` congela por defecto una escala con los individuos de **ese**
  experimento: sirve para su trayectoria, pero **no** para compararla con otra
  corrida (cada consolidado da límites distintos).
- Para comparar entre experimentos, congela una sola escala y pásala:

```bash
# Escala de la evaluación GE (o la que escriba --save-hv-scale)
python analyze_search.py Experiment_001 --output .../analysis_search \
    --hv-from-evaluation --hv-budget 50k --save-hv-scale results/hv_scale.json
python analyze_search.py Experiment_002 --output .../analysis_search \
    --hv-scale results/hv_scale.json

# Si train y test usan ficheros distintos, repite --hv-scale; se combinan
python analyze_search.py Experiment_002 --output .../analysis_search \
    --hv-scale results/hv_scale_train.json --hv-scale results/hv_scale_test.json

# Lo mismo para las generaciones intermedias de GE
python evaluate_generations.py Experiment_001 --hv-scale results/hv_scale.json
```

- `generation_comparison.py` comprueba los `vmin/vmax` **instancia por
  instancia** (no sólo `source`/`ref_point`) y falla si no coinciden. Con
  `--allow-different-scale` escribe los CSV igualmente, pero marcados como no
  comparables.
- Las instancias degeneradas (`vmax == vmin`) se excluyen, como hace
  `mohh-evaluate`, y quedan anotadas en `summary.json`.

## Valores de las figuras en CSV

```bash
python results/analysis_scripts/export_figure_values.py \
    --experiments-root results/experiments_larger
```

Genera:

```text
results/experiments_larger/Experiment_001/analysis/figure_values/
results/experiments_larger/Experiment_001/analysis_search/figure_values/
results/experiments_larger/Experiment_002/analysis/figure_values/
results/experiments_larger/Experiment_002/analysis_search/figure_values/
results/figure_values/ast_diversity_all.csv
```

Incluye, entre otros:

```text
trayectorias_media_{split}_{metrica}.csv
trayectorias_mejor_{split}_{metrica}.csv
boxplot_por_generacion_{split}.csv
mapa_calor_{split}_{metrica}.csv
medias_train_test_{metrica}.csv
```

## Referencias P*

Están en la raíz del proyecto:

```text
references_moea/train/reference_sets/
references_moea/test/reference_sets/
references_original/train/reference_sets/
references_original/test/reference_sets/
```

## Figuras con muchas instancias

`analyze_search.py` ya no dibuja una leyenda con las 80 instancias. En las
trayectorias agregadas usa una escala de color `viridis` y una colorbar
ordenada alfabéticamente; la línea negra es la media y es la única entrada de
la leyenda. Las figuras corregidas están en:

```text
results/experiments_larger/Experiment_*/analysis_search/07_figures/trayectorias/
```
