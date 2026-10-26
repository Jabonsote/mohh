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
