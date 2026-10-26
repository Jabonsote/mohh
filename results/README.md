# Resultados larger (80 train / 20 test)

Este directorio contiene los resultados y las herramientas para modificarlos.

## Experimentos

```text
results/experiments_larger/Experiment_001/
    generation_results/          # datos crudos de la corrida
    analysis/                    # análisis por individuo/generación + AST
    analysis_search/             # análisis de la búsqueda train -> test
results/experiments_larger/Experiment_002/
```

Dentro de cada `analysis/` y `analysis_search/`:

```text
analysis/07_figures/             # figuras
analysis/08_ast_diversity/       # CSV, JSON y PNG de diversidad AST
analysis/figure_values/          # valores de las figuras en CSV
analysis_search/07_figures/      # figuras
analysis_search/figure_values/   # valores de las figuras en CSV
```

## Scripts

```text
results/analysis_scripts/
```

Incluye `analyze_gen.py`, `analyze_search.py`, `evaluate_31.py`,
`ast_diversity.py`, `indicator_quality.py` y `export_figure_values.py`.

## Referencias y datos

Están en la raíz del proyecto:

```text
references_moea/train/reference_sets/      # 80 P*
references_moea/test/reference_sets/       # 20 P*
references_original/train/reference_sets/
references_original/test/reference_sets/
datasets/mqap/train/                       # 80
datasets/mqap/test/                        # 20
```

## Resumen

```text
results/larger_RESULTS.md
results/references_moea_counts.md
results/figure_values/ast_diversity_all.csv
```
