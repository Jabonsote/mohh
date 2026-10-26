# Scripts de análisis

Copia ejecutable de las herramientas de análisis, situada junto a los
resultados para poder modificarlas sin tocar la raíz del repositorio.

```text
analyze_gen.py          análisis por individuo y generación
analyze_search.py       análisis de la búsqueda train -> test
evaluate_31.py          evaluación de 31 réplicas
ast_diversity.py        diversidad AST
indicator_quality.py    calidad de indicadores
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

## Figuras con muchas instancias

`analyze_search.py` ya no dibuja una leyenda con las 80 instancias. En las
trayectorias agregadas usa una escala de color `viridis` y una colorbar
ordenada alfabéticamente; la línea negra es la media y es la única entrada de
la leyenda. Las figuras corregidas están en:

```text
results/experiments_larger/Experiment_*/analysis_search/07_figures/trayectorias/
```
