# Scripts de análisis

Los scripts de análisis viven en la **raíz** del repositorio (ver `ANALYSIS.md`).
Aquí están los que trabajan directamente sobre `results/evaluations/`.

```text
analyze_eval_partial.py   análisis de una evaluación parcial (la que se está corriendo)
tablas_evaluacion.py      tablas por MOEA, presupuesto, mutación e instancia
desglose_moea.py          HV por MOEA × mutación × presupuesto con contrastes pareados
wins_detalle.py           victorias por MOEA, mutación, presupuesto e instancia
```

## Datos

```text
results/evaluations/<experimento>/       una evaluación: MOEAD/ NSGAII/ SMSEMOA/
                                          10k/ 30k/ 50k/ <instancia>.csv
results/evaluations/<experimento>/fronts/ frentes .pof del layout del original
results/evaluations/<experimento>/initial_solutions/  poblaciones por repetición
```

## Uso

```bash
EVAL_DIR=results/evaluations/llamea_vs_ox_pmx_local \
python results/analysis_scripts/tablas_evaluacion.py
```

Escribe en el directorio de la evaluación:

```text
tabla_por_moea.csv       MOEA × presupuesto × mutación × base: victorias, diferencia, IC95, dz, p de Holm
tabla_por_instancia.csv  MOEA × mutación × instancia × base: W/P/E y HV medios
tabla_no_gana.csv        celdas donde el operador pierde la mayoría
tablas.md                las tres tablas en Markdown
```

El marco de normalización es el consolidado por instancia y presupuesto (como
en `mohh/evaluation/experiment.py`), el nadir es `[1.1] * K` y un fallo cuenta
como HV 0. Los empates son siempre fallos de los dos operadores.
