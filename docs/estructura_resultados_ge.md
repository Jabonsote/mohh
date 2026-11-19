# Estructura de resultados: original vs arnés de comparación

Referencia: `results/evaluations/experimentos_001_002/Experiment_001/evaluation_results/`
(copia local del original) y `experiment.py:28-47, 401-468` del repositorio
`AdrianMP1/MOHH-Specialized-Operators@main`.

## Estructura del original

```
evaluation_results/
├── initial_solutions/
│   ├── permutation/<inst>/experiment_001.txt ... experiment_031.txt, seeds.txt
│   └── real/<inst>/experiment_001.txt ..., seeds.txt
├── MOEAD/
│   ├── 10k/
│   │   ├── fronts/<inst>/<config>_<NM|WM>_<solver>.pof
│   │   │              <config>_<NM|WM>_<solver>_norm.pof
│   │   │              best_front_consolidated.pof
│   │   │              best_front_consolidated_norm.pof
│   │   ├── <inst>.csv
│   │   └── Figures/{NoMutation,WithMutation}/<inst>_<NM|WM>.png
│   ├── 30k/  (igual)
│   └── 50k/  (igual)
├── NSGAII/{10k,30k,50k}/   (igual)
├── SMSEMOA/{10k,30k,50k}/  (igual)
└── run_metrics/evaluation_<inst>.json
```

Los CSV llevan una columna por `(solver, configuración, mutación)`:
`MOEAD_Best_NM_MOEAD`, `NSGAII_PMX_WM_None`, etc. Los `.pof` son el frente de
la repetición mediana; `_norm` es el frente normalizado con los límites del
marco consolidado. `Figures/` lo genera después `make_plots.py`, no la
evaluación.

## Lo que escribe nuestro `write_ge_layout`

`scripts/compare_operators.py:1054-1140` reproduce:

- `<solver>/<presupuesto>/<inst>.csv` — una fila por repetición, columnas
  `<solver>_<operador>_<NM|WM>_<solver>`. ✔
- `<solver>/<presupuesto>/fronts/<inst>/*.pof` (y `_norm`,
  `best_front_consolidated*`). ✔
- `run_metrics/evaluation_<inst>.json`. ✔
- Formato `.pof` (cabecera `# N M`, notación científica). ✔

## Diferencia detectada (causa)

`llamea_vs_ge_best` se lanzó con `SOLVER=MOEAD`, así que solo tiene `MOEAD/`
y sus CSV solo 4 columnas (`MOEAD_llamea_NM_MOEAD`, `MOEAD_ge_best_NM_MOEAD`,
`MOEAD_llamea_WM_MOEAD`, `MOEAD_ge_best_WM_MOEAD`). El original tiene
`MOEAD/`, `NSGAII/` y `SMSEMOA/`.

**Causa:** `SOLVER=MOEAD` se pasa explícitamente al wrapper, que lo traduce en
`--solver MOEAD`. Sin ese override, `GE_PROTOCOL=1` activa por defecto
`MOEAD,NSGAII,SMSEMOA` y `--mutation-both` (6 protocolos), generando la
estructura completa del original.

**Corrección:** no pasar `SOLVER=MOEAD` en las comparaciones que deban seguir
el protocolo GE (ver `avance/PENDIENTE_CX.md`).

## Diferencias menores / adicionales

- `initial_solutions/real/`: el original lo escribe siempre; nuestro arnés
  solo escribe `permutation/` porque las instancias KC son de permutación.
- `Figures/`: el original lo genera `make_plots.py` como paso posterior; el
  arnés de comparación no lo genera.
- El arnés añade artefactos propios en la raíz (`report.md`, `contrastes.csv`,
  `evaluation.json`, `fronts/`, `hv_por_instancia/`, `operators.txt`,
  `seeds.txt`, `failures.json`). Son adicionales; no sustituyen ni cambian los
  archivos del formato original.
