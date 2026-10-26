# Resultados del pipeline larger

Generado por `larger_pipeline/summarize_results.py`.

## Referencias P*

| split | instancias | puntos ground set | puntos P* | singleton | media ground | mín | máx |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 80 | 113588 | 7113 | 0 | 1419.8 | 7 | 8932 |
| test | 20 | 50667 | 1544 | 0 | 2533.3 | 12 | 15941 |

Detalle por instancia: `larger_pipeline/references_moea/nondominated_counts.md`.

## Experiment_001

- Individuos: 1100; evaluaciones: 31; tiempo: 6.8 min.
- Diversidad AST media (todos): 0.7213; (elite): 0.7141.
- AST únicos medios (todos): 0.9606; (elite): 0.9591.

| indicador | media | std | tendencia | informativo |
|---|---:|---:|---:|:---:|
| HV | 0.3222 | 0.2522 | 0.895 | sí |
| IGD+ | 0.6963 | 1.671 | -0.825 | sí |
| DELTA_P | 0.9769 | 0.05555 | 0.864 | no |
| R2 | 0.9487 | 1.408 | -0.832 | sí |

> **Aviso:** DELTA_P: no informativo (std=0.05555, frac_1=0.247, frac_~1=0.699, frac_0=0.000)

> **Aviso:** R2: 9 valor(es) negativo(s) (mín=-0.6074); es posible si el frente mejora el ideal de P* y no es un error

Figuras principales:
- `results/experiments_larger/Experiment_001/analysis/07_figures/complexity_per_pair_Experiment_001.png`
- `results/experiments_larger/Experiment_001/analysis/07_figures/convergence_Experiment_001_DELTA_P.png`
- `results/experiments_larger/Experiment_001/analysis/07_figures/convergence_Experiment_001_HV.png`
- `results/experiments_larger/Experiment_001/analysis/07_figures/convergence_Experiment_001_IGD+.png`
- `results/experiments_larger/Experiment_001/analysis/07_figures/convergence_Experiment_001_R2.png`

- Diversidad AST: `results/experiments_larger/Experiment_001/analysis/08_ast_diversity/ast_diversity_Experiment_001.png`

## Experiment_002

- Individuos: 1100; evaluaciones: 31; tiempo: 5.9 min.
- Diversidad AST media (todos): 0.6858; (elite): 0.6675.
- AST únicos medios (todos): 0.9458; (elite): 0.9613.

| indicador | media | std | tendencia | informativo |
|---|---:|---:|---:|:---:|
| HV | 0.2869 | 0.2509 | 0.813 | sí |
| IGD+ | 0.9031 | 2.13 | -0.412 | sí |
| DELTA_P | 0.9729 | 0.06213 | 0.891 | no |
| R2 | 1.127 | 1.79 | -0.435 | sí |

> **Aviso:** DELTA_P: no informativo (std=0.06213, frac_1=0.246, frac_~1=0.681, frac_0=0.000)

> **Aviso:** R2: 1 valor(es) negativo(s) (mín=-0.4658); es posible si el frente mejora el ideal de P* y no es un error

Figuras principales:
- `results/experiments_larger/Experiment_002/analysis/07_figures/complexity_per_pair_Experiment_002.png`
- `results/experiments_larger/Experiment_002/analysis/07_figures/convergence_Experiment_002_DELTA_P.png`
- `results/experiments_larger/Experiment_002/analysis/07_figures/convergence_Experiment_002_HV.png`
- `results/experiments_larger/Experiment_002/analysis/07_figures/convergence_Experiment_002_IGD+.png`
- `results/experiments_larger/Experiment_002/analysis/07_figures/convergence_Experiment_002_R2.png`

- Diversidad AST: `results/experiments_larger/Experiment_002/analysis/08_ast_diversity/ast_diversity_Experiment_002.png`

## Nota sobre DELTA_P

En el dataset larger `DELTA_P` queda saturado: la mayoría de sus valores están en 1.0 o a menos de 0.01 del techo. El análisis lo calcula igual que el pipeline anterior, pero no se debe usar para comparar individuos. Usa **HV**, **IGD+** y **R2**.
