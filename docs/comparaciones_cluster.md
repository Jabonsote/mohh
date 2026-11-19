# Corridas de comparación en el cluster

Todas las comparaciones deben usar el **mismo protocolo GE** y producir el
mismo árbol de resultados (`docs/estructura_resultados_ge.md`), sin importar
qué operadores se comparen (LLaMEA vs GE, GE vs clásicos, LLaMEA vs clásicos).

## Regla crítica: no pasar `SOLVER`

`GE_PROTOCOL=1` activa por defecto `MOEAD,NSGAII,SMSEMOA` con ambas
mutaciones (NM/WM). Si se pasa `SOLVER=MOEAD`, solo se genera `MOEAD/` y el
resultado **no** es comparable con la evaluación GE. Por eso los comandos de
abajo no incluyen `SOLVER`.

## LLaMEA vs CX y PMX (sin OX)

```bash
OPERATORS="llamea=results/searches/glm-gptoss2/llamea_best_operator.py \
cx=results/ge_classics/cx_ge.py pmx=results/ge_classics/pmx_ge.py" \
GROUPS="base=cx,pmx llamea=llamea" CONTRASTS="llamea:base" \
GE_PROTOCOL=1 INSTANCES=all REPEATS=31 CPUS=52 \
NAME=llamea_vs_cx_pmx bash slurm/compare-operators.sh
```

`GROUPS` debe declarar los dos grupos: si falta `llamea=llamea`, el arnés
muere en FASE 3 (ya hay validación temprana).

## LLaMEA vs GE Best (repetir sin `SOLVER=MOEAD`)

La corrida anterior se hizo con `SOLVER=MOEAD` y solo generó `MOEAD/`. Para
que sea estrictamente comparable con la GE:

```bash
OPERATORS="llamea=results/searches/glm-gptoss2/llamea_best_operator.py \
ge_best=ge:Best@results/Experiment_002/MOEAD_operators.txt" \
GROUPS="base=ge_best llamea=llamea" CONTRASTS="llamea:base" \
GE_PROTOCOL=1 INSTANCES=all REPEATS=31 CPUS=52 \
NAME=llamea_vs_ge_best_completo bash slurm/compare-operators.sh
```

## GE vs clásicos

```bash
OPERATORS="ge_best=ge:Best@results/Experiment_002/MOEAD_operators.txt \
cx=results/ge_classics/cx_ge.py pmx=results/ge_classics/pmx_ge.py" \
GROUPS="clasicos=cx,pmx ge_best=ge_best" CONTRASTS="ge_best:clasicos" \
GE_PROTOCOL=1 INSTANCES=all REPEATS=31 CPUS=52 \
NAME=ge_best_vs_cx_pmx bash slurm/compare-operators.sh
```

## Coste y salida

- 3 operadores × 31 reps × 20 instancias × 6 protocolos = 11 160 tareas
  (≈30 min con 52 CPUs). Con 2 operadores, 7 440 tareas.
- Salida en `results/evaluations/<NAME>/` con el formato estándar:
  `initial_solutions/{permutation,real}/`, `MOEAD|NSGAII|SMSEMOA/10k|30k|50k/`
  (CSV, `fronts/`, `Figures/`) y `run_metrics/`.
- Monitoreo: `tail -f logs/compare-llamea-<JOBID>.out`.
- Reanálisis sin repetir el MOEA: añadir `LOAD_FRONTS=1` y un `NAME` nuevo.
- Los HV solo se comparan **dentro de una misma corrida** (mismo marco de
  normalización); ver `docs/hv_protocolo_original.md`.
