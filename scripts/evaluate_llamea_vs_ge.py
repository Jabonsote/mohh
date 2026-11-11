#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluacion GE + LLaMEA con el mismo codigo de `experiment.py`.

Compara los operadores Best/Middle/Worst de un experimento GE con el operador
generado por LLaMEA. La salida se escribe dentro de la corrida LLaMEA, en
`evaluation_results_vs_ge`, sin usar los resultados del experimento GE: solo se
lee su `MOEAD_operators.txt` (el codigo de los operadores).

Uso:
    python scripts/evaluate_llamea_vs_ge.py \
        --llamea-run results/searches/<RUN> \
        --llamea-operator results/searches/<RUN>/best_so_far.py \
        --ge-experiment experimentos_001_002/Experiment_001 \
        --dataset-path datasets/mqap
"""
import argparse
import os
from pathlib import Path

from mohh.evaluation.experiment import run_experiments


def read_ge_operators(path):
    labels = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if ":" in line:
                label, phenotype = line.split(":", 1)
                labels[label.strip()] = phenotype.strip()
    missing = [name for name in ("Best", "Middle", "Worst") if name not in labels]
    if missing:
        raise SystemExit(f"{path}: faltan {missing}")
    return [labels["Best"], labels["Middle"], labels["Worst"]]


def parse_budgets(value):
    budgets = {}
    for pair in value.split(","):
        label, budget = pair.split(":")
        budgets[label.strip()] = int(budget)
    return budgets


def main():
    parser = argparse.ArgumentParser(
        description="Evalua GE Best/Middle/Worst vs operador LLaMEA.")
    parser.add_argument("--llamea-run", required=True,
                        help="Carpeta de la corrida LLaMEA (salida).")
    parser.add_argument("--llamea-operator", default=None,
                        help="Operador .py de LLaMEA (por defecto best_so_far.py).")
    parser.add_argument("--ge-experiment", required=True,
                        help="Experimento GE con MOEAD_operators.txt.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--solvers", default="MOEAD,NSGAII,SMSEMOA")
    parser.add_argument("--n-experiments", type=int, default=31)
    parser.add_argument("--eval-budgets", default="10k:10500,30k:31500,50k:52500")
    parser.add_argument("--mo-generations", type=int, default=500)
    parser.add_argument("--mo-population-size", type=int, default=105)
    parser.add_argument("--evaluation-dir", default="evaluation_results_vs_ge")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="Instancias concretas (por defecto todas las de test).")
    args = parser.parse_args()

    run_dir = Path(args.llamea_run).resolve()
    operator = Path(args.llamea_operator) if args.llamea_operator else run_dir / "best_so_far.py"
    operator = operator.resolve()
    if not operator.is_file():
        raise SystemExit(f"no existe el operador LLaMEA: {operator}")

    ge_file = Path(args.ge_experiment).resolve() / "MOEAD_operators.txt"
    if not ge_file.is_file():
        raise SystemExit(f"no existe {ge_file}")

    ge_operators = read_ge_operators(ge_file)
    llamea_marker = f"llamea:{operator}"
    operators = ge_operators + [llamea_marker] * 3

    overrides = {
        "SOLVERS": [name.strip() for name in args.solvers.split(",") if name.strip()],
        "MODELS": ["MOEAD", "LLAMEA"],
        "N_EXPERIMENTS": args.n_experiments,
        "EVAL_BUDGETS": parse_budgets(args.eval_budgets),
        "MO_GENERATIONS": args.mo_generations,
        "MO_POPULATION_SIZE": args.mo_population_size,
        "DATASET_PATH": os.path.abspath(os.path.expanduser(args.dataset_path)),
        "EVALUATION_DIR": args.evaluation_dir,
    }

    print(f"LLaMEA:    {run_dir}")
    print(f"Operador:  {operator}")
    print(f"GE:        {ge_file}")
    print(f"Salida:    {run_dir / args.evaluation_dir}")
    print(f"Modelos:   {overrides['MODELS']}")
    print(f"Solvers:   {overrides['SOLVERS']}")
    print(f"Replicas:  {overrides['N_EXPERIMENTS']}")

    run_experiments(str(run_dir), "", operators, overrides,
                    instances=args.instances)
    print("Fin.")


if __name__ == "__main__":
    main()
