"""Evalua los operadores de cruce clasicos como referencia (linea base).

Sirve para dos cosas:

1. Comprobar que el HV normalizado del pipeline LLaMEA queda en el mismo orden
   de magnitud que el de la busqueda GE.
2. Tener una linea base contra la que comparar los operadores que invente el
   LLM (OX, PBX, PMX, CX, ERX y una permutacion aleatoria).

Escribe todo en `results/examples/reference_operators/`:

    reference_operators.json   HV por operador e instancia
    reference_operators.md     tabla legible

Uso:
    python scripts/llamea_examples.py
    python scripts/llamea_examples.py --train-instances all --mo-generations 300
"""

import argparse
import inspect
import json
import os

import numpy as np

from mohh.core.utilities.instance_utils import instance_paths
from mohh.llamea.fitness import build_evaluator, init_params
from mohh.llamea.reference import REFERENCE_OPERATORS


class _Solution:
    def __init__(self, code):
        self.code = code
        self.fitness = float("nan")
        self.feedback = ""
        self.error = None

    def set_scores(self, fitness, feedback="", error=None):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error


RANDOM_CODE = """import numpy as np

def crossover(x, y):
    return np.random.permutation(len(x))
"""


def _select(paths, amount, seed):
    if amount == "all":
        return list(paths)
    import random

    rng = random.Random(seed)
    return rng.sample(list(paths), min(int(amount), len(paths)))


def evaluate(name, code, paths, args, bounds):
    """Evalua un operador con los mismos limites de normalizacion."""
    evaluator = build_evaluator(
        paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
        model_name=args.solver,
        normalize=True,
        bounds=bounds,
    )
    solution = evaluator(_Solution(code))
    return {
        "operator": name,
        "hv": solution.fitness,
        "error": str(solution.error) if solution.error else None,
        "feedback": solution.feedback,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-instances", default="8")
    parser.add_argument("--mo-generations", type=int, default=30)
    parser.add_argument("--mo-population-size", type=int, default=30)
    parser.add_argument("--solver", default="MOEAD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results/examples/reference_operators")
    args = parser.parse_args()

    init_params(RANDOM_SEED=args.seed)
    paths = _select(instance_paths(train=True), args.train_instances, args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Instancias: {len(paths)}  solver: {args.solver}")

    # Los limites de normalizacion son los del panel de referencia completo,
    # los mismos que usaria una busqueda LLaMEA (HV justo y comparable).
    builder = build_evaluator(
        paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
        model_name=args.solver,
        normalize=True,
    )
    bounds = builder.ensure_reference_bounds()
    print("Limites de normalizacion calculados.")

    candidates = {}
    for name, function in REFERENCE_OPERATORS.items():
        source = inspect.getsource(function)
        # El contrato de LLaMEA exige que la funcion se llame `crossover`.
        candidates[name] = source.replace(
            f"def {function.__name__}(", "def crossover(", 1
        )
    candidates["random"] = RANDOM_CODE

    results = []
    for name, code in candidates.items():
        print(f"Evaluando {name}...")
        row = evaluate(name, code, paths, args, bounds)
        results.append(row)
        print(f"  HV={row['hv']:.6f}  error={row['error']}")

    results.sort(key=lambda row: (row["hv"] is None, -(row["hv"] or 0.0)))

    payload = {
        "instances": len(paths),
        "solver": args.solver,
        "mo_generations": args.mo_generations,
        "mo_population_size": args.mo_population_size,
        "seed": args.seed,
        "normalized": True,
        "nadir_factor": 1.1,
        "results": results,
    }
    json_path = os.path.join(args.results_dir, "reference_operators.json")
    with open(json_path, "w") as file:
        json.dump(payload, file, indent=2)

    lines = [
        "# Operadores de referencia (linea base)",
        "",
        f"- Instancias: {len(paths)}",
        f"- MOEA interno: {args.solver}",
        f"- MO_GENERATIONS / MO_POPULATION_SIZE: "
        f"{args.mo_generations} / {args.mo_population_size}",
        f"- Semilla: {args.seed}",
        "- HV normalizado como el proyecto original (nadir 1.1)",
        "",
        "| Operador | HV medio | Error |",
        "|----------|----------|-------|",
    ]
    for row in results:
        error = (row["error"] or "-").splitlines()[0]
        lines.append(f"| `{row['operator']}` | {row['hv']:.6f} | {error} |")
    markdown_path = os.path.join(args.results_dir, "reference_operators.md")
    with open(markdown_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    print(f"\nGuardado en {args.results_dir}")


if __name__ == "__main__":
    main()
