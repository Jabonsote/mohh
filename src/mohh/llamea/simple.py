"""LLaMEA (1+1) con strong priors, siguiendo BAG (Huang et al., 2026).

Un candidato por iteración, evaluación con el fitness de GE y reemplazo elitista.
El prompt se apoya en el código de un prior fuerte y en una estrategia
(refinar/crear); cada ``--prior-every`` iteraciones se refina un prior aleatorio.
"""
import json
import math
import os
import random
import time

import numpy as np

from mohh.llamea.fitness import build_evaluator, primitive_namespace
from mohh.llamea.parsing import parse_llm_response
from mohh.llamea.priors import load_priors

ROLE = (
    "You are an expert in evolutionary computation and Python. "
    "You design variation operators for permutation-encoded "
    "multi-objective optimization problems."
)

TASK = """Design a crossover (recombination) operator.

Write a module-level Python function:

    def crossover(x, y):

`x` and `y` are permutations of 0..n-1.
Return one child permutation, or a tuple (child1, child2).
Do not modify `x` or `y`.
The child must be a permutation of 0..n-1.
The operator must work for any n >= 2."""

STRATEGY_REFINE = (
    "Refine the reference operator. Keep its main mechanism and improve it. "
    "Change only what is needed."
)
STRATEGY_CREATE = (
    "Generate a new operator that is different from the reference. "
    "Use a different combination of order, position, adjacency or cycle ideas."
)

FORMAT = """Answer in exactly this format:

# Description: <one line>
# Code:
```python
<complete code with `import numpy as np` and `def crossover(x, y):`>
```"""


class Candidate:
    def __init__(self, code):
        self.code = code
        self.fitness = float("nan")
        self.feedback = ""
        self.error = None

    def set_scores(self, fitness, feedback="", error=None):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error


def build_prompt(parent_code, strategy):
    if strategy == "refine":
        strategy_text = STRATEGY_REFINE
    else:
        strategy_text = STRATEGY_CREATE
    return (
        f"{ROLE}\n\nTASK\n{TASK}\n\n"
        f"REFERENCE CODE\n```python\n{parent_code}\n```\n\n"
        f"STRATEGY\n{strategy_text}\n\n{FORMAT}"
    )


def minimal_compile(code):
    """Filtro mínimo: compila, define crossover y devuelve una permutación."""
    namespace = primitive_namespace()
    exec(compile(code, "<candidate>", "exec"), namespace)
    crossover = namespace.get("crossover")
    if not callable(crossover):
        raise ValueError("missing callable crossover(x, y)")
    n = 11
    x = np.arange(n)
    y = x[::-1].copy()
    child = crossover(x.copy(), y.copy())
    if isinstance(child, tuple):
        child = child[0]
    child = np.asarray(child)
    if child.shape != (n,):
        raise ValueError(f"wrong child shape: {child.shape}")
    if set(child.tolist()) != set(range(n)):
        raise ValueError("child is not a permutation of range(n)")
    return namespace


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _evaluate(evaluator, code):
    candidate = Candidate(code)
    try:
        result = evaluator(candidate)
    except Exception as error:  # noqa: BLE001
        return float("-inf"), f"{type(error).__name__}: {error}"
    if result is None or result.error is not None:
        return float("-inf"), getattr(result, "error", "evaluation failed")
    fitness = float(result.fitness)
    if not math.isfinite(fitness):
        return float("-inf"), "fitness is not finite"
    return fitness, None


def run_simple(args, exp_dir, logger, llm, train_paths):
    """Bucle (1+1) con strong priors. Devuelve el mejor código."""
    llm.operator_memory = 0
    evaluator = build_evaluator(
        train_paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
        model_name=args.solver,
        mutation=args.mutation,
        normalize=True,
        frame="stationary",
        diversity_feedback=False,
        repeat_penalty=1.0,
        patience=0,
        fitness_workers=args.fitness_workers,
    )

    priors = load_priors()
    best_code, best_fitness = None, float("-inf")
    for label, code in priors:
        fitness, error = _evaluate(evaluator, code)
        logger.info("Prior %s: fitness=%s error=%s", label, fitness, error)
        if fitness > best_fitness:
            best_code, best_fitness = code, fitness
    if best_code is None:
        raise RuntimeError("ningun prior pudo evaluarse")

    best_generation = 0
    _write_text(os.path.join(exp_dir, "best_so_far.py"), best_code + "\n")
    _write_json(os.path.join(exp_dir, "best_so_far.json"), {
        "generation": 0,
        "fitness": best_fitness,
        "name": "prior",
        "seed": args.seed,
    })

    progress_path = os.path.join(exp_dir, "progress.txt")
    started = time.time()
    for iteration in range(1, args.budget + 1):
        if iteration % args.prior_every == 0:
            parent_label, parent_code = random.choice(priors)
            strategy = "refine"
        elif random.random() < 0.5:
            parent_label, parent_code = "current-best", best_code
            strategy = "refine"
        else:
            parent_label, parent_code = "current-best", best_code
            strategy = "create"

        prompt = build_prompt(parent_code, strategy)
        try:
            response = llm.query([{"role": "user", "content": prompt}])
            code = parse_llm_response(response or "")
        except Exception as error:  # noqa: BLE001
            logger.warning("iter %d: respuesta invalida: %s", iteration, error)
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"iter {iteration} best={best_fitness:.6f} candidate=nan "
                    f"valid=0 error={type(error).__name__} "
                    f"parent={parent_label} strategy={strategy}\n")
            continue

        try:
            minimal_compile(code)
        except Exception as error:  # noqa: BLE001
            logger.warning("iter %d: filtro minimo: %s", iteration, error)
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"iter {iteration} best={best_fitness:.6f} candidate=nan "
                    f"valid=0 error={type(error).__name__} "
                    f"parent={parent_label} strategy={strategy}\n")
            continue

        fitness, error = _evaluate(evaluator, code)
        improved = fitness > best_fitness
        if improved:
            best_code, best_fitness, best_generation = code, fitness, iteration
            _write_text(os.path.join(exp_dir, "best_so_far.py"), best_code + "\n")
            _write_text(os.path.join(exp_dir, "llamea_best_operator.py"),
                        best_code + "\n")
            _write_json(os.path.join(exp_dir, "best_so_far.json"), {
                "generation": iteration,
                "fitness": best_fitness,
                "name": "crossover",
                "seed": args.seed,
            })

        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"iter {iteration} best={best_fitness:.6f} "
                f"candidate={fitness:.6f} valid=1 "
                f"error={error or '-'} parent={parent_label} "
                f"strategy={strategy} wall_s={time.time() - started:.1f}\n")

    _write_json(os.path.join(exp_dir, "llamea_best_info.json"), {
        "train_fitness": best_fitness,
        "best_generation": best_generation,
        "budget": args.budget,
        "solver": args.solver,
        "mo_generations": args.mo_generations,
        "mo_population_size": args.mo_population_size,
        "model": args.model,
        "best_operator": "best_so_far.py",
        "method": "llamea-1+1-strong-priors",
    })
    logger.info("Fin: best fitness=%s en iteracion %s", best_fitness, best_generation)
    print(f"Mejor fitness: {best_fitness:.6f} (iteracion {best_generation})")
    return best_code
