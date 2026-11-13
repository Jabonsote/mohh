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
from mohh.llamea.vllm import extract_description

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
    "Refine the strategy of the selected solution to improve it. "
    "Keep its main mechanism and change only what is needed."
)
STRATEGY_CREATE = (
    "Generate a new algorithm that is different from the algorithms you have "
    "tried before. Use a different combination of order, position, adjacency "
    "or cycle ideas."
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


def build_prompt(parent_code, strategy, last_error=""):
    if strategy == "refine":
        strategy_text = STRATEGY_REFINE
    else:
        strategy_text = STRATEGY_CREATE
    prompt = (
        f"{ROLE}\n\nTASK\n{TASK}\n\n"
        f"REFERENCE CODE\n```python\n{parent_code}\n```\n\n"
        f"STRATEGY\n{strategy_text}\n\n{FORMAT}"
    )
    if last_error:
        prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n"
            f"{last_error}\n"
            "Fix that error. Do not repeat the same invalid mechanism."
        )
    return prompt


def minimal_compile(code):
    """Filtro mínimo: compila, define crossover y devuelve permutaciones.

    Prueba tres tamaños y **ambos** hijos. No es el validador estricto de GE:
    solo comprueba el contrato mínimo antes de gastar el MOEA.
    """
    namespace = primitive_namespace()
    exec(compile(code, "<candidate>", "exec"), namespace)
    crossover = namespace.get("crossover")
    if not callable(crossover):
        raise ValueError("missing callable crossover(x, y)")

    for n in (11, 17, 23):
        x = np.arange(n)
        y = x[::-1].copy()
        out = crossover(x.copy(), y.copy())
        children = out if isinstance(out, tuple) else (out,)
        if isinstance(out, tuple) and len(out) != 2:
            raise ValueError(f"tuple must have two children, got {len(out)}")
        for index, child in enumerate(children):
            child = np.asarray(child)
            if child.shape != (n,):
                raise ValueError(
                    f"child{index + 1} has shape {child.shape}, expected {(n,)}")
            if set(child.tolist()) != set(range(n)):
                raise ValueError(
                    f"child{index + 1} is not a permutation of range({n})")
    return namespace


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _append_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


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


def run_simple(args, exp_dir, logger, llm, train_paths, bounds=None):
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
        bounds=bounds,
        frame="stationary",
        ranking=True,
        diversity_feedback=False,
        repeat_penalty=1.0,
        patience=0,
        fitness_workers=args.fitness_workers,
    )

    priors = load_priors()
    prior_queue = list(priors)
    random.shuffle(prior_queue)

    def next_prior():
        # Muestreo sin reemplazo; al agotarse, se reinicia el conjunto.
        if not prior_queue:
            prior_queue.extend(priors)
            random.shuffle(prior_queue)
        return prior_queue.pop()

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
    candidates_path = os.path.join(exp_dir, "candidates.jsonl")
    started = time.time()
    last_error = ""
    for iteration in range(1, args.budget + 1):
        if iteration % args.prior_every == 0:
            parent_label, parent_code = next_prior()
            strategy = "refine"
        elif random.random() < 0.5:
            parent_label, parent_code = "current-best", best_code
            strategy = "refine"
        else:
            parent_label, parent_code = "current-best", best_code
            strategy = "create"

        prompt = build_prompt(parent_code, strategy, last_error=last_error)
        try:
            response = llm.query([{"role": "user", "content": prompt}])
            code = parse_llm_response(response or "")
        except Exception as error:  # noqa: BLE001
            logger.warning("iter %d: respuesta invalida: %s", iteration, error)
            last_error = f"{type(error).__name__}: {error}"
            _append_jsonl(candidates_path, {
                "iteration": iteration, "valid": False,
                "error": f"{type(error).__name__}: {error}",
                "parent": parent_label, "strategy": strategy,
                "description": "", "code": (response or "")[:2000],
            })
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"iter {iteration} best={best_fitness:.6f} candidate=worst "
                    f"valid=0 error={type(error).__name__} "
                    f"parent={parent_label} strategy={strategy}\n")
            continue

        try:
            minimal_compile(code)
        except Exception as error:  # noqa: BLE001
            logger.warning("iter %d: filtro minimo: %s", iteration, error)
            last_error = f"{type(error).__name__}: {error}"
            _append_jsonl(candidates_path, {
                "iteration": iteration, "valid": False,
                "error": f"{type(error).__name__}: {error}",
                "parent": parent_label, "strategy": strategy,
                "description": extract_description(response or ""),
                "code": code[:2000],
            })
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"iter {iteration} best={best_fitness:.6f} candidate=worst "
                    f"valid=0 error={type(error).__name__} "
                    f"parent={parent_label} strategy={strategy}\n")
            continue

        fitness, error = _evaluate(evaluator, code)
        last_error = (str(error) if error else "")
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

        _append_jsonl(candidates_path, {
            "iteration": iteration, "valid": bool(math.isfinite(fitness)),
            "fitness": fitness, "error": (str(error) if error else None),
            "parent": parent_label, "strategy": strategy,
            "description": extract_description(response or ""),
            "code": code[:2000],
        })
        finite = math.isfinite(fitness)
        candidate_text = f"{fitness:.6f}" if finite else "worst"
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"iter {iteration} best={best_fitness:.6f} "
                f"candidate={candidate_text} valid={1 if finite else 0} "
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
    if hasattr(llm, "usage_snapshot"):
        _write_json(os.path.join(exp_dir, "tokens.json"), llm.usage_snapshot())
    logger.info("Fin: best fitness=%s en iteracion %s", best_fitness, best_generation)
    print(f"Mejor fitness: {best_fitness:.6f} (iteracion {best_generation})")
    return best_code
