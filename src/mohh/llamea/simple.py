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

from mohh.llamea.fitness import build_evaluator, compile_operator
from mohh.llamea.parsing import parse_llm_response
from mohh.llamea.priors import load_priors
from mohh.llamea.vllm import extract_description

ROLE = (
    "You are an expert in evolutionary computation and Python. "
    "You design variation operators for permutation-encoded "
    "multi-objective optimization problems."
)

TASK = """Design a crossover (recombination) operator.

Write a module-level Python function named EXACTLY:

    def crossover(x, y):

`x` and `y` are numpy arrays that are permutations of 0..n-1.
Return one child permutation, or a tuple (child1, child2).
Do not modify `x` or `y`.
The child must be a permutation of 0..n-1.
The operator must work for any n >= 2.

Do NOT call `.pop()` or `.index()` on a numpy array; convert with
`list(x)` first if needed. Ensure `len(child) == n`, every value appears
exactly once, and there are no -1 entries. It must also work when `x` is
`[0, 1, ..., n-1]` and `y` is `[n-1, ..., 1, 0]`."""

TASK_ONE = TASK + "\nReturn exactly one child permutation (not a tuple)."

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
def crossover(x, y):
    ...
```

The function must be named exactly `crossover`."""


class Candidate:
    def __init__(self, code):
        self.code = code
        self.fitness = float("nan")
        self.feedback = ""
        self.error = None
        self.hv = None
        self.hv_values = None

    def set_scores(self, fitness, feedback="", error=None):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error


def build_prompt(parent_code, strategy, last_error="", score=None,
                 description="", one_child=True, history=()):
    """Plantilla BAG: rol, tarea, historial, codigo, estrategia y formato."""
    task = TASK_ONE if one_child else TASK
    if strategy == "refine":
        strategy_text = STRATEGY_REFINE
    else:
        strategy_text = STRATEGY_CREATE
    history_text = ""
    if score is not None:
        rows = []
        for item in list(history)[-6:]:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                rows.append(f"- {item[0]} (Score: {item[1]:.4f})")
            else:
                rows.append(f"- {item}")
        listed = "\n".join(rows)
        history_text = (
            "The current population of algorithms already evaluated "
            "(name, description, score) is:\n"
            f"{listed or '- current best (Score: %.4f)' % score}\n\n"
            "The selected solution to update is:\n"
            f"{description or 'current best'} (Score: {score:.4f}; "
            "1 is best, 6 is worst)\n\n"
            "Do not repeat an operator already evaluated; change its mechanism.\n\n")
    prompt = (
        f"{ROLE}\n\nTASK\n{task}\n\n{history_text}"
        f"REFERENCE CODE\n```python\n{parent_code}\n```\n\n"
        f"STRATEGY\n{strategy_text}\n\n{FORMAT}"
    )
    if last_error:
        prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n"
            f"{last_error}\n"
            "Fix that exact error. Return a complete, runnable operator.")
    return prompt


def minimal_compile(code):
    """Filtro minimo con timeout y validacion de ambos hijos.

    Delega en ``compile_operator`` (el mismo camino que usa el evaluador):
    compila con limite de tiempo, comprueba que exista ``crossover`` y prueba
    tres tamanos. Asi un operador con ``while True`` no cuelga la busqueda.
    """
    return compile_operator(code)


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
        return float("-inf"), None, f"{type(error).__name__}: {error}"
    if result is None or result.error is not None:
        return float("-inf"), None, getattr(result, "error", "evaluation failed")
    fitness = float(result.fitness)
    if not math.isfinite(fitness):
        return float("-inf"), None, "fitness is not finite"
    hv = getattr(result, "hv", None)
    return fitness, hv, None


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

    best_code, best_fitness, best_hv = None, float("-inf"), None
    for label, code in priors:
        fitness, hv, error = _evaluate(evaluator, code)
        logger.info("Prior %s: fitness=%s hv=%s error=%s",
                    label, fitness, hv, error)
        if fitness > best_fitness:
            best_code, best_fitness, best_hv = code, fitness, hv
    if best_code is None:
        raise RuntimeError("ningun prior pudo evaluarse")

    best_generation = 0
    _write_text(os.path.join(exp_dir, "best_so_far.py"), best_code + "\n")
    _write_json(os.path.join(exp_dir, "best_so_far.json"), {
        "generation": 0,
        "fitness": best_fitness,
        "hv": best_hv,
        "name": "prior",
        "seed": args.seed,
    })

    progress_path = os.path.join(exp_dir, "progress.txt")
    candidates_path = os.path.join(exp_dir, "candidates.jsonl")
    started = time.time()
    last_error = ""
    history = []
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

        prompt = build_prompt(
            parent_code, strategy, last_error=last_error,
            score=best_fitness, description=parent_label, one_child=True,
            history=history)
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
                    f"iter {iteration} best={best_fitness:.6f} "
                    f"best_hv={best_hv} candidate=worst "
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
                    f"iter {iteration} best={best_fitness:.6f} "
                    f"best_hv={best_hv} candidate=worst "
                    f"valid=0 error={type(error).__name__} "
                    f"parent={parent_label} strategy={strategy}\n")
            continue

        fitness, hv, error = _evaluate(evaluator, code)
        last_error = (str(error) if error else "")
        # El rango GE manda. Solo si hay empate exacto se usa HV como desempate:
        # un candidato con el mismo rango pero mejor HV si avanza.
        improved = fitness > best_fitness or (
            fitness == best_fitness and hv is not None and best_hv is not None
            and hv > best_hv + 1e-12)
        if improved:
            best_code, best_fitness, best_hv = code, fitness, hv
            best_generation = iteration
            _write_text(os.path.join(exp_dir, "best_so_far.py"), best_code + "\n")
            _write_text(os.path.join(exp_dir, "llamea_best_operator.py"),
                        best_code + "\n")
            _write_json(os.path.join(exp_dir, "best_so_far.json"), {
                "generation": iteration,
                "fitness": best_fitness,
                "hv": best_hv,
                "name": "crossover",
                "seed": args.seed,
            })

        description = extract_description(response or "")
        if description:
            history.append((description, fitness))
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
                f"best_hv={best_hv} candidate={candidate_text} hv={hv} "
                f"valid={1 if finite else 0} error={error or '-'} parent={parent_label} "
                f"strategy={strategy} wall_s={time.time() - started:.1f}\n")

    _write_json(os.path.join(exp_dir, "llamea_best_info.json"), {
        "train_fitness": best_fitness,
        "train_hv": best_hv,
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
