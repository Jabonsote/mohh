"""LLaMEA (1+1) con strong priors, siguiendo BAG (Huang et al., 2026).

Un candidato por iteración, evaluación con el fitness exacto de GE (rango medio
por población, 1 = mejor) y reemplazo elitista. El prompt se apoya en el código
de un prior fuerte y en una estrategia (refinar/crear); cada
``--prior-every`` iteraciones se refina un prior aleatorio.
"""
import hashlib
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from mohh.llamea.fitness import build_population_evaluator, compile_operator
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
`[0, 1, ..., n-1]` and `y` is `[n-1, ..., 1, 0]`.

Use randomness ONLY through numpy's global RNG (`np.random.*`). Do NOT create
your own generator (`np.random.default_rng()`): the evaluation seeds
`np.random.seed(...)` and a private generator would ignore that seed, making
the operator irreproducible.

Answer with the code only: no long explanations, no examples of usage, no
self-checks that call `crossover` at import time. The operator must fit in a
few dozen lines."""

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

#: Esqueleto que se muestra cuando no hay un operador padre (estrategia
#: `create`): fija el contrato sin dar ningun mecanismo que copiar.
SKELETON = """import numpy as np

def crossover(x, y):
    n = len(x)
    child = np.empty(n, dtype=x.dtype)   # construye una permutacion NUEVA
    # ... tu mecanismo ...
    return child
"""

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
                 description="", one_child=True, history=(), best_score=None):
    """Plantilla BAG: rol, tarea, historial, codigo, estrategia y formato.

    `score` es el score **del padre que se muestra** (el incumbent cuando se
    refina, el prior cuando se refina un prior) y `best_score` el del mejor
    operador hasta ahora. Antes se confundian: se le atribuia al prior el
    score del incumbent, asi que el modelo creia que estaba refinando el
    mejor operador cuando en realidad era uno de los peores.

    `parent_code=None` (estrategia `create`) omite el bloque `REFERENCE CODE`
    y lo sustituye por un esqueleto: el modelo no tiene un operador que
    copiar, solo el contrato. Anclar la exploracion al incumbent era lo que
    hacia que la estrategia `create` fuera una copia anotada.
    """
    task = TASK_ONE if one_child else TASK
    if strategy == "refine":
        strategy_text = STRATEGY_REFINE
    else:
        strategy_text = STRATEGY_CREATE
    has_parent = parent_code is not None
    history_text = ""
    rows = []
    for item in list(history)[-6:]:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            rows.append(f"- {item[0]} (mean rank: {item[1]:.4f})")
        else:
            rows.append(f"- {item}")
    listed = "\n".join(rows)
    legend = ("The score is the GE mean rank over the training instances "
              "(1 = best, lower is better; 2 = worst).")
    if has_parent:
        shown = score if score is not None else best_score
        if shown is not None:
            history_text = (
                "The algorithms already evaluated "
                "(description, mean rank) are:\n"
                f"{listed or '- current best (mean rank: %.4f)' % shown}\n\n"
                f"{legend}\n\n"
                "The selected solution to update is:\n"
                f"{description or 'current best'} (mean rank: {shown:.4f})\n\n"
                "Do not repeat an operator already evaluated; change its mechanism.\n\n")
    else:
        history_text = (
            "The algorithms already evaluated "
            "(description, mean rank) are:\n"
            f"{listed or '- (none yet)'}\n\n"
            f"{legend}\n\n"
            "There is no parent operator: design a new one from scratch, "
            "with a mechanism different from the ones listed above.\n\n")
    if has_parent:
        reference = f"REFERENCE CODE\n```python\n{parent_code}\n```"
    else:
        reference = ("SKELETON (fill in the mechanism yourself; do not copy a "
                     "known operator verbatim)\n```python\n" + SKELETON + "```")
    prompt = (
        f"{ROLE}\n\nTASK\n{task}\n\n{history_text}"
        f"{reference}\n\n"
        f"STRATEGY\n{strategy_text}\n\n{FORMAT}"
    )
    if last_error:
        prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n"
            f"{last_error}\n"
            "Fix that exact error. Return a complete, runnable operator.")
    return prompt


def _response_truncated(llm):
    """True si la ultima respuesta del LLM se corto por el limite de tokens.

    Con modelos de razonamiento (`thinking=auto`) el pensamiento se come el
    presupuesto de `max_tokens` y la respuesta se corta antes de escribir el
    codigo: en `glm-gptoss` eso fue el 31,8 % de las peticiones y la causa
    de casi todos los fallos de parseo.
    """
    checker = getattr(llm, "last_truncated", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:  # noqa: BLE001 - nunca debe tumbar la corrida
            return False
    return False


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


def _summary(args, exp_dir, evaluator, llm, state):
    """Resumen consolidado de la corrida, en JSON.

    Reune en un solo archivo lo que hasta ahora estaba repartido: conteo de
    iteraciones y de exitosos/fallidos (con el motivo de cada fallo), tokens
    (total, entrada, salida, razonamiento, peticiones, segundos de
    inferencia), la traza de diversidad conductual, el mejor operador y la
    linea de tiempo por iteracion. Se escribe cada `summary_every` iteraciones
    y al final, para que una caida no se lleve la informacion.
    """
    tokens = {}
    if hasattr(llm, "usage_snapshot"):
        try:
            tokens = llm.usage_snapshot() or {}
        except Exception:  # noqa: BLE001 - el resumen no puede tumbar la corrida
            tokens = {}
    duplicates = {}
    if hasattr(evaluator, "duplicate_stats"):
        try:
            duplicates = evaluator.duplicate_stats()
        except Exception:  # noqa: BLE001
            duplicates = {}
    failures = state["failures"]
    total = state["valid"] + state["invalid"]
    return {
        "experiment": os.path.basename(os.path.normpath(exp_dir)),
        "model": getattr(args, "model", None),
        "method": "llamea-1+1-strong-priors",
        "budget": state["budget"],
        "solver": args.solver,
        "mo_generations": args.mo_generations,
        "mo_population_size": args.mo_population_size,
        "seed": args.seed,
        "iterations": {
            "total": total,
            "valid": state["valid"],
            "invalid": state["invalid"],
            "success_rate": (state["valid"] / total) if total else None,
            "improvements": state["improvements"],
            # Respuestas cortadas por `max_tokens`: el modelo no llego a
            # escribir el codigo. Es el motivo de casi todos los fallos.
            "truncated_responses": tokens.get("truncated_responses"),
        },
        "failures": {
            "total": state["invalid"],
            "by_reason": failures,
        },
        "tokens": tokens,
        "code_diversity": {
            "metric": "CrystalBLEU (Eghbali & Pradel, ASE 2022)",
            "value": state["code_diversity"],
            "n_codes": state["code_diversity_n"],
            "note": "1 - CrystalBLEU medio por pares; 0 = codigo identico",
        },
        "duplicates": duplicates,
        "best": {
            "generation": state["best_generation"],
            "fitness": state["best_fitness"],
            "fitness_metric": "GE population mean rank",
            "fitness_direction": "lower is better",
            "hv": state["best_hv"],
            "operator": "llamea_best_operator.py",
        },
        "timeline": state["timeline"],
        "artifacts": {
            "prompts_and_responses": "llm_requests.jsonl",
            "candidates": "candidates.jsonl",
            "progress": "progress.txt",
            "errors": "error.log",
            "code_diversity": "search_summary.json",
            "raw_fronts": "fronts/",
            "tokens": "tokens.json",
            "best_operator": "llamea_best_operator.py",
            "run_config": "run.json",
        },
    }


def _count_failure(state, stage, reason):
    """Suma un fallo al resumen, agrupado por etapa y tipo de error."""
    state["invalid"] += 1
    key = f"{stage}: {reason}"
    state["failures"][key] = state["failures"].get(key, 0) + 1


#: Codigos que se comparan entre si para el CrystalBLEU. La metricica es
#: O(n^2) en codigos, asi que se mide sobre una muestra uniforme.
CODE_DIVERSITY_SAMPLE = 300


def _schedule_code_diversity(state, pool):
    """Mide el CrystalBLEU en segundo plano (no bloquea la busqueda).

    CrystalBLEU (Eghbali & Pradel, ASE 2022) compara **codigo**, mientras
    que la diversidad conductual del evaluador compara **comportamiento**: son
    complements. Es O(n^2) en el numero de codigos, asi que corre aparte y
    sobre una muestra uniforme de los codigos distintos vistos.
    """
    with state["diversity_lock"]:
        if state["diversity_pending"]:
            return
        codes = list(state["codes"])
        state["diversity_pending"] = True

    def work():
        value = None
        try:
            from mohh.analysis.code_diversity import crystalbleu_diversity

            if len(codes) > 1:
                step = max(1, len(codes) // CODE_DIVERSITY_SAMPLE)
                sample = codes[::step][:CODE_DIVERSITY_SAMPLE]
                value = float(crystalbleu_diversity(sample))
        except Exception:  # noqa: BLE001 - la metrica no puede tumbar la corrida
            value = None
        finally:
            with state["diversity_lock"]:
                state["code_diversity"] = value
                state["code_diversity_n"] = len(codes)
                state["diversity_pending"] = False

    pool.submit(work)


def _generation_files(exp_dir):
    """Rutas del registro por generacion (mismo estilo que el log de GE)."""
    folder = os.path.join(exp_dir, "generations")
    os.makedirs(folder, exist_ok=True)
    return (os.path.join(folder, "individuals.jsonl"),
            os.path.join(folder, "generation_summary.jsonl"))


def _log_individual(path, iteration, candidate, code, description, fitness,
                    error, parent, strategy, evaluator, llm):
    """Una linea por operador de la generacion (`individuals.jsonl`).

    Guarda todo lo que identifica al operador: generacion, hash del codigo,
    descripcion, fitness, error, la huella conductual que dejo el evaluador y
    los tokens consumidos hasta ese momento. Es el mismo registro por
    operador que escribe el log de la GE, para que la corrida se pueda
    analizar sin volver a ejecutar nada.
    """
    digest = hashlib.sha1((code or "").encode("utf-8")).hexdigest()[:12]
    tokens = {}
    if hasattr(llm, "usage_snapshot"):
        try:
            tokens = llm.usage_snapshot() or {}
        except Exception:  # noqa: BLE001 - el registro no puede tumbar la corrida
            tokens = {}
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "generation": iteration,
        "index": 0,
        "name": getattr(candidate, "name", None) if candidate else None,
        "code_sha1": digest,
        "code_chars": len(code or ""),
        "description": description,
        "fitness": (None if fitness is None or not math.isfinite(fitness)
                    else float(fitness)),
        "error": (str(error)[:300] if error else None),
        "parent": parent,
        "strategy": strategy,
        "tokens_total": tokens.get("total_tokens"),
        "tokens_prompt": tokens.get("prompt_tokens"),
        "tokens_completion": tokens.get("completion_tokens"),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_generation_summary(path, iteration, state, evaluator, llm, started):
    """Una linea por generacion con las metricas agregadas (`generation_summary.jsonl`)."""
    tokens = {}
    if hasattr(llm, "usage_snapshot"):
        try:
            tokens = llm.usage_snapshot() or {}
        except Exception:  # noqa: BLE001
            tokens = {}
    valid = [item["fitness"] for item in state["timeline"]
              if item.get("valid") and item.get("fitness") is not None]
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "generation": iteration,
        "population_size": 1,
        "evaluated": len(valid),
        "errors": state["invalid"],
        "valid_rate_gen": (len(valid) / iteration) if iteration else None,
        "best_fitness": state["best_fitness"],
        "mean_fitness": (float(sum(valid) / len(valid)) if valid else None),
        "improvements": state["improvements"],
        "distinct_codes_so_far": len(state["codes"]),
        "code_diversity_crystalbleu": state["code_diversity"],
        "tokens_total": tokens.get("total_tokens"),
        "tokens_prompt": tokens.get("prompt_tokens"),
        "tokens_completion": tokens.get("completion_tokens"),
        "wall_seconds": round(time.time() - started, 3),
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_simple(args, exp_dir, logger, llm, train_paths, bounds=None):
    """Bucle (1+1) con strong priors. Devuelve el mejor código."""
    llm.operator_memory = 0
    evaluator = build_population_evaluator(
        train_paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
        model_name=args.solver,
        mutation=args.mutation,
        normalize=True,
        bounds=None,
        frame="ge",
        raw_rank=True,
        repeat_penalty=1.0,
        patience=0,
        fitness_workers=args.fitness_workers,
        fronts_dir=os.path.join(exp_dir, "fronts"),
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

    prior_solutions = [Candidate(code) for _, code in priors]
    evaluator.evaluate_population(prior_solutions)
    valid_priors = [
        candidate for candidate in prior_solutions
        if candidate.error is None and math.isfinite(float(candidate.fitness))
    ]
    if not valid_priors:
        raise RuntimeError("ningun prior pudo evaluarse")
    best_solution = min(
        valid_priors,
        key=lambda candidate: (
            candidate.fitness,
            -(candidate.hv if candidate.hv is not None else float("-inf")),
        ),
    )
    for (label, _), candidate in zip(priors, prior_solutions):
        logger.info("Prior %s: rank=%s hv=%s error=%s",
                    label, candidate.fitness, candidate.hv, candidate.error)

    # Score de cada prior: el prompt debe mostrar el score del operador que
    # se pide refinar, no el del incumbent.
    prior_scores = {}
    for (label, _), candidate in zip(priors, prior_solutions):
        value = float(candidate.fitness)
        prior_scores[label] = value if math.isfinite(value) else None

    best_code = best_solution.code
    best_fitness = float(best_solution.fitness)
    best_hv = best_solution.hv

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
    summary_path = os.path.join(exp_dir, "search_summary.json")
    individuals_path, generations_path = _generation_files(exp_dir)
    started = time.time()
    last_error = ""
    history = []
    # Contadores del resumen consolidado (exitos/fallos/diversidad/tokens).
    state = {
        "budget": args.budget,
        "valid": 0,
        "invalid": 0,
        "improvements": 0,
        "failures": {},
        "timeline": [],
        "best_generation": 0,
        "best_fitness": best_fitness,
        "best_hv": best_hv,
        # Diversidad de codigo (CrystalBLEU) y su medicion en segundo plano.
        "codes": [],
        "code_diversity": None,
        "code_diversity_n": 0,
        "diversity_lock": threading.Lock(),
        "diversity_pending": False,
    }
    diversity_pool = ThreadPoolExecutor(max_workers=1)
    summary_every = max(1, int(getattr(args, "summary_every", 25) or 25))
    for iteration in range(1, args.budget + 1):
        if iteration % args.prior_every == 0:
            parent_label, parent_code = next_prior()
            strategy = "refine"
            # Si el prior no se pudo evaluar, su score es desconocido: se
            # muestra el del incumbent (que es lo unico que hay).
            parent_score = prior_scores.get(parent_label)
            if parent_score is None:
                parent_score = best_fitness
        elif random.random() < 0.5:
            parent_label, parent_code = "current-best", best_code
            strategy = "refine"
            parent_score = best_fitness
        else:
            # `create` no lleva codigo de referencia: sin un operador que
            # copiar, el modelo explora de verdad. Antes receive el incumbent
            # y "crear" era una copia anotada de su mecanismo.
            parent_label, parent_code = "new-operator", None
            strategy = "create"
            parent_score = None

        prompt = build_prompt(
            parent_code, strategy, last_error=last_error,
            score=parent_score, description=parent_label, one_child=True,
            history=history, best_score=best_fitness)
        try:
            response = llm.query([{"role": "user", "content": prompt}])
            code = parse_llm_response(response or "")
        except Exception as error:  # noqa: BLE001
            truncated = _response_truncated(llm)
            if truncated:
                # Se agotaron los tokens: el modelo no llego a escribir el
                # codigo. Es un fallo del presupuesto, no del operador, asi
                # que se cuenta aparte y el feedback pide una respuesta corta.
                stage = "respuesta_truncada"
                reason = "max_tokens agotado"
                last_error = (
                    "the previous answer was cut off because it ran out of "
                    "tokens before the code was written. Answer with the code "
                    "only: a few dozen lines, no explanations, no usage "
                    "examples, no self-check at import time.")
                logger.warning("iter %d: respuesta truncada (max_tokens)",
                               iteration)
            else:
                stage = "respuesta_invalida"
                reason = type(error).__name__
                last_error = f"{type(error).__name__}: {error}"
                logger.warning("iter %d: respuesta invalida: %s", iteration, error)
            _append_jsonl(candidates_path, {
                "iteration": iteration, "valid": False,
                "error": last_error,
                "parent": parent_label, "strategy": strategy,
                "description": "", "code": (response or "")[:2000],
            })
            with open(progress_path, "a", encoding="utf-8") as handle:
                handle.write(
                    f"iter {iteration} best={best_fitness:.6f} "
                    f"best_hv={best_hv} candidate=worst "
                    f"valid=0 error={stage} "
                    f"parent={parent_label} strategy={strategy}\n")
            _count_failure(state, stage, reason)
            state["timeline"].append({
                "iteration": iteration, "valid": False,
                "fitness": None, "hv": None,
                "reason": type(error).__name__,
            })
            _log_individual(individuals_path, iteration, None, response or "",
                            "", None, error, parent_label, strategy,
                            evaluator, llm)
            _log_generation_summary(generations_path, iteration, state,
                                    evaluator, llm, started)
            if iteration % summary_every == 0:
                _write_json(summary_path,
                            _summary(args, exp_dir, evaluator, llm, state))
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
            _count_failure(state, "filtro_minimo", type(error).__name__)
            state["timeline"].append({
                "iteration": iteration, "valid": False,
                "fitness": None, "hv": None,
                "reason": type(error).__name__,
            })
            _log_individual(individuals_path, iteration, None, code[:2000],
                            extract_description(response or ""), None, error,
                            parent_label, strategy, evaluator, llm)
            _log_generation_summary(generations_path, iteration, state,
                                    evaluator, llm, started)
            if iteration % summary_every == 0:
                _write_json(summary_path,
                            _summary(args, exp_dir, evaluator, llm, state))
            continue

        candidate_solution = Candidate(code)
        try:
            evaluator.evaluate_population([candidate_solution], [best_solution])
        except Exception as error:  # noqa: BLE001
            candidate_solution.set_scores(
                float("inf"), f"{type(error).__name__}: {error}", error)

        if candidate_solution.error is not None:
            fitness = float("inf")
            hv = None
            error = candidate_solution.error
        else:
            fitness = float(candidate_solution.fitness)
            hv = candidate_solution.hv
            error = None
        last_error = (str(error) if error else "")

        # GE: menor rango es mejor. Solo si hay empate exacto se usa HV.
        improved = (
            error is None
            and math.isfinite(fitness)
            and (
                fitness < best_solution.fitness
                or (
                    fitness == best_solution.fitness
                    and hv is not None
                    and best_solution.hv is not None
                    and hv > best_solution.hv + 1e-12
                )
            )
        )
        if improved:
            best_solution = candidate_solution
            best_generation = iteration
            _write_text(os.path.join(exp_dir, "best_so_far.py"), best_solution.code + "\n")
            _write_text(os.path.join(exp_dir, "llamea_best_operator.py"),
                        best_solution.code + "\n")
            _write_json(os.path.join(exp_dir, "best_so_far.json"), {
                "generation": iteration,
                "fitness": best_solution.fitness,
                "hv": best_solution.hv,
                "name": "crossover",
                "seed": args.seed,
            })
        best_code = best_solution.code
        best_fitness = float(best_solution.fitness)
        best_hv = best_solution.hv

        description = extract_description(response or "")
        if description and math.isfinite(fitness):
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
        # `progress.txt` es `clave=valor` separado por espacios: el motivo va
        # como tipo de excepcion (sin espacios). El mensaje completo queda en
        # `candidates.jsonl`.
        error_label = type(error).__name__ if error is not None else "-"
        with open(progress_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"iter {iteration} best={best_fitness:.6f} "
                f"best_hv={best_hv} candidate={candidate_text} hv={hv} "
                f"valid={1 if finite else 0} error={error_label} parent={parent_label} "
                f"strategy={strategy} wall_s={time.time() - started:.1f}\n")
        if finite:
            state["valid"] += 1
        else:
            _count_failure(state, "evaluacion", "sin_frente_valido")
        if improved:
            state["improvements"] += 1
        state["timeline"].append({
            "iteration": iteration, "valid": bool(finite),
            "fitness": fitness if finite else None, "hv": hv,
            "reason": str(error) if error else None,
        })
        state["best_generation"] = best_generation
        state["best_fitness"] = best_fitness
        state["best_hv"] = best_hv
        # Todos los operadores de la generacion quedan registrados (codigo,
        # fitness, error, huella conductual y tokens), como en el log de la GE.
        _log_individual(individuals_path, iteration, candidate_solution, code,
                        description, fitness if finite else None, error,
                        parent_label, strategy, evaluator, llm)
        if code not in state["codes"]:
            state["codes"].append(code)
        _log_generation_summary(generations_path, iteration, state, evaluator,
                                llm, started)
        if iteration % summary_every == 0:
            _schedule_code_diversity(state, diversity_pool)
            _write_json(summary_path, _summary(args, exp_dir, evaluator, llm, state))

    _write_json(os.path.join(exp_dir, "llamea_best_info.json"), {
        "train_fitness": best_fitness,
        "train_fitness_metric": "GE population mean rank",
        "train_fitness_direction": "lower is better",
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
    # El CrystalBLEU se mide en segundo plano: se espera al hilo para que el
    # resumen final tenga el valor (si no, sale `null`).
    diversity_pool.shutdown(wait=True)
    # Resumen consolidado final (tokens, exitosos/fallos, diversidad, mejor).
    _write_json(summary_path, _summary(args, exp_dir, evaluator, llm, state))
    logger.info("Fin: best fitness=%s en iteracion %s", best_fitness, best_generation)
    print(f"Mejor fitness: {best_fitness:.6f} (iteracion {best_generation})")
    print(f"Resumen: {summary_path}")
    return best_code
