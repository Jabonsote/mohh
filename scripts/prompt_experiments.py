#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimentos A/B de plantillas LLaMEA/BAG con un modelo pequeno.

Objetivo: separar fallos de **sintaxis**, **nombre/contrato**, **logica** y
**evaluacion**. Escribe cada candidato en JSONL al instante (resume), mide el
ranking de los validos y calcula diversidad CrystalBLEU en un hilo aparte.

Uso:
    python scripts/prompt_experiments.py --model qwen2.5-coder:7b-instruct \
        --n-per-config 6 --out /tmp/prompt_round1
"""
import argparse
import concurrent.futures
import glob
import json
import os
import re
import time

import numpy as np

from mohh.llamea.fitness import build_evaluator
from mohh.llamea.parsing import parse_llm_response
from mohh.llamea.priors import load_ge_priors
from mohh.llamea.simple import Candidate, minimal_compile
from mohh.llamea.simple_priors import load_simple_priors
from mohh.llamea.vllm import VLLM_LLM

ROLE = (
    "You are an excellent Python programmer. You are a Python expert working "
    "on a new optimization algorithm."
)

TASK = """Your task is to design a crossover (recombination) operator for
permutation-encoded multi-objective optimization problems. The operator must
work for any n >= 2. `x` and `y` are numpy arrays that are permutations of
0..n-1. Do not modify `x` or `y`. Return a child that is a permutation of
0..n-1."""

TASK_HARD = TASK + """

HARD REQUIREMENTS
- The function MUST be named exactly `crossover(x, y)`.
- `x` and `y` are numpy arrays, not Python lists. Do NOT call `.index()` or
  `.pop()` on them; if you need list methods, use `list(x)` / `list(y)`.
- Return exactly one child (not a tuple) unless the task explicitly asks for two.
- The child must have length n, contain every value in 0..n-1 exactly once,
  and contain no -1 or duplicate.
- Avoid FANCY INDEXING with arrays of a different length. If you use
  `child[a:b] = ...`, both sides must have the same length.
- Do not write an infinite loop or an example/self-test at module level.
- To fill the empty positions, use exactly
  `positions = [i for i in range(n) if child[i] == -1]` and zip those positions
  with the remaining values. Do NOT use `np.random.permutation(n)` for the
  positions: it can overwrite positions that already contain a value.
- Do not write `range(a, b) + range(c, d)`; use
  `list(range(a, b)) + list(range(c, d))`.
- If you use a `for i in range(n)` loop, do not modify `i` inside it.
- It must work when `x = [0, 1, ..., n-1]` and `y = [n-1, ..., 1, 0]`.
  Test that case mentally before returning."""

TASK_ONE = TASK + "\nReturn exactly one child permutation (not a tuple)."
TASK_ONE_HARD = TASK_HARD

STRATEGY_REFINE = "Refine the strategy of the selected solution to improve it."
STRATEGY_CREATE = (
    "Generate a new algorithm that is different from the algorithms you have "
    "tried before."
)

FORMAT = """Provide the Python code and also give it a one-line description,
describing the main idea. Give the response in the format:

# Description: <short-description>
# Code:
```python
<code>
```"""

FORMAT_HARD = """Give exactly this format and nothing else:

# Description: <one line>
# Code:
```python
import numpy as np

def crossover(x, y):
    ...
```"""

SIMPLE = """import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]
    used = set(x[a:b].tolist())
    fill = [v for v in y if v not in used]
    positions = list(range(b, n)) + list(range(0, a))
    for pos, value in zip(positions, fill):
        child[pos] = value
    return child
"""


def bag_prompt(reference, strategy, description="", score=None, error="",
               one_child=False, hard=False):
    task = (TASK_ONE_HARD if one_child and hard else
            TASK_ONE if one_child else TASK_HARD if hard else TASK)
    fmt = FORMAT_HARD if hard else FORMAT
    history = ""
    if score is not None:
        history = (
            "The current population of algorithms already evaluated "
            f"(name, description, score) is:\n- {description or 'candidate'} "
            f"(Score: {score:.4f}, where 1 is best and 6 is worst)\n\n"
            "The selected solution to update is:\n"
            f"{description or 'candidate'}\n\nWith code:\n")
    strategy_text = STRATEGY_REFINE if strategy == "refine" else STRATEGY_CREATE
    prompt = (
        f"{ROLE}\n\nTASK\n{task}\n\n{history}"
        f"```python\n{reference}\n```\n\n"
        f"STRATEGY\n{strategy_text}\n\n{fmt}"
    )
    if error:
        prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n"
            f"{error}\n"
            "Fix that exact error. Return a complete, runnable operator.")
    return prompt


def current_prompt(reference, strategy, error="", **_kwargs):
    from mohh.llamea.simple import build_prompt
    return build_prompt(reference, strategy, last_error=error)


CONFIGS = [
    dict(name="current_ge", builder=current_prompt, temp=0.7,
         simple=False, feedback=False, one_child=False, hard=False),
    dict(name="bag_ge", builder=bag_prompt, temp=0.7,
         simple=False, feedback=False, one_child=False, hard=True),
    dict(name="bag_ge_fb", builder=bag_prompt, temp=0.7,
         simple=False, feedback=True, one_child=False, hard=True),
    dict(name="bag_simple", builder=bag_prompt, temp=0.3,
         simple=True, feedback=False, one_child=True, hard=False),
    dict(name="bag_simple_fb", builder=bag_prompt, temp=0.3,
         simple=True, feedback=True, one_child=True, hard=False),
    dict(name="hard_simple", builder=bag_prompt, temp=0.2,
         simple=True, feedback=True, one_child=True, hard=True),
    dict(name="hard_simple_tuple", builder=bag_prompt, temp=0.2,
         simple=True, feedback=True, one_child=False, hard=True),
    dict(name="hard_ge", builder=bag_prompt, temp=0.3,
         simple=False, feedback=True, one_child=False, hard=True),
    # Ronda 2: priors clasicos simples, que son validos y diversos.
    dict(name="prior_mix", builder=bag_prompt, temp=0.3,
         simple=True, feedback=True, one_child=True, hard=True),
    dict(name="prior_mix_tuple", builder=bag_prompt, temp=0.3,
         simple=True, feedback=True, one_child=False, hard=True),
    dict(name="prior_mix_low", builder=bag_prompt, temp=0.15,
         simple=True, feedback=True, one_child=True, hard=True),
    dict(name="prior_mix_nofb", builder=bag_prompt, temp=0.3,
         simple=True, feedback=False, one_child=True, hard=True),
    dict(name="prior_corner", builder=bag_prompt, temp=0.3,
         simple=True, feedback=True, one_child=True, hard=True),
    dict(name="prior_corner_tuple", builder=bag_prompt, temp=0.3,
         simple=True, feedback=True, one_child=False, hard=True),
]


def classify(response):
    """Devuelve (status, detail, code). Separa sintaxis, nombre y contrato."""
    try:
        code = parse_llm_response(response or "")
    except Exception as error:  # noqa: BLE001
        text = f"{type(error).__name__}: {error}"
        if "SyntaxError" in text:
            return "syntax", text, ""
        if "No code block" in text:
            return "parse", text, ""
        return "name", text, ""
    try:
        minimal_compile(code)
    except Exception as error:  # noqa: BLE001
        return "contract", f"{type(error).__name__}: {error}", code
    return "ok", None, code


def _eval(evaluator, code):
    candidate = Candidate(code)
    try:
        result = evaluator(candidate)
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {error}"
    if result is None or result.error is not None:
        return None, str(getattr(result, "error", "evaluation failed"))
    fitness = float(result.fitness)
    if not np.isfinite(fitness):
        return None, "fitness not finite"
    return fitness, None


def _description(response):
    match = re.search(r"#\s*Description\s*:\s*(.+)", response or "")
    return match.group(1).strip()[:200] if match else ""


def _diversity(codes_by_config, path):
    try:
        from mohh.analysis.code_diversity import crystalbleu_diversity
    except Exception as error:  # noqa: BLE001
        with open(path, "w") as handle:
            json.dump({"error": str(error)}, handle)
        return {}
    result = {}
    for name, codes in codes_by_config.items():
        unique = list(dict.fromkeys(codes))
        result[name] = {
            "n_codes": len(unique),
            "diversity": crystalbleu_diversity(unique) if len(unique) > 1 else 0.0,
        }
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--n-per-config", type=int, default=6)
    parser.add_argument("--train-instances", type=int, default=2)
    parser.add_argument("--out", default="/tmp/prompt_experiments")
    parser.add_argument("--only", default="", help="nombres separados por coma")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "results.jsonl")
    existing = set()
    rows = []
    if args.resume and os.path.exists(rows_path):
        for line in open(rows_path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
            existing.add((row["config"], row["index"]))
        print(f"resume: {len(rows)} filas existentes", flush=True)

    train = sorted(glob.glob("datasets/mqap/train/*.txt"))[:args.train_instances]
    evaluator = None
    if not args.no_eval:
        evaluator = build_evaluator(
            train, mo_generations=10, mo_population_size=10, seed=42,
            model_name="MOEAD", mutation=False, normalize=True,
            frame="stationary", ranking=True, fitness_workers=1,
        )
    priors = load_ge_priors()
    simple_priors = load_simple_priors()
    prior_scores = []
    if evaluator is not None:
        for label, code in simple_priors + priors:
            score, error = _eval(evaluator, code)
            prior_scores.append((label, score))
    best_prior = max((score for _, score in prior_scores), default=None)
    print("priors:", prior_scores, "best:", best_prior, flush=True)

    llm = VLLM_LLM(
        model=args.model, base_url=args.base_url, api_key="ollama",
        temperature=0.7, max_tokens=1536, timeout=180, operator_memory=0,
    )

    selected = [c for c in CONFIGS if not args.only
                or c["name"] in set(args.only.split(","))]
    codes_by_config = {c["name"]: [] for c in selected}
    handle = open(rows_path, "a")
    for config in selected:
        name = config["name"]
        llm.temperature = config["temp"]
        last_error = ""
        for index in range(args.n_per_config):
            if (name, index) in existing:
                continue
            strategy = "refine" if index % 2 == 0 else "create"
            if config["simple"]:
                label, reference = simple_priors[index % len(simple_priors)]
            else:
                label, reference = priors[index % len(priors)]
            kwargs = {}
            if config["builder"] is bag_prompt:
                kwargs = {
                    "description": label,
                    "score": best_prior,
                    "one_child": config["one_child"],
                    "hard": config["hard"],
                }
            prompt = config["builder"](
                reference, strategy,
                error=last_error if config["feedback"] else "", **kwargs)
            started = time.time()
            try:
                response = llm.query([{"role": "user", "content": prompt}])
            except Exception as error:  # noqa: BLE001
                row = {"config": name, "index": index,
                       "status": "request_error", "error": str(error),
                       "code": "", "seconds": time.time() - started}
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                last_error = str(error)
                print(f"{name}[{index}] request_error", flush=True)
                continue

            status, detail, code = classify(response)
            fitness = None
            if status == "ok" and evaluator is not None:
                fitness, eval_error = _eval(evaluator, code)
                if fitness is None:
                    status, detail = "evaluation", eval_error
                else:
                    last_error = ""
            if status != "ok":
                last_error = detail or status
            if code:
                codes_by_config[name].append(code)
            row = {
                "config": name, "index": index, "status": status,
                "error": detail, "fitness": fitness, "code": code[:6000],
                "description": _description(response or ""),
                "seconds": round(time.time() - started, 2),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{name}[{index}] {status} fit={fitness} "
                  f"{row['seconds']:.1f}s {(detail or '')[:90]}", flush=True)
    handle.close()

    # CrystalBLEU en segundo plano, mientras se resume.
    diversity_path = os.path.join(args.out, "diversity.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_diversity, codes_by_config, diversity_path)
        print("\nconfig                 n  syntax name contract evaluation  ok   rate  best",
              flush=True)
        for config in selected:
            name = config["name"]
            subset = [row for row in rows if row.get("config") == name]
            counts = {}
            for row in subset:
                counts[row.get("status", "unknown")] = (
                    counts.get(row.get("status", "unknown"), 0) + 1)
            ok = counts.get("ok", 0)
            scores = [row.get("fitness") for row in subset
                      if row.get("fitness") is not None]
            best = max(scores) if scores else None
            rate = 100.0 * ok / len(subset) if subset else 0.0
            print(f"{name:22s} {len(subset):2d}  {counts.get('syntax', 0):6d} "
                  f"{counts.get('name', 0):4d} {counts.get('contract', 0):8d} "
                  f"{counts.get('evaluation', 0):10d} {ok:4d} {rate:5.1f}%  {best}",
                  flush=True)
        diversity = future.result()
    print("\ndiversity:", json.dumps(diversity, indent=2), flush=True)

    # Errores mas comunes entre los rechazados.
    common = {}
    for row in rows:
        if row.get("status") == "ok":
            continue
        text = row.get("error") or row.get("status") or ""
        key = text.split("\n")[0][:90]
        common[key] = common.get(key, 0) + 1
    print("\ncommon failures:", flush=True)
    for key, count in sorted(common.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{count:3d}  {key}", flush=True)


if __name__ == "__main__":
    main()
