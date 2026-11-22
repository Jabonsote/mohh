"""Prompts del bucle (1+1) desde config/prompts.yaml."""
import os
import subprocess
from pathlib import Path

from mohh.llamea import prompt_config, simple
from mohh.llamea.prompt_config import build_prompt, prompt

ROOT = Path(__file__).resolve().parents[1]

# Copia literal de las constantes y de build_prompt antes del refactor.
LEGACY = {}
LEGACY["ROLE"] = (
    "You are an expert in evolutionary computation and Python. "
    "You design variation operators for permutation-encoded "
    "multi-objective optimization problems."
)
LEGACY["TASK"] = """Design a crossover (recombination) operator.

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
LEGACY["TASK_ONE"] = LEGACY["TASK"] + "\nReturn exactly one child permutation (not a tuple)."
LEGACY["STRATEGY_REFINE"] = (
    "Refine the strategy of the selected solution to improve it. "
    "Keep its main mechanism and change only what is needed."
)
LEGACY["STRATEGY_CREATE"] = (
    "Generate a new algorithm that is different from the algorithms you have "
    "tried before. Use a different combination of order, position, adjacency "
    "or cycle ideas."
)
LEGACY["SKELETON"] = """import numpy as np

def crossover(x, y):
    n = len(x)
    child = np.empty(n, dtype=x.dtype)   # construye una permutacion NUEVA
    # ... tu mecanismo ...
    return child
"""
LEGACY["FORMAT"] = """Answer in exactly this format:

# Description: <one line>
# Code:
```python
def crossover(x, y):
    ...
```

The function must be named exactly `crossover`."""


def _legacy_build_prompt(parent_code, strategy, last_error="", score=None,
                         description="", one_child=True, history=(),
                         best_score=None):
    task = LEGACY["TASK_ONE"] if one_child else LEGACY["TASK"]
    if strategy == "refine":
        strategy_text = LEGACY["STRATEGY_REFINE"]
    else:
        strategy_text = LEGACY["STRATEGY_CREATE"]
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
                     "known operator verbatim)\n```python\n" + LEGACY["SKELETON"] + "```")
    result = (
        f"{LEGACY['ROLE']}\n\nTASK\n{task}\n\n{history_text}"
        f"{reference}\n\n"
        f"STRATEGY\n{strategy_text}\n\n{LEGACY['FORMAT']}"
    )
    if last_error:
        result += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n"
            f"{last_error}\n"
            "Fix that exact error. Return a complete, runnable operator.")
    return result


PARENT = "import numpy as np\n\ndef crossover(x, y):\n    return x.copy()\n"


def test_refine_con_parent_es_identico():
    kwargs = dict(last_error="ValueError: not a permutation", score=2.5,
                  description="OX", history=[("OX", 3.0)], best_score=1.1)
    assert build_prompt(PARENT, "refine", **kwargs) == \
        _legacy_build_prompt(PARENT, "refine", **kwargs)


def test_create_sin_parent_es_identico():
    kwargs = dict(score=None, description="new-operator",
                  history=[("OX", 3.0)], best_score=1.1)
    assert build_prompt(None, "create", **kwargs) == \
        _legacy_build_prompt(None, "create", **kwargs)


def test_constantes_y_task_one_derivado():
    assert prompt_config.REGISTRY.name == "prompts.yaml"
    assert prompt_config.REGISTRY.exists()
    for key, value in LEGACY.items():
        assert prompt(key) == value
        assert getattr(simple, key) == value
    assert prompt("TASK_ONE") == prompt("TASK") + prompt_config.registry()["TASK_ONE_SUFFIX"]


def test_clave_nueva_sin_tocar_codigo(tmp_path):
    texto = (ROOT / "config" / "prompts.yaml").read_text(encoding="utf-8")
    texto += '\nNUEVA_CLAVE: "texto futuro"\n'
    custom = tmp_path / "prompts.yaml"
    custom.write_text(texto, encoding="utf-8")
    env = os.environ.copy()
    env["MOHH_PROMPTS_REGISTRY"] = str(custom)
    result = subprocess.run(
        ["python", "-c",
         "from mohh.llamea.prompt_config import prompt; print(prompt('NUEVA_CLAVE'))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "texto futuro"
