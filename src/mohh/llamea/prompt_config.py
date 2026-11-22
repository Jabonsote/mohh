"""Prompts del bucle (1+1) desde ``config/prompts.yaml``.

**Anadir una clave no requiere tocar codigo**: ``prompt(key)`` lee el YAML.
Unico lector del archivo; ``simple.py`` importa de aqui por compatibilidad.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = Path(os.environ.get(
    "MOHH_PROMPTS_REGISTRY", str(_ROOT / "config" / "prompts.yaml")))


@lru_cache(maxsize=1)
def registry() -> dict:
    """Carga el YAML. Se puede recargar con ``registry.cache_clear()``."""
    with open(REGISTRY, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=256)
def prompt(key: str) -> str:
    """Texto de una clave. ``TASK_ONE`` se deriva sin duplicar ``TASK``."""
    data = registry()
    if key == "TASK_ONE":
        return data["TASK"] + data["TASK_ONE_SUFFIX"]
    if key not in data:
        raise KeyError(f"prompt '{key}' no existe en {REGISTRY}")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"prompt '{key}' no es texto en {REGISTRY}")
    return value


ROLE = prompt("ROLE")
TASK = prompt("TASK")
TASK_ONE = prompt("TASK_ONE")
STRATEGY_REFINE = prompt("STRATEGY_REFINE")
STRATEGY_CREATE = prompt("STRATEGY_CREATE")
SKELETON = prompt("SKELETON")
FORMAT = prompt("FORMAT")


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
    task = prompt("TASK_ONE") if one_child else prompt("TASK")
    if strategy == "refine":
        strategy_text = prompt("STRATEGY_REFINE")
    else:
        strategy_text = prompt("STRATEGY_CREATE")
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
                     "known operator verbatim)\n```python\n" + prompt("SKELETON") + "```")
    prompt_text = (
        f"{prompt('ROLE')}\n\nTASK\n{task}\n\n{history_text}"
        f"{reference}\n\n"
        f"STRATEGY\n{strategy_text}\n\n{prompt('FORMAT')}"
    )
    if last_error:
        prompt_text += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n"
            f"{last_error}\n"
            "Fix that exact error. Return a complete, runnable operator.")
    return prompt_text
