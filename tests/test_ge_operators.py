"""Operadores GE como fuente comparable (mohh.llamea.ge_operators)."""

import numpy as np
import pytest

from mohh.llamea.ge_operators import (
    discover_operator_files,
    ge_callable,
    ge_namespace,
    load_ge_operators,
)

# Fenotipo real del proyecto original (`MOEAD_operators.txt`).
BEST = ("order_based(fill_first_occurring,preserve_elements,y,"
        "order_based(fill_first_occurring,preserve_elements,y,reverse(y)))")


def _write_operators(tmp_path, name="MOEAD_operators.txt"):
    path = tmp_path / name
    path.write_text(
        f"Best: {BEST}\n"
        "Middle: order_based(fill_first_occurring,preserve_segments,x,y)\n"
        "Worst: order_based(fill_first_occurring,preserve_elements,y,x)\n",
        encoding="utf-8")
    return path


def test_discover_and_load_ge_operators(tmp_path):
    _write_operators(tmp_path)
    files = discover_operator_files(str(tmp_path))
    assert len(files) == 1
    operators = load_ge_operators(str(tmp_path))
    assert set(operators) == {"MOEAD_Best", "MOEAD_Middle", "MOEAD_Worst"}
    assert operators["MOEAD_Best"] == BEST


def test_load_ge_operators_can_filter_by_model(tmp_path):
    _write_operators(tmp_path)
    _write_operators(tmp_path, "NSGAII_operators.txt")
    assert len(load_ge_operators(str(tmp_path))) == 6
    filtered = load_ge_operators(str(tmp_path), model="NSGAII")
    assert set(filtered) == {"NSGAII_Best", "NSGAII_Middle", "NSGAII_Worst"}


def test_ge_callable_produces_a_permutation():
    crossover = ge_callable(BEST)
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([5, 3, 1, 4, 2, 0])
    child = crossover(x, y)
    # `order_based` devuelve dos hijos (tupla), como en la ruta GE.
    children = child if isinstance(child, tuple) else (child,)
    for item in children:
        assert sorted(np.asarray(item).tolist()) == list(range(6))


def test_ge_namespace_exposes_crossover():
    namespace = ge_namespace(BEST)
    assert callable(namespace["crossover"])
    assert "np" in namespace
