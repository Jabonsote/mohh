"""Pruebas del script que siembra un operador (RNG global)."""
import numpy as np
import pytest

from mohh.llamea.fitness import compile_operator, uses_unseeded_rng
from scripts.seed_operator import seed_operator

UNSEEDED = """import numpy as np

def crossover(x, y):
    rng = np.random.default_rng()
    n = len(x)
    start = rng.integers(0, n - 1)
    end = rng.integers(start + 1, n)
    child = np.empty(n, dtype=x.dtype)
    child[start:end + 1] = x[start:end + 1]
    for pos in range(n):
        if start <= pos <= end:
            continue
        child[pos] = y[pos]
    seen = set(child.tolist())
    for pos in range(n):
        if child[pos] in seen and list(child.tolist()).count(child[pos]) > 1:
            child[pos] = next(v for v in range(n) if v not in seen)
            seen.add(child[pos])
    return child
"""


def test_seed_operator_replaces_the_private_generator():
    seeded, warnings = seed_operator(UNSEEDED)
    assert not uses_unseeded_rng(seeded)
    assert "np.random.randint(" in seeded
    assert "default_rng" not in seeded
    assert warnings == []


def test_seeded_operator_is_valid():
    seeded, _ = seed_operator(UNSEEDED)
    namespace = compile_operator(seeded)  # valida que siga siendo un operador
    assert callable(namespace["crossover"])


def test_seeded_operator_is_reproducible():
    seeded, _ = seed_operator(UNSEEDED)
    crossover = compile_operator(seeded)["crossover"]

    def hijo(seed):
        np.random.seed(seed)
        x = np.random.permutation(20)
        y = np.random.permutation(20)
        return crossover(x, y).copy()

    np.testing.assert_array_equal(hijo(5), hijo(5))


def test_unseeded_operator_is_not_reproducible():
    """El motivo de sembrar: el generador propio ignora la semilla del MOEA."""
    namespace = compile_operator(UNSEEDED, check_seeded_rng=False)
    crossover = namespace["crossover"]

    def hijo():
        np.random.seed(5)
        x = np.random.permutation(20)
        y = np.random.permutation(20)
        return crossover(x, y).copy()

    assert not np.array_equal(hijo(), hijo())


def test_seed_operator_keeps_a_seeded_generator():
    code = """import numpy as np

def crossover(x, y):
    rng = np.random.default_rng(42)
    child = np.array(x, copy=True)
    child[0], child[1] = child[1], child[0]
    return child
"""
    seeded, warnings = seed_operator(code)
    assert not uses_unseeded_rng(seeded)
    assert "default_rng(42)" in seeded
    assert warnings == []
