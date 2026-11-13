"""Pruebas de la diversidad de código con CrystalBLEU."""
import pytest

from mohh.analysis.code_diversity import (
    code_tokens, crystal_ignoring, crystalbleu_diversity,
    crystalbleu_similarity)

A = """import numpy as np
def crossover(x, y):
    c = np.array(x, copy=True)
    c[0], c[1] = c[1], c[0]
    return c
"""

B = """import numpy as np
def crossover(x, y):
    return np.roll(x, 1)
"""


def test_identical_code_has_zero_diversity():
    assert crystalbleu_diversity([A, A]) == 0.0


def test_different_code_has_positive_diversity():
    assert crystalbleu_diversity([A, B]) > 0.0


def test_similarity_is_symmetric():
    left = crystalbleu_similarity(A, B)
    right = crystalbleu_similarity(B, A)
    assert left == pytest.approx(right)


def test_ignoring_is_not_empty():
    ignoring = crystal_ignoring([A, B])
    assert ignoring
    assert all(isinstance(item, tuple) for item in ignoring)


def test_tokenizer_falls_back_on_invalid_code():
    assert code_tokens("def broken(:\n")  # regex fallback
