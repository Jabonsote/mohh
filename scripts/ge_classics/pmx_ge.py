"""Adaptador del PMX_Cross de la evaluacion GE a la interfaz del arnes.

Devuelve los DOS hijos del mismo corte, como el `PMX_Cross` original
(`mohh.evaluation.operators.crossover.PMX_Cross`). `LLMOperator` usa la
tupla directamente, sin `child[::-1]`.
"""
import numpy as np

from mohh.evaluation.operators.crossover import PMX_Cross


def crossover(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    X = np.zeros((2, 1, n), dtype=x.dtype)
    X[0, 0] = x
    X[1, 0] = y
    out = PMX_Cross(prob=1.0)._do(None, X)
    return (np.asarray(out[0, 0], dtype=x.dtype),
            np.asarray(out[1, 0], dtype=x.dtype))
