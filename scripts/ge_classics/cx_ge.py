"""Adaptador del CX_Cross de la evaluacion GE a la interfaz del arnes.

Devuelve los DOS hijos complementarios del mismo ciclo. `LLMOperator`
detecta la tupla y los usa directamente, sin reemplazar el segundo por
`child[::-1]`. Por tanto reproduce exactamente el operador que usa la
evaluacion gramatical original (`mohh.evaluation.operators.crossover.CX_Cross`).
"""
import numpy as np

from mohh.evaluation.operators.crossover import CX_Cross


def crossover(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    X = np.zeros((2, 1, n), dtype=x.dtype)
    X[0, 0] = x
    X[1, 0] = y
    out = CX_Cross(prob=1.0)._do(None, X)
    return (np.asarray(out[0, 0], dtype=x.dtype),
            np.asarray(out[1, 0], dtype=x.dtype))
