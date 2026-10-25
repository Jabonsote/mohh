"""
IGD e IGD+ para problemas de minimizacion multiobjetivo.

Convencion (importante)
-----------------------
Todos los problemas del proyecto (mQAP) son de MINIMIZACION, luego la
direccion de preferencia es "menor es mejor". La mayoria de la literatura de
Falcon-Cardona (RSEHH / RSEIterative) trabaja con MAXIMIZACION, y `pymoo`
implementa sus indicadores en esa convencion. Aqui se implementa la version de
minimizacion:

    IGD(A, P)     = (1/|P|) * sum_{p in P} min_{a in A} ||p - a||_2
    IGD+(A, P)    = (1/|P|) * sum_{p in P} min_{a in A} ||max(0, a - p)||_2

`A` es el frente de aproximacion (la poblacion del individual's) y `P` el
conjunto de referencia. El minimo se toma SIEMPRE sobre `A`, de modo que un
individuo con mas soluciones (mejor cobertura) obtiene un valor menor: es
decir, IGD e IGD+ son "menor es mejor", como debe ser un indicador de
distancia al frente de referencia dentro de la fase de generacion.

La direccion de IGD+ es la clave: `max(0, a - p)` penaliza solo cuando el
frente de aproximacion es PEOR que el punto de referencia. Si `a` domina a `p`
(mejor o igual en todos los objetivos) la contribucion es 0, que es la
propiedad de preferencia debil que define a IGD+. IGD, en cambio, es simetrico
y penaliza tambien cuando el frente es mejor que la referencia.
"""

import numpy as np

__all__ = ["igd", "igd_plus"]


def _check(front, reference):
    if front is None or reference is None:
        raise ValueError("front y reference no pueden ser None")

    front = np.atleast_2d(np.asarray(front, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))

    if front.size == 0:
        raise ValueError("front vacio")
    if reference.size == 0:
        raise ValueError("reference vacio")
    if front.ndim != 2 or reference.ndim != 2:
        raise ValueError("front y reference deben ser matrices (n, m)")
    if front.shape[1] != reference.shape[1]:
        raise ValueError(
            f"numero de objetivos distinto: front={front.shape[1]}, "
            f"reference={reference.shape[1]}"
        )
    if not np.all(np.isfinite(front)) or not np.all(np.isfinite(reference)):
        raise ValueError("front y reference contienen valores no finitos")

    return front, reference


def _pairwise(front, reference):
    """Distancias de cada punto de referencia a cada punto del frente.

    Devuelve una matriz de forma (|reference|, |front|).
    """
    # diff[i, j, k] = reference[i, k] - front[j, k]
    return reference[:, None, :] - front[None, :, :]


def igd(front, reference, norm=None):
    """
    Inverted Generational Distance (minimizacion).

    @param front: (n, m) frente de aproximacion.
    @param reference: (r, m) conjunto de referencia.
    @param norm: (m,) factor de normalizacion por objetivo; si es None se usa 1.
    @return: escalar >= 0. Menor es mejor.
    """
    front, reference = _check(front, reference)

    diff = _pairwise(front, reference)
    if norm is not None:
        diff = diff / np.asarray(norm, dtype=float)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    return float(np.mean(np.min(dist, axis=1)))


def igd_plus(front, reference, norm=None):
    """
    IGD+ weakly preference-based (minimizacion).

    Solo penaliza el error en el sentido de la preferencia: si `front` tiene
    objetivos estrictamente peores que `p` en alguna dimension, la distancia
    crece; si los tiene mejores o iguales, no contribuye.

    @param front: (n, m) frente de aproximacion.
    @param reference: (r, m) conjunto de referencia.
    @param norm: (m,) factor de normalizacion por objetivo.
    @return: escalar >= 0. Menor es mejor.
    """
    front, reference = _check(front, reference)

    # max(0, a - p): penaliza solo cuando el frente es peor que la referencia.
    diff = _pairwise(front, reference) * -1.0
    diff = np.maximum(diff, 0.0)
    if norm is not None:
        diff = diff / np.asarray(norm, dtype=float)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    return float(np.mean(np.min(dist, axis=1)))
