"""
IGD e IGD+ para problemas de minimizacion multiobjetivo.

Implementacion oficial de `pymoo` (`pymoo.indicators.igd.IGD` y
`pymoo.indicators.igd_plus.IGDPlus`), que a su vez delegan el calculo en
`moocore`. Aqui solo se mantiene la validacion de entradas y la escala por
objetivo, para no cambiar el contrato que ya usan `ReferenceFitness` y
`compute_indicators`.

Convencion (importante)
-----------------------
Todos los problemas del proyecto (mQAP) son de MINIMIZACION, luego la
direccion de preferencia es "menor es mejor". `pymoo` devuelve el valor CRUDO
del indicador: no invierte el signo ni asume ninguna direccion, asi que la
lectura "menor es mejor" la impone este proyecto (ver
`mohh.core.indicators.directions`). Definiciones:

    IGD(A, P)     = (1/|P|) * sum_{p in P} min_{a in A} ||p - a||_2
    IGD+(A, P)    = (1/|P|) * sum_{p in P} min_{a in A} ||max(0, a - p)||_2

`A` es el frente de aproximacion (la poblacion del individuo) y `P` el
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

from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

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


def _aplicar_norm(front, reference, norm):
    """
    Escala ambos conjuntos por objetivo.

    Dividir `front` y `reference` por `norm` es equivalente a dividir las
    distancias por `norm`, que es lo que hacia la version previa, pero asi el
    calculo lo hace pymoo y no hay dos caminos distintos.
    """
    if norm is None:
        return front, reference

    factor = np.asarray(norm, dtype=float)
    if factor.shape != (front.shape[1],):
        raise ValueError(
            f"norm debe tener una entrada por objetivo: {factor.shape} vs "
            f"{front.shape[1]}"
        )
    if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
        raise ValueError("norm debe ser finito y estrictamente positivo")

    return front / factor, reference / factor


def igd(front, reference, norm=None):
    """
    Inverted Generational Distance (minimizacion), via `pymoo.indicators.igd`.

    @param front: (n, m) frente de aproximacion.
    @param reference: (r, m) conjunto de referencia.
    @param norm: (m,) factor de normalizacion por objetivo; si es None se usa 1.
    @return: escalar >= 0. Menor es mejor.
    """
    front, reference = _check(front, reference)
    front, reference = _aplicar_norm(front, reference, norm)

    return float(IGD(reference)(front))


def igd_plus(front, reference, norm=None):
    """
    IGD+ weakly preference-based (minimizacion), via
    `pymoo.indicators.igd_plus`.

    Solo penaliza el error en el sentido de la preferencia: si `front` tiene
    objetivos estrictamente peores que `p` en alguna dimension, la distancia
    crece; si los tiene mejores o iguales, no contribuye.

    @param front: (n, m) frente de aproximacion.
    @param reference: (r, m) conjunto de referencia.
    @param norm: (m,) factor de normalizacion por objetivo.
    @return: escalar >= 0. Menor es mejor.
    """
    front, reference = _check(front, reference)
    front, reference = _aplicar_norm(front, reference, norm)

    return float(IGDPlus(reference)(front))
