"""
DeltaP (Dp) para problemas de minimizacion multiobjetivo.

Referencia
----------
Ishibuchi et al. (2006), "Performance Assessment of Multiobjective Optimizers
Based on Fairness Measures and an Application to Benchmark Problems", IEEE
SMC. Es la componente de la familia R-metric (Ishigami et al., 2011) que mide
cuanto se aproxima el frente de aproximacion al conjunto de referencia en
sentido *ortogonal*.

Definicion (geometria ortogonal)
--------------------------------
Para un punto de referencia p y un nadir n (n >= p componente a componente),
la region dominada por p es la caja

    B_p = { y : p <= y <= n },        volumen  V_box = prod_j (n_j - p_j)

`Dp` es la fraccion de `B_p` que NO queda cubierta por la region ortogonal
definida por el frente de aproximacion A. Dicha region es la escalera
axis-aligned construida sobre los puntos no dominados de A. La escalera arranca
en el NADIR (la "esquina buena" en minimizacion) y baja hacia p:

    se ordenan los puntos no dominados a^(1), a^(2), ... por el objetivo 1 de
    forma DESCENDENTE (del nadir hacia p) y se define a^(0)_1 = n_1
    el tramo i ocupa  max(a^(i)_1, p_1) <= y_1 <= min(a^(i-1)_1, n_1)
    y en el resto de objetivos   max(a^(i)_j, p_j) <= y_j <= a^(i)_j

    V_orth = sum_i  ancho_i * prod_{j >= 2} (a^(i)_j - p_j)

    Dp(A, P) = (1/|P|) * sum_{p in P} (1 - V_orth(A, p) / V_box(p))

En dos objetivos la formula es exactamente el volumen ortogonal publicado por
Ishibuchi et al., espejado a minimizacion, lo que se verifica en
`tests/test_indicators.py`. La extension a m objetivos mantiene la
descomposicion en tramos disjuntos a lo largo del primer objetivo, por lo que
el calculo es EXACTO (no es muestreo).

Interpretacion
--------------
* Dp = 0: la region ortogonal cubre toda la caja; el frente alcanza el
  reference point en todas las direcciones.
* Dp -> 1: el frente no cubre nada de la caja.
* Menor es mejor.

Convencion: minimizacion (problemas mQAP del proyecto).
"""

import numpy as np

__all__ = ["delta_p", "non_dominated"]


def non_dominated(front):
    """
    Filtro de no dominados para minimizacion.

    Un punto a es eliminado si existe b con b <= a y b != a.

    @param front: (n, m)
    @return: (k, m) con k <= n, conservando el orden de entrada.
    """
    front = np.atleast_2d(np.asarray(front, dtype=float))
    if front.size == 0:
        return front

    n = front.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        others = front[keep]
        # b domina a a si b <= a y al menos una componente es estricta.
        le = np.all(others <= front[i], axis=1)
        lt = np.any(others < front[i], axis=1)
        if np.any(le & lt):
            keep[i] = False

    return front[keep]


def delta_p(front, reference, nadir=None):
    """
    DeltaP (Dp) para minimizacion.

    @param front: (n, m) frente de aproximacion del individual's.
    @param reference: (r, m) conjunto de referencia P_i*.
    @param nadir: (m,) nadir de la region de interes; None => maximo de
        `reference` componente a componente, que es >= cada punto de
        referencia y por tanto da una region bien definida.
    @return: escalar en [0, 1]. Menor es mejor.
    """
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
        raise ValueError("front o reference contienen valores no finitos")

    if nadir is None:
        nadir = np.max(reference, axis=0)
    else:
        nadir = np.asarray(nadir, dtype=float).reshape(-1)
        if nadir.shape[0] != front.shape[1]:
            raise ValueError(
                f"nadir con {nadir.shape[0]} componentes, se esperaban "
                f"{front.shape[1]}"
            )
        if np.any(reference > nadir):
            raise ValueError("hay puntos de referencia por encima del nadir")

    m = front.shape[1]

    # Un punto de referencia que ya toca el nadir en alguna dimension genera
    # una caja de volumen cero: no hay region que cubrir, luego no aporta a la
    # media y se descarta. Para que la region sea informativa, el llamante
    # debe pasar un nadir que domine estrictamente a la referencia.
    spans = nadir[None, :] - reference
    non_degenerate = np.all(spans > 0, axis=1)
    if not np.any(non_degenerate):
        return 0.0
    reference = reference[non_degenerate]
    spans = spans[non_degenerate]

    # Solo son tiles los puntos dentro de la region y no dominados: un punto
    # dominado no puede aportar a la escalera y solo ensuciaria los anchos.
    inside = front[np.all(front <= nadir, axis=1)]
    active = non_dominated(inside)

    if active.shape[0] == 0:
        return 1.0

    # La escalera se construye desde el nadir (la esquina buena en
    # minimizacion) hacia el punto de referencia: orden descendente por el
    # primer objetivo.
    order = np.argsort(-active[:, 0], kind="stable")
    active = active[order]

    values = np.empty(reference.shape[0], dtype=float)

    for idx, p in enumerate(reference):
        # Volumen de la caja B_p.
        box = float(np.prod(spans[idx]))

        # Anchos de los tramos, recortados a [p_1, n_1]. El primer tramo
        # arranca en el nadir.
        upper = np.concatenate(([nadir[0]], active[:-1, 0]))
        lower = active[:, 0]
        widths = np.maximum(
            np.minimum(upper, nadir[0]) - np.maximum(lower, p[0]), 0.0
        )

        # Secciones: producto de (a_j - p_j) para j >= 2, recortado a 0.
        tail = active[:, 1:]
        if m > 2:
            sections = np.prod(np.maximum(tail - p[1:], 0.0), axis=1)
        elif m == 2:
            sections = np.maximum(tail[:, 0] - p[1], 0.0)
        else:
            sections = np.ones(active.shape[0], dtype=float)

        orth = float(np.sum(widths * sections))

        values[idx] = 1.0 - min(orth / box, 1.0)

    return float(np.mean(values))
