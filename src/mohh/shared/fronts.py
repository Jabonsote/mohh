"""Frentes no dominados, normalizacion y recorte al nadir.

Utilidades comunes de la evaluacion GE y de LLaMEA. Reexportadas desde
`mohh.llamea.fitness` para mantener la firma publica existente.
"""

import numpy as np

from mohh.constants import NADIR

#: Punto de nadir fijo sobre frentes normalizados.
NADIR_FACTOR = NADIR


def normalize_front(front, bounds):
    """Normaliza un frente con limites (vmin, vmax) por objetivo.

    Formula ``(f - vmin) / (vmax - vmin)``. Un objetivo degenerado
    (``vmax == vmin``) da NaN: quien llama descarta esas instancias antes.
    """
    vmin = np.asarray(bounds[0], dtype=float)
    vmax = np.asarray(bounds[1], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(front, dtype=float) - vmin) / (vmax - vmin)


def clip_to_reference_box(front, nadir=NADIR_FACTOR):
    """Recorta objetivos a la caja [0, nadir] antes de calcular HV.

    Evita que coordenadas negativas de un candidato mejor que el panel
    inflen el HV por encima de ``nadir**K``. Puntos peores que el nadir
    aportan 0.
    """
    return np.clip(np.asarray(front, dtype=float), 0.0, float(nadir))


def _consolidated_frame(*point_sets):
    """Frente no dominado de la union de varios conjuntos de puntos.

    Envolvente no dominada de todo lo evaluado en una instancia. Los puntos
    repetidos no cuentan. Devuelve `None` si no hay puntos.
    """
    from mohh.core.utilities.algorithm.MO import non_dominated_sorting_vectorized

    points = [np.asarray(points, dtype=float) for points in point_sets
              if points is not None and len(points)]
    if not points:
        return None
    stacked = np.vstack(points)
    stacked = np.unique(stacked, axis=0)
    front = np.asarray(non_dominated_sorting_vectorized(stacked)[0], dtype=float)
    return front


def _nondominated(points):
    """Filtra los puntos no dominados (misma regla que `MO.py`).

    Barrido tras ordenar lexicograficamente: `O(n log n)`. Conserva el
    conjunto y el orden de la version de broadcasting.
    """
    from mohh.core.utilities.algorithm.MO import sign

    array = np.asarray(points, dtype=float)
    if len(array) <= 1:
        return array
    adjusted = array * sign
    k = adjusted.shape[1]
    # En `adjusted` el "mejor" es el MAYOR (`sign` = -1 para minimizacion).
    order = np.lexsort([adjusted[:, j] for j in range(k - 1, -1, -1)])
    sorted_p = adjusted[order]
    front = []
    for i in range(len(sorted_p)):
        p = sorted_p[i]
        if front:
            f = sorted_p[front]
            if np.any(np.all(f >= p, axis=1) & np.any(f > p, axis=1)):
                continue                      # p esta dominado por el frente
            kill = np.all(p >= f, axis=1) & np.any(p > f, axis=1)
            if np.any(kill):
                front = [idx for idx, dead in zip(front, kill) if not dead]
        front.append(i)
    mask = np.zeros(len(array), dtype=bool)
    mask[order[front]] = True
    return array[mask]


def comprobate_frame(points):
    """`comprobate` de `experiment.py`: peela el frente mientras tenga <= 2 puntos.

    Evita calcular el nadir desde un frente que domina a todos los demas
    puntos. Se detiene cuando no queda nada fuera del frente actual.
    """
    if points is None:
        return None
    array = np.asarray(points, dtype=float)
    if len(array) == 0:
        return array
    remaining = {tuple(point) for point in array}
    front = _nondominated(array)
    while len(front) <= 2 and len(remaining) > len(front):
        for point in front:
            remaining.discard(tuple(point))
        if not remaining:
            break
        front = _nondominated(np.asarray(sorted(remaining), dtype=float))
    return front
