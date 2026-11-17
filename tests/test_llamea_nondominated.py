"""El filtro no dominado rapido debe ser EXACTAMENTE el de la referencia O(n^2).

La fase "calculando HV..." de `scripts/compare_operators.py` consolida ~28 000
puntos por instancia y paga esa operacion ~120 veces; con el filtro O(n^2) (la
matriz `n x n x k` de booleanos) tardaba horas. La version rapida (barrido
tras ordenar) devuelve el mismo conjunto y el mismo orden, y estos tests lo
comprueban contra la implementacion de referencia.
"""
import numpy as np
import pytest

from mohh.core.utilities.algorithm.MO import sign
from mohh.llamea.fitness import _nondominated


def _nondominated_referencia(points):
    """Implementacion O(n^2) de referencia (la que habia antes del barrido)."""
    array = np.asarray(points, dtype=float)
    if len(array) <= 1:
        return array
    adjusted = array * sign
    dominates = (
        np.all(adjusted[:, None, :] >= adjusted[None, :, :], axis=2)
        & np.any(adjusted[:, None, :] > adjusted[None, :, :], axis=2)
    )
    np.fill_diagonal(dominates, False)
    return array[~dominates.any(axis=0)]


def _filas(front):
    return {tuple(np.round(row, 12)) for row in front}


@pytest.mark.parametrize("n,k", [(50, 3), (200, 3), (200, 4), (500, 3),
                                 (1000, 2), (300, 3), (5, 3)])
def test_mismo_conjunto_que_la_referencia(n, k):
    rng = np.random.default_rng(0)
    points = rng.normal(0, 1, size=(n, k))
    assert _filas(_nondominated(points)) == _filas(_nondominated_referencia(points))


def test_mismo_orden_que_la_referencia():
    """El marco consolidado se compara byte a byte: tambien el orden."""
    rng = np.random.default_rng(7)
    points = rng.normal(0, 1, size=(400, 3))
    assert np.array_equal(_nondominated(points), _nondominated_referencia(points))


def test_duplicados_no_se_eliminan():
    """`comprobate_frame` espera que los repetidos sobrevivan al filtro."""
    points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    front = _nondominated(points)
    assert len(front) == 2
    assert _filas(front) == _filas(_nondominated_referencia(points))


def test_casos_degenerate():
    assert len(_nondominated(np.array([[1.0, 2.0]]))) == 1
    assert len(_nondominated(np.empty((0, 3)))) == 0
    # un frente donde todo es igual: no se dominan entre si
    points = np.ones((5, 3))
    assert len(_nondominated(points)) == 5


def test_escala():
    """Con el tamano real de una instancia consolidada, el barrido es viable."""
    rng = np.random.default_rng(3)
    points = rng.normal(0, 1, size=(28000, 3))
    front = _nondominated(points)
    reference = _nondominated_referencia(points)
    assert _filas(front) == _filas(reference)
