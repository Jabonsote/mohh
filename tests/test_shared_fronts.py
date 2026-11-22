"""Contrato de los frentes compartidos por GE y LLaMEA."""
import numpy as np
import pytest

from mohh.core.utilities.algorithm.MO import sign
from mohh.shared.fronts import (
    NADIR_FACTOR,
    _nondominated,
    clip_to_reference_box,
    comprobate_frame,
    normalize_front,
)


def _nondominated_referencia(points):
    """Filtro O(n^2) de referencia (matriz de dominancia)."""
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


def test_normalize_formula():
    front = np.array([[1.0, 4.0], [3.0, 0.0]])
    bounds = (np.array([1.0, 0.0]), np.array([5.0, 8.0]))
    expected = np.array([[0.0, 0.5], [0.5, 0.0]])
    np.testing.assert_allclose(normalize_front(front, bounds), expected)
    assert NADIR_FACTOR == pytest.approx(1.1)


def test_clip_bounds():
    front = np.array([[-2.0, 0.5], [0.4, 3.0]])
    expected = np.array([[0.0, 0.5], [0.4, 1.1]])
    np.testing.assert_allclose(clip_to_reference_box(front), expected)


def test_comprobate_peels_small_fronts():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    np.testing.assert_allclose(comprobate_frame(points), [[2.0, 2.0]])
    assert comprobate_frame(None) is None


@pytest.mark.parametrize("seed,n,k", [(0, 40, 2), (1, 80, 3), (2, 120, 4)])
def test_nondominated_contra_referencia_aleatoria(seed, n, k):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, k))
    np.testing.assert_array_equal(
        _nondominated(points), _nondominated_referencia(points)
    )


def test_llamea_fitness_reexporta_fronts():
    from mohh.llamea import fitness
    from mohh.shared import fronts

    names = ("normalize_front", "clip_to_reference_box", "_consolidated_frame",
             "_nondominated", "comprobate_frame", "NADIR_FACTOR")
    for name in names:
        assert getattr(fitness, name) is getattr(fronts, name)
