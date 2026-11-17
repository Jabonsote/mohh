"""
Los indicadores de distancia usan la implementacion OFICIAL de pymoo.

`mohh.core.indicators.igd` delega en `pymoo.indicators.igd.IGD` y
`pymoo.indicators.igd_plus.IGDPlus` (a su vez sobre `moocore`). Este test fija
dos propiedades:

1. los valores coinciden con la definicion literal (la version numpy que el
   proyecto usaba antes), asi que cambiar de implementacion no mueve ningun
   numero del analisis;
2. se mantiene el contrato de la casa: validacion de entradas, `norm` por
   objetivo y convencion "menor es mejor".

HV no se comprueba aqui: ya venia de `pymoo.indicators.hv.HV`.
"""

import numpy as np
import pytest

from mohh.core.indicators.igd import igd, igd_plus


def _igd_referencia(front, reference, norm=None):
    """Definicion literal de IGD (minimizacion), sin pymoo."""
    diff = reference[:, None, :] - front[None, :, :]
    if norm is not None:
        diff = diff / np.asarray(norm, dtype=float)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(np.min(dist, axis=1)))


def _igd_plus_referencia(front, reference, norm=None):
    """Definicion literal de IGD+ (minimizacion), sin pymoo."""
    # max(0, a - p): solo penaliza cuando el frente es PEOR que la referencia.
    diff = front[None, :, :] - reference[:, None, :]
    diff = np.maximum(diff, 0.0)
    if norm is not None:
        diff = diff / np.asarray(norm, dtype=float)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(np.min(dist, axis=1)))


@pytest.mark.parametrize("semilla", range(5))
@pytest.mark.parametrize("n_objetivos", [2, 3, 5])
def test_igd_coincide_con_la_definicion(semilla, n_objetivos):
    rng = np.random.default_rng(semilla)
    referencia = rng.random((40, n_objetivos)) * 100.0
    frente = rng.random((25, n_objetivos)) * 100.0

    assert igd(frente, referencia) == pytest.approx(
        _igd_referencia(frente, referencia), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("semilla", range(5))
@pytest.mark.parametrize("n_objetivos", [2, 3, 5])
def test_igd_plus_coincide_con_la_definicion(semilla, n_objetivos):
    rng = np.random.default_rng(100 + semilla)
    referencia = rng.random((40, n_objetivos)) * 100.0
    frente = rng.random((25, n_objetivos)) * 100.0

    assert igd_plus(frente, referencia) == pytest.approx(
        _igd_plus_referencia(frente, referencia), rel=1e-12, abs=1e-12)


def test_igd_plus_no_penaliza_cuando_el_frente_es_mejor():
    """Propiedad de preferencia debil: si `a` domina a `p`, la distancia es 0."""
    referencia = np.array([[5.0, 5.0]])
    frente = np.array([[1.0, 1.0], [2.0, 3.0]])

    assert igd_plus(frente, referencia) == pytest.approx(0.0, abs=1e-12)


def test_igd_si_penaliza_aunque_el_frente_sea_mejor():
    """IGD es simetrico: un frente mejor que la referencia tambien puntua."""
    referencia = np.array([[5.0, 5.0]])
    frente = np.array([[1.0, 1.0]])

    assert igd(frente, referencia) > 0.0


def test_norm_escala_por_objetivo():
    """`norm` divide las distancias por objetivo, igual que la version previa."""
    rng = np.random.default_rng(7)
    referencia = rng.random((10, 3)) * 10.0
    frente = rng.random((8, 3)) * 10.0
    factor = np.array([2.0, 4.0, 5.0])

    assert igd(frente, referencia, norm=factor) == pytest.approx(
        _igd_referencia(frente, referencia, norm=factor), rel=1e-12)
    assert igd_plus(frente, referencia, norm=factor) == pytest.approx(
        _igd_plus_referencia(frente, referencia, norm=factor), rel=1e-12)

    with pytest.raises(ValueError):
        igd(frente, referencia, norm=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        igd(frente, referencia, norm=np.zeros(3))


def test_referencia_de_un_solo_punto():
    """Un P* de un punto sigue definido (aunque su caja de escala colapse)."""
    referencia = np.array([[3.0, 4.0]])
    # Los dos puntos del frente estan PEOR que la referencia, asi que la
    # distancia minima es estrictamente positiva.
    frente = np.array([[4.0, 5.0], [6.0, 8.0]])

    esperado = _igd_plus_referencia(frente, referencia)
    assert esperado > 0.0
    assert igd_plus(frente, referencia) == pytest.approx(esperado, rel=1e-12)

    # Si el frente contiene un punto que domina al de referencia, IGD+ es 0
    # (propiedad de preferencia debil) y sigue estando bien definido.
    frente_mejor = np.array([[0.0, 0.0], [6.0, 8.0]])
    assert igd_plus(frente_mejor, referencia) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("mala", [
    None,
    np.empty((0, 2)),
    np.array([[np.nan, 1.0]]),
    np.array([[np.inf, 1.0]]),
])
def test_entradas_invalidas(mala):
    buena = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        igd(mala, buena)
    with pytest.raises(ValueError):
        igd(buena, mala)


def test_numero_de_objetivos_distinto():
    with pytest.raises(ValueError):
        igd(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0, 3.0]]))


def test_compute_indicators_sigue_funcionando():
    """De punta a punta: frente + P* -> los cuatro indicadores."""
    from mohh.analysis.indicators_all import compute_indicators

    pstar = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0]])
    frente = np.array([[1.2, 1.1], [2.1, 2.0]])
    referencias = {"inst": None}

    # ReferenceSet real, no un None: se construye desde el propio P*.
    from mohh.core.indicators.reference_sets import ReferenceSet
    ideal = pstar.min(axis=0)
    span = np.where(pstar.max(axis=0) - ideal > 0,
                    pstar.max(axis=0) - ideal, 1.0)
    referencias["inst"] = ReferenceSet(
        name="inst", front=pstar, ideal=ideal,
        nadir=pstar.max(axis=0) + 0.1 * span, span=span)

    medidas = compute_indicators({"inst": frente}, references=referencias,
                                 metrics=("HV", "IGD+", "DELTA_P", "R2"),
                                 require_references=True)

    fila = medidas.loc["inst"]
    assert np.isfinite(fila["HV"]) and fila["HV"] > 0.0
    assert np.isfinite(fila["IGD+"]) and fila["IGD+"] >= 0.0
    assert np.isfinite(fila["DELTA_P"]) and 0.0 <= fila["DELTA_P"] <= 1.0
    assert np.isfinite(fila["R2"]) and fila["R2"] >= 0.0
