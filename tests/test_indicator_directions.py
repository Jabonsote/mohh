"""
Direccion unica de cada indicador: que se MAXIMIZA y que se MINIMIZAN.

El registro es `mohh.core.indicators.directions`. Estos tests impiden que las
copias locales (`sdd`, `indicators_all`) y el codigo que rankea (`stats`,
`analyze_search`) se desincronicen: si alguien anade un indicador de
maximizacion y olvida `stats`, el Wilcoxon y los rangos salen al reves y el
test revienta.
"""

import numpy as np
import pandas as pd
import pytest

from mohh.core.indicators.directions import (
    DIRECTIONS,
    MAXIMIZATION_METRICS,
    MINIMIZATION_METRICS,
    aggregation_for_metric,
    axis_hint,
    direction_label,
    indicator_minimization,
)


def test_registro_de_direcciones():
    assert MAXIMIZATION_METRICS == {"HV"}
    assert MINIMIZATION_METRICS == {"IGD+", "DELTA_P", "R2"}
    assert set(DIRECTIONS) == {"HV", "IGD+", "DELTA_P", "R2"}
    assert DIRECTIONS["HV"] == "maximizar"
    assert all(DIRECTIONS[m] == "minimizar"
               for m in ("IGD+", "DELTA_P", "R2"))


def test_helpers_de_direccion():
    assert indicator_minimization("HV") is False
    assert indicator_minimization("IGD+") is True
    assert indicator_minimization("DELTA_P") is True
    assert indicator_minimization("R2") is True

    assert direction_label("HV") == "maximizar"
    assert direction_label("R2") == "minimizar"

    assert axis_hint("HV") == "(mayor es mejor)"
    assert axis_hint("IGD+") == "(menor es mejor)"

    assert aggregation_for_metric("HV") == "max"
    assert aggregation_for_metric("IGD+") == "min"
    assert aggregation_for_metric("DELTA_P") == "min"
    assert aggregation_for_metric("R2") == "min"


def test_sdd_comparte_el_registro():
    """`sdd` no importa `mohh` a nivel de modulo, pero no puede desviarse."""
    from mohh.core.utilities.algorithm.sdd import SDD, _MINIMIZATION_METRICS

    assert set(_MINIMIZATION_METRICS) == set(MINIMIZATION_METRICS)
    assert SDD(metric_type="HV").is_minimization is False
    assert SDD(metric_type="IGD+").is_minimization is True


def test_indicators_all_deriva_del_registro():
    from mohh.analysis.indicators_all import (
        INDICATORS,
        _MINIMIZATION,
        indicator_minimization as helper,
    )

    assert set(_MINIMIZATION) == set(MINIMIZATION_METRICS)
    assert [m for m in INDICATORS if helper(m)] == list(_MINIMIZATION)
    for metrica in INDICATORS:
        assert helper(metrica) == indicator_minimization(metrica)


def _marco(valores_a, valores_b, etiquetas=("A", "B")):
    """DataFrame instancia x configuracion."""
    return pd.DataFrame({etiquetas[0]: valores_a, etiquetas[1]: valores_b})


def test_rangos_aplican_la_direccion():
    from mohh.analysis.stats import average_ranks

    # A siempre MEJOR en HV (mas alto) y PEOR en IGD+ (mas alto).
    alto = np.arange(1.0, 11.0)
    bajo = np.arange(0.5, 10.5, 1.0)
    marco = _marco(alto, bajo)

    rangos_hv = average_ranks(marco, metric="HV")
    assert (rangos_hv["A"] == 1).all(), "HV se maximiza: el mayor es el 1"

    rangos_igd = average_ranks(marco, metric="IGD+")
    assert (rangos_igd["B"] == 1).all(), "IGD+ se minimiza: el menor es el 1"


def test_wilcoxon_no_invierte_la_direccion():
    from mohh.analysis.stats import wilcoxon_vs_reference

    # A con IGD+ claramente MENOR que B (mejor), 12 instancias.
    rng = np.random.default_rng(0)
    peor = np.full(12, 5.0) + rng.normal(0, 0.05, 12)
    mejor = np.full(12, 1.0) + rng.normal(0, 0.05, 12)
    marco = _marco(mejor, peor)

    # Referencia = B (la peor): A tiene que salir como mejor.
    tabla = wilcoxon_vs_reference(marco, reference="B", metric="IGD+")
    fila = tabla.set_index("config").loc["A"]
    assert fila["mean_diff"] < 0.0
    assert bool(fila["better"]) is True

    # HV: A con valores MAS ALTOS que B (mejor). Sin negar saldria al reves.
    marco_hv = _marco(peor + 10.0, peor)
    tabla_hv = wilcoxon_vs_reference(marco_hv, reference="B", metric="HV")
    fila_hv = tabla_hv.set_index("config").loc["A"]
    assert fila_hv["mean_diff"] < 0.0
    assert bool(fila_hv["better"]) is True


def test_mejor_por_generacion_usa_la_direccion():
    """analyze_search toma max para HV y min para IGD+/DELTA_P/R2."""
    from mohh.postanalysis import analyze_search

    filas = []
    for generacion in (0, 1):
        filas.append({"instance": "i1", "generation": generacion,
                      "metric": "HV", "value": 0.1 * generacion})
        filas.append({"instance": "i1", "generation": generacion,
                      "metric": "IGD+", "value": 1.0 - 0.1 * generacion})
    largo = pd.DataFrame(filas)

    mejores = analyze_search.mejor_por_generacion(largo)
    por_metrica = mejores.set_index(["metric", "generation"])["mejor"]

    # HV sube de 0.0 a 0.1: el mejor de la gen 1 es el mayor.
    assert por_metrica[("HV", 0)] == pytest.approx(0.0)
    assert por_metrica[("HV", 1)] == pytest.approx(0.1)
    # IGD+ baja de 1.0 a 0.9: el mejor de la gen 1 es el menor.
    assert por_metrica[("IGD+", 0)] == pytest.approx(1.0)
    assert por_metrica[("IGD+", 1)] == pytest.approx(0.9)
