"""Regresiones de la escala de HV compartida del analisis por generacion.

El HV solo es comparable entre generaciones y entre experimentos si todos se
miden con los mismos `vmin/vmax` y el mismo `ref_point`. Estos tests fijan esa
propiedad: detectar instancias degeneradas, restringir una escala sin mover sus
limites y rechazar escalas que difieren aunque compartan `source` y `ref_point`.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

from mohh.analysis.hv_scale import (
    HvScale,
    HvScaleError,
    diferencias_escala,
    instancias_sin_escala,
    load_hv_scale,
    restringir_escala,
    save_hv_scale,
)

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SPEC = importlib.util.spec_from_file_location(
    "generation_comparison",
    os.path.join(ROOT, "results", "analysis_scripts", "generation_comparison.py"),
)
generation_comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation_comparison)

SEARCH_SPEC = importlib.util.spec_from_file_location(
    "analyze_search", os.path.join(ROOT, "analyze_search.py")
)
analyze_search = importlib.util.module_from_spec(SEARCH_SPEC)
SEARCH_SPEC.loader.exec_module(analyze_search)


def _escala(bounds=None, ref_point=1.1, source="consolidado"):
    bounds = bounds or {
        "A": (np.array([0.0, 0.0]), np.array([10.0, 20.0])),
        "B": (np.array([1.0, 1.0]), np.array([2.0, 2.0])),
    }
    return HvScale(bounds, ref_point=ref_point, source=source)


def test_instancias_sin_escala_detecta_objetivo_colapsado():
    escala = _escala({
        "A": (np.array([0.0, 0.0]), np.array([10.0, 20.0])),
        "B": (np.array([1.0, 1.0]), np.array([2.0, 2.0])),
        "C": (np.array([1.0, 5.0]), np.array([1.0, 9.0])),  # objetivo 0 constante
    })
    assert instancias_sin_escala(escala) == ["C"]
    # Las dos instancias restantes siguen siendo puntuables.
    assert set(restringir_escala(escala, ["A", "B"]).bounds) == {"A", "B"}


def test_restringir_escala_no_cambia_los_limites():
    escala = _escala()
    reducida = restringir_escala(escala, ["A"])
    assert list(reducida.bounds) == ["A"]
    assert np.array_equal(reducida.bounds["A"][0], escala.bounds["A"][0])
    assert np.array_equal(reducida.bounds["A"][1], escala.bounds["A"][1])
    assert reducida.ref_point == escala.ref_point
    assert reducida.source == escala.source


def test_restringir_escala_exige_cobertura():
    escala = _escala()
    with pytest.raises(HvScaleError, match="no cubre"):
        restringir_escala(escala, ["A", "Z"])


def test_diferencias_escala_detecta_mismos_source_pero_limites_distintos():
    """El fallo real: 'consolidado' + 1.1 en los dos, pero 80/80 limites cambian."""
    left = _escala(source="consolidado")
    right = _escala({
        "A": (np.array([0.0, 0.0]), np.array([11.0, 20.0])),
        "B": (np.array([1.0, 1.0]), np.array([2.0, 2.0])),
    }, source="consolidado")
    assert diferencias_escala(left, right) == ["A"]


def test_diferencias_escala_ok_con_los_mismos_limites():
    assert diferencias_escala(_escala(), _escala()) == []


def test_diferencias_escala_detecta_instancia_ausente():
    left = _escala()
    right = _escala({"A": left.bounds["A"]})
    assert diferencias_escala(left, right) == ["B"]


def test_save_load_roundtrip(tmp_path):
    escala = _escala()
    path = save_hv_scale(escala, str(tmp_path / "hv_scale.json"))
    cargada = load_hv_scale(path)
    assert cargada.ref_point == escala.ref_point
    assert diferencias_escala(escala, cargada) == []


def test_generation_comparison_rechaza_escalas_distintas():
    """`generation_comparison` no debe comparar HV con escalas distintas."""
    left = _escala().to_dict()
    right = _escala({
        "A": (np.array([0.0, 0.0]), np.array([12.0, 20.0])),
        "B": (np.array([1.0, 1.0]), np.array([2.0, 2.0])),
    }).to_dict()
    assert generation_comparison._limites_distintos(
        {"exp1": left, "exp2": right}) == ["A"]
    assert generation_comparison._limites_distintos(
        {"exp1": left, "exp2": left}) == []


def test_preparar_escala_split_excluye_degeneradas():
    escala = _escala({
        "A": (np.array([0.0, 0.0]), np.array([10.0, 20.0])),
        "B": (np.array([1.0, 1.0]), np.array([1.0, 2.0])),  # colapsa en obj 0
    })
    util, degeneradas = analyze_search.preparar_escala_split(
        escala, ["A", "B"], "train", verbose=False)
    assert degeneradas == ["B"]
    assert list(util.bounds) == ["A"]


def test_preparar_escala_split_falla_si_no_cubre():
    escala = _escala({"A": _escala().bounds["A"]})
    with pytest.raises(analyze_search.AnalisisSearchError, match="no cubre"):
        analyze_search.preparar_escala_split(
            escala, ["A", "Z"], "test", verbose=False)


def test_preparar_escala_split_no_estricto_omite_sin_frente():
    escala = _escala({"A": _escala().bounds["A"]})
    util, degeneradas = analyze_search.preparar_escala_split(
        escala, ["A", "Z"], "train", verbose=False, strict=False)
    assert list(util.bounds) == ["A"]
    assert degeneradas == []


def test_combinar_escalas_disjuntas():
    train = _escala({"A": (np.array([0.0, 0.0]), np.array([1.0, 1.0]))})
    test = _escala({"T": (np.array([0.0, 0.0]), np.array([2.0, 2.0]))})
    combinada = analyze_search.combinar_escalas([train, test])
    assert set(combinada.bounds) == {"A", "T"}
    assert combinada.ref_point == 1.1


def test_combinar_escalas_rechaza_conflicto_en_instancia_comun():
    left = _escala({"A": (np.array([0.0, 0.0]), np.array([1.0, 1.0]))})
    right = _escala({"A": (np.array([0.0, 0.0]), np.array([2.0, 2.0]))})
    with pytest.raises(analyze_search.AnalisisSearchError, match="se contradicen"):
        analyze_search.combinar_escalas([left, right])


def test_combinar_escalas_rechaza_ref_point_distinto():
    left = _escala({"A": (np.array([0.0, 0.0]), np.array([1.0, 1.0]))})
    right = _escala({"T": (np.array([0.0, 0.0]), np.array([2.0, 2.0]))},
                    ref_point=1.2)
    with pytest.raises(analyze_search.AnalisisSearchError, match="ref_point"):
        analyze_search.combinar_escalas([left, right])


def test_escala_guardada_completa_se_puede_reutilizar(tmp_path):
    """Una escala recortada fallaria por cobertura; la completa no."""
    escala = _escala({
        "A": (np.array([0.0, 0.0]), np.array([10.0, 20.0])),
        "B": (np.array([1.0, 1.0]), np.array([1.0, 2.0])),  # degenerada
    })
    path = save_hv_scale(escala, str(tmp_path / "hv_scale.json"))
    cargada = load_hv_scale(path)
    util, degeneradas = analyze_search.preparar_escala_split(
        cargada, ["A", "B"], "train", verbose=False)
    assert degeneradas == ["B"]
    assert list(util.bounds) == ["A"]
