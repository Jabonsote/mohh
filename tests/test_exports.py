"""Los exports publicos de los paquetes principales no estan rotos.

Un nombre en ``__all__`` que no es atributo del paquete rompe
``from <paquete> import *`` con ``AttributeError``. Estos tests lo detectan
para los tres paquetes que se exportan de forma agregada.
"""

import importlib

import pytest

PAQUETES = ("mohh.llamea", "mohh.analysis", "mohh.core.indicators")


@pytest.mark.parametrize("paquete", PAQUETES)
def test_star_import_no_falla(paquete):
    """``from <paquete> import *`` tiene que ejecutarse sin error."""
    exec(f"from {paquete} import *", {})


@pytest.mark.parametrize("paquete", PAQUETES)
def test_todo_en_all_es_atributo(paquete):
    """Cada nombre de ``__all__`` debe ser atributo del modulo."""
    modulo = importlib.import_module(paquete)
    # El star import es lo que materializa los submodulos listados en __all__.
    exec(f"from {paquete} import *", {})
    for nombre in modulo.__all__:
        assert hasattr(modulo, nombre), (
            f"{paquete}.__all__ incluye {nombre!r} pero no es atributo"
        )


def test_analysis_expone_modulos_de_analisis():
    from mohh import analysis

    for nombre in ("code_diversity", "evaluation_readers", "hv_scale"):
        assert nombre in analysis.__all__


def test_indicators_expone_reference_sets():
    from mohh.core import indicators

    for nombre in (
        "NADIR_MARGIN",
        "ReferenceSet",
        "ReferenceSetError",
        "load_reference_sets",
        "pooled_reference_sets",
        "read_pof",
    ):
        assert nombre in indicators.__all__
