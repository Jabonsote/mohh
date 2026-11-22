"""Contratos del split de `fitness.py` en `compilation.py` y `archive.py`.

Los casos de compilacion y validacion se importan de `test_llamea.py` para no
duplicarlos: aqui solo se comprueba que los alias siguen apuntando al mismo
objeto que las implementaciones movidas.
"""

import importlib

from mohh.llamea import archive, compilation, fitness

# Suite existente, reutilizada sin copiar su cuerpo.
from test_llamea import (
    test_compile_operator_accepts_valid_permutation_operator,
    test_compile_operator_rejects_non_permutation,
    test_validation_feedback_points_at_unfilled_slots,
)

NOMBRES_COMPILACION = (
    "_set_metadata", "EarlyStop", "COMPILE_TIMEOUT", "OPERATOR_FILENAME",
    "_describe_child_problem", "_exception_feedback", "validate_operator",
    "DEGENERACY_TRIALS", "recombination_novelty", "primitive_namespace",
    "_compile_with_source", "validate_operator_minimal", "_UNSEEDED_RNG_RE",
    "uses_unseeded_rng", "compile_operator", "_cached_operator",
)

NOMBRES_ARCHIVE = ("HVArchive", "_nd_merge", "_same_points")


def test_modulos_importables():
    for modulo in (compilation, archive, fitness):
        assert importlib.import_module(modulo.__name__) is modulo


def test_aliases_de_compilacion_son_el_mismo_objeto():
    for nombre in NOMBRES_COMPILACION:
        assert getattr(fitness, nombre) is getattr(compilation, nombre), nombre


def test_aliases_de_archive_son_el_mismo_objeto():
    for nombre in NOMBRES_ARCHIVE:
        assert getattr(fitness, nombre) is getattr(archive, nombre), nombre


def test_solve_front_task_sigue_en_fitness():
    assert "_solve_front_task" not in compilation.__dict__
    assert "_solve_front_task" not in archive.__dict__
    assert callable(fitness._solve_front_task)
