"""Contrato del codigo compartido: estadistica, instancias y operadores GE."""

from mohh.evaluation import ge_operators
from mohh.llamea import ge_operators as legacy_ge_operators
from mohh.llamea import statistics as legacy_statistics
from mohh.llamea.run import _select_instances
from mohh.shared import statistics
from mohh.shared.instances import select_instances

# Los tests equivalentes ya existen: se reutilizan en vez de duplicarlos.
from test_compare_operators import test_instance_selection_is_sorted_before_sampling  # noqa: F401
from test_llamea_statistics import test_add_holm_marks_significant_rows  # noqa: F401


def test_alias_de_compatibilidad_identidad():
    for name in (
        "finite_pairs",
        "wilcoxon_signed_rank",
        "paired_summary",
        "holm_adjust",
        "add_holm",
        "minimum_detectable_difference",
    ):
        assert getattr(legacy_statistics, name) is getattr(statistics, name)
    for name in (
        "OPERATORS_SUFFIX",
        "discover_operator_files",
        "load_ge_operators",
        "ge_callable",
        "ge_namespace",
    ):
        assert getattr(legacy_ge_operators, name) is getattr(ge_operators, name)
    assert _select_instances is select_instances


def test_select_instances_determinista_y_orden_estable():
    paths = ["/data/c.txt", "/data/a.txt", "/data/d.txt", "/data/b.txt"]
    expected = select_instances(paths, 3, 7)
    assert expected == select_instances(paths, 3, 7)
    assert expected == select_instances(list(reversed(paths)), 3, 7)
    assert select_instances(paths, "all", 7) == sorted(paths)
