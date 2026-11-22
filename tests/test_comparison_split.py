"""Contratos del split de los scripts de comparacion.

Los tests de regresion se importan: el objetivo es comprobar que el traslado no
cambio el comportamiento, no duplicar los casos.
"""

import scripts.compare_ge_frame as ge_frame
import scripts.compare_operators as compare
from mohh.comparison import engine, writers
from mohh.shared import mo

# Reutilizacion literal de las regresiones del arnes original.
from test_compare_operators import (  # noqa: F401
    test_failed_front_counts_as_zero_or_is_excluded,
    test_ge_layout_matches_the_original_evaluation,
    test_group_contrasts_compares_two_groups,
    test_load_operator_reads_a_ge_operator,
    test_operator_catalog_keeps_the_project_prefix,
    test_parse_groups_and_contrasts,
    test_protocols_include_solver_and_mutation_variants,
    test_shared_frame_ignores_failed_front_but_uses_the_rest,
    test_solve_task_injects_the_project_crossover,
)

ENGINE_NAMES = (
    "protocol_name",
    "budget_label",
    "_log",
    "_fmt",
    "_p",
    "_ci",
    "_frame_from_points",
    "_safe_frame",
    "shared_frame",
    "hv_values",
    "per_run_scores",
    "_comparison_pairs",
    "pairwise",
    "pairwise_design",
    "_group_scores",
    "group_contrasts",
    "summarize",
    "rank_stability",
)
WRITER_NAMES = (
    "_fronts_payload",
    "save_fronts",
    "save_failures",
    "save_json",
    "write_operator_catalog",
    "write_run_metrics_summary",
    "write_ge_layout",
    "write_instance_artifacts",
    "_median_front",
    "write_csvs",
    "_csv_num",
    "write_report",
)


def test_scripts_exponen_el_motor_y_los_escritores():
    for name in ENGINE_NAMES:
        assert getattr(compare, name) is getattr(engine, name)
    for name in WRITER_NAMES:
        assert getattr(compare, name) is getattr(writers, name)
    assert compare.hv_values.__module__ == "mohh.comparison.engine"
    assert compare.write_csvs.__module__ == "mohh.comparison.writers"


def test_solve_front_snapshots_sigue_siendo_atributo_del_script():
    assert compare.solve_front_snapshots is mo.solve_front_snapshots


def test_solve_task_lee_solve_front_snapshots_del_namespace_del_script(
        monkeypatch):
    captured = {}

    def fake_snapshots(instance, namespace, **kwargs):
        captured["namespace"] = namespace
        captured["crossover"] = kwargs.get("crossover")
        return {}

    monkeypatch.setattr(compare, "solve_front_snapshots", fake_snapshots)
    key = ("MOEAD", "pmx", 0, "30k", 0)
    task = ([(key)], "PMX_Cross", "project", object(), 42, "MOEAD", False,
            0, {}, None)
    result = compare._solve_task(task)
    assert captured["namespace"] is None
    assert captured["crossover"] is not None
    assert result[0][0] == key


def test_ge_frame_reutiliza_el_formateo_y_conserva_su_marco():
    assert ge_frame._fmt is engine._fmt
    # Estas funciones difieren del motor comun (peeling de GE, sin clipping,
    # operadores como dict): se conservan en el script.
    for name in ("_ge_frame", "_shared_frames", "_hv_values",
                 "_per_run_scores", "_summarize", "_comparisons",
                 "_protocol_name", "_write_report"):
        assert callable(getattr(ge_frame, name))
