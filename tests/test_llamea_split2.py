"""Tests del reparto de modulos: `run.py` y `reporting.py` siguen siendo la API."""

from mohh.llamea import observability, report_io, reporting, run, setup
from test_llamea import (
    test_llamea_kwargs_enable_diversity_mechanisms,
    test_update_latest_points_to_current_run,
)
from test_llamea_reporting import (
    test_bounds_roundtrip,
    test_prompt_log_uses_exact_request_and_reasoning,
    test_reference_panel_rejects_unknown_names,
    test_reference_panel_selects_subset,
    test_tokens_summary_from_generation_log,
    test_tokens_summary_none_without_token_fields,
    test_write_report_builds_csv_markdown_and_plot,
)

_SETUP_NAMES = (
    "_EvalSolution", "_merge_reference_scores", "_load_bounds", "_save_bounds",
    "_reference_panel", "_model_revision", "_make_evaluator", "_llamea_kwargs",
)
_OBSERVABILITY_NAMES = (
    "_Tee", "_setup_console_log", "_setup_logging", "_update_latest",
    "_rotate_request_log", "_start_stall_watchdog",
)
_IO_NAMES = (
    "_atomic_write_text", "_atomic_write_json", "_metadata", "_read_jsonl",
    "load_evolution", "tokens_summary", "_pair_turns", "_request_records",
)


def test_modulos_nuevos_importables():
    for module in (setup, observability, report_io):
        assert module.__name__.startswith("mohh.llamea.")
        assert module.__file__.endswith(".py")


def test_aliases_run_setup():
    for name in _SETUP_NAMES:
        assert getattr(run, name) is getattr(setup, name)


def test_aliases_run_observability():
    for name in _OBSERVABILITY_NAMES:
        assert getattr(run, name) is getattr(observability, name)


def test_aliases_reporting_report_io():
    for name in _IO_NAMES:
        assert getattr(reporting, name) is getattr(report_io, name)


def test_nucleo_permanece_en_los_modulos_originales():
    assert callable(run.run_llamea)
    assert callable(run.main)
    assert run.ResultsExperimentLogger is reporting.ResultsExperimentLogger
    assert run._atomic_write_json is report_io._atomic_write_json
    for name in ("write_prompt_log", "write_csv", "write_markdown", "write_plot",
                 "write_generation_csv", "write_seeds", "write_report"):
        assert callable(getattr(reporting, name))
