"""Regresiones del arnes de comparacion (fallos y protocolos)."""

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
SPEC = importlib.util.spec_from_file_location(
    "compare_operators", os.path.join(ROOT, "scripts", "compare_operators.py")
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def _frame():
    return np.array([[0.0, 0.0], [1.0, 1.0]])


def _operators():
    return [("a", None, "llamea", "code"), ("b", None, "classic", "ox")]


def test_failed_front_counts_as_zero_or_is_excluded():
    protocols = [("MOEAD", False)]
    fronts = {
        ("MOEAD", "a", 0, 0): np.empty((0, 0)),
        ("MOEAD", "b", 0, 0): np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    failures = {("MOEAD", "a", 0, 0): "RuntimeError: boom"}

    zero = compare.hv_values(fronts, _operators(), protocols, [_frame()], 1,
                             [None], failures=failures, fail_policy="zero")
    assert zero[("MOEAD", "a", 0)] == [0.0]
    assert zero[("MOEAD", "b", 0)][0] > 0.0

    excluded = compare.hv_values(fronts, _operators(), protocols, [_frame()], 1,
                                 [None], failures=failures, fail_policy="exclude")
    assert excluded[("MOEAD", "a", 0)] == [None]

    with pytest.raises(RuntimeError, match="fallo en MOEAD a"):
        compare.hv_values(fronts, _operators(), protocols, [_frame()], 1,
                          [None], failures=failures, fail_policy="abort")


def test_shared_frame_ignores_failed_front_but_uses_the_rest():
    protocols = [("MOEAD", False)]
    fronts = {
        ("MOEAD", "a", 0, 0): np.empty((0, 0)),
        ("MOEAD", "b", 0, 0): np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    frames = compare.shared_frame(fronts, _operators(), protocols, 1, 1,
                                  [None])
    frame = frames[(None, 0)]
    assert frame is not None
    assert len(frame) == 2


def test_protocols_include_solver_and_mutation_variants():
    class Args:
        solver = "MOEAD,NSGAII"
        mutation = False
        mutation_both = True

    protocols = compare.build_protocols(Args())
    assert protocols == [
        ("MOEAD", False), ("NSGAII", False),
        ("MOEAD", True), ("NSGAII", True),
    ]
    assert compare.protocol_name("MOEAD", True) == "MOEAD+mut"


def test_load_operator_reads_a_ge_operator(tmp_path):
    path = tmp_path / "MOEAD_operators.txt"
    path.write_text(
        "Best: order_based(fill_first_occurring,preserve_elements,y,"
        "order_based(fill_first_occurring,preserve_elements,y,reverse(y)))\n",
        encoding="utf-8")
    name, namespace, kind, source = compare.load_operator(
        f"ge_best=ge:Best@{path}")
    assert name == "ge_best"
    assert kind == "ge"
    assert source.startswith("order_based(")
    assert callable(namespace["crossover"])


def test_load_operator_rejects_unknown_ge_label(tmp_path):
    path = tmp_path / "MOEAD_operators.txt"
    path.write_text("Best: reverse(y)\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        compare.load_operator(f"ge=ge:Missing@{path}")


def test_parse_groups_and_contrasts():
    groups = compare.parse_groups(["base=ox,pmx", "ge=ge_best"])
    assert groups == {"base": ["ox", "pmx"], "ge": ["ge_best"]}
    assert compare.parse_contrasts(["base:ge"]) == [("base", "ge")]
    with pytest.raises(SystemExit):
        compare.parse_contrasts(["base_ge"])


def test_group_contrasts_compares_two_groups():
    protocols = [("MOEAD", False)]
    operators = [("ox", None, "classic", "ox"), ("llamea", None, "llamea", "code")]
    fronts = {
        ("MOEAD", "ox", 0, 0): np.array([[0.0, 1.0], [1.0, 0.0]]),
        ("MOEAD", "llamea", 0, 0): np.array([[0.5, 0.5]]),
    }
    rows = [
        {"operator": "ox", "kind": "classic", "runs": 1, "mean": 0.1},
        {"operator": "llamea", "kind": "llamea", "runs": 1, "mean": 0.4},
    ]
    results = compare.group_contrasts(
        fronts, rows, protocols, 1, 1,
        {"base": ["ox"], "nuevo": ["llamea"]}, [("nuevo", "base")])
    assert len(results) == 1
    contrast = results[0]
    assert contrast["group"] == "nuevo"
    assert contrast["baseline_group"] == "base"
    assert contrast["pairs"] == 1
    assert contrast["wins"] == 1


def test_reference_history_stays_aligned_when_a_classic_appears_late():
    from mohh.llamea.run import _merge_reference_scores

    history = {}
    _merge_reference_scores(history, {"ox": 1.0}, 0)
    _merge_reference_scores(history, {"ox": 2.0, "pbx": 3.0}, 1)
    _merge_reference_scores(history, {"ox": 4.0, "pbx": 5.0}, 2)
    assert history["ox"] == [1.0, 2.0, 4.0]
    assert history["pbx"] == [None, 3.0, 5.0]


def test_instance_selection_is_sorted_before_sampling():
    from mohh.llamea.run import _select_instances

    paths = ["/data/b.txt", "/data/a.txt", "/data/c.txt"]
    assert _select_instances(paths, "all", 7) == sorted(paths)
    # Mismo seed y mismo conjunto: la eleccion no depende del orden de listdir.
    assert _select_instances(paths, 2, 7) == _select_instances(
        list(reversed(paths)), 2, 7
    )


def test_ge_layout_matches_the_original_evaluation(tmp_path):
    """La salida replica la estructura de `experiment.py`."""
    from scripts.compare_operators import (budget_label, ge_column_name,
                                          write_ge_layout,
                                          write_operator_catalog)

    assert budget_label(300, 105) == "30k"
    assert ge_column_name("MOEAD", False, "ox") == "MOEAD_ox_NM_MOEAD"
    assert ge_column_name("NSGAII", True, "llamea") == "NSGAII_llamea_WM_NSGAII"

    class Args:
        mo_generations = 300
        mo_population_size = 105
        repeats = 2
        paths = ["/tmp/KC11-2fl-1rl.txt"]

    operators = [("ox", None, "classic", "ox"), ("llamea", None, "llamea", "code")]
    protocols = [("MOEAD", False)]
    values = {("MOEAD", "ox", 0, "30k"): [0.5], ("MOEAD", "ox", 1, "30k"): [0.6],
              ("MOEAD", "llamea", 0, "30k"): [0.7],
              ("MOEAD", "llamea", 1, "30k"): [0.8]}
    # La repeticion mediana de "ox" es la 1 (0.6 > 0.5): ahi debe estar el frente.
    front = np.array([[0.0, 0.0], [1.0, 1.0]])
    fronts = {("MOEAD", "ox", 1, "30k", 0): front,
              ("MOEAD", "llamea", 1, "30k", 0): front}
    frames = {("30k", 0): np.array([[0.0, 0.0], [1.0, 1.0]])}

    write_ge_layout(str(tmp_path), Args(), fronts, values, frames, operators,
                    protocols, labels=["30k"])
    csv_path = tmp_path / "MOEAD" / "30k" / "KC11-2fl-1rl.csv"
    assert csv_path.is_file()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == ("Experiment,MOEAD_ox_NM_MOEAD,MOEAD_llamea_NM_MOEAD")
    assert lines[1] == "1,0.5,0.7"
    assert (tmp_path / "run_metrics" / "evaluation_KC11-2fl-1rl.json").is_file()
    front = (tmp_path / "MOEAD" / "30k" / "fronts" / "KC11-2fl-1rl"
             / "MOEAD_ox_NM_MOEAD.pof")
    assert front.is_file()

    catalog = write_operator_catalog(str(tmp_path), operators)
    text = open(catalog, encoding="utf-8").read()
    assert "ox: classic:ox" in text
    assert "llamea: code" in text
