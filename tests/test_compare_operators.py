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
    return [("a", None, "discovered"), ("b", None, "classic")]


def test_failed_front_counts_as_zero_or_is_excluded():
    protocols = [("MOEAD", False)]
    fronts = {
        ("MOEAD", "a", 0, 0): np.empty((0, 0)),
        ("MOEAD", "b", 0, 0): np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    failures = {("MOEAD", "a", 0, 0): "RuntimeError: boom"}

    zero = compare.hv_values(fronts, _operators(), protocols, [_frame()], 1,
                             failures=failures, fail_policy="zero")
    assert zero[("MOEAD", "a", 0)] == [0.0]
    assert zero[("MOEAD", "b", 0)][0] > 0.0

    excluded = compare.hv_values(fronts, _operators(), protocols, [_frame()], 1,
                                 failures=failures, fail_policy="exclude")
    assert excluded[("MOEAD", "a", 0)] == [None]

    with pytest.raises(RuntimeError, match="fallo en MOEAD a"):
        compare.hv_values(fronts, _operators(), protocols, [_frame()], 1,
                          failures=failures, fail_policy="abort")


def test_shared_frame_ignores_failed_front_but_uses_the_rest():
    protocols = [("MOEAD", False)]
    fronts = {
        ("MOEAD", "a", 0, 0): np.empty((0, 0)),
        ("MOEAD", "b", 0, 0): np.array([[0.0, 1.0], [1.0, 0.0]]),
    }
    frame = compare.shared_frame(fronts, _operators(), protocols, 1, 1)[0]
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
