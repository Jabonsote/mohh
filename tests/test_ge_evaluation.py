"""Pruebas unitarias de la evaluación GE y de sus utilidades compartidas."""
import importlib.util
import json
import os

import numpy as np
import pytest

from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.evaluation.algorithms import MOEA_Decomposition, NSGAII, SMS_MOEA
from mohh.evaluation.experiment import boundaries_from_points, normalize_values
from mohh.evaluation.experiment import build_solver
from mohh.llamea.operator import LLMOperator

ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, "scripts", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boundaries_from_points_include_dominated_points():
    good = {(0.0, 2.0), (1.0, 1.0), (2.0, 0.0)}
    bad = {(10.0, 10.0)}
    vmin, vmax = boundaries_from_points(good | bad)
    np.testing.assert_allclose(vmin, [0.0, 0.0])
    np.testing.assert_allclose(vmax, [10.0, 10.0])

    normalized_bad = normalize_values(np.array([[10.0, 10.0]]), vmin, vmax)
    hv = compute_hypervolume(np.array([1.1, 1.1]), normalized_bad)
    assert hv > 0.0

    # Marco antiguo (solo el frente no dominado): el punto malo queda fuera.
    vmin_old, vmax_old = boundaries_from_points(good)
    normalized_old = normalize_values(np.array([[10.0, 10.0]]), vmin_old, vmax_old)
    assert np.all(normalized_old > 1.1)
    assert compute_hypervolume(np.array([1.1, 1.1]), normalized_old) == 0.0


def test_build_solver_accepts_llamea_operator(tmp_path):
    path = tmp_path / "op.py"
    path.write_text(
        "import numpy as np\n"
        "def crossover(x, y):\n"
        "    c = np.array(x, copy=True)\n"
        "    c[0], c[1] = c[1], c[0]\n"
        "    return c\n"
    )
    solver = build_solver("MOEAD", "llamea", f"llamea:{path}", "NullMutation")
    assert isinstance(solver.crossover, LLMOperator)


class _FakeCallback:
    populations = {}


class _FakeAlgorithm:
    def __init__(self):
        self.callback = _FakeCallback()

    def has_next(self):
        return True

    def ask(self):
        return None


@pytest.mark.parametrize("solver_class", [MOEA_Decomposition, NSGAII, SMS_MOEA])
def test_solve_instance_handles_none_from_ask(solver_class):
    solver = solver_class.__new__(solver_class)
    solver.algorithm = _FakeAlgorithm()
    assert solver.solve_instance() == {}


def test_read_ge_operators_and_budgets(tmp_path):
    module = _load_script("evaluate_llamea_vs_ge")
    path = tmp_path / "MOEAD_operators.txt"
    path.write_text("Best: best_code\nMiddle: middle_code\nWorst: worst_code\n")
    assert module.read_ge_operators(path) == ["best_code", "middle_code", "worst_code"]
    assert module.parse_budgets("10k:10500,30k:31500") == {
        "10k": 10500, "30k": 31500}

    path.write_text("Best: only_best\n")
    with pytest.raises(SystemExit):
        module.read_ge_operators(path)


def test_compare_stats_pairs_by_experiment(tmp_path):
    module = _load_script("compare_ge_llamea_stats")
    root = tmp_path / "run_metrics"
    root.mkdir()
    records = []
    for experiment in range(1, 6):
        records.append({
            "instance": "I1", "label": "10k",
            "config": "MOEAD_Best_NM_MOEAD",
            "experiment": f"Experiment {experiment:03d}",
            "hypervolume": 0.5,
        })
        records.append({
            "instance": "I1", "label": "10k",
            "config": "MOEAD_Best_NM_LLAMEA",
            "experiment": f"Experiment {experiment:03d}",
            "hypervolume": 0.6,
        })
    json.dump({"sections": {"hypervolumes": records}},
              open(root / "evaluation_test.json", "w"))

    frame = module.load(str(tmp_path))
    table = module.compare(frame, {}, ("Best",))
    assert len(table) == 1
    assert table.iloc[0]["pairs"] == 5
    assert table.iloc[0]["wins"] == 5
    assert table.iloc[0]["median_difference"] == pytest.approx(0.1)


def test_population_evaluator_returns_positive_hv(monkeypatch):
    """El fitness de LLaMEA debe ser HV positivo, no un ranking negativo."""
    from mohh.llamea.fitness import PopulationEvaluator

    evaluator = PopulationEvaluator([], 1, 1, normalize=True)
    monkeypatch.setattr(evaluator, "_ensure_instances", lambda: [None])
    monkeypatch.setattr(
        evaluator, "_evaluate_code",
        lambda solution, compute_hypervolume: {
            "fronts": [np.array([[0.0, 1.0], [1.0, 0.0]])],
            "error": None,
        })

    class Solution:
        def __init__(self, code):
            self.code = code
            self.fitness = None
            self.feedback = ""
            self.error = None
            self.name = "x"
            self.description = ""

        def set_scores(self, fitness, feedback="", error=None):
            self.fitness = fitness
            self.feedback = feedback
            self.error = error

    first = Solution("first")
    second = Solution("second")
    evaluator._evaluate_batch([second], [first])

    assert first.fitness >= 0.0
    assert second.fitness >= 0.0
    assert "GE population ranking" in first.feedback
    assert "Fitness" in first.feedback or "score" in first.feedback


def test_build_population_evaluator_constructs():
    from mohh.llamea.fitness import build_population_evaluator
    evaluator = build_population_evaluator([], 1, 1)
    assert evaluator is not None


def test_build_evaluator_ranking_flag():
    from mohh.llamea.fitness import build_evaluator
    evaluator = build_evaluator([], 1, 1, ranking=True)
    assert evaluator.ranking is True


def test_rank_score_is_positive_and_rank1_is_best(monkeypatch):
    from mohh.llamea.fitness import build_evaluator

    evaluator = build_evaluator([], 1, 1, ranking=True)
    monkeypatch.setattr(evaluator, "_hv_values", lambda fronts: [1.0])
    monkeypatch.setattr(
        evaluator, "_reference_hv_values",
        lambda: {"a": [0.5], "b": [0.2]},
    )
    score, mean_rank, k = evaluator._rank_score([np.array([[0.0, 1.0]])])
    assert k == 3
    assert mean_rank == pytest.approx(1.0)
    assert score == pytest.approx(3.0)

    monkeypatch.setattr(evaluator, "_hv_values", lambda fronts: [0.1])
    score, mean_rank, k = evaluator._rank_score([np.array([[0.0, 1.0]])])
    assert mean_rank == pytest.approx(3.0)
    assert score == pytest.approx(1.0)


def test_clip_to_nadir_bounds_hv_after_operator_failure():
    from mohh.evaluation.experiment import clip_to_nadir

    nadir = np.array([1.1, 1.1, 1.1, 1.1])
    front = np.array([[-17.0, -16.0, -17.0, -14.0], [0.5, 0.5, 0.5, 0.5]])
    clipped = clip_to_nadir(front, nadir)
    assert clipped.min() >= 0.0
    assert clipped.max() <= 1.1
    hv = compute_hypervolume(nadir, clipped)
    assert hv <= 1.1 ** 4 + 1e-9
