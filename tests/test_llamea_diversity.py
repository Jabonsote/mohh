"""Tests de diversidad conductual de operadores de cruce.

La diversidad que importa aqui no es la del texto: dos operadores distintos
pueden hacer exactamente lo mismo. Estos tests fijan el contrato de la huella
conductual, el archivo MAP-Elites, las familias y los hooks de LLaMEA.
"""

import json
import pickle

import numpy as np
import pytest

from llamea.solution import Solution

from mohh.llamea import diversity
from mohh.llamea.behavioral import (
    least_represented_family,
    select_behavioral_focal,
    select_behavioral_partner,
)
from mohh.llamea.diversity import (
    BehaviorTracker,
    behavior_descriptor,
    behavior_distance,
    behavior_fingerprint,
    family_from_fingerprint,
    solution_behavior_descriptor,
    solution_distance,
)
from mohh.llamea.fitness import OperatorEvaluator, validate_operator
from mohh.llamea.prompts import make_operators
from mohh.llamea.reference import REFERENCE_OPERATORS


def _copy_x(x, y):
    return np.array(x, copy=True)


def _random_permutation(x, y):
    return np.random.permutation(len(x))


def _fingerprint(name):
    return behavior_fingerprint(REFERENCE_OPERATORS[name])


def test_behavior_fingerprint_is_deterministic_and_restores_rng():
    before = np.random.get_state()
    first = behavior_fingerprint(REFERENCE_OPERATORS["ox"])
    second = behavior_fingerprint(REFERENCE_OPERATORS["ox"])
    after = np.random.get_state()

    assert np.allclose(first, second)
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_behavior_families_match_the_classic_mechanism():
    expected = {
        "ox": "order",
        "pbx": "position",
        "pmx": "position",
        "cx": "cycle",
        "erx": "adjacency",
    }
    for name, family in expected.items():
        fingerprint = _fingerprint(name)
        assert family_from_fingerprint(fingerprint) == family, name


def test_copy_and_random_are_detected_as_degenerate_and_exploratory():
    copy_fp = behavior_fingerprint(_copy_x)
    random_fp = behavior_fingerprint(_random_permutation)

    assert family_from_fingerprint(copy_fp) == "copy"
    assert family_from_fingerprint(random_fp) == "exploratory"
    assert copy_fp[diversity.FEATURE_INDEX["recombination"]] < 0.05
    assert random_fp[diversity.FEATURE_INDEX["recombination"]] > 0.80


def test_behavior_distance_contract():
    ox = _fingerprint("ox")
    pbx = _fingerprint("pbx")

    assert behavior_distance(ox, ox) == pytest.approx(0.0)
    assert behavior_distance(ox, pbx) == behavior_distance(pbx, ox)
    assert behavior_distance(ox, pbx) > 0.05
    with pytest.raises(ValueError):
        behavior_distance(np.zeros(3), np.zeros(4))


def test_descriptor_is_bounded_and_uses_behavior():
    for name in ("ox", "pbx", "cx", "erx"):
        descriptor = behavior_descriptor(_fingerprint(name))
        assert len(descriptor) == 2
        assert all(0.0 <= value <= 1.0 for value in descriptor)


def test_tracker_counts_novelty_clusters_and_archive_coverage():
    tracker = BehaviorTracker(threshold=0.05, grid_shape=(4, 4))
    ox = _fingerprint("ox")
    pbx = _fingerprint("pbx")

    first = tracker.add("ox", ox, 0.80, generation=1, description="OX")
    second = tracker.add("pbx", pbx, 0.70, generation=2, description="PBX")
    duplicate = tracker.add("ox-rewritten", ox, 0.60, generation=3,
                            description="same behavior, other text")

    assert first["novelty"] == pytest.approx(1.0)
    assert 0.0 < second["novelty"] <= 1.0
    assert duplicate["novelty"] == pytest.approx(0.0)
    assert duplicate["cached"] is False  # distinto codigo, mismo comportamiento

    snapshot = tracker.snapshot()
    assert snapshot["n"] == 3
    assert snapshot["unique_behavior"] == 2
    assert snapshot["near_duplicates"] == 1
    assert 0.0 < snapshot["archive_coverage"] <= 1.0
    # El duplicado conductual tambien es de la familia order: la diversidad
    # conductual no lo cuenta como mecanismo nuevo, pero el histograma si.
    assert snapshot["family_counts"]["order"] == 2
    assert snapshot["family_counts"]["position"] == 1
    assert snapshot["mean_pairwise_distance"] > 0.0

    entries = tracker.entries_snapshot()
    assert len(entries) == 3
    json.dumps(entries)  # serializable para diversity.json


def test_tracker_caches_exact_code_without_inflating_diversity():
    tracker = BehaviorTracker()
    fingerprint = _fingerprint("ox")
    tracker.add("same-code", fingerprint, 0.80)
    cached = tracker.add("same-code", fingerprint, 0.90)

    assert cached["cached"] is True
    assert cached["novelty"] == pytest.approx(0.0)
    assert cached["fitness"] == pytest.approx(0.90)
    assert tracker.n == 1
    assert tracker.snapshot()["unique_behavior"] == 1


def test_tracker_hint_points_to_an_unexplored_family():
    tracker = BehaviorTracker()
    tracker.add("ox", _fingerprint("ox"), 0.8, generation=1)
    tracker.add("pbx", _fingerprint("pbx"), 0.7, generation=2)

    hint = tracker.hint()
    assert "cycle" in hint
    assert "try" in hint


def test_tracker_is_picklable_and_keeps_its_matrix():
    tracker = BehaviorTracker()
    tracker.add("ox", _fingerprint("ox"), 0.8)
    tracker.add("pbx", _fingerprint("pbx"), 0.7)

    restored = pickle.loads(pickle.dumps(tracker))
    assert restored.snapshot() == tracker.snapshot()
    restored.add("cx", _fingerprint("cx"), 0.6)
    assert restored.n == tracker.n + 1


def test_solution_distance_and_descriptor_use_metadata():
    left = Solution(code="left")
    right = Solution(code="right")
    ox = _fingerprint("ox")
    pbx = _fingerprint("pbx")
    left.add_metadata("behavior_fingerprint", ox.tolist())
    right.add_metadata("behavior_fingerprint", pbx.tolist())

    assert solution_distance(left, right) == pytest.approx(
        behavior_distance(ox, pbx)
    )
    descriptor = solution_behavior_descriptor(left)
    assert descriptor == behavior_descriptor(ox)
    # Sin fingerprint no rompe: cae a la distancia/descriptor de codigo.
    assert solution_distance(Solution(code="a"), Solution(code="b")) >= 0.0
    assert len(solution_behavior_descriptor(Solution(code="a"))) == 2


def test_validate_operator_checks_the_largest_instance():
    def fails_on_100(x, y):
        if len(x) >= 100:
            raise ValueError("operator only supports n < 100")
        return np.roll(x, 1)

    ok, reason = validate_operator(fails_on_100)
    assert not ok
    assert "n=100" in reason


def test_validity_rate_uses_all_evaluations_not_the_budget():
    evaluator = OperatorEvaluator([], 1, 1, normalize=False)
    evaluator._evaluations = 8
    evaluator._valid_evaluations = 6

    stats = evaluator.duplicate_stats()
    assert stats["validity_rate"] == pytest.approx(0.75)
    assert stats["evaluations"] == 8
    assert stats["valid_evaluations"] == 6


def test_behavioral_feedback_flags_a_behavioral_duplicate():
    evaluator = OperatorEvaluator([], 1, 1, normalize=False)
    fingerprint = _fingerprint("ox")
    evaluator.diversity.add("first", fingerprint, 0.8, generation=1,
                            description="classic OX")
    second = evaluator.diversity.add("second", fingerprint, 0.7, generation=2,
                                     description="OX with another variable name")

    note = evaluator._behavioral_feedback(second, {"fingerprint": fingerprint})
    assert "Behavioral profile" in note
    assert "BEHAVIORALLY REDUNDANT" in note
    assert "coverage" in note


def test_repeated_code_does_not_retroactively_lower_the_archive_best(tmp_path):
    """Regresion: una repeticion penalizada no debe sobrescribir el HV original."""
    fingerprint = _fingerprint("ox")
    evaluator = OperatorEvaluator([], 1, 1, normalize=False,
                                  diversity_log=str(tmp_path / "diversity.jsonl"))
    evaluator._evaluate_code = lambda solution, compute_hv: {
        "fronts": None,
        "per_instance": [1.0],
        "score": 1.0,
        "error": None,
        "exception": None,
        "novelty": 0.5,
        "fingerprint": fingerprint,
    }

    first = Solution(code="same operator", generation=1)
    second = Solution(code="same operator", generation=2)
    evaluator(first)
    evaluator(second)

    stats = evaluator.duplicate_stats()
    assert len(evaluator.archive) == 1
    assert evaluator.archive.best() == pytest.approx(1.0)
    assert stats["duplicate_evaluations"] == 1
    assert stats["valid_evaluations"] == 2
    assert stats["validity_rate"] == pytest.approx(1.0)
    assert second.fitness == pytest.approx(0.5)  # ranking penalizado
    assert "DUPLICATE" in second.feedback
    assert "Behavioral profile" in second.feedback

    # La traza incremental existe y es JSON valido (fuente para diversity.json
    # si la corrida se corta antes de terminar).
    trace = tmp_path / "diversity.jsonl"
    assert trace.is_file()
    rows = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["family"] == "order"
    assert rows[0]["fingerprint"]
    assert rows[1]["novelty"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Extension propia: combine con seleccion conductual y familia objetivo
# --------------------------------------------------------------------------- #

def _solution(code, fingerprint, family, fitness, novelty=0.5):
    solution = Solution(code=code, name=code, description=code, generation=1)
    solution.fitness = fitness
    solution.add_metadata("behavior_fingerprint", fingerprint.tolist())
    solution.add_metadata("behavior_family", family)
    solution.add_metadata("behavioral_novelty", novelty)
    return solution


def test_least_represented_family_is_deterministic_and_ignores_copy():
    tracker = BehaviorTracker()
    tracker.add("ox", _fingerprint("ox"), 0.8, generation=1)
    tracker.add("pbx", _fingerprint("pbx"), 0.7, generation=2)

    # order y position ya tienen representante; cycle es la primera familia
    # sin cubrir en el orden canonico.
    assert least_represented_family(tracker) == "cycle"


def test_behavioral_partner_prefers_the_target_family_when_available():
    focal = _solution("focal", _fingerprint("ox"), "order", 0.8)
    near = _solution("near", _fingerprint("pbx"), "position", 0.7)
    target = _solution("target", _fingerprint("cx"), "cycle", 0.6)

    assert select_behavioral_partner(
        focal, [near, target], target_family="cycle"
    ) is target


def test_behavioral_partner_maximizes_distance_without_a_target():
    focal = _solution("focal", _fingerprint("ox"), "order", 0.8)
    near = _solution("near", _fingerprint("pbx"), "position", 0.7)
    far = _solution("far", _fingerprint("cx"), "cycle", 0.1)

    # La distancia conductual manda: no gana el de mejor fitness.
    assert select_behavioral_partner(focal, [near, far]) is far


def test_behavioral_focal_tournament_rewards_novelty_at_equal_fitness():
    old = _solution("old", _fingerprint("ox"), "order", 0.7, novelty=0.0)
    novel = _solution("novel", _fingerprint("cx"), "cycle", 0.7, novelty=0.9)

    assert select_behavioral_focal(
        [old, novel], tournament_size=2, rng=__import__("random").Random(0)
    ) is novel


def test_make_operators_can_include_the_two_parent_combine():
    operators = make_operators(include_combine=True)
    assert len(operators) == 3
    assert {op.name for op in operators} == {"refine", "redesign", "combine"}
    combine = next(op for op in operators if op.name == "combine")
    assert combine.number_of_parents == 2
    assert "TARGET BEHAVIOR FAMILY" in repr(combine)


def test_evaluator_records_a_target_family_hit():
    fingerprint = _fingerprint("ox")
    evaluator = OperatorEvaluator([], 1, 1, normalize=False)
    evaluator._evaluate_code = lambda solution, compute_hv: {
        "fronts": None, "per_instance": [1.0], "score": 1.0,
        "error": None, "exception": None, "novelty": 0.5,
        "fingerprint": fingerprint,
    }
    solution = Solution(code="target-hit", generation=1)
    solution.add_metadata("target_family", "order")

    evaluator(solution)

    assert "Target family check: PASS" in solution.feedback
    assert solution.get_metadata("target_family_hit") is True
    assert evaluator.duplicate_stats()["target_family_hits"] == 1
    assert evaluator.duplicate_stats()["target_family_attempts"] == 1


def test_evaluator_flags_a_target_family_miss_and_can_penalize_ranking():
    fingerprint = _fingerprint("ox")
    evaluator = OperatorEvaluator([], 1, 1, normalize=False,
                                  target_penalty=0.5)
    evaluator._evaluate_code = lambda solution, compute_hv: {
        "fronts": None, "per_instance": [1.0], "score": 1.0,
        "error": None, "exception": None, "novelty": 0.5,
        "fingerprint": fingerprint,
    }
    solution = Solution(code="target-miss", generation=1)
    solution.add_metadata("target_family", "cycle")

    evaluator(solution)

    assert "Target family check: MISS" in solution.feedback
    assert solution.get_metadata("target_family_hit") is False
    # El HV crudo sigue en 1.0; solo el ranking que ve LLaMEA se penaliza.
    assert evaluator._cache["target-miss"]["score"] == pytest.approx(1.0)
    assert solution.fitness == pytest.approx(0.5)
    assert evaluator.duplicate_stats()["target_family_hit_rate"] == 0.0
