"""Pruebas unitarias del bucle LLaMEA (1+1) con strong priors."""
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

import mohh.llamea.simple as simple
from mohh.llamea.fitness import compile_operator
from mohh.llamea.operator import LLMOperator

VALID = """import numpy as np

def crossover(x, y):
    child = np.array(x, copy=True)
    child[0], child[1] = child[1], child[0]
    return child
"""

BEST = """# BEST
import numpy as np

def crossover(x, y):
    child = np.array(x, copy=True)
    child[0], child[1] = child[1], child[0]
    return child
"""

BAD = """# BAD
import numpy as np

def crossover(x, y):
    return np.array(x, copy=True)
"""

NO_IMPORT = """def crossover(x, y):
    child = np.array(x, copy=True)
    child[0], child[1] = child[1], child[0]
    return child
"""

NOT_PERMUTATION = """import numpy as np

def crossover(x, y):
    return np.zeros(len(x), dtype=int)
"""

MISSING_FUNCTION = "import numpy as np\n\nvalue = 3\n"


def _args(budget=4, prior_every=2):
    return SimpleNamespace(
        budget=budget, prior_every=prior_every, mo_generations=1,
        mo_population_size=4, seed=42, solver="MOEAD", mutation=False,
        fitness_workers=1, model="fake-small",
    )


class FakeLLM:
    operator_memory = 0

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def query(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return self.responses.pop(0)


class _FakeEvaluator:
    def evaluate_population(self, population, parent_population=None, **kwargs):
        for candidate in list(population) + list(parent_population or []):
            code = candidate.code or ""
            if "BEST" in code:
                fitness = 1.0
            elif "BAD" in code:
                fitness = 3.0
            else:
                fitness = 2.0
            candidate.set_scores(fitness)


def _fake_builder(*args, **kwargs):
    return _FakeEvaluator()


def test_load_priors_returns_code():
    priors = simple.load_priors()
    assert priors
    assert all("def crossover" in code for _, code in priors)


def test_build_prompt_has_reference_and_strategy():
    prompt = simple.build_prompt(VALID, "refine")
    assert "REFERENCE CODE" in prompt
    assert VALID in prompt
    assert "Refine the strategy" in prompt
    assert "mQAP" not in prompt
    create = simple.build_prompt(VALID, "create")
    assert "Generate a new algorithm" in create


def test_minimal_compile_accepts_permutation_and_tuple():
    simple.minimal_compile(VALID)
    tuple_code = VALID + "\n"
    tuple_code = tuple_code.replace("return child", "return (child, np.array(y, copy=True))")
    simple.minimal_compile(tuple_code)


def test_minimal_compile_accepts_missing_numpy_import():
    # El namespace de primitivas ya expone np: un despiste del modelo pequeno
    # no debe invalidar el operador.
    simple.minimal_compile(NO_IMPORT)


def test_minimal_compile_rejects_non_permutation():
    with pytest.raises(ValueError):
        simple.minimal_compile(NOT_PERMUTATION)


def test_minimal_compile_rejects_missing_crossover():
    with pytest.raises(ValueError):
        simple.minimal_compile(MISSING_FUNCTION)


def test_minimal_compile_rejects_invalid_second_child():
    code = VALID.replace(
        "return child",
        "return (child, np.zeros(len(x), dtype=int))",
    )
    with pytest.raises(ValueError):
        simple.minimal_compile(code)


def test_compile_operator_checks_larger_sizes():
    """Un operador que falla en n>=24 ya no pasa el filtro minimo."""
    code = """import numpy as np

def crossover(x, y):
    if len(x) >= 24:
        raise ValueError('small model operator')
    return np.array(x, copy=True)
"""
    with pytest.raises(ValueError):
        compile_operator(code)


def test_llm_operator_copies_inputs():
    namespace = {}
    exec("""import numpy as np

def crossover(x, y):
    x[0] = 999
    return np.array(x, copy=True)
""", namespace)
    operator = LLMOperator(namespace)
    X = np.array([[[0, 1, 2, 3]], [[3, 2, 1, 0]]])
    before = X.copy()
    operator._do(None, X)
    np.testing.assert_array_equal(X, before)


def test_run_simple_uses_ge_population_rank(tmp_path, monkeypatch):
    captured = {}

    def builder(*args, **kwargs):
        captured.update(kwargs)
        return _FakeEvaluator()

    monkeypatch.setattr(simple, "build_population_evaluator", builder)
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    simple.run_simple(
        _args(budget=0), str(tmp_path), logging.getLogger("test-ge-rank"),
        FakeLLM([]), [],
    )
    assert captured["raw_rank"] is True
    assert captured["frame"] == "ge"
    assert captured["bounds"] is None


def test_run_simple_small_model_failures_and_best(tmp_path, monkeypatch):
    monkeypatch.setattr(simple, "build_population_evaluator", _fake_builder)
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    llm = FakeLLM([
        NOT_PERMUTATION,       # filtro mínimo: inválido
        "",                    # respuesta vacía
        BEST,                  # válido y mejor
        BAD,                   # válido y peor
        NO_IMPORT,             # válido, sin import explícito
    ])
    logger = logging.getLogger("test-simple")
    logger.setLevel(logging.INFO)
    best = simple.run_simple(_args(budget=5), str(tmp_path), logger, llm, [])
    assert "BEST" in best
    saved = (tmp_path / "best_so_far.py").read_text()
    assert "BEST" in saved
    info = json.loads((tmp_path / "best_so_far.json").read_text())
    assert info["fitness"] == pytest.approx(1.0)
    progress = (tmp_path / "progress.txt").read_text().splitlines()
    assert any("valid=0" in line for line in progress)
    assert any("valid=1" in line for line in progress)
    assert (tmp_path / "llamea_best_info.json").is_file()


def test_run_simple_keeps_global_best(tmp_path, monkeypatch):
    monkeypatch.setattr(simple, "build_population_evaluator", _fake_builder)
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    llm = FakeLLM([BEST, BAD, BAD])
    logger = logging.getLogger("test-simple-keep")
    simple.run_simple(_args(budget=3), str(tmp_path), logger, llm, [])
    assert "BEST" in (tmp_path / "best_so_far.py").read_text()
    assert json.loads((tmp_path / "best_so_far.json").read_text())["fitness"] == 1.0


def test_run_simple_prior_injection_changes_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(simple, "build_population_evaluator", _fake_builder)
    priors = [("p1", VALID), ("p2", BEST)]
    monkeypatch.setattr(simple, "load_priors", lambda: priors)
    llm = FakeLLM([VALID, VALID])
    logger = logging.getLogger("test-simple-prior")
    simple.run_simple(_args(budget=4, prior_every=2), str(tmp_path), logger, llm, [])
    # En la iteración 2 debe usar el prompt de un prior, no el del mejor.
    assert "REFERENCE CODE" in llm.prompts[1]


def test_build_prompt_includes_previous_error():
    prompt = simple.build_prompt(
        VALID, "refine", "ValueError: not a permutation")
    assert "PREVIOUS ATTEMPT FAILED" in prompt
    assert "not a permutation" in prompt


def test_run_simple_uses_hv_only_as_tiebreak(tmp_path, monkeypatch):
    class TieEvaluator:
        def evaluate_population(self, population, parent_population=None, **kwargs):
            for candidate in list(population) + list(parent_population or []):
                candidate.set_scores(1.0)
                candidate.hv = 2.0 if "BEST" in (candidate.code or "") else 1.0

    monkeypatch.setattr(simple, "build_population_evaluator", lambda *a, **k: TieEvaluator())
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    llm = FakeLLM([BEST])
    logger = logging.getLogger("test-simple-tiebreak")
    simple.run_simple(_args(budget=1), str(tmp_path), logger, llm, [])
    assert "BEST" in (tmp_path / "best_so_far.py").read_text()
    info = json.loads((tmp_path / "best_so_far.json").read_text())
    assert info["fitness"] == 1.0
    assert info["hv"] == 2.0


def test_simple_priors_are_valid_for_random_permutations():
    from mohh.llamea.fitness import compile_operator
    from mohh.llamea.simple_priors import load_simple_priors

    priors = load_simple_priors()
    assert len(priors) >= 5
    for _, code in priors:
        namespace = compile_operator(code)
        crossover = namespace["crossover"]
        for _ in range(10):
            n = int(np.random.randint(2, 30))
            x = np.random.permutation(n)
            y = np.random.permutation(n)
            child = np.asarray(crossover(x.copy(), y.copy()))
            assert child.shape == (n,)
            assert set(child.tolist()) == set(range(n))


def test_build_prompt_includes_history():
    prompt = simple.build_prompt(
        VALID, "refine", score=4.0, description="best",
        history=[("order crossover", 3.5), ("cycle crossover", 2.0)])
    assert "order crossover" in prompt
    assert "cycle crossover" in prompt
    assert "Do not repeat" in prompt
