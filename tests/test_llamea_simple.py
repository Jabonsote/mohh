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

    def __init__(self, responses, truncated=False):
        self.responses = list(responses)
        self.prompts = []
        self.truncated = truncated

    def query(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return self.responses.pop(0)

    def last_truncated(self):
        return self.truncated


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


def test_build_prompt_create_has_no_reference_code():
    """`create` no muestra ningun operador que copiar (si no, es una copia)."""
    prompt = simple.build_prompt(None, "create")
    assert "REFERENCE CODE" not in prompt
    assert "SKELETON" in prompt
    assert "from scratch" in prompt
    assert "Generate a new algorithm" in prompt


def test_build_prompt_shows_the_score_of_the_parent():
    """El score mostrado es el del padre, no el del incumbent."""
    prompt = simple.build_prompt(VALID, "refine", score=2.5,
                                 description="CX", best_score=1.1)
    assert "CX (mean rank: 2.5000)" in prompt
    assert "1.1000" not in prompt.split("The selected solution to update is:")[1]


def test_build_prompt_history_explains_the_scale():
    prompt = simple.build_prompt(VALID, "refine", score=1.0,
                                 description="current best",
                                 history=[("OX-like", 1.9)], best_score=1.0)
    assert "mean rank" in prompt
    assert "1 = best" in prompt


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


def test_run_simple_writes_summary_and_generation_log(tmp_path, monkeypatch):
    """La corrida deja resumen consolidado y todos los operadores por generacion."""
    monkeypatch.setattr(simple, "build_population_evaluator", _fake_builder)
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    llm = FakeLLM([VALID, BAD, VALID])
    logger = logging.getLogger("test-simple")
    logger.setLevel(logging.INFO)
    simple.run_simple(_args(budget=3), str(tmp_path), logger, llm, [])

    summary = json.loads((tmp_path / "search_summary.json").read_text())
    assert summary["iterations"]["total"] == 3
    assert summary["iterations"]["valid"] + summary["iterations"]["invalid"] == 3
    assert "success_rate" in summary["iterations"]
    assert "tokens" in summary and "code_diversity" in summary
    assert summary["code_diversity"]["metric"].startswith("CrystalBLEU")
    assert summary["best"]["fitness_metric"] == "GE population mean rank"
    assert summary["failures"]["total"] == summary["iterations"]["invalid"]

    # Todos los operadores de cada generacion quedan registrados.
    individuals = (tmp_path / "generations" / "individuals.jsonl").read_text()
    assert len(individuals.strip().splitlines()) == 3
    generations = (tmp_path / "generations" / "generation_summary.jsonl")
    assert len(generations.read_text().strip().splitlines()) == 3
    last = json.loads(generations.read_text().strip().splitlines()[-1])
    assert last["generation"] == 3
    assert last["best_fitness"] is not None


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


def test_run_simple_create_prompt_has_no_reference_code(tmp_path, monkeypatch):
    """En la estrategia `create` el prompt no lleva el codigo del incumbent."""
    monkeypatch.setattr(simple, "build_population_evaluator", _fake_builder)
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    monkeypatch.setattr(simple.random, "random", lambda: 0.9)  # -> create
    llm = FakeLLM([VALID])
    logger = logging.getLogger("test-simple-create")
    simple.run_simple(_args(budget=1, prior_every=1000), str(tmp_path),
                      logger, llm, [])
    assert "REFERENCE CODE" not in llm.prompts[0]
    assert "SKELETON" in llm.prompts[0]


def test_run_simple_counts_truncated_responses(tmp_path, monkeypatch):
    """Una respuesta cortada por `max_tokens` se cuenta como tal."""
    monkeypatch.setattr(simple, "build_population_evaluator", _fake_builder)
    monkeypatch.setattr(simple, "load_priors", lambda: [("prior", VALID)])
    llm = FakeLLM(["sin codigo"], truncated=True)
    logger = logging.getLogger("test-simple-truncated")
    simple.run_simple(_args(budget=1), str(tmp_path), logger, llm, [])
    summary = json.loads((tmp_path / "search_summary.json").read_text())
    reasons = summary["failures"]["by_reason"]
    assert any("truncada" in key for key in reasons), reasons
    progress = (tmp_path / "progress.txt").read_text()
    assert "respuesta_truncada" in progress


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
