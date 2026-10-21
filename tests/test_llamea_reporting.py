"""Tests de normalizacion, ranking, logging y reportes de `mohh.llamea`.

Complementan `test_llamea.py` (adaptador, parseo, vLLM) con lo necesario para
que el hipervolumen sea comparable con el proyecto original y para que todos
los artefactos queden bajo `results/`.
"""

import json
import os

import numpy as np
import pytest

from mohh.generation.utilities.algorithm.HH_auxiliars import is_permutation
from mohh.llamea.fitness import HVArchive, NADIR_FACTOR, normalize_front
from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for


# --------------------------------------------------------------------------- #
# Operadores de referencia
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(REFERENCE_OPERATORS))
@pytest.mark.parametrize("n", [5, 11, 24, 50])
def test_reference_operators_return_permutations(name, n):
    rng = np.random.RandomState(0)
    function = REFERENCE_OPERATORS[name]
    for _ in range(20):
        x = rng.permutation(n)
        y = rng.permutation(n)
        child = np.asarray(function(x, y))
        assert child.shape == (n,)
        assert is_permutation(child)


def test_reference_namespace_is_minimal():
    namespace = namespace_for(REFERENCE_OPERATORS["ox"])
    assert callable(namespace["crossover"])
    assert namespace["np"] is np


# --------------------------------------------------------------------------- #
# Normalizacion y ranking
# --------------------------------------------------------------------------- #

def test_normalize_front_matches_project_formula():
    front = np.array([[2.0, 10.0], [4.0, 6.0], [6.0, 2.0]])
    vmin = np.array([2.0, 2.0])
    vmax = np.array([6.0, 10.0])
    normalized = normalize_front(front, (vmin, vmax))
    expected = (front - vmin) / (vmax - vmin)
    np.testing.assert_allclose(normalized, expected)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_clip_to_reference_box_bounds_hv_of_better_front():
    """Un candidato mejor que el panel no debe inflar el HV por encima de nadir**K."""
    from mohh.core.utilities.algorithm.MO import compute_hypervolume
    from mohh.llamea.fitness import clip_to_reference_box

    # Marco del panel: [10, 10]..[20, 20]. El candidato esta en [0, 0],
    # fuera del panel; sin recorte el HV seria (1.1-(-1))^2 > 1.1^2.
    frame = np.array([[10.0, 10.0], [20.0, 20.0]])
    candidate = np.array([[0.0, 0.0]])
    vmin, vmax = frame.min(axis=0), frame.max(axis=0)
    normalized = normalize_front(candidate, (vmin, vmax))
    bounded = clip_to_reference_box(normalized)
    hv = compute_hypervolume(np.full(2, 1.1), bounded)
    assert hv == pytest.approx(1.1 ** 2)
    assert hv <= 1.1 ** 2 + 1e-12


def test_normalize_front_matches_original_formula_with_degenerate_column():
    """Formula del original; una columna degenerada da NaN (instancia invalida).

    Es lo que hace `normalize_values` en `mohh/evaluation/experiment.py` con
    `vmax == vmin`; el bucle descarta esas instancias antes de normalizar
    (`OperatorEvaluator._hv_values`).
    """
    front = np.array([[3.0, 1.0], [3.0, 5.0]])
    normalized = normalize_front(front, (np.array([3.0, 1.0]), np.array([3.0, 5.0])))
    assert np.all(np.isnan(normalized[:, 0]))
    np.testing.assert_allclose(normalized[:, 1], [0.0, 1.0])


def test_consolidated_frame_is_non_dominated_envelope():
    from mohh.llamea.fitness import _consolidated_frame

    left = np.array([[0.0, 2.0], [1.0, 1.0]])
    right = np.array([[2.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    frame = _consolidated_frame(left, right)
    # El punto dominado (5, 5) queda fuera; los demas forman la envolvente.
    assert frame.shape == (3, 2)
    np.testing.assert_allclose(sorted(frame.tolist()), [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])


def test_hv_archive_refreshes_values_by_key():
    """En el marco GE el valor de un candidato se refresca y el ranking cambia."""
    archive = HVArchive()
    assert archive.add(0.5, 0.0, key="a") == 1
    assert archive.add(0.2, 0.0, key="b") == 2
    archive.update("b", 0.9)
    assert archive.rank("b") == 1
    assert archive.rank("a") == 2
    assert archive.best() == 0.9


def test_evaluator_seeds_ge_frame_from_saved_bounds():
    """Con `--bounds-file` la caja guardada se convierte en semilla del marco GE."""
    from mohh.llamea.fitness import build_evaluator

    bounds = [
        (np.array([0.0, 0.0]), np.array([2.0, 2.0])),
        (np.array([1.0, 1.0]), np.array([3.0, 1.0])),  # objetivo 2 degenerado
    ]
    evaluator = build_evaluator(["a", "b"], 1, 1, normalize=True, bounds=bounds)
    evaluator.ensure_reference_bounds()
    frame_a, frame_b = evaluator._frame_points
    # Las esquinas de la caja: la envolvente tiene el min/max originales.
    np.testing.assert_allclose(frame_a.min(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(frame_a.max(axis=0), [2.0, 2.0])
    # La instancia con un objetivo sin escala queda marcada y sin HV.
    assert frame_b.shape == (1, 2)
    values = evaluator._hv_values([np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]])])
    assert values[0] == pytest.approx(1.1 ** 2)
    assert values[1] is None
    assert evaluator.frame_degenerate_instances() == ["b"]


def test_evaluator_ge_frame_grows_and_shrinks_hv_without_moea():
    """El marco GE incorpora a los candidatos y recalcula su HV (sin MOEA)."""
    from mohh.llamea.fitness import build_evaluator

    evaluator = build_evaluator(["instancia"], 1, 1, seed=1, normalize=True)
    # Marco inicial: semilla tipo panel (una caja 0..1).
    evaluator._frame_points = [np.array([[0.0, 0.0], [1.0, 1.0]])]
    evaluator._scores = {}
    first = np.array([[0.2, 0.8], [0.8, 0.2]])
    second = np.array([[-1.0, 0.5], [0.5, -1.0]])   # mejor que el panel
    evaluator._cache = {
        "first": {"fronts": [first], "error": None, "novelty": 0.0},
        "second": {"fronts": [second], "error": None, "novelty": 0.0},
    }

    evaluator._recompute_scores()
    hv_first_before = evaluator.final_scores()["first"][0]

    # El segundo candidato amplia la envolvente: el HV del primero baja.
    assert evaluator._refresh_frame([second]) is True
    evaluator._recompute_scores()
    hv_first_after = evaluator.final_scores()["first"][0]
    assert hv_first_after < hv_first_before
    assert evaluator.best_key_in_final_frame() == "second"
    assert evaluator.final_scores()["second"][0] > hv_first_after

    # Un candidato dominado no cambia el marco.
    dominated = np.array([[5.0, 5.0]])
    assert evaluator._refresh_frame([dominated]) is False


def test_nadir_factor_matches_original_project():
    assert NADIR_FACTOR == 1.1


def test_hv_archive_ranks_descending_and_survives_pickle():
    import pickle

    archive = HVArchive()
    assert archive.add(0.5) == 1
    assert archive.add(0.1) == 2
    assert archive.add(0.9) == 1
    assert archive.best() == 0.9
    assert len(archive) == 3

    restored = pickle.loads(pickle.dumps(archive))
    assert len(restored) == 3
    assert restored.add(0.7) == 2


def test_hv_archive_breaks_ties_by_novelty():
    archive = HVArchive()
    assert archive.add(0.5, novelty=0.1) == 1
    # Mismo HV pero menos novedad: queda detras.
    assert archive.add(0.5, novelty=0.05) == 2
    # Mismo HV y mas novedad: se pone primero.
    assert archive.add(0.5, novelty=0.9) == 1
    assert archive.best() == 0.5
    assert len(archive) == 3


def test_hv_archive_accepts_legacy_state_without_novelty():
    import pickle

    # Estado viejo: `_values` con floats sueltos (antes del desempate).
    restored = HVArchive.__new__(HVArchive)
    restored.__setstate__({"_values": [0.4, 0.2]})
    assert len(restored) == 2
    assert restored.add(0.3) == 2
    assert restored.best() == 0.4

    archive = HVArchive()
    archive.add(0.5, novelty=0.3)
    roundtrip = pickle.loads(pickle.dumps(archive))
    assert len(roundtrip) == 1
    assert roundtrip.add(0.5, novelty=0.1) == 2


# --------------------------------------------------------------------------- #
# Logging dentro de results/ y reportes
# --------------------------------------------------------------------------- #

class _FakeIndividual:
    def __init__(self, generation, fitness, name="op", description="", error=None):
        self.generation = generation
        self.fitness = fitness
        self.name = name
        self.description = description
        self.error = error
        self.code = "import numpy as np\n\ndef crossover(x, y):\n    return x\n"
        self.configspace = None

    def to_dict(self):
        return {
            "generation": self.generation,
            "fitness": self.fitness,
            "name": self.name,
            "description": self.description,
            "error": self.error,
            "code": self.code,
            "id": f"{self.generation}-{self.name}",
        }


def test_logger_writes_under_results_dir(tmp_path):
    from mohh.llamea.reporting import ResultsExperimentLogger

    ResultsExperimentLogger.results_dir = str(tmp_path / "llamea_logs")
    ResultsExperimentLogger.verbose = False
    ResultsExperimentLogger.seed = 7
    ResultsExperimentLogger.solver = "MOEAD"
    ResultsExperimentLogger.mo_generations = 2
    ResultsExperimentLogger.mo_population_size = 6
    ResultsExperimentLogger.mutation = False
    logger = ResultsExperimentLogger("demo")

    assert os.path.commonpath([logger.dirname, str(tmp_path)]) == str(tmp_path)
    assert os.path.isdir(os.path.join(logger.dirname, "code"))
    assert os.path.isdir(os.path.join(logger.dirname, "configspace"))

    first = _FakeIndividual(1, 0.4, name="a")
    second = _FakeIndividual(2, 0.8, name="b")
    second.code = "import numpy as np\n\ndef crossover(x, y):\n    return y\n"
    logger.log_population([first])
    logger.log_population([second])

    with open(os.path.join(logger.dirname, "generation_summary.jsonl")) as file:
        summaries = [json.loads(line) for line in file if line.strip()]
    assert [row["generation"] for row in summaries] == [1, 2]
    assert summaries[1]["best_fitness"] == pytest.approx(0.8)
    # Trazabilidad: la semilla y los acumulados van en cada generacion.
    assert summaries[1]["seed"] == 7
    assert summaries[1]["solver"] == "MOEAD"
    assert summaries[1]["distinct_operators_so_far"] == 2
    assert summaries[1]["evaluated_so_far"] == 2
    assert summaries[1]["best_so_far"] == pytest.approx(0.8)

    # Un registro por individuo, con hash del codigo y semilla.
    with open(os.path.join(logger.dirname, "individuals.jsonl")) as file:
        individuals = [json.loads(line) for line in file if line.strip()]
    assert [row["name"] for row in individuals] == ["a", "b"]
    assert individuals[0]["seed"] == 7
    assert len(individuals[0]["code_sha1"]) == 12
    assert individuals[0]["code_sha1"] != individuals[1]["code_sha1"]

    # Checkpoint durable: progreso + mejor operador hasta ahora.
    with open(os.path.join(logger.dirname, "checkpoint.json")) as file:
        checkpoint = json.load(file)
    assert checkpoint["generation"] == 2
    assert checkpoint["counters"]["distinct_operators"] == 2
    assert open(os.path.join(logger.dirname, "best_so_far.py")).read() == second.code

    # Progreso en un archivo de una linea (para `tail -f` en el cluster).
    assert "gen 2" in open(os.path.join(logger.dirname, "progress.txt")).read()

    with open(os.path.join(logger.dirname, "log.jsonl")) as file:
        assert sum(1 for line in file if line.strip()) == 2
    # El logger no debe dejar nada en el directorio de trabajo.
    assert not list(tmp_path.glob("exp-*"))


def test_logger_mirrors_progress_and_checkpoint_to_run_root(tmp_path):
    """El monitoreo y el watchdog leen la raiz del run, no solo `exp-*`."""
    from mohh.llamea.reporting import ResultsExperimentLogger

    run_root = tmp_path / "run"
    ResultsExperimentLogger.results_dir = str(tmp_path / "llamea_logs")
    ResultsExperimentLogger.progress_root = str(run_root)
    ResultsExperimentLogger.state_root = str(run_root)
    ResultsExperimentLogger.verbose = False
    ResultsExperimentLogger.seed = 11
    ResultsExperimentLogger.solver = "MOEAD"
    ResultsExperimentLogger.mo_generations = 2
    ResultsExperimentLogger.mo_population_size = 6
    ResultsExperimentLogger.mutation = False
    try:
        logger = ResultsExperimentLogger("root")
        logger.log_population([_FakeIndividual(1, 0.25, name="a")])
        assert "gen 1" in (run_root / "progress.txt").read_text()
        assert (run_root / "checkpoint.json").is_file()
        assert (run_root / "best_so_far.py").is_file()
    finally:
        ResultsExperimentLogger.progress_root = None
        ResultsExperimentLogger.state_root = None


def test_write_report_builds_csv_markdown_and_plot(tmp_path):
    from mohh.llamea.reporting import write_report

    records = [
        {"generation": 1, "id": "1-a", "name": "a", "description": "first",
         "fitness": 0.2, "error": None, "code": "def crossover(x, y):\n    return x\n"},
        {"generation": 2, "id": "2-b", "name": "b", "description": "worse",
         "fitness": 0.0, "error": "ValueError: not a permutation", "code": ""},
        {"generation": 3, "id": "3-c", "name": "c", "description": "best",
         "fitness": 0.7, "error": None, "code": "def crossover(x, y):\n    return y\n"},
    ]
    with open(os.path.join(tmp_path, "log.jsonl"), "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    paths = write_report(str(tmp_path), title="demo")
    assert os.path.isfile(paths["csv"])
    assert os.path.isfile(paths["markdown"])

    csv_text = open(paths["csv"]).read()
    assert "best_so_far" in csv_text
    assert "0.7" in csv_text

    markdown = open(paths["markdown"]).read()
    assert "Validas (HV > 0): 2" in markdown
    assert "Invalidas: 1" in markdown
    assert "not a permutation" in markdown

    if "plot" in paths:
        assert os.path.getsize(paths["plot"]) > 0


def test_write_generation_csv_keeps_generations_and_tokens(tmp_path):
    from mohh.llamea.reporting import write_generation_csv

    records = [
        {"generation": 0, "best_fitness": 0.1, "tokens_total": 100,
         "tokens_total_gen": 100},
        {"generation": 1, "best_fitness": 0.2, "tokens_total": 260,
         "tokens_total_gen": 160},
    ]
    with open(os.path.join(tmp_path, "generation_summary.jsonl"), "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    path = write_generation_csv(str(tmp_path))
    assert path == os.path.join(str(tmp_path), "generations.csv")
    lines = open(path).read().strip().splitlines()
    assert lines[0].split(",") == ["generation", "best_fitness", "tokens_total",
                                   "tokens_total_gen"]
    assert len(lines) == 3
    assert "160" in lines[2]


def test_write_generation_csv_without_log_returns_none(tmp_path):
    from mohh.llamea.reporting import write_generation_csv

    assert write_generation_csv(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #

def test_vllm_llm_records_usage():
    from mohh.llamea.vllm import VLLM_LLM

    llm = VLLM_LLM.__new__(VLLM_LLM)
    llm.reset_usage()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Message:
        content = "hello"

    class _Choice:
        message = _Message()

    class _Response:
        usage = _Usage()
        choices = [_Choice()]

    llm._last_message_chars = 40
    llm._record_usage(_Response(), elapsed=1.5)
    llm._record_usage(_Response(), elapsed=0.5)
    snapshot = llm.usage_snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["prompt_tokens"] == 20
    assert snapshot["completion_tokens"] == 10
    assert snapshot["total_tokens"] == 30
    # Instrumentacion de costo: latencia acumulada y peticiones fallidas.
    assert snapshot["inference_seconds"] == 2.0
    assert snapshot["failed_requests"] == 0
    assert len(llm.usage_history) == 2
    assert llm.usage_history[0]["seconds"] == 1.5
    assert llm.usage_history[1]["seconds"] == 0.5
    assert llm.usage_history[1]["completion_tokens"] == 5


def test_vllm_llm_estimates_usage_when_missing():
    from mohh.llamea.vllm import VLLM_LLM

    llm = VLLM_LLM.__new__(VLLM_LLM)
    llm.reset_usage()

    class _Message:
        content = "x" * 40  # ~10 tokens estimados

    class _Choice:
        message = _Message()

    class _Response:
        usage = None
        choices = [_Choice()]

    llm._last_message_chars = 80  # ~20 tokens de prompt
    llm._record_usage(_Response())
    snapshot = llm.usage_snapshot()
    assert snapshot["prompt_tokens"] == 20
    assert snapshot["completion_tokens"] == 10
    assert snapshot["total_tokens"] == 30


def test_vllm_llm_logs_raw_response_reasoning_and_params(tmp_path):
    import threading

    from mohh.llamea.vllm import VLLM_LLM

    llm = VLLM_LLM.__new__(VLLM_LLM)
    llm.reset_usage()
    llm.model = "qwen-test"
    llm.base_url = "http://localhost:9/v1"
    llm.temperature = 0.3
    llm.max_tokens = 512
    llm.top_p = 0.8
    llm.top_k = 20
    llm.sampling_repetition_penalty = 1.05
    llm.thinking = True
    llm.request_context = {"experiment": "prueba"}
    llm.request_log_path = os.path.join(tmp_path, "llm_requests.jsonl")
    llm._request_lock = threading.Lock()
    llm._request_index = 0
    llm._last_elapsed = 1.25

    class _Message:
        content = "```python\ndef crossover(x, y):\n    return x\n```"
        reasoning_content = "Primero considero el orden."

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Response:
        id = "resp-1"
        model = "qwen-test"
        system_fingerprint = None
        choices = [_Choice()]
        usage = type("_Usage", (), {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "completion_tokens_details": type("_Details", (), {"reasoning_tokens": 3})(),
            "prompt_tokens_details": None,
            "model_dump": lambda self: {"prompt_tokens": 10,
                                        "completion_tokens": 5,
                                        "total_tokens": 15},
        })()

    llm._last_response = _Response()
    llm._write_request_log(
        1, [{"role": "user", "content": "prompt completo"}], {},
        raw=_Message.content, cleaned=_Message.content,
    )
    with open(llm.request_log_path) as file:
        record = json.loads(file.readline())
    assert record["prompt"] == "prompt completo"
    assert record["response"]["reasoning_content"] == "Primero considero el orden."
    assert record["response"]["raw_content"] == _Message.content
    assert record["params"]["temperature"] == 0.3
    assert record["params"]["max_tokens"] == 512
    assert record["params"]["thinking"] is True
    assert record["response"]["usage"]["total_tokens"] == 15


def test_prompt_log_uses_exact_request_and_reasoning(tmp_path):
    from mohh.llamea.reporting import write_prompt_log

    with open(os.path.join(tmp_path, "conversationlog.jsonl"), "w") as file:
        file.write(json.dumps({"role": "client", "content": "prompt LLaMEA",
                               "time": 1.0}) + "\n")
        file.write(json.dumps({"role": "qwen", "content": "respuesta limpia",
                               "time": 2.0}) + "\n")
    request = {
        "request": 1,
        "model": "qwen-test",
        "prompt": "prompt exacto con memoria",
        "messages": [{"role": "user", "content": "prompt exacto con memoria"}],
        "params": {"temperature": 0.3, "max_tokens": 512, "thinking": "on"},
        "response": {
            "raw_content": "respuesta cruda con <thinking>razonamiento</thinking>",
            "cleaned_content": "respuesta limpia",
            "reasoning_content": "razonamiento interno",
            "finish_reason": "stop",
        },
    }
    with open(os.path.join(tmp_path, "llm_requests.jsonl"), "w") as file:
        file.write(json.dumps(request) + "\n")

    records = write_prompt_log(str(tmp_path))
    assert records[0]["request"] == 1
    assert records[0]["model"] == "qwen-test"
    assert records[0]["params"]["temperature"] == 0.3
    assert records[0]["reasoning_chars"] == len("razonamiento interno")
    with open(os.path.join(tmp_path, "prompts", "gen-0000.md")) as file:
        markdown = file.read()
    assert "prompt exacto con memoria" in markdown
    assert "razonamiento interno" in markdown
    assert "respuesta cruda con" in markdown


def test_vllm_llm_is_picklable_after_request_logging(tmp_path):
    """LLaMEA picklea el objeto; el lock y la ultima respuesta no deben romperlo."""
    import pickle

    from mohh.llamea.vllm import VLLM_LLM

    llm = VLLM_LLM(
        model="qwen-test",
        base_url="http://localhost:9/v1",
        request_log_path=os.path.join(tmp_path, "llm_requests.jsonl"),
    )
    restored = pickle.loads(pickle.dumps(llm))
    assert restored.model == "qwen-test"
    assert restored.request_log_path == llm.request_log_path
    assert restored.usage["requests"] == 0
    assert hasattr(restored, "_request_lock")


def test_tokens_summary_from_generation_log(tmp_path):
    from mohh.llamea.reporting import tokens_summary

    records = [
        {"generation": 0, "tokens_total": 100, "tokens_prompt": 80,
         "tokens_completion": 20, "requests": 1, "tokens_total_gen": 100},
        {"generation": 1, "tokens_total": 260, "tokens_prompt": 210,
         "tokens_completion": 50, "requests": 2, "tokens_total_gen": 160},
    ]
    with open(os.path.join(tmp_path, "generation_summary.jsonl"), "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    summary = tokens_summary(str(tmp_path))
    assert summary["tokens_total"] == 260
    assert summary["requests"] == 2
    assert summary["tokens_total_gen_mean"] == pytest.approx(130.0)
    assert summary["tokens_total_gen_max"] == 160


def test_tokens_summary_none_without_token_fields(tmp_path):
    from mohh.llamea.reporting import tokens_summary

    with open(os.path.join(tmp_path, "generation_summary.jsonl"), "w") as file:
        file.write(json.dumps({"generation": 0, "best_fitness": 0.1}) + "\n")
    assert tokens_summary(str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# CLI: opciones de normalizacion y limites
# --------------------------------------------------------------------------- #

def test_bounds_roundtrip(tmp_path):
    from mohh.llamea.run import _load_bounds, _save_bounds

    path = tmp_path / "bounds.json"
    bounds = [(np.array([1.0, 2.0]), np.array([3.0, 4.0]))]
    instances = [os.path.join("datasets", "mqap", "train", "KC11.txt")]
    _save_bounds(str(path), bounds, instances)

    loaded = _load_bounds(str(path))
    key = os.path.basename(instances[0])
    np.testing.assert_allclose(loaded[key][0], [1.0, 2.0])
    np.testing.assert_allclose(loaded[key][1], [3.0, 4.0])


def test_reference_panel_rejects_unknown_names():
    from mohh.llamea.run import _reference_panel

    with pytest.raises(ValueError):
        _reference_panel("ox,does-not-exist")


def test_reference_panel_selects_subset():
    from mohh.llamea.run import _reference_panel

    panel = _reference_panel("ox, erx")
    assert set(panel) == {"ox", "erx"}
    assert _reference_panel("") is None
