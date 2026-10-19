"""Tests de la integracion LLaMEA + MOHH-R (mohh.llamea).

Cubren el adaptador de operadores, el parseo/reparacion de respuestas, el
conector vLLM (sin servidor) y un bucle LLaMEA de extremo a extremo con un
LLM simulado (sin red), ademas de confirmar que el nucleo de evolucion
gramatical NO fue modificado.
"""

import inspect
import os
import subprocess

import numpy as np
import pytest

from mohh.core.operators.operator_template import HH_Operator
from mohh.llamea.fitness import build_evaluator, compile_operator
from mohh.llamea.operator import LLMOperator
from mohh.llamea.vllm import model_is_available, VLLM_LLM, wait_for_server

INSTANCE = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "mqap", "train", "KC23-2fl-1rl.txt"
)

EXAMPLE_CODE = """import numpy as np

def crossover(x, y):
    n = len(x)
    start, end = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1)
    child[start:end] = x[start:end]
    taken = set(child[start:end].tolist())
    fill = [v for v in y if v not in taken]
    positions = list(range(end, n)) + list(range(0, start))
    for pos, val in zip(positions, fill):
        child[pos] = val
    return child
"""

INVALID_CODE = """import numpy as np

def crossover(x, y):
    return np.zeros(len(x), dtype=int)
"""

NO_FUNCTION_CODE = "import numpy as np\n\nvalue = 3\n"


# --------------------------------------------------------------------------- #
# Adaptador de operadores (mohh.llamea.operator)
# --------------------------------------------------------------------------- #

def test_llm_operator_applies_generated_crossover():
    namespace = {}
    exec(EXAMPLE_CODE, namespace)
    op = LLMOperator(namespace, prob=1.0)

    class FakeProblem:
        n_var = 6
        n_obj = 2
        n_constr = 0

    X = np.array([[[0, 1, 2, 3, 4, 5]], [[5, 4, 3, 2, 1, 0]]])
    Y = op._do(FakeProblem(), X)
    child = Y[0, 0]
    assert set(child) == set(range(6))
    assert set(Y[1, 0]) == set(range(6))


def test_ge_core_is_untouched():
    """El nucleo de evolucion gramatical no fue modificado por la integracion."""
    params = inspect.signature(HH_Operator.__init__).parameters
    assert "eval_namespace" not in params, "HH_Operator no debe cambiar para LLaMEA"

    op = HH_Operator("reverse(x)", "Natural", prob=1.0)

    class FakeProblem:
        n_var = 4
        n_obj = 2
        n_constr = 0

    X = np.array([[[0, 1, 2, 3]], [[3, 2, 1, 0]]])
    Y = op._do(FakeProblem(), X)
    assert list(Y[0, 0]) == [3, 2, 1, 0]


# --------------------------------------------------------------------------- #
# Compilacion / validacion de codigo
# --------------------------------------------------------------------------- #

def test_compile_operator_accepts_valid_permutation_operator():
    namespace = compile_operator(EXAMPLE_CODE)
    assert callable(namespace["crossover"])


def test_compile_operator_rejects_non_permutation():
    with pytest.raises(ValueError):
        compile_operator(INVALID_CODE)


def test_compile_operator_rejects_missing_function():
    with pytest.raises(ValueError):
        compile_operator(NO_FUNCTION_CODE)


def test_compile_operator_reports_syntax_errors():
    with pytest.raises(ValueError):
        compile_operator("def crossover(x, y)\n    return x\n")


def test_compile_operator_rejects_infinite_loop(monkeypatch):
    """Un operador que no termina (`while True`) se rechaza, no cuelga."""
    import mohh.llamea.fitness as fit

    monkeypatch.setattr(fit, "OPERATOR_CALL_TIMEOUT", 0.5)
    code = (
        "import numpy as np\n\n"
        "def crossover(x, y):\n"
        "    while True:\n"
        "        pass\n"
    )
    with pytest.raises(ValueError):
        fit.compile_operator(code)


def test_compile_operator_exposes_mohh_primitives():
    """El LLM puede reutilizar los primitivos de permutacion de MOHH-R.

    Se usa `move_right` (devuelve un array nuevo). `reverse(x)` tambien es un
    primitivo de MOHH-R, pero devuelve una *vista* del padre (`x[::-1]`) y el
    validador la rechaza: el algoritmo modifica los descendientes in-place, asi
    que una vista corromperia a la poblacion (ver `validate_operator`).
    """
    namespace = compile_operator("def crossover(x, y):\n    return move_right(x)\n")
    child = namespace["crossover"](np.array([0, 1, 2, 3]), np.array([3, 2, 1, 0]))
    assert set(np.asarray(child)) == set(range(4))


def test_compile_operator_bounds_module_level_code():
    """Un bucle infinito a nivel de modulo no puede colgar la corrida.

    El LLM suele anexar un "self-check" que llama a `crossover` al importar. Si
    ese codigo no termina, antes se colgaba el proceso entero (paso en
    `qwen25coder7b_moead_b30_st_r3`, con un `while` que giraba buscando una
    adyacencia inexistente). Ahora el `exec` esta acotado y el LLM recibe un
    error explicito.
    """
    code = (
        "import numpy as np\n\n"
        "def crossover(x, y):\n"
        "    return x\n\n"
        "while True:  # self-check que nunca termina\n"
        "    pass\n"
    )
    with pytest.raises(ValueError, match="module-level"):
        compile_operator(code, timeout=1.0)


def test_mutation_flag_is_not_a_noop():
    """`--mutation` tiene que cambiar la corrida (no ser un no-op).

    El nucleo GE tiene `with_mutation` comentado y el arnes cargaba siempre
    `NullMutation`, asi que `--mutation` no cambiaba nada: la ablacion con y sin
    la bandera daba frentes **identicos**. Ahora carga `Swap_Mutation`, la
    mutacion de permutacion que usa `experiment.py`, inyectada por atributo para
    no tocar el nucleo.
    """
    fronts = {}
    for mutation in (False, True):
        fitness = build_evaluator([INSTANCE], mo_generations=5,
                                  mo_population_size=10, seed=42,
                                  normalize=False, mutation=mutation)
        fronts[mutation] = fitness._solve_front(
            fitness._ensure_instances()[0], compile_operator(EXAMPLE_CODE)
        )
    assert not np.array_equal(fronts[False], fronts[True])


def test_compile_operator_provides_numpy_without_import():
    """El LLM suele olvidar `import numpy as np`; el namespace lo aporta."""
    namespace = compile_operator(
        "def crossover(x, y):\n    return np.arange(len(x))\n"
    )
    child = namespace["crossover"](np.arange(5), np.arange(5))
    assert set(np.asarray(child)) == set(range(5))


def test_compile_operator_provides_typing_without_import():
    """El LLM usa anotaciones (`List[int]`) sin importar `typing`; se aporta."""
    code = (
        "def crossover(x: List[int], y: List[int]) -> List[int]:\n"
        "    return np.arange(len(x))\n"
    )
    namespace = compile_operator(code)
    child = namespace["crossover"](np.arange(4), np.arange(4))
    assert set(np.asarray(child)) == set(range(4))


def test_compile_operator_accepts_tuple_from_order_based():
    """Un operador puede devolver la tupla (child1, child2) como la gramatica."""
    code = (
        "def crossover(x, y):\n"
        "    return order_based(fill_first_occurring, preserve_segments, x, y)\n"
    )
    namespace = compile_operator(code)
    x = np.arange(8)
    y = np.arange(8)[::-1]
    child = namespace["crossover"](x, y)
    assert isinstance(child, tuple) and len(child) == 2
    assert set(np.asarray(child[0])) == set(range(8))
    assert set(np.asarray(child[1])) == set(range(8))


# --------------------------------------------------------------------------- #
# Feedback de validacion (lo que recibe el LLM cuando su operador no vale)
# --------------------------------------------------------------------------- #

def _message_from(code):
    with pytest.raises(ValueError) as excinfo:
        compile_operator(code)
    return str(excinfo.value)


def test_validation_feedback_points_at_unfilled_slots():
    """El clasico OBX mal cerrado: quedan huecos con -1."""
    code = (
        "import numpy as np\n"
        "def crossover(x, y):\n"
        "    n = len(x)\n"
        "    child = np.full(n, -1)\n"
        "    child[2:5] = x[2:5]\n"
        "    return child\n"
    )
    message = _message_from(code)
    assert "n=11" in message
    assert "unfilled" in message and "-1" in message
    assert "missing value(s)" in message


def test_validation_feedback_reports_failing_line_of_generated_code():
    """Un `assert` del propio LLM se reporta con su numero de linea."""
    code = (
        "import numpy as np\n"
        "def crossover(x, y):\n"
        "    n = len(x)\n"
        "    child = np.arange(n)\n"
        "    assert np.array_equal(np.sort(child), y)\n"
        "    return child\n"
    )
    message = _message_from(code)
    assert "AssertionError" in message
    assert "line 5" in message and "assert np.array_equal" in message


def test_validation_feedback_rejects_in_place_mutation_of_parents():
    code = (
        "def crossover(x, y):\n"
        "    x[0], y[0] = y[0], x[0]\n"
        "    return x.copy()\n"
    )
    message = _message_from(code)
    assert "modified its inputs" in message and "read-only" in message


def test_validation_feedback_rejects_view_of_a_parent():
    code = "def crossover(x, y):\n    return x[::-1]\n"
    message = _message_from(code)
    assert "view of a parent" in message


def test_validation_feedback_reports_duplicates_and_missing_values():
    code = (
        "import numpy as np\n"
        "def crossover(x, y):\n"
        "    n = len(x)\n"
        "    child = np.arange(n)\n"
        "    child[0] = child[1]\n"
        "    return child\n"
    )
    message = _message_from(code)
    assert "duplicated value(s)" in message
    assert "missing value(s)" in message


def test_validation_feedback_rejects_tuple_of_wrong_size():
    code = (
        "import numpy as np\n"
        "def crossover(x, y):\n"
        "    return x.copy(), y.copy(), x.copy()\n"
    )
    message = _message_from(code)
    assert "tuple of 3 elements" in message



def test_make_task_prompt_is_solver_specific():
    from mohh.llamea.prompts import make_task_prompt

    assert "MOEA/D" in make_task_prompt("MOEAD")
    assert "NSGA-II" in make_task_prompt("NSGAII")
    assert "SMS-EMOA" in make_task_prompt("SMSEMOA")
    # default = MOEA/D
    assert "MOEA/D" in make_task_prompt()


def test_task_prompt_covers_interface_and_domain_knowledge():
    """El prompt fija el contrato y aporta conocimiento de dominio (familias)."""
    from mohh.llamea.prompts import make_task_prompt

    prompt = make_task_prompt("MOEAD")
    # Contrato y objetivo.
    assert "def crossover(x, y)" in prompt
    assert "permutation" in prompt
    assert "hypervolume" in prompt
    # Conocimiento de dominio (familias de operadores de permutacion).
    for family in ("Order Crossover", "Position-Based", "Edge Recombination", "Cycle"):
        assert family in prompt


def test_task_prompt_is_not_grammar_based():
    """El prompt no debe depender de la gramatica de MOHH-R."""
    from mohh.llamea.prompts import make_task_prompt

    prompt = make_task_prompt("MOEAD").lower()
    for grammar_term in ("bnf", "grammar", "order_based", "fill_first_occurring",
                         "preserve_segments", "map_list", "alternate_elements"):
        assert grammar_term not in prompt


def test_make_operators_are_single_parent():
    """Exploracion/explotacion: dos operadores de un solo padre."""
    from mohh.llamea.prompts import make_operators

    operators = make_operators()
    assert len(operators) == 2
    assert {op.name for op in operators} == {"refine", "redesign"}
    assert all(op.number_of_parents == 1 for op in operators)
    # `__repr__` (lo que LLaMEA concatena al prompt) es la instruccion real.
    assert all("crossover operator" in repr(op) for op in operators)


# --------------------------------------------------------------------------- #
# Fitness (reutiliza MOSolver + hipervolumen de MOHH-R)
# --------------------------------------------------------------------------- #

class _FakeSolution:
    def __init__(self, code):
        self.code = code
        self.fitness = None
        self.feedback = None
        self.error = None

    def set_scores(self, fitness, feedback="", error=None):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error


def test_evaluator_scores_valid_operator():
    fitness = build_evaluator([INSTANCE], mo_generations=3, mo_population_size=10, seed=42)
    sol = fitness(_FakeSolution(EXAMPLE_CODE))
    assert np.isfinite(sol.fitness)
    assert sol.fitness > 0
    assert sol.error is None
    assert "hypervolume" in sol.feedback.lower()


def test_evaluator_rejects_invalid_operator():
    fitness = build_evaluator([INSTANCE], mo_generations=3, mo_population_size=10, seed=42)
    sol = fitness(_FakeSolution(INVALID_CODE))
    assert sol.fitness == 0.0
    assert sol.error is not None


def test_evaluator_is_picklable_for_parallel_workers():
    """El fitness debe sobrevivir un pickle (LLaMEA con max_workers > 1)."""
    import pickle

    fitness = build_evaluator([INSTANCE], mo_generations=3, mo_population_size=10, seed=42)
    # Simular que el padre ya construyo sus instancias antes del pickle.
    fitness._ensure_instances()
    restored = pickle.loads(pickle.dumps(fitness))
    assert restored._instances is None
    sol = restored(_FakeSolution(EXAMPLE_CODE))
    assert sol.fitness > 0


def test_evaluator_caches_repeated_code():
    """Un mismo codigo no se re-evalua; el ranking penaliza la repeticion."""
    fitness = build_evaluator([INSTANCE], mo_generations=2, mo_population_size=6, seed=42)
    first = fitness(_FakeSolution(EXAMPLE_CODE))
    second = fitness(_FakeSolution(EXAMPLE_CODE))
    # El HV crudo se cachea y no cambia...
    assert first.fitness == float(fitness._scores[EXAMPLE_CODE.strip()][0]) > 0
    # ...pero la repeticion exacta se penaliza para el ranking (presion de
    # diversidad) y el feedback lo dice.
    assert second.fitness == pytest.approx(first.fitness * fitness.repeat_penalty)
    assert "DUPLICATE" in second.feedback
    assert "reused" in second.feedback
    assert len(fitness._cache) == 1
    assert fitness.duplicate_stats()["duplicate_evaluations"] == 1


def test_repeat_penalty_can_be_disabled():
    fitness = build_evaluator([INSTANCE], mo_generations=2, mo_population_size=6,
                              seed=42, repeat_penalty=1.0)
    first = fitness(_FakeSolution(EXAMPLE_CODE))
    second = fitness(_FakeSolution(EXAMPLE_CODE))
    assert first.fitness == second.fitness > 0
    assert "DUPLICATE" not in second.feedback


def test_recombination_novelty_flags_parent_copies():
    from mohh.llamea.fitness import recombination_novelty

    def copy_x(x, y):
        return np.array(x, copy=False)

    def copy_x_fresh(x, y):
        return np.array(x, copy=True)

    ok, novelty, reason = recombination_novelty(copy_x)
    assert ok is False
    assert novelty == 0.0
    assert "view of a parent" in reason

    ok, novelty, reason = recombination_novelty(copy_x_fresh)
    assert ok is False
    assert novelty == 0.0
    assert "degenerate" in reason


def test_recombination_novelty_accepts_real_crossover():
    from mohh.llamea.fitness import recombination_novelty
    from mohh.llamea.reference import ox

    ok, novelty, reason = recombination_novelty(ox)
    assert ok is True
    assert reason is None
    assert 0.0 < novelty <= 1.0


def test_evaluator_rejects_degenerate_operator_without_running_moea():
    fitness = build_evaluator([INSTANCE], mo_generations=2,
                              mo_population_size=6, seed=42)
    code = "import numpy as np\n\ndef crossover(x, y):\n    return np.array(x, copy=False)\n"
    sol = fitness(_FakeSolution(code))
    assert sol.fitness == 0.0
    assert "view of a parent" in sol.feedback or "degenerate" in sol.feedback


def test_evaluator_scores_zero_when_operator_fails_on_instance():
    """Un operador valido en tamanos chicos que falla en una instancia real
    no debe tumbar la busqueda: score 0 + feedback."""
    code = (
        "import numpy as np\n\n"
        "def crossover(x, y):\n"
        "    n = len(x)\n"
        "    if n == 23:\n"
        "        raise RuntimeError('boom on the real instance')\n"
        "    return np.arange(n)\n"
    )
    fitness = build_evaluator([INSTANCE], mo_generations=3, mo_population_size=10, seed=42)
    sol = fitness(_FakeSolution(code))
    assert sol.fitness == 0.0
    assert "failed on an instance" in sol.feedback


# --------------------------------------------------------------------------- #
# Conector vLLM / verificacion del modelo
# --------------------------------------------------------------------------- #

def test_model_is_available_local_path(tmp_path):
    model_dir = tmp_path / "models--org--model"
    model_dir.mkdir()
    assert model_is_available(str(model_dir))
    assert not model_is_available("orgo/definitely-not-downloaded-model-xyz")


def test_model_is_available_respects_hf_home(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    (hub / "models--org--cached").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert model_is_available("org/cached")
    assert not model_is_available("org/absent")


def test_vllm_llm_constructs_without_server():
    llm = VLLM_LLM(model="tiny-model", base_url="http://localhost:9/v1")
    assert llm.model == "tiny-model"
    assert llm.base_url == "http://localhost:9/v1"


def test_wait_for_server_times_out_quickly():
    assert wait_for_server("http://localhost:9/v1", timeout=0.5, interval=0.1) is False


def test_vllm_llm_repairs_responses(monkeypatch):
    llm = VLLM_LLM(model="m", base_url="http://localhost:9/v1")

    import llamea.llm as llamea_llm
    monkeypatch.setattr(
        llamea_llm.OpenAI_LLM,
        "query",
        lambda self, session, **kw: "<think>bla</think>```python\npass\n```",
    )
    cleaned = llm.query([{"role": "user", "content": "x"}])
    assert "<think>" not in cleaned

    repaired = llm.extract_algorithm_code(
        "```python\ndef crossover(x, y):\n    return x\n\nif __name__ == '__main__':\n    pass\n```\n"
    )
    assert "__main__" not in repaired


def test_vllm_llm_extract_falls_back_without_closing_fence():
    """Si el modelo no cierra el fence, se usa el extractor tolerante."""
    llm = VLLM_LLM(model="m", base_url="http://localhost:9/v1")
    code = llm.extract_algorithm_code("```python\ndef crossover(x, y):\n    return x\n")
    assert "def crossover" in code


def test_vllm_llm_extract_raises_without_code():
    from llamea.utils import NoCodeException

    llm = VLLM_LLM(model="m", base_url="http://localhost:9/v1")
    with pytest.raises(NoCodeException):
        llm.extract_algorithm_code("solo prosa, sin ningun bloque de codigo")


def test_vllm_llm_extract_requires_module_level_crossover():
    """Codigo sin `def crossover` se rechaza con NoCodeException (no IndexError)."""
    from llamea.utils import NoCodeException

    llm = VLLM_LLM(model="m", base_url="http://localhost:9/v1")
    with pytest.raises(NoCodeException):
        llm.extract_algorithm_code("```python\nimport numpy as np\nvalue = 3\n```\n")


# --------------------------------------------------------------------------- #
# Conexion viva con un servidor local OpenAI-compatible (opt-in)
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(
    not os.environ.get("LLAMEA_SERVER_URL"),
    reason="define LLAMEA_SERVER_URL para probar contra un servidor local",
)
def test_live_server_generates_valid_operator():
    """Genera un operador contra un servidor local (vLLM o el de desarrollo).

    Se activa con `LLAMEA_SERVER_URL=http://127.0.0.1:8001/v1` (y opcionalmente
    `LLAMEA_SERVER_MODEL`). Verifica conexion + generacion + parseo + contrato.
    """
    from mohh.llamea.prompts import (
        ROLE_PROMPT, make_task_prompt, EXAMPLE_PROMPT, OUTPUT_FORMAT_PROMPT,
    )

    url = os.environ["LLAMEA_SERVER_URL"]
    model = os.environ.get("LLAMEA_SERVER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    llm = VLLM_LLM(model=model, base_url=url, temperature=0.2)
    prompt = ROLE_PROMPT + make_task_prompt("MOEAD") + EXAMPLE_PROMPT + OUTPUT_FORMAT_PROMPT

    # Un modelo chico local (p.ej. 0.5B) no siempre cumple el contrato; lo que
    # se prueba aqui es la conexion extremo a extremo (peticion HTTP -> parseo ->
    # contrato -> operador valido). Se exige al menos una muestra valida.
    valid = 0
    for _ in range(5):
        try:
            solution = llm.sample_solution([{"role": "user", "content": prompt}])
            compile_operator(solution.code)
        except Exception:
            continue
        assert "def crossover" in solution.code
        valid += 1

    assert valid >= 1, "El servidor no produjo ningun operador valido en 5 intentos"


# --------------------------------------------------------------------------- #
# Parseo de respuestas del LLM
# --------------------------------------------------------------------------- #

def test_strip_thinking_removes_reasoning_blocks():
    from mohh.llamea.parsing import strip_thinking

    text = "Pienso...\n<think>razonamiento largo</think>\n```python\ndef crossover(x, y):\n    return x\n```"
    cleaned = strip_thinking(text)
    assert "razonamiento" not in cleaned
    assert "<think>" not in cleaned
    assert "```python" in cleaned


def test_extract_code_block_handles_unclosed_fence():
    from mohh.llamea.parsing import extract_code_block

    text = "Aqui:\n```python\ndef crossover(x, y):\n    return x\n"
    assert "def crossover" in extract_code_block(text)


def test_extract_code_block_takes_largest():
    from mohh.llamea.parsing import extract_code_block

    text = (
        "```python\nx = 1\n```\nmas texto\n"
        "```python\ndef crossover(x, y):\n    return x\n```"
    )
    assert "def crossover" in extract_code_block(text)


def test_repair_code_removes_main_guard_and_closes_brackets():
    from mohh.llamea.parsing import repair_code

    code = "def crossover(x, y):\n    child = x.copy(\n    return child\n\nif __name__ == '__main__':\n    pass\n"
    repaired = repair_code(code)
    assert "__main__" not in repaired
    assert repaired.rstrip().endswith(")")


def test_parse_llm_response_end_to_end():
    from mohh.llamea.parsing import parse_llm_response

    raw = "<think>analizando...</think>\n# Description: OX\n# Code:\n```python\n" + EXAMPLE_CODE + "```\n"
    code = parse_llm_response(raw)
    assert "def crossover" in code
    compile(code, "<test>", "exec")


def test_parse_llm_response_rejects_junk():
    from mohh.llamea.parsing import parse_llm_response

    with pytest.raises(ValueError):
        parse_llm_response("no hay codigo aqui, solo prosa")


# --------------------------------------------------------------------------- #
# Bucle LLaMEA de extremo a extremo (LLM simulado, sin red)
# --------------------------------------------------------------------------- #

class _ScriptedLLM:
    """LLM simulado que devuelve una respuesta de crossover valida."""

    def __init__(self, response):
        self.model = "scripted"
        self.response = response

    def set_logger(self, logger):
        self.logger = logger

    def sample_solution(self, session_messages, parent_ids=None, HPO=False,
                        base_code=None, diff_mode=False):
        from llamea.solution import Solution
        import re

        code = re.search(r"```python\n(.*?)\n```", self.response, re.DOTALL).group(1)
        name = re.findall(r"(?:def|class)\s*(\w*).*\:", code)[0]
        return Solution(name=name, description="scripted", code=code)


def test_llamea_end_to_end_with_scripted_llm(tmp_path, monkeypatch):
    from llamea import LLaMEA

    monkeypatch.chdir(tmp_path)
    response = "# Description: OX crossover\n# Code:\n```python\n" + EXAMPLE_CODE + "\n```\n"
    llm = _ScriptedLLM(response)

    fitness = build_evaluator([INSTANCE], mo_generations=2, mo_population_size=6, seed=1)

    llamea = LLaMEA(
        f=fitness,
        llm=llm,
        n_parents=1,
        n_offspring=1,
        role_prompt="role",
        task_prompt="task",
        budget=2,
        elitism=True,
        minimization=False,
        max_workers=1,
        experiment_name="e2e",
    )
    best = llamea.run()
    assert best.code.strip()
    assert np.isfinite(best.fitness)
    assert best.fitness > 0


# --------------------------------------------------------------------------- #
# Scripts SLURM de la integracion
# --------------------------------------------------------------------------- #

def test_llamea_scripts_exist_and_are_valid():
    slurm_dir = os.path.join(os.path.dirname(__file__), "..", "slurm")
    for script in ("llamea-vllm-server.slrm", "llamea-generate.slrm"):
        path = os.path.join(slurm_dir, script)
        assert os.path.isfile(path), f"missing {script}"
        result = subprocess.run(["bash", "-n", path], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        content = open(path).read()
        assert content.count("#SBATCH") >= 5
        assert "#SBATCH -p" in content

    server = open(os.path.join(slurm_dir, "llamea-vllm-server.slrm")).read()
    assert "#SBATCH -p gpu" in server
    generate = open(os.path.join(slurm_dir, "llamea-generate.slrm")).read()
    assert "LLAMEA_SERVER_JOBID" in generate

    chain = os.path.join(slurm_dir, "llamea-full-chain.sh")
    assert os.path.isfile(chain)
    result = subprocess.run(["bash", "-n", chain], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
