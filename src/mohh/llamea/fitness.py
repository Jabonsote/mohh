"""Fitness de LLaMEA: evalua un operador de cruce generado por el LLM.

Reutiliza la logica actual de MOHH-R: el mismo `MOSolver` interno (MOEA/D,
NSGA-II o SMS-EMOA de pymoo) y la misma metrica de hipervolumen que usa la
busqueda gramatical. La unica diferencia es que el operador candidato ya no
se decodifica de una gramatica: es una funcion `crossover(x, y)` escrita por
el LLM, inyectada con `LLMOperator` (ver `mohh.llamea.operator`).

El fitness es el hipervolumen medio sobre las instancias (por defecto train).
Es comparable con la generacion GE porque usa la misma funcion objetivo mQAP,
el mismo presupuesto de MOEA y la misma metrica.

El evaluador es una clase (picklable) para que LLaMEA pueda paralelizar la
evaluacion con `max_workers > 1`; cada llamada re-inicializa `Params` de forma
idempotente, de modo que funciona tambien en procesos hijos.
"""

import hashlib
import json
import linecache
import os
import signal
import tempfile
import threading
import traceback
from datetime import datetime

import numpy as np

from mohh.generation.utilities.algorithm.HH_auxiliars import is_permutation
from mohh.llamea.diversity import (
    BehaviorTracker,
    behavior_descriptor,
    behavior_fingerprint,
    family_from_fingerprint,
)

FALLBACK_SCORE = 0.0


def _set_metadata(solution, key, value):
    """`Solution.add_metadata` si existe; no rompe con soluciones de prueba."""
    add = getattr(solution, "add_metadata", None)
    if callable(add):
        add(key, value)


class EarlyStop(BaseException):
    """Parada temprana: el mejor ranking no mejoro en `patience` evaluaciones.

    El evaluador la lanza desde `__call__` y `run.py` la captura para cerrar la
    corrida con el mejor operador guardado (no se pierde nada y no se gastan
    cientos de generaciones repitiendo el mismo anclaje).

    Hereda de `BaseException` a proposito: LLaMEA envuelve la evaluacion en
    `try/except Exception` (llamea.py, `evolve_solution`) y convertiria la
    parada en un simple candidato con score 0. Con `BaseException` la senal
    atraviesa ese `except` y llega a `run.py`. Requiere `max_workers=1` (el
    default); con joblib/loky el transporte de BaseException no es confiable.
    """

# Limites de tiempo para el codigo generado por el LLM. Protegen la busqueda de
# operadores que no terminan (`while True`, recursiones sin corte). Son
# necesarios porque con `max_workers=1` joblib usa un backend secuencial que NO
# aplica el `eval_timeout` de LLaMEA.
OPERATOR_CALL_TIMEOUT = 10.0   # segundos por llamada de validacion (n <= 100)
COMPILE_TIMEOUT = 30.0         # segundos para `exec` del codigo del LLM (código de módulo)
INSTANCE_TIMEOUT = 300.0       # segundos por instancia del MOEA interno (configurable)

#: Nombre de fichero ficticio con el que se compila el codigo del LLM. Se
#: registra en `linecache` para que el traceback muestre la linea exacta que
#: fallo, que es la parte mas util del feedback (ver `_exception_feedback`).
OPERATOR_FILENAME = "<llm-operator>"


class _TimeoutError(Exception):
    """El codigo del LLM excedio el tiempo permitido."""


def _call_with_timeout(func, timeout, *args, **kwargs):
    """Ejecuta `func(*args, **kwargs)` abortando si excede `timeout` segundos.

    Usa `SIGALRM`, disponible solo en el hilo principal de plataformas POSIX.
    Si no se puede usar (otro hilo, otra plataforma o `timeout` no positivo) se
    ejecuta sin limite, como antes de que existiera este guardia.
    """
    usable = (
        timeout is not None
        and timeout > 0
        and hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        return func(*args, **kwargs)

    def _on_alarm(signum, frame):  # pragma: no cover - depende del reloj
        raise _TimeoutError(f"exceeded {timeout} seconds")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def init_params(**overrides):
    """Inicializa el singleton `Params` con los defaults de MOHH-R.

    Reutiliza `mohh.generation.params.params_dict` (los mismos defaults que la
    busqueda gramatical: DATASET, PROBLEM_NAME, etc.) y aplica overrides. Es
    idempotente y deja el entorno consistente para `Instance` y `MOSolver`.
    """
    from mohh.generation.params import params_dict
    from mohh.core.params import Params

    params = Params()
    params.update_params(params_dict)
    if overrides:
        params.update_params(overrides)
    return params


def _describe_child_problem(child, n, x, y):
    """Explica por que `child` no es una permutacion valida, o devuelve None.

    El texto es feedback para el LLM: dice que se esperaba y que se ha
    encontrado (longitud, valores repetidos, huecos sin rellenar, vista de un
    padre). Ver `docs/llamea-prompts/areas-de-mejora.md`.
    """
    try:
        values = np.asarray(child)
    except Exception as e:  # noqa: BLE001 - se reporta como feedback
        return f"it is not array-like ({type(e).__name__}: {e})"

    if values.ndim != 1:
        return (f"expected a 1-D array of length {n}, got shape "
                f"{tuple(values.shape)}")

    length = int(values.shape[0])
    if length != n:
        return f"expected length {n}, got length {length}"

    if np.shares_memory(values, x) or np.shares_memory(values, y):
        return ("it is a view of a parent; the child must be a NEW array "
                "(np.empty_like, np.array(x, copy=True), ...)")

    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            return (f"it has dtype {values.dtype} and contains non-integer "
                    "values; use an integer array with the values 0..n-1")
        values = values.astype(np.int64)

    unique, counts = np.unique(values, return_counts=True)
    in_range = (unique >= 0) & (unique < n)
    out_of_range = unique[~in_range]
    repeated = unique[in_range & (counts > 1)]
    repeated_counts = counts[in_range & (counts > 1)]
    missing = np.setdiff1d(np.arange(n), unique[in_range])

    reasons = []
    if out_of_range.size:
        shown = ", ".join(str(int(v)) for v in out_of_range[:4])
        reasons.append(
            f"{out_of_range.size} value(s) outside range(n) ({shown})"
            + (" - a -1 usually means a slot left unfilled by the copy loop"
               if -1 in out_of_range else "")
        )
    if repeated.size:
        shown = ", ".join(f"{int(v)} appears {int(c)} times"
                          for v, c in zip(repeated[:4], repeated_counts[:4]))
        reasons.append(f"{repeated.size} duplicated value(s) ({shown})")
    if missing.size:
        shown = ", ".join(str(int(v)) for v in missing[:6])
        reasons.append(f"{missing.size} missing value(s) ({shown})")

    if not reasons:
        return None
    return "; ".join(reasons)


def _exception_feedback(n, exc):
    """Feedback de una excepcion lanzada por el operador del LLM.

    Incluye el tipo y el mensaje, y la linea del codigo generado que la
    provoco (p. ej. un `assert` del propio LLM o un indice fuera de rango).
    """
    head = f"crossover(x, y) raised {type(exc).__name__} for n={n}: {exc}"
    frames = [
        frame for frame in traceback.extract_tb(exc.__traceback__)
        if frame.filename == OPERATOR_FILENAME
    ]
    if not frames:
        return head
    details = " | ".join(
        f"line {frame.lineno}: {(frame.line or '').strip()}" for frame in frames[-3:]
    )
    return f"{head}; the failing line is {details}"


def validate_operator(crossover_func):
    """Valida `crossover` en tamanos reales del dominio: `(ok, motivo)`.

    Prueba n = 11, 24, 50 y 100 (el rango de las instancias mQAP) y comprueba:
    que no se modifiquen los padres, que devuelva una permutacion (o la tupla
    `(child1, child2)`), que no sea una vista de un padre y que no lance
    excepciones. `motivo` es el texto que se le da al LLM como feedback; antes
    era un unico mensaje generico que dejaba al modelo adivinando por que su
    operador era invalido (y lo llevaba a repetir el mismo bug).
    """
    rng = np.random.RandomState(0)
    # Tamanos del dominio mQAP (las instancias van de n=11 a n=100). No se usan
    # tamanos muy chicos: algunas primitivas de MOHH-R (p.ej. preserve_segments)
    # no estan definidas para n<3 y no se debe tocar el nucleo. Se incluye 100
    # porque un operador puede funcionar hasta 50 y romper en la instancia mas
    # grande; sin esta prueba el fallo aparecia despues, gastando el MOEA.
    for n in (11, 24, 50, 100):
        x = rng.permutation(n)
        y = rng.permutation(n)
        x_before = x.copy()
        y_before = y.copy()
        try:
            child = _call_with_timeout(crossover_func, OPERATOR_CALL_TIMEOUT, x, y)
        except Exception as e:  # noqa: BLE001 - se reporta como feedback
            return False, _exception_feedback(n, e)

        if not (np.array_equal(x, x_before) and np.array_equal(y, y_before)):
            return False, (
                f"crossover modified its inputs for n={n}: x and y are read-only. "
                "In-place writes such as `x[i] = ...`, `x.sort()`, `x[idx] = ...` or "
                "`np.random.shuffle(x)` corrupt the algorithm's population; build the "
                "child in a NEW array (np.empty_like, np.full, np.array(x, copy=True), ...)"
            )

        if isinstance(child, tuple):
            if len(child) != 2:
                return False, (
                    f"for n={n} crossover returned a tuple of {len(child)} elements; "
                    "return either one permutation or the tuple (child1, child2)"
                )
            for label, candidate in zip(("child1", "child2"), child):
                problem = _describe_child_problem(candidate, n, x, y)
                if problem is not None:
                    return False, (f"for n={n}, {label} is not a permutation of "
                                   f"range({n}): {problem}")
        else:
            problem = _describe_child_problem(child, n, x, y)
            if problem is not None:
                return False, (f"for n={n}, the returned child is not a permutation "
                               f"of range({n}): {problem}")
    return True, None


def _valid_operator(crossover_func) -> bool:
    """True si `crossover(x, y)` devuelve permutacion(es) valida(s).

    Acepta tanto una permutacion como una tupla `(child1, child2)`, igual que
    los primitivos de la gramatica (`order_based`, `map_list`, ...).
    """
    return validate_operator(crossover_func)[0]



#: Numero de pares aleatorios que usa el chequeo de degeneracion.
DEGENERACY_TRIALS = 8


def recombination_novelty(crossover_func, trials=DEGENERACY_TRIALS, seed=1234):
    """Mide cuanto recombina el operador y detecta copias de un padre.

    Aplica `crossover` a pares aleatorios y devuelve `(ok, novelty, reason)`.
    `novelty` es la distancia de Hamming normalizada media al padre mas cercano
    (0 = devuelve siempre un padre, 1 = el hijo no se parece a ninguno).

    `ok=False` cuando el operador es degenerado: en *todos* los intentos
    devuelve un padre sin cambios (o una vista de el). Es el modo de fallo
    tipico de los modelos chicos (`return x`, `np.array(x, copy=False)`):
    pasa la validacion "es una permutacion" pero no recombina nada, y gasta el
    MOEA interno completo sin aportar nada. Ver `docs/llamea-prompts/areas-de-mejora.md`.
    """
    rng = np.random.RandomState(seed)
    distances = []
    for _ in range(trials):
        n = int(rng.choice([11, 16, 24, 50, 100]))
        x = rng.permutation(n)
        y = rng.permutation(n)
        try:
            child = _call_with_timeout(crossover_func, OPERATOR_CALL_TIMEOUT, x, y)
        except Exception as e:  # noqa: BLE001 - se reporta como feedback
            return False, 0.0, (
                f"crossover raised {type(e).__name__} during the novelty check: {e}"
            )
        if isinstance(child, tuple):
            child = child[0]
        try:
            child = np.asarray(child)
        except Exception:  # noqa: BLE001 - no es un array
            return False, 0.0, "crossover did not return an array"
        if child.shape != x.shape:
            return False, 0.0, "crossover returned a child of the wrong length"
        if np.shares_memory(child, x) or np.shares_memory(child, y):
            return False, 0.0, (
                "crossover returned a view of a parent; build a NEW array "
                "(np.empty_like, np.array(x, copy=True), ...)"
            )
        distance = min(int(np.sum(child != x)), int(np.sum(child != y))) / n
        distances.append(distance)

    novelty = float(np.mean(distances)) if distances else 0.0
    if novelty == 0.0:
        return False, 0.0, (
            "degenerate crossover: it returns a parent unchanged for every test "
            "pair; it must recombine information from BOTH parents"
        )
    return True, novelty, None


def primitive_namespace() -> dict:
    """Namespace con los primitivos de permutacion de MOHH-R + ayudas.

    Por un lado expone los mismos bloques que usa la busqueda gramatical (por
    compatibilidad). Por otro, deja `np`/`numpy` y nombres comunes de `typing`
    disponibles: son descuidos muy comunes del LLM (olvidar `import numpy as np`
    o usar `List[int]` sin importar), y tenerlos en el namespace evita rechazar
    operadores correctos por ese detalle (la logica del operador no cambia).
    """
    import numpy as np

    from mohh.core.utilities.algorithm import HH_functions as hh

    # Nombres de `typing` que el LLM usa a menudo en anotaciones sin importarlos
    # (`def crossover(x: List[int], ...)`); evitarlos como error de compilacion.
    import typing

    return {
        "np": np,
        "numpy": np,
        "List": typing.List,
        "Tuple": typing.Tuple,
        "Dict": typing.Dict,
        "Set": typing.Set,
        "Optional": typing.Optional,
        "Sequence": typing.Sequence,
        "Any": typing.Any,
        "Callable": typing.Callable,
        "swap": hh.swap,
        "move_right": hh.move_right,
        "reverse": hh.reverse,
        "map_list": hh.map_list,
        "alternate_elements": hh.alternate_elements,
        "preserve_elements": hh.preserve_elements,
        "preserve_segments": hh.preserve_segments,
        "fill_first_occurring": hh.fill_first_occurring,
        "order_based": hh.order_based,
    }


def _compile_with_source(code: str):
    """Compila el codigo del LLM registrando su fuente en `linecache`.

    Sin ese registro, un traceback de codigo pasado a `exec` no puede mostrar
    la linea que fallo (`linecache.getline` no encuentra el fichero), y el
    feedback al LLM se queda en un `AssertionError` sin contexto.
    """
    source_lines = code.splitlines(keepends=True)
    linecache.cache[OPERATOR_FILENAME] = (
        len(code), None, source_lines, OPERATOR_FILENAME,
    )
    return compile(code, OPERATOR_FILENAME, "exec")


def compile_operator(code: str, timeout=COMPILE_TIMEOUT) -> dict:
    """`exec` del codigo del LLM; devuelve el namespace o lanza ValueError.

    El namespace incluye los primitivos de permutacion de MOHH-R, de modo que
    el operador generado puede reutilizar la logica actual. Debe existir
    `crossover(x, y)` y devolver permutacion(es) valida(s).

    Los mensajes de error son el feedback que recibe el LLM, asi que dicen que
    tamano fallo, que se encontro en lugar de una permutacion y, si el codigo
    lanzo una excepcion, la linea que la provoco.

    El `exec` va acotado por `timeout`: el LLM suele dejar codigo de modulo
    ("self-check" que llama a `crossover` al importar) y un bucle infinito ahi
    colgaba la corrida entera (paso en
    `results/searches/qwen25coder7b_moead_b30_st_r3`, ver
    `docs/llamea-prompts/areas-de-mejora.md` §14).
    """
    namespace = primitive_namespace()
    try:
        compiled = _compile_with_source(code)
    except SyntaxError as e:
        raise ValueError(
            f"Code did not compile: {type(e).__name__} on line {e.lineno}: {e.msg}"
        ) from e
    try:
        _call_with_timeout(exec, timeout, compiled, namespace)
    except _TimeoutError as e:
        raise ValueError(
            "Code did not finish running its module-level statements "
            f"(they execute when the module is loaded, before any evaluation): {e}. "
            "Remove any example/self-check that calls crossover at import time."
        ) from e
    except Exception as e:  # noqa: BLE001 - se reporta como feedback al LLM
        raise ValueError(f"Code did not execute: {type(e).__name__}: {e}") from e

    crossover = namespace.get("crossover")
    if not callable(crossover):
        raise ValueError("Code must define a callable 'crossover(x, y)' function.")

    ok, reason = validate_operator(crossover)
    if not ok:
        raise ValueError(f"Invalid crossover operator: {reason}.")

    return namespace


#: Punto de nadir fijo sobre frentes normalizados, igual que el proyecto
#: original (`mohh/evaluation/experiment.py` usa `[1.1] * n_obj`).
NADIR_FACTOR = 1.1


def normalize_front(front, bounds):
    """Normaliza un frente con sus limites (vmin, vmax) por objetivo.

    Formula **identica** a `normalize_values` del proyecto original
    (`mohh/evaluation/experiment.py`): ``(f - vmin) / (vmax - vmin)``. Si un
    objetivo es degenerado (``vmax == vmin``) no hay escala y el resultado es
    NaN, igual que en el original (alli la instancia queda invalidada). Quien
    llama debe descartar esas instancias *antes* de normalizar; ver
    `OperatorEvaluator._hv_values`.
    """
    vmin = np.asarray(bounds[0], dtype=float)
    vmax = np.asarray(bounds[1], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.asarray(front, dtype=float) - vmin) / (vmax - vmin)


def _consolidated_frame(*point_sets):
    """Frente no dominado de la union de varios conjuntos de puntos.

    Reproduce el `consolidated[label]` del proyecto original: el marco de
    normalizacion de una instancia es la envolvente no dominada de **todo** lo
    evaluado en ella. Devuelve `None` si no hay puntos.
    """
    from mohh.core.utilities.algorithm.MO import non_dominated_sorting_vectorized

    points = [np.asarray(points, dtype=float) for points in point_sets
              if points is not None and len(points)]
    if not points:
        return None
    stacked = np.vstack(points)
    # Como el `set` del proyecto original: los puntos repetidos no cuentan.
    stacked = np.unique(stacked, axis=0)
    front = np.asarray(non_dominated_sorting_vectorized(stacked)[0], dtype=float)
    return front


def _same_points(left, right):
    """True si dos conjuntos de puntos coinciden (independiente del orden)."""
    if left is None or right is None:
        return left is right
    if left.shape != right.shape:
        return False
    order_left = np.lexsort(left.T[::-1])
    order_right = np.lexsort(right.T[::-1])
    return np.array_equal(left[order_left], right[order_right])


class HVArchive:
    """Archivo de evaluaciones para calcular el ranking (1 = mejor).

    Cada evaluacion es una entrada `(clave, novedad)` y el valor de HV se
    consulta al calcular el ranking. En el marco GE esto importa: cuando el
    marco de normalizacion cambia (porque un candidato amplia la envolvente),
    todos los candidatos se reordenan con su valor actualizado. En el marco
    estacionario cada evaluacion usa una clave propia y su valor queda fijo,
    que es el comportamiento historico.
    """

    def __init__(self):
        self._entries = []   # lista de (clave, novedad)
        self._values = {}    # clave -> HV actual
        self._lock = threading.Lock()

    def __getstate__(self):
        """El lock no se puede picklear; se conservan entradas y valores."""
        return {"_entries": list(self._entries), "_values": dict(self._values)}

    def __setstate__(self, state):
        entries = state.get("_entries")
        if entries is None:
            # Estado legado: `_values` era una lista de (valor, novedad) o floats.
            entries, values = [], {}
            for index, item in enumerate(state.get("_values", [])):
                value, novelty = item if isinstance(item, tuple) else (float(item), 0.0)
                key = f"legacy-{index}"
                entries.append((key, float(novelty)))
                values[key] = float(value)
        else:
            values = dict(state.get("_values", {}))
        self._entries = [(str(key), float(novelty)) for key, novelty in entries]
        self._values = {str(key): float(value) for key, value in values.items()}
        self._lock = threading.Lock()

    def add(self, value: float, novelty: float = 0.0, key=None) -> int:
        """Registra un HV (y su novedad) y devuelve su ranking (1 = mejor).

        Los empates en HV se desempatan por novedad de recombinacion: un
        operador que de verdad mezcla a los padres queda por delante de otro
        que solo los copia y casualmente obtiene el mismo hipervolumen.

        Con `key` (marco GE) varias evaluaciones del mismo codigo comparten
        valor: `update` lo refresca para todas ellas a la vez.
        """
        key = f"ev-{len(self._entries)}" if key is None else str(key)
        with self._lock:
            self._entries.append((key, float(novelty)))
            self._values[key] = float(value)
            return self._rank_locked(key)

    def update(self, key, value: float, novelty=None) -> None:
        """Refresca el HV (y opcionalmente la novedad) de todas las entradas de `key`."""
        key = str(key)
        with self._lock:
            if key not in self._values:
                return
            self._values[key] = float(value)
            if novelty is not None:
                self._entries = [
                    (entry_key, float(novelty) if entry_key == key else entry_novelty)
                    for entry_key, entry_novelty in self._entries
                ]

    def rank(self, key) -> int:
        with self._lock:
            return self._rank_locked(str(key))

    def _rank_locked(self, key: str) -> int:
        value = self._values[key]
        novelty = next(
            (novelty for entry_key, novelty in reversed(self._entries) if entry_key == key), 0.0
        )
        better = sum(
            1 for entry_key, entry_novelty in self._entries
            if self._values.get(entry_key, float("-inf")) > value
            or (self._values.get(entry_key, float("-inf")) == value and entry_novelty > novelty)
        )
        return better + 1

    def best(self) -> float:
        with self._lock:
            return max(self._values.values(), default=float("nan"))

    def __len__(self) -> int:
        return len(self._entries)


class OperatorEvaluator:
    """Fitness picklable que evalua operadores generados por el LLM.

    Reutiliza la logica interna de MOHH-R (MOSolver + hipervolumen) sobre las
    instancias indicadas (train durante la busqueda; test en la verificacion
    final). Crea las instancias de forma perezosa y determinista.

    El hipervolumen se calcula **igual que en la evaluacion gramatical**
    (`mohh/evaluation/experiment.py`): por instancia se toma el frente no
    dominado consolidado de todo lo evaluado como marco de normalizacion
    (`vmin`/`vmax` por objetivo), cada frente se normaliza con
    `normalize_front` y se integra hasta el nadir fijo `[1.1] * K`. En el
    proyecto original ese marco son *todas* las configuraciones evaluadas; aqui
    son los operadores de referencia (`mohh.llamea.reference`) mas todos los
    candidatos ya evaluados (`frame="ge"`, por defecto). Asi los HV de todos
    los candidatos y de la linea base clasica son comparables entre si y con
    los de la busqueda GE, y un candidato que supera el panel lo supera tambien
    en la tabla (no en una escala abierta).

    Con `frame="stationary"` se usa solo el panel de referencia como marco
    (comportamiento anterior): los HV quedan fijos, pero pueden superar
    ampliamente los del panel y no son comparables con la tabla GE.

    Cachea el frente por instancia de cada codigo: si el LLM repite un operador
    no se vuelve a correr el MOEA interno, y si el marco cambia se recalculan
    los HV de todos los candidatos a partir de esos frentes (sin MOEA).
    """

    def __init__(self, instance_paths, mo_generations, mo_population_size,
                 seed=42, model_name="MOEAD", mutation=False,
                 instance_timeout=INSTANCE_TIMEOUT, normalize=True,
                 bounds=None, reference_operators=None, archive=None,
                 frame="ge", fronts_dir=None, repeat_penalty=0.5, patience=0,
                 diversity_feedback=True, diversity_log=None,
                 target_penalty=1.0, require_target_family=False):
        self.instance_paths = [str(p) for p in instance_paths]
        self.mo_generations = mo_generations
        self.mo_population_size = mo_population_size
        self.seed = seed
        self.model_name = model_name
        self.mutation = mutation
        self.instance_timeout = instance_timeout
        self.normalize = normalize
        self.bounds = bounds
        self.reference_operators = reference_operators
        self.archive = archive if archive is not None else HVArchive()
        self.frame = frame if normalize else "none"
        # Presion de diversidad: un candidato identico a uno ya evaluado recibe
        # su HV multiplicado por `repeat_penalty` **solo para el ranking** (el HV
        # crudo se conserva para el reporte). `1.0` desactiva la penalizacion.
        self.repeat_penalty = float(repeat_penalty)
        # Parada temprana: si el mejor ranking no mejora en `patience`
        # evaluaciones seguidas, se lanza `EarlyStop` y `run.py` cierra con el
        # mejor operador guardado (evita gastar 1000 generaciones en un anclaje).
        self.patience = int(patience)
        # Validacion contra la familia conductual objetivo del combine. El HV
        # crudo nunca se altera: `target_penalty` solo afecta al ranking que ve
        # LLaMEA (como `repeat_penalty`), y `require_target_family` permite
        # rechazar el hijo antes de gastar el MOEA.
        self.target_penalty = float(target_penalty)
        self.require_target_family = bool(require_target_family)
        self._target_attempts = 0
        self._target_hits = 0
        self._best_ranking = None
        self._best_solution = None
        self._since_improvement = 0
        self._first_seen = {}            # clave -> generacion de la primera vez
        self._duplicates = 0             # repeticiones exactas evaluadas
        # Diversidad conductual (no textual): huella de lo que hace el operador
        # sobre pares de padres fijos, novedad, familias y archivo MAP-Elites.
        # `diversity_feedback` agrega el diagnostico al feedback que ve el LLM.
        self.diversity = BehaviorTracker()
        self.diversity_feedback = bool(diversity_feedback)
        self._evaluations = 0            # llamadas a __call__ (incluye invalidas)
        self._valid_evaluations = 0      # llamadas con HV definido
        # Frentes crudos por candidato (evidencia de la evaluacion). Si es None
        # no se guarda nada (`--no-save-fronts`).
        self.fronts_dir = fronts_dir
        if fronts_dir:
            os.makedirs(fronts_dir, exist_ok=True)
        # `diversity.jsonl` es una traza independiente del manifest de frentes.
        # `run.py` pasa `exp_dir/diversity.jsonl` para conservarla incluso con
        # `--no-save-fronts`; si no se indica, se deriva del directorio de
        # frentes (compatibilidad con llamadas antiguas).
        self.diversity_log = diversity_log or (
            os.path.join(os.path.dirname(os.path.abspath(fronts_dir)),
                         "diversity.jsonl")
            if fronts_dir else None
        )
        if self.diversity_log:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.diversity_log)), exist_ok=True
            )
        self._diversity_lock = threading.Lock()
        self._fronts_lock = threading.Lock()
        self._cache = {}
        self._instances = None
        self._init_dir = tempfile.mkdtemp(prefix="llamea_init_")
        # Marco GE: frentes de referencia por instancia y envolvente actual.
        self._reference_fronts = None    # lista por instancia: {operador: frente}
        self._reference_points = None    # lista por instancia: frente consolidado
        self._frame_points = None        # lista por instancia: envolvente actual
        self._scores = {}                # clave -> (score, per_instance, degeneradas)

    # -- infraestructura -------------------------------------------------- #
    def __getstate__(self):
        """Al picklear (joblib, `max_workers > 1`) no se envian las instancias.

        Se recrean de forma perezosa en el proceso hijo. Asi el evaluador es
        siempre picklable aunque el padre ya haya construido sus instancias.
        """
        state = self.__dict__.copy()
        state["_instances"] = None
        state.pop("_diversity_lock", None)
        state.pop("_fronts_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._diversity_lock = threading.Lock()
        self._fronts_lock = threading.Lock()

    def _params_overrides(self):
        return {
            "RANDOM_SEED": self.seed,
            "MO_MODEL": self.model_name,
            "MO_GENERATIONS": self.mo_generations,
            "MO_POPULATION_SIZE": self.mo_population_size,
            "MO_CROSSOVER_PROBABILITY": 1.0,
            "MO_MUTATION_BOOL": self.mutation,
            "NUM_NEGIHBORS": 10,
            "PROB_NEIGHBORS": 0.9,
            "FILE_PATH_INITIAL_SOLUTIONS": self._init_dir,
        }

    def _ensure_instances(self):
        if self._instances is not None:
            return self._instances

        from mohh.generation.problem.instance import Instance

        init_params(**self._params_overrides())
        # Semilla determinista para las soluciones iniciales de cada instancia.
        np.random.seed(self.seed)

        instances = [Instance(self.mo_population_size, "Natural")
                     for _ in self.instance_paths]
        for inst, path in zip(instances, self.instance_paths):
            inst.load_problem("QAP", path)

        self._instances = instances
        return instances

    # -- evaluacion de un operador sobre una instancia -------------------- #
    def _solve_front(self, instance, namespace):
        """Corre el MOEA interno con `namespace['crossover']` y devuelve el frente."""
        from mohh.generation.algorithm.multiobjective import MOSolver
        from mohh.llamea.operator import LLMOperator

        # Semilla determinista por corrida (igual que el worker GE).
        np.random.seed(self.seed)
        solver = MOSolver(seed=self.seed)
        # El operador del LLM se inyecta directamente en el solver; la ruta GE
        # (load_operator -> HH_Operator) no se toca.
        solver.crossover = LLMOperator(namespace, prob=1.0)
        # Mutacion: el nucleo GE tiene `with_mutation` comentado y el arnes
        # siempre cargaba `NullMutation`, asi que `--mutation` era un no-op.
        # Ahora `--mutation` carga la mutacion de permutacion que si usa la
        # evaluacion GE (`Swap_Mutation`) con la probabilidad de cada solver,
        # sin tocar el nucleo (se inyecta por el mismo `load_operator`).
        if self.mutation:
            # `Swap_Mutation` (la mutacion de permutacion que usa la evaluacion
            # GE) vive en `mohh.evaluation.operators.mutation`; se inyecta por
            # atributo igual que el cruce, sin tocar el nucleo.
            from mohh.evaluation.operators.mutation import Swap_Mutation

            key = {
                "MOEAD": "MOEAD_MUTATION_PROBABILITY",
                "NSGAII": "NSGA_MUTATION_PROBABILITY",
                "SMSEMOA": "SMS_MUTATION_PROBABILITY",
            }.get(self.model_name, "MOEAD_MUTATION_PROBABILITY")
            from mohh.evaluation.params import mo_params

            prob = mo_params.get(key, 0.05)
            solver.mutation = Swap_Mutation(prob)
        else:
            solver.load_operator("mutation", "NullMutation")
        solver.load_instance(instance)
        solver.start_model(model_name=self.model_name)
        _, pareto_front, *_ = _call_with_timeout(
            solver.solve_instance, self.instance_timeout
        )
        return np.asarray(pareto_front, dtype=float)

    def _bounds_for(self, index):
        """Limites de normalizacion de la instancia `index` (o None)."""
        if self.bounds is None:
            return None
        if isinstance(self.bounds, dict):
            key = os.path.basename(self.instance_paths[index])
            return self.bounds.get(key)
        return self.bounds[index]

    def ensure_reference_bounds(self):
        """Construye (una vez) el marco de normalizacion por instancia.

        Evalua el panel de operadores de referencia sobre cada instancia, une
        sus frentes, toma el frente no dominado consolidado y de ahi los
        limites (min/max por objetivo). Es el mismo procedimiento que
        `experiment.py` del proyecto original con el frente consolidado de todas
        las configuraciones evaluadas; en el bucle LLaMEA el panel es el unico
        conjunto conocido de antemano (marco estacionario) y sirve de semilla
        del marco GE, que ademas incorpora los candidatos ya evaluados.
        """
        if not self.normalize or self.bounds is not None:
            self._seed_frame_from_bounds()
            return self.bounds

        from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for

        operators = self.reference_operators or REFERENCE_OPERATORS
        instances = self._ensure_instances()
        init_params(**self._params_overrides())

        bounds = []
        reference_fronts = []
        frame_points = []
        for inst in instances:
            fronts = {}
            for name, func in operators.items():
                try:
                    front = self._solve_front(inst, namespace_for(func))
                except Exception:  # noqa: BLE001 - una referencia que falle no corta
                    continue
                if front is not None and len(front):
                    fronts[name] = front
            consolidated = _consolidated_frame(*fronts.values())
            if consolidated is not None:
                bounds.append((consolidated.min(axis=0), consolidated.max(axis=0)))
            else:
                k = inst.k_objectives
                bounds.append((np.zeros(k), np.ones(k)))
            reference_fronts.append(fronts)
            frame_points.append(consolidated)
        self.bounds = bounds
        self._reference_fronts = reference_fronts
        self._reference_points = frame_points
        self._frame_points = [None if f is None else f.copy() for f in frame_points]
        return bounds

    def _seed_frame_from_bounds(self):
        """Semilla del marco a partir de limites ya calculados (`--bounds-file`).

        Se usan las K esquinas de la caja `(vmin, vmax)` (cada una con el maximo
        en un objetivo y el minimo en los demas): su envolvente no dominada
        reproduce exactamente la caja, asi que la semilla del marco es la misma
        que la del panel guardado y los candidatos solo la amplian si mejoran
        esos valores. Se calcula una sola vez.
        """
        if not self.normalize or self._frame_points is not None:
            return
        self._reference_fronts = [{} for _ in self.instance_paths]
        self._reference_points = []
        frame_points = []
        for index in range(len(self.instance_paths)):
            bounds = self._bounds_for(index)
            if bounds is None:
                frame_points.append(None)
                self._reference_points.append(None)
                continue
            vmin = np.asarray(bounds[0], dtype=float)
            vmax = np.asarray(bounds[1], dtype=float)
            corners = np.tile(vmin, (len(vmin), 1))
            for objective in range(len(vmin)):
                corners[objective, objective] = vmax[objective]
            consolidated = _consolidated_frame(corners)
            frame_points.append(consolidated)
            self._reference_points.append(consolidated)
        self._frame_points = frame_points

    # -- marco GE --------------------------------------------------------- #
    def _hv_values(self, fronts):
        """HV por instancia en el marco actual (None en instancias sin escala).

        Reproduce el calculo del proyecto original: normalizar el frente con
        los `vmin`/`vmax` del marco comun de esa instancia y calcular el
        hipervolumen hasta el nadir fijo `[1.1] * K`. Si el marco es degenerado
        (`vmax == vmin` en algun objetivo) la instancia no tiene escala y se
        excluye, igual que el original la invalida con NaN.
        """
        from mohh.core.utilities.algorithm.MO import compute_hypervolume

        if self._frame_points is None:
            self.ensure_reference_bounds()

        values = []
        for index, front in enumerate(fronts):
            frame = self._frame_points[index] if self._frame_points else None
            if frame is None or len(frame) == 0 or front is None or len(front) == 0:
                values.append(None)
                continue
            vmin = frame.min(axis=0)
            vmax = frame.max(axis=0)
            if np.any(vmax == vmin):
                values.append(None)
                continue
            normalized = normalize_front(front, (vmin, vmax))
            if not np.all(np.isfinite(normalized)):  # pragma: no cover - salvaguarda
                values.append(None)
                continue
            values.append(float(compute_hypervolume(
                np.full(normalized.shape[1], NADIR_FACTOR), normalized
            )))
        return values

    def _mean_hv(self, values):
        """Media de los HV por instancia; None si ninguna instancia es puntuable."""
        valid = [value for value in values if value is not None]
        return float(np.mean(valid)) if valid else None

    def _degenerate_instances(self, values):
        return [index for index, value in enumerate(values) if value is None]

    def _refresh_frame(self, fronts):
        """Amplia la envolvente (marco GE) con el frente de un candidato.

        Devuelve True si el marco cambio, lo que obliga a recalcular los HV de
        todos los candidatos ya evaluados.
        """
        if self.frame != "ge":
            return False
        if self._frame_points is None:
            self.ensure_reference_bounds()
        changed = False
        for index, front in enumerate(fronts):
            if front is None or not len(front):
                continue
            current = self._frame_points[index]
            merged = _consolidated_frame(current, front)
            if merged is not None and not _same_points(merged, current):
                self._frame_points[index] = merged
                changed = True
        return changed

    def _recompute_scores(self):
        """Recalcula el HV de todos los candidatos cacheados en el marco actual."""
        for key, entry in self._cache.items():
            fronts = entry.get("fronts")
            if entry.get("error") is not None or fronts is None:
                continue
            values = self._hv_values(fronts)
            self._scores[key] = (self._mean_hv(values), values,
                                 self._degenerate_instances(values))
            if self._scores[key][0] is not None:
                self.archive.update(key, self._scores[key][0])

    # -- informes --------------------------------------------------------- #
    def final_scores(self):
        """HV por candidato en el marco final: `{codigo: (score, por_instancia)}`."""
        return {key: (score, values) for key, (score, values, _) in self._scores.items()}

    def duplicate_stats(self):
        """Estadisticas de diversidad: texto, comportamiento y repeticiones.

        `distinct_evaluated` cuenta codigos distintos; `unique_behavior` cuenta
        mecanismos distintos (con un umbral de distancia conductual). La
        diferencia entre ambos es la senal de anclaje: muchos codigos nuevos que
        hacen lo mismo. `validity_rate` usa todas las evaluaciones (incluidas las
        invalidas) como denominador; antes se dividia por el presupuesto, lo que
        podia dar >100% cuando el MOEA evalua varios individuos por generacion.
        """
        stats = {
            "distinct_evaluated": len(self._cache),
            "duplicate_evaluations": self._duplicates,
            "repeat_penalty": self.repeat_penalty,
            "patience": self.patience,
            "since_improvement": self._since_improvement,
            "evaluations": self._evaluations,
            "valid_evaluations": self._valid_evaluations,
            "validity_rate": (
                self._valid_evaluations / self._evaluations
                if self._evaluations else 0.0
            ),
            "target_family_attempts": self._target_attempts,
            "target_family_hits": self._target_hits,
            "target_family_hit_rate": (
                self._target_hits / self._target_attempts
                if self._target_attempts else 0.0
            ),
            "target_penalty": self.target_penalty,
            "require_target_family": self.require_target_family,
        }
        stats.update(self.diversity.snapshot())
        return stats

    def diversity_snapshot(self):
        """Resumen de diversidad conductual (para logger y reporte)."""
        return self.diversity.snapshot()

    def diversity_entries(self):
        """Entradas de diversidad serializables (fingerprints incluidos)."""
        return self.diversity.entries_snapshot()

    @staticmethod
    def _target_family(solution):
        getter = getattr(solution, "get_metadata", None)
        if not callable(getter):
            return None
        try:
            value = getter("target_family")
        except Exception:  # noqa: BLE001 - metadata opcional
            return None
        return str(value) if value else None

    @staticmethod
    def _target_note(target_family, observed_family, hit):
        """Texto accionable para el LLM tras validar la familia objetivo."""
        if not target_family:
            return ""
        if observed_family is None:
            return (
                f" Target family check: expected {target_family}, but the "
                "behavior could not be measured (the fixed-pair probe failed)."
            )
        if hit:
            return (
                f" Target family check: PASS (expected {target_family}, observed "
                f"{observed_family}). Keep this mechanism and improve its HV."
            )
        return (
            f" Target family check: MISS (expected {target_family}, observed "
            f"{observed_family}). The child did not behave as requested; change "
            "the recombination mechanism (not just the syntax) so the measured "
            "family matches the target."
        )

    def _behavioral_feedback(self, info, entry):
        """Diagnostico conductual accionable para el LLM (reflexion tipo ReEvo)."""
        if not self.diversity_feedback:
            return ""
        fingerprint = entry.get("fingerprint")
        if fingerprint is None or not info:
            return (
                " Behavioral profile unavailable: the operator passed the basic "
                "checks but failed the fixed-pair behavior probe. Keep it simple "
                "and deterministic."
            )

        recombination, structure = behavior_descriptor(fingerprint)
        family = info.get("family", "mixed")
        closest = info.get("closest_reference")
        closest_distance = info.get("closest_distance", 0.0)
        novelty = float(info.get("novelty", 0.0))
        coverage = float(info.get("coverage", 0.0))
        hint = info.get("hint", "")
        note = (
            f" Behavioral profile: family={family}, closest classic={closest} "
            f"(distance {closest_distance:.2f}), recombination={recombination:.2f}, "
            f"structure preservation={structure:.2f}."
        )
        if info.get("cached"):
            note += (
                " This is the same code as a previous candidate; changing the "
                "mechanism is the only way to get a new behavior."
            )
        elif novelty < self.diversity.threshold:
            nearest = self.diversity.nearest(info.get("key"))
            where = ""
            if nearest:
                where = (
                    f" (generation {nearest.get('generation')}, family "
                    f"{nearest.get('family')}, description "
                    f"'{nearest.get('description')[:80]}')"
                )
            note += (
                f" BEHAVIORALLY REDUNDANT: novelty {novelty:.3f} is below "
                f"{self.diversity.threshold:.2f}; this behaves almost exactly like "
                f"a previously evaluated operator{where}. A syntactic rewrite will "
                f"not help: change the recombination mechanism."
            )
        else:
            note += (
                f" Behavioral novelty: {novelty:.3f} (0 = same behavior as an "
                f"existing operator, 1 = unlike any tried)."
            )
        note += f" Behavior archive coverage: {coverage:.0%}."
        if hint:
            note += f" Under-explored direction: {hint}."
        return note

    def _log_diversity(self, key, info, entry, score, target_family=None,
                       observed_family=None, target_hit=None):
        """Traza durable por operador, aunque la corrida se corte a mitad."""
        if not self.diversity_log:
            return
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "key_sha1": hashlib.sha1(key.encode("utf-8")).hexdigest()[:12],
            "generation": info.get("generation"),
            "family": info.get("family"),
            "closest_reference": info.get("closest_reference"),
            "closest_distance": info.get("closest_distance"),
            "novelty": info.get("novelty"),
            "cell": list(info.get("cell", ())),
            "coverage": info.get("coverage"),
            "fitness": float(score),
            "fingerprint": [float(value) for value in entry["fingerprint"]],
            "description": info.get("description", ""),
            "target_family": target_family,
            "observed_family": observed_family,
            "target_family_hit": target_hit,
        }
        with self._diversity_lock:
            with open(self.diversity_log, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def best_solution(self):
        """El mejor `Solution` visto (para cerrar tras una parada temprana)."""
        return self._best_solution

    def reference_scores(self):
        """HV de cada operador de referencia en el marco final (tabla comparable).

        Es la comparacion justa: el mejor candidato y la linea base clasica
        medidos en el mismo marco, como en la tabla de la busqueda GE.
        """
        if self._reference_fronts is None:
            self.ensure_reference_bounds()
        scores = {}
        for index, fronts in enumerate(self._reference_fronts or []):
            for name, front in fronts.items():
                values = scores.setdefault(name, [None] * len(self.instance_paths))
                frame = self._frame_points[index] if self._frame_points else None
                if frame is None or len(frame) == 0:
                    continue
                vmin = frame.min(axis=0)
                vmax = frame.max(axis=0)
                if np.any(vmax == vmin):
                    continue
                from mohh.core.utilities.algorithm.MO import compute_hypervolume

                normalized = normalize_front(front, (vmin, vmax))
                values[index] = float(compute_hypervolume(
                    np.full(normalized.shape[1], NADIR_FACTOR), normalized
                ))
        return {name: self._mean_hv(values) for name, values in scores.items()}

    def frame_degenerate_instances(self):
        """Instancias del marco final sin escala (`vmin == vmax`)."""
        if self._frame_points is None:
            return []
        names = []
        for index, frame in enumerate(self._frame_points):
            if frame is None or len(frame) == 0 or np.any(frame.max(axis=0) == frame.min(axis=0)):
                names.append(os.path.basename(self.instance_paths[index]))
        return names

    def best_key_in_final_frame(self):
        """Clave (codigo) del mejor candidato en el marco final."""
        scored = [(score, key) for key, (score, _) in self.final_scores().items()
                  if score is not None]
        if not scored:
            return None
        return max(scored)[1]


    # -- fitness ---------------------------------------------------------- #
    def _track_best(self, ranking_score, solution):
        """Actualiza el mejor-hasta-ahora y aplica la parada por paciencia.

        Se llama en TODOS los caminos de evaluacion (exito, codigo invalido y
        HV indefinido) para que `--patience` cuente tambien las generaciones
        que no producen nada util.
        """
        if self._best_ranking is None or ranking_score > self._best_ranking + 1e-12:
            self._best_ranking = ranking_score
            self._best_solution = solution
            self._since_improvement = 0
        else:
            self._since_improvement += 1
            if self.patience and self._since_improvement >= self.patience:
                raise EarlyStop(
                    f"sin mejora del mejor HV en {self.patience} evaluaciones "
                    f"(mejor={self._best_ranking:.6f})"
                )

    def __call__(self, solution, logger=None):
        from mohh.core.utilities.algorithm.MO import compute_hypervolume

        self._evaluations += 1
        # Cache por codigo: en corridas largas el LLM repite el mismo operador
        # muchas veces (p. ej. el OX del ejemplo). Reutilizar los frentes
        # ahorra todo el MOEA interno sin cambiar el resultado.
        key = (solution.code or "").strip()
        target_family = self._target_family(solution)
        entry = self._cache.get(key)
        cached_hit = entry is not None

        if entry is None:
            entry = self._evaluate_code(solution, compute_hypervolume)
            self._cache[key] = entry
            self._save_candidate_fronts(key, entry, solution)

        if entry["error"] is not None:
            message = entry["error"]
            if cached_hit:
                message = f"(reused) {message}"
            observed_family = entry.get("observed_family")
            target_hit = entry.get("target_hit")
            if target_family:
                self._target_attempts += 1
                fingerprint = entry.get("fingerprint")
                if target_hit is None and fingerprint is not None:
                    observed_family = family_from_fingerprint(fingerprint)
                    target_hit = observed_family == target_family
                self._target_hits += int(bool(target_hit))
                _set_metadata(solution, "target_family", target_family)
                _set_metadata(solution, "target_family_hit", target_hit)
                if observed_family is not None:
                    _set_metadata(solution, "target_family_observed", observed_family)
                message += self._target_note(target_family, observed_family,
                                             target_hit)
                # Dejar evidencia tambien cuando el hijo se rechaza antes del
                # MOEA: el `diversity.jsonl` debe explicar por que no entro.
                if entry.get("fingerprint") is not None:
                    tracker = getattr(self, "diversity", None)
                    self._log_diversity(
                        key,
                        {
                            "generation": getattr(solution, "generation", None),
                            "family": observed_family,
                            "closest_reference": None,
                            "closest_distance": None,
                            "novelty": entry.get("novelty"),
                            "cell": (),
                            "coverage": getattr(tracker, "coverage", None),
                            "description": "",
                        },
                        entry,
                        FALLBACK_SCORE,
                        target_family,
                        observed_family,
                        target_hit,
                    )
            error = entry.get("exception")
            if error is None:
                error = ValueError(message)
            solution.set_scores(FALLBACK_SCORE, message, error)
            self._track_best(FALLBACK_SCORE, solution)
            return solution

        # Marco GE: el candidato puede ampliar la envolvente de la instancia, en
        # cuyo caso los HV de todos los candidatos ya evaluados se recalculan
        # (solo hipervolumen: los frentes estan cacheados).
        if not self.normalize:
            score = entry["score"]
            per_instance = entry["per_instance"]
            degenerate = []
        else:
            if self.frame == "ge" and self._refresh_frame(entry["fronts"]):
                self._recompute_scores()
            if key not in self._scores:
                values = self._hv_values(entry["fronts"])
                self._scores[key] = (self._mean_hv(values), values,
                                     self._degenerate_instances(values))
            score, per_instance, degenerate = self._scores[key]
        novelty = float(entry.get("novelty", 0.0))

        if score is None:
            reason = (
                "no instance has a usable normalization frame: the consolidated "
                "reference front collapsed (vmin == vmax) on every instance, so "
                "hypervolume is undefined (the original pipeline invalidates "
                "those instances as NaN)"
            )
            solution.set_scores(FALLBACK_SCORE, reason, ValueError(reason))
            self._track_best(FALLBACK_SCORE, solution)
            return solution

        self._valid_evaluations += 1

        # Presion de diversidad (no de prompt): si el codigo ya se habia
        # evaluado, se penaliza **solo el ranking** (el HV crudo se conserva
        # para el reporte) y el feedback lo dice de forma accionable. Es la
        # respuesta directa a la evidencia del anclaje: en la corrida b200 el
        # mismo operador se repitio 108 veces pese a ir por debajo de PBX.
        generation = getattr(solution, "generation", None)
        if generation is not None and key not in self._first_seen:
            self._first_seen[key] = generation
        duplicate_note = ""
        ranking_score = score
        if cached_hit and self.repeat_penalty < 1.0 and score is not None:
            self._duplicates += 1
            ranking_score = score * self.repeat_penalty
            first = self._first_seen.get(key)
            where = f" (first proposed at generation {first})" if first is not None else ""
            duplicate_note = (
                f" DUPLICATE: this exact code was already evaluated{where}; its "
                f"ranking score is penalized by x{self.repeat_penalty:.2f} "
                f"({score:.6f} -> {ranking_score:.6f}). Repeating an operator does "
                f"not improve the search: propose a STRUCTURALLY DIFFERENT crossover."
            )

        # Diversidad conductual: se mide una sola vez por codigo (el fingerprint
        # se calculo en `_evaluate_code`) y se expone en `Solution.metadata`
        # para que LLaMEA use `distance_metric` / `behavior_descriptor`
        # conductuales en novelty, clearing y MAP-Elites.
        diversity_info = None
        fingerprint = entry.get("fingerprint")
        if fingerprint is not None:
            diversity_info = self.diversity.add(
                key, fingerprint, score,
                generation=generation,
                description=getattr(solution, "description", ""),
            )
            _set_metadata(solution, "behavior_fingerprint",
                          [float(value) for value in fingerprint])
            _set_metadata(solution, "behavior_descriptor",
                          behavior_descriptor(fingerprint))
            _set_metadata(solution, "behavior_family", diversity_info["family"])
            _set_metadata(solution, "behavioral_novelty",
                          float(diversity_info["novelty"]))
            _set_metadata(solution, "behavior_cell",
                          list(diversity_info["cell"]))

        # Validacion contra la familia objetivo del combine. Se mide con el
        # fingerprint que ya calculo el evaluador; no se vuelve a ejecutar el
        # operador. El HV crudo queda intacto y la penalizacion, si se activa,
        # solo afecta al ranking que ve LLaMEA.
        observed_family = None
        target_hit = None
        target_note = ""
        if target_family:
            self._target_attempts += 1
            if diversity_info is not None:
                observed_family = diversity_info.get("family")
                target_hit = observed_family == target_family
                self._target_hits += int(target_hit)
            _set_metadata(solution, "target_family", target_family)
            _set_metadata(solution, "target_family_hit", target_hit)
            if observed_family is not None:
                _set_metadata(solution, "target_family_observed", observed_family)
            target_note = self._target_note(target_family, observed_family,
                                             target_hit)
            if self.require_target_family and target_hit is False:
                reason = (
                    "Target family validation failed: expected "
                    f"{target_family}, observed {observed_family}. The child was "
                    "rejected before it could enter the population."
                )
                solution.set_scores(FALLBACK_SCORE, reason + target_note,
                                    ValueError(reason))
                self._track_best(FALLBACK_SCORE, solution)
                if diversity_info is not None:
                    self._log_diversity(key, diversity_info, entry, score,
                                        target_family, observed_family, target_hit)
                return solution
            if target_hit is False and self.target_penalty < 1.0:
                ranking_score *= self.target_penalty
                target_note += (
                    f" The ranking score was multiplied by x{self.target_penalty:.2f} "
                    f"because the target family was missed ({score:.6f} -> "
                    f"{ranking_score:.6f}); the raw HV is unchanged."
                )

        if diversity_info is not None:
            self._log_diversity(key, diversity_info, entry, score,
                                target_family, observed_family, target_hit)

        # El archivo de ranking guarda candidatos unicos. Antes una repeticion
        # se agregaba con la misma clave y su HV penalizado sobrescribia el de
        # la primera evaluacion (el mejor podia bajar solo por repetir codigo).
        # Ahora la repeticion reutiliza el ranking del original y la penalizacion
        # queda solo en `ranking_score`, que es lo que ve LLaMEA.
        if cached_hit:
            rank = self.archive.rank(key)
        else:
            rank = self.archive.add(ranking_score, novelty, key=key)
        total = len(self.archive)
        best = self.archive.best()

        # Parada temprana y mejor-hasta-ahora: se sigue el mejor ranking. Si no
        # mejora en `patience` evaluaciones seguidas, se corta la corrida.
        self._track_best(ranking_score, solution)
        reuse = " (reused cached evaluation)" if cached_hit else ""
        frame_note = (
            "normalized with the consolidated front of the reference operators and "
            "every candidate evaluated so far, exactly like the MOHH-R grammar "
            "pipeline" if self.frame == "ge" else
            "normalized with the reference operators only (stationary frame)"
        )
        degenerate_note = (
            " Instances without scale (frame collapsed, excluded as the grammar "
            f"pipeline does): {[os.path.basename(self.instance_paths[i]) for i in degenerate]}."
            if degenerate else ""
        )
        # Mejora: decirle al LLM como va contra la linea base clasica medida en el
        # MISMO marco. Sin esto el feedback solo compara al candidato con los
        # otros candidatos, y el modelo no sabe que OX/PBX existen ni si va
        # perdiendo contra ellos (se ancla en el OX del ejemplo sin saber que PBX
        # lo supera).
        baseline_note = ""
        if self.normalize:
            reference = {name: value for name, value in self.reference_scores().items()
                         if value is not None}
            if reference and score is not None:
                best_name, best_value = max(reference.items(), key=lambda item: item[1])
                delta = 100.0 * (score - best_value) / best_value if best_value else 0.0
                relation = ("ABOVE" if delta > 0 else "BELOW" if delta < 0 else "LEVEL with")
                classics = ", ".join(
                    f"{name.upper()} {value:.6f}" for name, value in
                    sorted(reference.items(), key=lambda item: -item[1])
                )
                baseline_note = (
                    f" Reference operators in the same frame (higher is better): "
                    f"you are {abs(delta):.1f}% {relation} the best classic "
                    f"{best_name.upper()} = {best_value:.6f} [{classics}]. "
                    f"Your goal is to beat {best_name.upper()}."
                )
        hv_detail = ""
        if score is not None and ranking_score != score:
            hv_detail = (
                f" (raw mean HV {score:.6f}; the ranking score above also "
                "includes the duplicate/target penalties)"
            )
        feedback = (
            f"Ranking: this crossover is #{rank} of {total} evaluated so far by "
            f"mean hypervolume (1 = best; higher is better; HV ties are broken by "
            f"recombination novelty). "
            f"Mean hypervolume ({frame_note}): {ranking_score:.6f}{hv_detail}. "
            f"Best so far: {best:.6f}.{baseline_note} "
            f"Recombination novelty: {novelty:.2f} (0 = copies a parent, 1 = child "
            f"unlike both parents). "
            f"Per-instance HV: {[None if hv is None else round(hv, 6) for hv in per_instance]}."
            f"{degenerate_note}{duplicate_note}{reuse}"
        )
        feedback += self._behavioral_feedback(diversity_info, entry)
        feedback += target_note
        solution.set_scores(ranking_score, feedback)
        return solution

    def _evaluate_code(self, solution, compute_hypervolume):
        """Evalua los frentes del codigo en las instancias (o devuelve el error).

        Solo corre el MOEA interno: el hipervolumen se calcula despues, en el
        marco de normalizacion vigente (`_hv_values`), porque en el marco GE
        puede cambiar cuando otro candidato amplia la envolvente de la instancia.
        """
        try:
            namespace = compile_operator(solution.code)
        except ValueError as e:
            return {"fronts": None, "per_instance": [], "score": None,
                    "error": str(e), "exception": e, "novelty": 0.0,
                    "fingerprint": None}

        # Chequeo barato de degeneracion ANTES del MOEA: si el operador solo
        # copia a un padre no vale la pena correr las instancias.
        ok, novelty, reason = recombination_novelty(namespace["crossover"])
        if not ok:
            return {"fronts": None, "per_instance": [], "score": None,
                    "error": reason, "exception": ValueError(reason), "novelty": 0.0,
                    "fingerprint": None}

        # Huella conductual sobre pares de padres fijos. Es independiente del
        # MOEA y se reutiliza si el LLM repite el codigo. Si el operador falla
        # en algun par, `behavior_fingerprint` devuelve None y la evaluacion
        # sigue: la diversidad es un diagnostico, no un requisito de validez.
        fingerprint = behavior_fingerprint(
            namespace["crossover"], timeout=OPERATOR_CALL_TIMEOUT
        )

        # Validacion dura opcional: si el hijo no pertenece a la familia que
        # pidio el combine, no se gasta el MOEA interno. El fingerprint ya se
        # calculo sobre pares fijos y es barato comparado con las instancias.
        target_family = self._target_family(solution)
        if target_family and self.require_target_family and fingerprint is not None:
            observed_family = family_from_fingerprint(fingerprint)
            if observed_family != target_family:
                reason = (
                    "Target family validation failed: expected "
                    f"{target_family}, observed {observed_family}. The child did "
                    "not behave as requested and was rejected before the MOEA."
                )
                return {
                    "fronts": None, "per_instance": [], "score": None,
                    "error": reason, "exception": ValueError(reason),
                    "novelty": novelty, "fingerprint": fingerprint,
                    "target_family": target_family,
                    "observed_family": observed_family, "target_hit": False,
                }

        instances = self._ensure_instances()
        init_params(**self._params_overrides())
        if self.normalize:
            self.ensure_reference_bounds()

        fronts = []
        for index, inst in enumerate(instances):
            try:
                front = self._solve_front(inst, namespace)
            except Exception as e:  # noqa: BLE001 - se reporta como feedback
                # Un operador puede pasar la validacion en tamanos chicos y
                # fallar en una instancia real (p.ej. mutar los padres in-place
                # o indexar fuera de rango). No debe tumbar la busqueda.
                return {
                    "fronts": None,
                    "per_instance": [],
                    "score": None,
                    "error": f"Crossover failed on an instance: {type(e).__name__}: {e}",
                    "exception": e,
                    "novelty": novelty,
                    "fingerprint": fingerprint,
                }
            if self.normalize:
                fronts.append(front)
                continue
            # Sin normalizacion (`--no-normalize`): HV crudo por instancia.
            ref_point = np.asarray(front, dtype=float).max(axis=0) * NADIR_FACTOR
            ref_point[ref_point == 0] = 1.0
            fronts.append(float(compute_hypervolume(ref_point, front)))

        if self.normalize:
            return {"fronts": fronts, "per_instance": [], "score": None,
                    "error": None, "exception": None, "novelty": novelty,
                    "fingerprint": fingerprint}

        return {"fronts": None, "per_instance": list(fronts),
                "score": float(np.mean(fronts)), "error": None,
                "exception": None, "novelty": novelty,
                "fingerprint": fingerprint}

    # -- artefactos ------------------------------------------------------- #
    def _save_candidate_fronts(self, key, entry, solution):
        """Guarda los frentes crudos del candidato en `fronts/`.

        Escribe `candidate-<sha1>.npz` (un arreglo por instancia) y agrega una
        linea a `fronts/manifest.jsonl` con la generacion, el candidato y el
        archivo. Es la evidencia cruda de la evaluacion: permite recalcular el
        HV en cualquier marco sin volver a correr el MOEA interno.
        """
        if not self.fronts_dir or entry.get("fronts") is None:
            return
        fronts = entry["fronts"]
        arrays = {
            f"instance_{index}": np.asarray(front, dtype=float)
            for index, front in enumerate(fronts)
            if front is not None and len(front)
        }
        if not arrays:
            return
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        filename = f"candidate-{digest}.npz"
        np.savez_compressed(os.path.join(self.fronts_dir, filename), **arrays)
        generation = getattr(solution, "generation", None)
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "key_sha1": digest,
            "generation": int(generation) if generation is not None else None,
            "name": getattr(solution, "name", None),
            "instances": [os.path.basename(path) for path in self.instance_paths],
            "saved_instances": sorted(arrays),
            "file": filename,
            "score_at_evaluation": entry.get("score"),
            "novelty": entry.get("novelty"),
        }
        with self._fronts_lock:
            with open(os.path.join(self.fronts_dir, "manifest.jsonl"), "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def save_frame(self, path=None):
        """Guarda el marco de normalizacion final por instancia.

        `frame.npz` con `vmin`/`vmax` de cada instancia puntuable, mas un JSON
        legible (`frame.json`). Es el marco con el que se midieron los HV que
        reporta `llamea_best_info.json`.
        """
        if self._frame_points is None:
            self.ensure_reference_bounds()
        if not self._frame_points:
            return None
        path = path or os.path.join(self.fronts_dir or ".", "frame.npz")
        arrays = {}
        readable = {}
        for index, frame in enumerate(self._frame_points):
            name = os.path.basename(self.instance_paths[index])
            if frame is None or len(frame) == 0:
                readable[name] = None
                continue
            arrays[f"vmin_{index}"] = frame.min(axis=0)
            arrays[f"vmax_{index}"] = frame.max(axis=0)
            readable[name] = {
                "vmin": frame.min(axis=0).tolist(),
                "vmax": frame.max(axis=0).tolist(),
            }
        if not arrays:
            return None
        np.savez_compressed(path, **arrays)
        with open(os.path.splitext(path)[0] + ".json", "w") as handle:
            json.dump({"frame": self.frame, "instances": readable}, handle, indent=2)
        return path

    # -- utilidades ------------------------------------------------------- #
    def __repr__(self):  # pragma: no cover - solo para logs
        return (f"OperatorEvaluator(n_instances={len(self.instance_paths)}, "
                f"model={self.model_name}, seed={self.seed}, "
                f"normalize={self.normalize})")


def build_evaluator(train_instances, mo_generations, mo_population_size,
                    seed=42, model_name="MOEAD", mutation=False,
                    instance_timeout=INSTANCE_TIMEOUT, normalize=True,
                    bounds=None, reference_operators=None, archive=None,
                    frame="ge", fronts_dir=None, repeat_penalty=0.5, patience=0,
                    diversity_feedback=True, diversity_log=None,
                    target_penalty=1.0, require_target_family=False):
    """Construye la funcion de fitness que consume LLaMEA.

    @param train_instances: lista de rutas de instancias (train o test).
    @param mo_generations: generaciones del MOEA interno por instancia.
    @param mo_population_size: tamano de poblacion del MOEA interno.
    @param seed: semilla base (reproducibilidad intra-nodo).
    @param model_name: MOEA interno (MOEAD, NSGAII, SMSEMOA).
    @param mutation: si True, agrega mutacion al MOEA interno (por defecto
        solo cruce, igual que la generacion GE con `MO_MUTATION_BOOL=False`).
    @param normalize: normalizar el HV con un frente de referencia (justo y
        comparable, como el proyecto original). Si False, usa el nadir
        `max(frente) * 1.1` de cada frente.
    @param bounds: limites de normalizacion precalculados (lista por instancia o
        dict {nombre_de_instancia: (vmin, vmax)}). Si None y `normalize`, se
        construyen con `ensure_reference_bounds()`.
    @param reference_operators: dict nombre -> funcion `crossover` para el frente
        de referencia (default: `mohh.llamea.reference.REFERENCE_OPERATORS`).
    @param archive: `HVArchive` compartido para el ranking (uno nuevo si None).
    @param frame: `"ge"` (por defecto) normaliza con el frente consolidado de los
        operadores de referencia y todos los candidatos ya evaluados, como la
        evaluacion gramatical; `"stationary"` usa solo los operadores de
        referencia (HV fijos, escala abierta).
    @param fronts_dir: si se indica, guarda los frentes crudos de cada candidato
        (`candidate-*.npz` + `manifest.jsonl`) en ese directorio.
    """
    return OperatorEvaluator(
        train_instances,
        mo_generations=mo_generations,
        mo_population_size=mo_population_size,
        seed=seed,
        model_name=model_name,
        mutation=mutation,
        instance_timeout=instance_timeout,
        normalize=normalize,
        bounds=bounds,
        reference_operators=reference_operators,
        archive=archive,
        frame=frame,
        fronts_dir=fronts_dir,
        repeat_penalty=repeat_penalty,
        patience=patience,
        diversity_feedback=diversity_feedback,
        diversity_log=diversity_log,
        target_penalty=target_penalty,
        require_target_family=require_target_family,
    )
