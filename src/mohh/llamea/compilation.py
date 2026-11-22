"""Compilacion y validacion del operador generado por el LLM.

Copia literal del bloque original de `mohh.llamea.fitness`; ese modulo
reexporta los nombres para no romper los importadores existentes.
"""

import linecache
import re
import traceback
from functools import lru_cache

import numpy as np

from mohh.constants import OPERATOR_CALL_TIMEOUT
from mohh.shared.mo import _TimeoutError, _call_with_timeout


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

# Los limites de validacion y de instancia viven en `mohh.constants`. Protegen
# de operadores que no terminan: con `max_workers=1` joblib usa un backend
# secuencial que NO aplica el `eval_timeout` de LLaMEA.
COMPILE_TIMEOUT = 30.0         # segundos para `exec` del codigo del LLM (código de módulo)

#: Nombre de fichero ficticio con el que se compila el codigo del LLM. Se
#: registra en `linecache` para que el traceback muestre la linea exacta que
#: fallo, que es la parte mas util del feedback (ver `_exception_feedback`).
OPERATOR_FILENAME = "<llm-operator>"


# `_TimeoutError` se reexporta desde `mohh.shared.mo`: `compile_operator`
# captura la misma clase que lanza `_call_with_timeout`.


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
        "collapse": hh.collapse,
        "verify_inputs": hh.verify_inputs,
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


def validate_operator_minimal(crossover_func):
    """Filtro minimo: una sola talla y una permutacion.

    No comprueba mutacion de entradas (el adaptador pasa copias), ni varios
    tamanos, ni novedad, ni familia objetivo. La evaluacion decide.
    """
    rng = np.random.RandomState(0)
    for n in (11, 17, 23, 50):
        pairs = [(np.arange(n), np.arange(n)[::-1].copy())]
        pairs.extend(
            (rng.permutation(n), rng.permutation(n)) for _ in range(3)
        )
        for x, y in pairs:
            try:
                out = _call_with_timeout(crossover_func, OPERATOR_CALL_TIMEOUT,
                                         x.copy(), y.copy())
            except Exception as error:  # noqa: BLE001
                return False, f"crossover raised {type(error).__name__} at n={n}: {error}"
            if isinstance(out, tuple):
                if len(out) != 2:
                    return False, f"tuple of {len(out)} elements; expected two children"
                children = out
            else:
                children = (out,)
            for index, child in enumerate(children):
                try:
                    child = np.asarray(child)
                except Exception:  # noqa: BLE001
                    return False, f"child{index + 1} did not return an array at n={n}"
                if child.shape != (n,):
                    return False, (
                        f"child{index + 1} has shape {child.shape}; expected {(n,)} at n={n}")
                if set(child.tolist()) != set(range(n)):
                    return False, (
                        f"child{index + 1} is not a permutation of range({n}) at n={n}")
    return True, None


#: Generador propio sin semilla: `np.random.default_rng()` sin argumentos.
#: El arnes siembra el RNG global (`np.random.seed`), pero un generador propio
#: se inicializa con la entropia del sistema: el operador deja de ser
#: reproducible y la comparacion pareada (mismas semillas) pierde sentido.
_UNSEEDED_RNG_RE = re.compile(r"default_rng\s*\(\s*\)")


def uses_unseeded_rng(code: str) -> bool:
    """True si el codigo crea un generador propio sin semilla.

    `np.random.default_rng(42)` si es determinista y no se rechaza; lo que
    rompe la reproducibilidad es la llamada sin argumentos.
    """
    return bool(_UNSEEDED_RNG_RE.search(code or ""))


def compile_operator(code: str, timeout=COMPILE_TIMEOUT,
                     check_seeded_rng=True) -> dict:
    """`exec` del codigo del LLM; devuelve el namespace o lanza ValueError.

    El namespace incluye los primitivos de permutacion de MOHH-R, de modo que
    el operador generado puede reutilizar la logica actual. Debe existir
    `crossover(x, y)` y devolver permutacion(es) valida(s).

    Los mensajes de error son el feedback que recibe el LLM, asi que dicen que
    tamano fallo, que se encontro en lugar de una permutacion y, si el codigo
    lanzo una excepcion, la linea que la provoco.

    `check_seeded_rng=True` rechaza los operadores que crean su propio
    generador sin semilla (`np.random.default_rng()`): el arnes siembra el RNG
    global, pero ese generador se inicializa con la entropia del sistema y el
    operador dejaria de ser reproducible (rompe la comparacion pareada). La
    comparacion de operadores lo desactiva para poder cargar el operador y
    avisar.

    El `exec` va acotado por `timeout`: el LLM suele dejar codigo de modulo
    ("self-check" que llama a `crossover` al importar) y un bucle infinito ahi
    colgaba la corrida entera (paso en
    `results/searches/qwen25coder7b_moead_b30_st_r3`, ver
    `docs/llamea-prompts/areas-de-mejora.md` §14).
    """
    if check_seeded_rng and uses_unseeded_rng(code):
        raise ValueError(
            "Operator creates its own random generator without a seed "
            "(np.random.default_rng()). The evaluation seeds numpy's global "
            "RNG (np.random.seed), but a private generator ignores it and the "
            "operator would not be reproducible. Use the global RNG "
            "(np.random.*) instead."
        )
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

    ok, reason = validate_operator_minimal(crossover)
    if not ok:
        raise ValueError(f"Invalid crossover operator: {reason}.")

    return namespace


@lru_cache(maxsize=8)
def _cached_operator(code: str):
    """Compila una vez por proceso el codigo que se evalua en varias instancias.

    Los workers de ``ProcessPoolExecutor`` reciben el codigo (texto), no el
    namespace: las funciones creadas con ``exec`` no se pueden picklear por
    referencia. Este cache evita repetir la compilacion y la validacion en cada
    instancia del mismo candidato.
    """
    return compile_operator(code)
