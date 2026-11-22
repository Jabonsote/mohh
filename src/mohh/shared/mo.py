"""Protocolo MOEA comun a la busqueda GE y LLaMEA.

Copia literal del arnes de `mohh.llamea.fitness`; ese modulo reexporta los
simbolos para no romper importadores existentes.
"""

import signal
import threading

import numpy as np

from mohh.constants import EVAL_BUDGETS, MO_POPULATION_SIZE
from mohh.shared.fronts import _consolidated_frame


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

    def _on_alarm(_signum, _frame):  # pragma: no cover - depende del reloj
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


def _load_mutation(solver, model_name, mutation):
    """Carga la mutacion del MOEA interno, igual para el modo secuencial y el pool."""
    if mutation:
        # `Swap_Mutation` (la mutacion de permutacion que usa la evaluacion GE)
        # vive en `mohh.evaluation.operators.mutation`; se inyecta por atributo
        # igual que el cruce, sin tocar el nucleo.
        from mohh.evaluation.operators.mutation import Swap_Mutation

        key = {
            "MOEAD": "MOEAD_MUTATION_PROBABILITY",
            "NSGAII": "NSGA_MUTATION_PROBABILITY",
            "SMSEMOA": "SMS_MUTATION_PROBABILITY",
        }.get(model_name, "MOEAD_MUTATION_PROBABILITY")
        from mohh.evaluation.params import mo_params

        prob = mo_params.get(key, 0.05)
        solver.mutation = Swap_Mutation(prob)
    else:
        solver.load_operator("mutation", "NullMutation")


# `EVAL_BUDGETS` se re-exporta desde `mohh.constants` para los importadores
# existentes: 100/300/500 generaciones con poblacion 105 del original.


def solve_front_snapshots(instance, namespace, seed, model_name, mutation,
                          instance_timeout, budgets=None, crossover=None):
    """Como `_solve_front_namespace`, pero con los snapshots del original.

    Corre el MOEA hasta el ultimo presupuesto y devuelve ``{label: frente}``
    con el frente de la poblacion al alcanzar cada uno (10k/30k/50k). Una
    sola corrida da los tres presupuestos: el coste no se multiplica (correr
    hasta 500 generaciones cuesta ~1.7x una a 300).

    ``crossover`` permite inyectar un operador pymoo ya construido (por
    ejemplo ``PMX_Cross`` del proyecto) en vez de envolver ``namespace`` en
    ``LLMOperator``. Es lo que usa ``compare_operators.py`` para el tipo
    ``project:``.
    """
    from pymoo.core.callback import Callback

    class _Snapshots(Callback):
        """El `MyCallback` del original: guarda el frente en cada presupuesto."""

        def __init__(self, population_size, budget_map):
            super().__init__()
            self.populations = {}
            self.checkpoints = {
                label: max(1, budget // population_size)
                for label, budget in budget_map.items()
            }

        def notify(self, algorithm):
            for label, generation in self.checkpoints.items():
                if algorithm.n_gen == generation:
                    # `F` puede ser 1D si el frente tiene un solo punto.
                    front = np.atleast_2d(np.asarray(algorithm.result().F,
                                                     dtype=float))
                    if front.size:
                        front, _ = np.unique(front, axis=0, return_index=True)
                        self.populations[label] = front

    budgets = dict(EVAL_BUDGETS) if budgets is None else dict(budgets)
    from mohh.generation.algorithm.multiobjective import MOSolver
    from mohh.shared.operators import LLMOperator

    np.random.seed(seed)
    population_size, _current_generations = _params_size()
    generations = max(budget // population_size for budget in budgets.values())

    solver = MOSolver(seed=seed)
    solver.crossover = (crossover if crossover is not None
                        else LLMOperator(namespace, prob=1.0))
    _load_mutation(solver, model_name, mutation)
    solver.load_instance(instance)
    # El solver usa `params["MO_GENERATIONS"]` como terminacion: el snapshot
    # necesita llegar al ultimo presupuesto, no solo a los 300 del default.
    solver.generations = generations
    solver.start_model(model_name=model_name)
    callback = _Snapshots(population_size, budgets)
    solver.algorithm.callback = callback
    _call_with_timeout(solver.solve_instance, instance_timeout)
    return {label: front for label, front in callback.populations.items()
            if front is not None and len(front)}


def _params_size():
    """Poblacion y generaciones vigentes (Params es un dict-like)."""
    from mohh.core.params import Params
    params = Params()
    try:
        population_size = int(params["MO_POPULATION_SIZE"])
    except Exception:  # noqa: BLE001 - sin params, el default del original
        population_size = MO_POPULATION_SIZE
    try:
        generations = int(params["MO_GENERATIONS"])
    except Exception:  # noqa: BLE001
        generations = 300
    return population_size, generations


def _solve_front_namespace(instance, namespace, seed, model_name, mutation,
                           instance_timeout, crossover=None):
    """Corre el MOEA interno sobre una instancia y devuelve su frente.

    Es la unidad de trabajo comun de la busqueda secuencial y del pool de
    instancias. No modifica la instancia recibida: ``MOSolver.load_instance``
    copia el problema y la poblacion inicial.

    ``crossover`` permite inyectar un operador pymoo ya construido (por
    ejemplo ``PMX_Cross`` del proyecto) en vez de envolver ``namespace`` en
    ``LLMOperator``.
    """
    from mohh.generation.algorithm.multiobjective import MOSolver
    from mohh.shared.operators import LLMOperator

    # Semilla determinista por corrida (igual que el worker GE).
    np.random.seed(seed)
    solver = MOSolver(seed=seed)
    # El operador se inyecta directamente en el solver; la ruta GE
    # (load_operator -> HH_Operator) no se toca.
    solver.crossover = (crossover if crossover is not None
                        else LLMOperator(namespace, prob=1.0))
    _load_mutation(solver, model_name, mutation)
    solver.load_instance(instance)
    solver.start_model(model_name=model_name)
    _, pareto_front, *_ = _call_with_timeout(
        solver.solve_instance, instance_timeout
    )
    return np.asarray(pareto_front, dtype=float)


def _solve_front_protocols(instance, namespace, seed, protocols,
                           instance_timeout):
    """Corre el MOEA bajo cada protocolo (solver, mutacion) y consolida.

    Es el protocolo de la evaluacion GE: un frente por cada combinacion de
    solver y mutacion, unidos en un solo frente por instancia antes de medir
    el HV. Un protocolo que falle no cancela los demas.
    """
    fronts = []
    for model_name, mutation in protocols:
        try:
            front = _solve_front_namespace(
                instance, namespace, seed=seed, model_name=model_name,
                mutation=mutation, instance_timeout=instance_timeout,
            )
        except Exception:  # noqa: BLE001 - se omite el protocolo fallido
            continue
        if front is not None and len(front):
            fronts.append(front)
    if not fronts:
        return None
    return _consolidated_frame(*fronts)
