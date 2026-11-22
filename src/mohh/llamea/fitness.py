"""Evaluacion de operadores `crossover(x, y)` propuestos por un LLM."""

import hashlib
import json
import linecache
import multiprocessing
import os
import re
import signal
import tempfile
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache

import numpy as np

from mohh.constants import (
    EVAL_BUDGETS,
    INSTANCE_TIMEOUT,
    MO_POPULATION_SIZE,
    NADIR,
    OPERATOR_CALL_TIMEOUT,
    REPEAT_PENALTY,
)
from mohh.shared.fronts import (
    NADIR_FACTOR,
    _consolidated_frame,
    _nondominated,
    clip_to_reference_box,
    comprobate_frame,
    normalize_front,
)
from mohh.shared.mo import (
    _TimeoutError,
    _call_with_timeout,
    _load_mutation,
    _params_size,
    _solve_front_namespace,
    _solve_front_protocols,
    init_params,
    solve_front_snapshots,
)

FALLBACK_SCORE = 0.0


# Compilacion y validacion: reexportadas desde `mohh.llamea.compilation`.
from mohh.llamea.compilation import (
    _set_metadata,
    EarlyStop,
    COMPILE_TIMEOUT,
    OPERATOR_FILENAME,
    _describe_child_problem,
    _exception_feedback,
    validate_operator,
    DEGENERACY_TRIALS,
    recombination_novelty,
    primitive_namespace,
    _compile_with_source,
    validate_operator_minimal,
    _UNSEEDED_RNG_RE,
    uses_unseeded_rng,
    compile_operator,
    _cached_operator,
)


# `EVAL_BUDGETS` se re-exporta desde `mohh.constants` para los importadores
# existentes: 100/300/500 generaciones con poblacion 105 del original.
# Protocolo MOEA: reexportado desde `mohh.shared.mo`.


def _solve_front_task(task):
    """Worker picklable: una instancia de un candidato o de una referencia.

    Devuelve un diccionario en vez de lanzar la excepcion para que un fallo en
    una instancia no aborte el resto del lote (el comportamiento secuencial
    tambien sigue con las demas referencias).
    """
    try:
        init_params(**task["params"])
        if task.get("reference") is not None:
            from mohh.llamea.reference import namespace_for

            namespace = namespace_for(task["reference"])
        else:
            namespace = _cached_operator(task["code"])
        protocols = task.get("protocols")
        if protocols:
            front = _solve_front_protocols(
                task["instance"], namespace, seed=task["seed"],
                protocols=protocols,
                instance_timeout=task["instance_timeout"],
            )
        else:
            front = _solve_front_namespace(
                task["instance"],
                namespace,
                seed=task["seed"],
                model_name=task["model_name"],
                mutation=task["mutation"],
                instance_timeout=task["instance_timeout"],
            )
        return {"front": front, "error": None}
    except Exception as error:  # noqa: BLE001 - se reporta como feedback
        return {
            "front": None,
            "error": f"{type(error).__name__}: {error}",
        }


# Frentes, normalizacion y recorte: reexportados desde `mohh.shared.fronts`.


# Archivo y utilidades de frentes: reexportadas desde `mohh.llamea.archive`.
from mohh.llamea.archive import HVArchive, _nd_merge, _same_points


class OperatorEvaluator:
    """Evalua operadores generados por un LLM con el mismo MOEA e HV.

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
    sin repetir el MOEA; si el marco cambia se recalculan
    los HV de todos los candidatos a partir de esos frentes (sin MOEA).
    """

    def __init__(self, instance_paths, mo_generations, mo_population_size,
                 seed=42, model_name="MOEAD", mutation=False,
                 instance_timeout=INSTANCE_TIMEOUT, normalize=True,
                 bounds=None, reference_operators=None, archive=None,
                 frame="ge", fronts_dir=None, repeat_penalty=REPEAT_PENALTY, patience=0,
                 fitness_workers=1, model_names=None, mutations=None,
                 ranking=False):
        self.instance_paths = [str(p) for p in instance_paths]
        self.mo_generations = mo_generations
        self.mo_population_size = mo_population_size
        self.seed = seed
        self.model_name = model_name
        self.mutation = mutation
        # Protocolo GE: todos los solvers y ambas mutaciones. Si no se indica,
        # se conserva el comportamiento de un solo solver/mutacion.
        if model_names:
            names = list(model_names)
        else:
            names = [model_name]
        if mutations is None:
            muts = [bool(mutation)]
        else:
            muts = [bool(value) for value in mutations]
        self.protocols = [(name, mut) for name in names for mut in muts]
        self.instance_timeout = instance_timeout
        # Paralelismo dentro de un candidato: una tarea por instancia. Con
        # `1` se conserva el bucle secuencial historico. El cluster lo activa
        # con `FITNESS_WORKERS` (ver slurm/llamea-single.slrm).
        self.fitness_workers = max(1, int(fitness_workers or 1))
        self.normalize = normalize
        self.bounds = bounds
        self.reference_operators = reference_operators
        self.archive = archive if archive is not None else HVArchive()
        self.frame = frame if normalize else "none"
        self.ranking = bool(ranking)
        # Presion de diversidad: un candidato identico a uno ya evaluado recibe
        # su HV multiplicado por `repeat_penalty` **solo para el ranking** (el HV
        # crudo se conserva para el reporte). `1.0` desactiva la penalizacion.
        self.repeat_penalty = float(repeat_penalty)
        # Parada temprana: si el mejor ranking no mejora en `patience`
        # evaluaciones seguidas, se lanza `EarlyStop` y `run.py` cierra con el
        # mejor operador guardado (evita gastar 1000 generaciones en un anclaje).
        self.patience = int(patience)
        self._best_ranking = None
        self._best_solution = None
        self._since_improvement = 0
        self._first_seen = {}            # clave -> generacion de la primera vez
        self._duplicates = 0             # repeticiones exactas evaluadas
        self._evaluations = 0            # llamadas a __call__ (incluye invalidas)
        self._valid_evaluations = 0      # llamadas con HV definido
        # Frentes crudos por candidato (evidencia de la evaluacion). Si es None
        # no se guarda nada (`--no-save-fronts`).
        self.fronts_dir = fronts_dir
        if fronts_dir:
            os.makedirs(fronts_dir, exist_ok=True)
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
        """Recrea las instancias en el proceso hijo al deserializar."""
        state = self.__dict__.copy()
        state["_instances"] = None
        state.pop("_fronts_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
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
        if len(self.protocols) > 1:
            return _solve_front_protocols(
                instance,
                namespace,
                seed=self.seed,
                protocols=self.protocols,
                instance_timeout=self.instance_timeout,
            )
        model_name, mutation = self.protocols[0]
        return _solve_front_namespace(
            instance,
            namespace,
            seed=self.seed,
            model_name=model_name,
            mutation=mutation,
            instance_timeout=self.instance_timeout,
        )

    def _instance_task(self, instance, code=None, reference=None):
        """Tarea serializable para evaluar una instancia en otro proceso."""
        return {
            "instance": instance,
            "code": code,
            "reference": reference,
            "params": self._params_overrides(),
            "seed": self.seed,
            "model_name": self.model_name,
            "mutation": self.mutation,
            "protocols": self.protocols,
            "instance_timeout": self.instance_timeout,
        }

    def _run_tasks(self, tasks):
        """Ejecuta tareas de instancia en paralelo, preservando el orden.

        `fitness_workers` es el paralelismo **dentro de un candidato** (una
        instancia por tarea). No se debe confundir con `--max-workers`, que
        paraleliza candidatos en LLaMEA; usar ambos a la vez anida pools de
        procesos y por eso `run.py` los declara incompatibles.
        """
        workers = max(1, int(self.fitness_workers or 1))
        if workers <= 1 or len(tasks) <= 1:
            return [_solve_front_task(task) for task in tasks]

        workers = min(workers, len(tasks))
        try:
            # `forkserver` (no `fork`): la corrida tiene el hilo del watchdog
            # de parada y el cliente de vLLM con hilos; forkear un proceso
            # multihilo puede bloquear al hijo. El servidor de forkserver es
            # limpio y ademas evita reimportar todo en cada lote como `spawn`.
            try:
                context = multiprocessing.get_context("forkserver")
            except ValueError:  # pragma: no cover - plataformas sin forkserver
                context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as pool:
                return list(pool.map(_solve_front_task, tasks))
        except Exception:
            # Compatibilidad: si una tarea no se puede picklear (p. ej. un
            # operador de referencia definido como closure), se degrada a
            # secuencial en vez de tumbar la corrida.
            return [_solve_front_task(task) for task in tasks]

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

        from mohh.llamea.reference import REFERENCE_OPERATORS

        operators = self.reference_operators or REFERENCE_OPERATORS
        instances = self._ensure_instances()
        init_params(**self._params_overrides())

        # El panel de referencia tambien se paraleliza por instancia: con 5
        # operadores y 80 instancias son 400 MOEAs antes del primer candidato.
        # GE ya hacia esto en su `evaluation_step`; LLaMEA lo hacia secuencial.
        contexts = []
        tasks = []
        for index, inst in enumerate(instances):
            for name, func in operators.items():
                contexts.append((index, name, inst))
                tasks.append(self._instance_task(inst, reference=func))
        results = self._run_tasks(tasks)

        reference_fronts = [{} for _ in instances]
        for (index, name, _inst), result in zip(contexts, results):
            if result["error"] is not None:
                # Una referencia que falle no corta el marco: se omite, igual
                # que el bucle secuencial anterior.
                continue
            front = result["front"]
            if front is not None and len(front):
                reference_fronts[index][name] = front

        bounds = []
        frame_points = []
        for inst, fronts in zip(instances, reference_fronts):
            consolidated = _consolidated_frame(*fronts.values())
            if consolidated is not None:
                bounds.append((consolidated.min(axis=0), consolidated.max(axis=0)))
            else:
                k = inst.k_objectives
                bounds.append((np.zeros(k), np.ones(k)))
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
        reproduce la caja, asi que la semilla del marco es la misma
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
            normalized = clip_to_reference_box(normalized)
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
        }
        return stats

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
                normalized = clip_to_reference_box(normalized)
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

    def _reference_hv_values(self):
        """HV por instancia de cada operador del panel de referencia."""
        from mohh.core.utilities.algorithm.MO import compute_hypervolume

        if self._reference_fronts is None:
            self.ensure_reference_bounds()
        names = sorted({name for fronts in (self._reference_fronts or [])
                        for name in fronts})
        values = {name: [] for name in names}
        for index, fronts in enumerate(self._reference_fronts or []):
            frame = self._frame_points[index] if self._frame_points else None
            for name in names:
                front = fronts.get(name)
                if frame is None or front is None or len(front) == 0:
                    values[name].append(None)
                    continue
                vmin = frame.min(axis=0)
                vmax = frame.max(axis=0)
                if np.any(vmax == vmin):
                    values[name].append(None)
                    continue
                normalized = normalize_front(front, (vmin, vmax))
                normalized = clip_to_reference_box(normalized)
                values[name].append(float(compute_hypervolume(
                    np.full(normalized.shape[1], NADIR_FACTOR), normalized)))
        return values

    def _rank_score(self, fronts):
        """Fitness GE: (K + 1 - rango medio) del candidato frente al panel."""
        from scipy.stats import rankdata

        if fronts is None:
            return 0.0, None, 1
        candidate_values = self._hv_values(fronts)
        reference = self._reference_hv_values()
        names = list(reference)
        ranks = []
        for index, candidate in enumerate(candidate_values):
            if candidate is None:
                continue
            values = [candidate]
            values.extend(reference[name][index] for name in names
                          if reference[name][index] is not None)
            if len(values) < 2:
                continue
            ranks.append(float(rankdata([-value for value in values])[0]))
        if not ranks:
            return 0.0, None, len(names) + 1
        mean_rank = float(np.mean(ranks))
        k = len(names) + 1
        return float(k + 1) - mean_rank, mean_rank, k

    def __call__(self, solution, logger=None):
        from mohh.core.utilities.algorithm.MO import compute_hypervolume

        self._evaluations += 1
        # Cache por codigo: en corridas largas el LLM repite el mismo operador
        # muchas veces (p. ej. el OX del ejemplo). Reutilizar los frentes
        # ahorra todo el MOEA interno sin cambiar el resultado.
        key = (solution.code or "").strip()
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
            error = entry.get("exception")
            if error is None:
                error = ValueError(message)
            solution.set_scores(FALLBACK_SCORE, message, error)
            self._track_best(FALLBACK_SCORE, solution)
            return solution

        # Fitness por rango GE: positivo y con el mismo orden que GE
        # (rank 1 = mejor). Se evalua el candidato contra el panel de referencia.
        if self.ranking:
            rank_score, mean_rank, k = self._rank_score(entry["fronts"])
            # HV solo como desempate cuando dos codigos empatan en rango.
            # No es el fitness principal: el orden GE sigue mandando.
            hv_values = self._hv_values(entry["fronts"])
            hv_mean = self._mean_hv(hv_values)
            solution.hv = hv_mean
            solution.hv_values = hv_values
            self._valid_evaluations += 1
            if mean_rank is None:
                feedback = ("GE ranking: no valid fronts/panel; fitness 0.0.")
            else:
                hv_text = (f" HV tie-break {hv_mean:.6f}." if hv_mean is not None
                           else "")
                feedback = (
                    f"GE ranking: mean rank {mean_rank:.4f} of {k} "
                    f"(1 = best; lower is better). "
                    f"Fitness = (K + 1 - mean rank) = {rank_score:.4f}."
                    f"{hv_text}")
            solution.set_scores(rank_score, feedback)
            self._track_best(rank_score, solution)
            self.archive.update(key, rank_score)
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
            "every candidate evaluated so far, as in the grammar search "
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
                    "error": str(e), "exception": e, "novelty": 0.0}

        # Chequeo barato de degeneracion ANTES del MOEA: si el operador solo
        # copia a un padre no vale la pena correr las instancias.
        # La novedad se registra como diagnostico, pero no se rechaza el
        # candidato: el fitness decide.
        _ok, novelty, _reason = recombination_novelty(namespace["crossover"])

        instances = self._ensure_instances()
        init_params(**self._params_overrides())
        if self.normalize:
            self.ensure_reference_bounds()

        # Cada instancia es una tarea independiente. Con `fitness_workers > 1`
        # se evaluan en paralelo; con 1 se usa el namespace ya compilado y se
        # conserva el camino secuencial historico.
        if self.fitness_workers > 1 and len(instances) > 1:
            tasks = [self._instance_task(inst, code=solution.code)
                     for inst in instances]
            results = self._run_tasks(tasks)
            fronts = []
            for result in results:
                if result["error"] is not None:
                    # Un operador puede pasar la validacion en tamanos chicos y
                    # fallar en una instancia real (p.ej. mutar los padres
                    # in-place o indexar fuera de rango). No debe tumbar la
                    # busqueda.
                    detail = result["error"]
                    return {
                        "fronts": None,
                        "per_instance": [],
                        "score": None,
                        "error": f"Crossover failed on an instance: {detail}",
                        "exception": ValueError(detail),
                        "novelty": novelty,
                    }
                fronts.append(result["front"])
        else:
            fronts = []
            for inst in instances:
                try:
                    front = self._solve_front(inst, namespace)
                except Exception as e:  # noqa: BLE001 - se reporta como feedback
                    return {
                        "fronts": None,
                        "per_instance": [],
                        "score": None,
                        "error": (f"Crossover failed on an instance: "
                                  f"{type(e).__name__}: {e}"),
                        "exception": e,
                        "novelty": novelty,
                    }
                fronts.append(front)

        if self.normalize:
            return {"fronts": fronts, "per_instance": [], "score": None,
                    "error": None, "exception": None, "novelty": novelty}

        # Sin normalizacion (`--no-normalize`): HV crudo por instancia.
        raw = []
        for front in fronts:
            ref_point = np.asarray(front, dtype=float).max(axis=0) * NADIR_FACTOR
            ref_point[ref_point == 0] = 1.0
            raw.append(float(compute_hypervolume(ref_point, front)))
        return {"fronts": None, "per_instance": raw,
                "score": float(np.mean(raw)), "error": None,
                "exception": None, "novelty": novelty}

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


class PopulationEvaluator(OperatorEvaluator):
    """Fitness por poblacion, con el HV de la evaluacion gramatical (GE).

    El hipervolumen se calcula **igual que en `mohh/evaluation/experiment.py`**:
    por instancia se consolida **todo lo evaluado** (los priors de arranque y
    cada candidato ya evaluado), de ahi sale el frente no dominado consolidado
    (con su `comprobate`), ese frente da los limites (min/max por objetivo) con
    los que se normaliza cada frente, y el frente normalizado se integra hasta
    el nadir fijo ``[1.1] * K``. Ese frente consolidado es el mismo marco que
    usa la tabla final de GE, asi que el valor que ve el LLM durante la
    busqueda es comparable con el que se reporta al final.

    Sobre esos HV se aplica el rango de GE (`compute_rank`): el fitness es el
    **rango medio** entre instancias, 1 es el mejor y, con `raw_rank=True`,
    menor es mejor (el mismo valor que usa GE). Con `raw_rank=False` se
    invierte a `(K + 1) - rango_medio` para clientes que maximizan.

    El marco crece con cada operador; los HV se recalculan con frentes
    cacheados. Padres y descendientes comparten marco en cada evaluacion.
    """

    def __init__(self, *args, raw_rank=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_rank = bool(raw_rank)
        # Marco consolidado (como `experiment.py`): frente no dominado de todo
        # lo evaluado, por instancia. `_ge_folded` marca los codigos ya
        # incorporados.
        self._ge_frame = None
        self._ge_folded = set()

    def evaluate_population(self, population, parent_population=None,
                            logger=None):
        if parent_population is None:
            parents = []
        elif isinstance(parent_population, (list, tuple)):
            parents = list(parent_population)
        else:
            parents = [parent_population]
        return self._evaluate_batch(list(population), parents, logger)

    def __call__(self, new_population, parent_population=None, logger=None):
        if isinstance(new_population, (list, tuple)):
            return self.evaluate_population(new_population, parent_population, logger)
        return super().__call__(new_population, logger=logger)

    def _fold_ge_frame(self, instances):
        """Incorpora al frente consolidado los frentes de todo lo evaluado.

        `experiment.py` consolida por instancia los puntos de todos los
        operadores evaluados y saca de ahi el frente no dominado (el marco) y
        sus limites. Aqui el consolidado crece de forma incremental: cada
        frente nuevo se fusiona con el marco vigente mediante `_nd_merge`, que
        es equivalente a recalcular el filtro no dominado sobre todo el
        historico pero en `O(|marco| * |nuevo|)`.
        """
        if self._ge_frame is None:
            self._ge_frame = [None] * len(instances)
        for key, entry in self._cache.items():
            if key in self._ge_folded:
                continue
            fronts = entry.get("fronts") if entry else None
            if fronts:
                for index, front in enumerate(fronts):
                    if front is None or len(front) == 0:
                        continue
                    if index >= len(self._ge_frame):
                        continue
                    self._ge_frame[index] = _nd_merge(self._ge_frame[index],
                                                        front)
            self._ge_folded.add(key)

    def _ge_consolidated(self, index):
        """Union de los puntos evaluados en la instancia `index`."""
        pieces = []
        for entry in self._cache.values():
            fronts = entry.get("fronts") if entry else None
            if not fronts or index >= len(fronts):
                continue
            front = fronts[index]
            if front is None or len(front) == 0:
                continue
            pieces.append(np.asarray(front, dtype=float))
        if not pieces:
            return None
        return np.unique(np.vstack(pieces), axis=0)

    def _ge_comprobate(self, index):
        """Aplica el `comprobate` del original a la instancia `index`."""
        return comprobate_frame(self._ge_consolidated(index))

    def _ge_bounds(self, index):
        """`vmin`/`vmax` del marco consolidado de la instancia.

        Como en `experiment.py`: los limites salen del frente no dominado
        consolidado (`np.min(best_front)`, `np.max(best_front)`), no de todos
        los puntos. Devuelve `None` si la instancia no tiene escala.
        """
        frame = self._ge_frame[index] if self._ge_frame else None
        if frame is None or len(frame) == 0:
            return None
        if len(frame) <= 2:
            frame = self._ge_comprobate(index)
            if frame is None or len(frame) == 0:
                return None
        vmin = frame.min(axis=0)
        vmax = frame.max(axis=0)
        if np.any(vmax == vmin):
            return None
        return vmin, vmax

    def _evaluate_batch(self, new_population, parent_population, logger=None):
        from mohh.core.utilities.algorithm.MO import compute_hypervolume
        from mohh.generation.utilities.algorithm.HH_auxiliars import compute_rank

        population = list(parent_population) + list(new_population)
        unique = {}
        for solution in population:
            key = (solution.code or "").strip()
            unique.setdefault(key, solution)

        instances = self._ensure_instances()
        if self.normalize and self.bounds is None:
            # El marco se construye con todo lo evaluado (`_fold_ge_limits`),
            # no con el panel de referencia: asi no hace falta evaluar los
            # clasicos aparte y el marco coincide con el de la tabla GE.
            self.bounds = [None] * len(instances)
        for key, solution in unique.items():
            if key not in self._cache:
                self._evaluations += 1
                entry = self._evaluate_code(solution, compute_hypervolume)
                self._cache[key] = entry
                if entry.get("fronts") is not None:
                    self._valid_evaluations += 1
                self._save_candidate_fronts(key, entry, solution)

        # Marco consolidado con todo lo evaluado (priors + candidatos).
        self._fold_ge_frame(instances)

        ranks_by_key = {key: [] for key in unique}
        hv_by_key = {key: [] for key in unique}
        for index in range(len(instances)):
            bounds = self._ge_bounds(index)
            if bounds is None:
                # Instancia sin escala: se excluye del promedio, como en
                # `experiment.py` (que la invalida con NaN).
                continue
            vmin, vmax = bounds
            hypervolumes = []
            keys = []
            for key in unique:
                entry = self._cache.get(key)
                if entry is None or entry.get("fronts") is None:
                    continue
                front = entry["fronts"][index]
                if front is None or len(front) == 0:
                    continue
                normalized = normalize_front(front, (vmin, vmax))
                if not np.all(np.isfinite(normalized)):
                    continue
                normalized = clip_to_reference_box(normalized)
                hypervolumes.append(float(compute_hypervolume(
                    np.full(normalized.shape[1], NADIR_FACTOR), normalized)))
                keys.append(key)
            if not hypervolumes:
                continue
            ranks = compute_rank(hypervolumes)
            for key, hv, rank in zip(keys, hypervolumes, ranks):
                hv_by_key[key].append(float(hv))
                ranks_by_key[key].append(rank)

        for solution in population:
            key = (solution.code or "").strip()
            entry = self._cache.get(key) or {}
            values = ranks_by_key.get(key) or []
            hvs = hv_by_key.get(key) or []
            if entry.get("error") is not None or not values or not hvs:
                reason = entry.get("error") or (
                    "no valid fronts on any instance")
                # En modo GE el peor valor es +inf; en modo maximizador, 0.
                score = float("inf") if self.raw_rank else 0.0
                error = entry.get("exception") or ValueError(reason)
                solution.set_scores(
                    score,
                    f"GE population ranking: {reason}; assigned the worst score.",
                    error=error,
                )
                continue

            mean_rank = float(np.mean(values))
            mean_hv = float(np.mean(hvs))
            k = len(unique)
            solution.hv = mean_hv
            solution.hv_values = hvs
            best = min(values)
            worst = max(values)
            if self.raw_rank:
                # Valor exacto de GE: rango medio, 1 = mejor.
                score = mean_rank
                feedback = (
                    f"GE population ranking: mean rank {mean_rank:.4f} "
                    f"(1 = best; lower is better) over {len(values)} instances. "
                    f"Best instance rank {best}, worst {worst}, K={k}. "
                    f"Reference point: fixed nadir [{NADIR_FACTOR}]*K over the "
                    f"consolidated frame of everything evaluated so far."
                )
            else:
                # Inversion solo para clientes que maximizan; mismo orden.
                score = float(k + 1) - mean_rank
                feedback = (
                    f"GE population ranking: mean rank {mean_rank:.4f} "
                    f"(1 = best; lower is better). Fitness = "
                    f"(K + 1 - mean rank) = {score:.4f} with K={k}. "
                    f"Best instance rank {best}, worst {worst}, "
                    f"over {len(values)} instances. Reference point: fixed "
                    f"nadir [{NADIR_FACTOR}]*K over the consolidated frame of "
                    f"everything evaluated so far."
                )
            solution.set_scores(score, feedback=feedback)
            # Diversidad conductual: el operador entra en el archivo
            if not self.raw_rank:
                self._track_best(score, solution)
                self.archive.update(key, score)

        return new_population, parent_population


def build_population_evaluator(train_instances, mo_generations,
                               mo_population_size, seed=42,
                               model_name="MOEAD", mutation=False,
                               instance_timeout=INSTANCE_TIMEOUT,
                               normalize=True, bounds=None,
                               reference_operators=None, archive=None,
                               frame="stationary", fronts_dir=None,
                               repeat_penalty=REPEAT_PENALTY, patience=0,
                               fitness_workers=1, model_names=None,
                               mutations=None, raw_rank=False):
    """Construye el evaluador batch con el fitness de la busqueda GE.

    `raw_rank=True` devuelve el rango medio GE sin invertir (1 = mejor, menor
    es mejor). Es el modo que usa el bucle simple para comparar candidatos.
    """
    return PopulationEvaluator(
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
        fitness_workers=fitness_workers,
        model_names=model_names,
        mutations=mutations,
        raw_rank=raw_rank,
    )


def build_evaluator(train_instances, mo_generations, mo_population_size,
                    seed=42, model_name="MOEAD", mutation=False,
                    instance_timeout=INSTANCE_TIMEOUT, normalize=True,
                    bounds=None, reference_operators=None, archive=None,
                    frame="ge", fronts_dir=None, repeat_penalty=REPEAT_PENALTY, patience=0,
                    fitness_workers=1, model_names=None, mutations=None,
                    ranking=False):
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
    @param fitness_workers: procesos para evaluar las instancias de un mismo
        candidato (default 1 = secuencial). No combinar con `--max-workers > 1`
        de LLaMEA: seria un pool dentro de otro pool.
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
        fitness_workers=fitness_workers,
        model_names=model_names,
        mutations=mutations,
        ranking=ranking,
    )
