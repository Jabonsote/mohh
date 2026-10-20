"""Seleccion conductual de padres y validacion de la familia objetivo.

LLaMEA 1.4 permite declarar un `Operator(number_of_parents=2)`, pero su
seleccion de padres es por fitness: el segundo padre se elige con el mismo
torneo/roulette que el primero. Eso convierte el "combine" en un refinamiento
caro de dos individuos parecidos, no en una recombinacion que explore el
espacio de comportamientos.

Esta extension (propia, no del LLaMEA original) agrega tres piezas:

1. Seleccion conductual del padre focal: un torneo que combina calidad (HV) y
   novedad conductual. No se elige solo al de mayor fitness.
2. Seleccion conductual del companero de combine: se maximiza la distancia de
   fingerprint al padre focal; si hay una familia objetivo, se prefiere un
   padre de esa familia (si existe) para aportar su mecanismo.
3. Validacion del hijo contra la familia objetivo: el evaluador mide el
   fingerprint del hijo y registra `target_family_hit`; el feedback dice si
   acerto o no. Opcionalmente (`--require-target-family`) el hijo que no
   cumple se rechaza antes de gastar el MOEA, y con `--target-penalty` se
   puede penalizar solo el ranking sin tocar el HV crudo.

No modifica el nucleo GE ni el paquete LLaMEA: la subclase solo reemplaza
`_select_parents`, agrega metadata antes de evaluar y delega el resto en
`LLaMEA`.
"""

from __future__ import annotations

import random

import numpy as np

from llamea import LLaMEA

from mohh.llamea.diversity import behavior_distance, solution_fingerprint

#: Familias que tienen sentido como objetivo de una recombinacion. `copy` es
#: un fallo y `mixed` no da una direccion concreta.
TARGET_FAMILIES = ("order", "position", "cycle", "adjacency", "exploratory")


def _metadata(solution, key, default=None):
    """Lee metadata de una `Solution` sin romper con objetos de prueba."""
    getter = getattr(solution, "get_metadata", None)
    if not callable(getter):
        return default
    try:
        value = getter(key)
    except Exception:  # noqa: BLE001 - metadata opcional
        return default
    return default if value is None else value


def _fitness(solution):
    try:
        value = float(solution.fitness)
    except (TypeError, ValueError):
        return float("-inf")
    return value if np.isfinite(value) else float("-inf")


def _family(solution):
    return _metadata(solution, "behavior_family") or "mixed"


def _novelty(solution):
    try:
        return float(_metadata(solution, "behavioral_novelty", 0.0))
    except (TypeError, ValueError):
        return 0.0


def least_represented_family(tracker, families=TARGET_FAMILIES):
    """Familia con menos operadores en el archivo conductual.

    Es determinista: en empate respeta el orden de `families`. Si el tracker no
    existe o aun no tiene entradas, empieza por `order`, la familia mas simple
    de explicar al LLM.
    """
    if tracker is None:
        return families[0]
    try:
        counts = tracker.snapshot().get("family_counts", {})
    except Exception:  # noqa: BLE001 - un tracker de prueba no debe romper
        counts = {}
    if not counts:
        return families[0]
    return min(families, key=lambda name: (int(counts.get(name, 0)),
                                           families.index(name)))


def behavioral_parent_quality(solution, novelty_weight=0.5):
    """Calidad para el torneo focal: HV con un bono por novedad conductual.

    No se usa para el ranking ni para el reporte: solo para elegir que padres
    entran a la siguiente ronda. Un operador con HV 0 y mucha novedad no debe
    ganarle a uno con HV alto; por eso el bono es multiplicativo sobre el HV y
    no un termino aditivo de la misma escala.
    """
    fitness = _fitness(solution)
    if not np.isfinite(fitness):
        return float("-inf")
    novelty = max(0.0, min(1.0, _novelty(solution)))
    return fitness * (1.0 + float(novelty_weight) * novelty)


def select_behavioral_focal(population, tournament_size=3, novelty_weight=0.5,
                            rng=None):
    """Elige un padre focal por torneo calidad + novedad conductual.

    `population` son `Solution` ya evaluadas. Si no hay fingerprints, el bono
    de novedad es cero y el comportamiento es el torneo por calidad de LLaMEA.
    """
    rng = rng or random
    population = list(population)
    if not population:
        raise ValueError("population is empty")
    size = max(1, min(int(tournament_size), len(population)))
    if size == len(population):
        candidates = population
    else:
        candidates = rng.sample(population, size)
    return max(candidates, key=lambda s: behavioral_parent_quality(s,
                                                                    novelty_weight))


def select_behavioral_partner(focal, population, target_family=None,
                              novelty_weight=0.5, exclude_ids=None, rng=None):
    """Elige el segundo padre de un combine por distancia conductual.

    Reglas, en orden:
    - se excluye el padre focal y los ya usados (`exclude_ids`);
    - si `target_family` existe y hay candidatos de esa familia distintos al
      focal, se restringe el pool a ellos (aportan el mecanismo que se quiere
      recombinar);
    - dentro del pool se maximiza la distancia de fingerprint, con un pequeno
      bono por novedad y por no repetir la familia del focal.

    Devuelve `None` si no hay un candidato medible; el llamador decide el
    fallback (LLaMEA elige con su metodo original).
    """
    rng = rng or random
    if focal is None:
        return None
    exclude_ids = set(exclude_ids or ())
    exclude_ids.add(id(focal))
    focal_code = getattr(focal, "code", None)
    focal_fp = solution_fingerprint(focal)
    focal_family = _family(focal)

    candidates = []
    for candidate in population or ():
        if id(candidate) in exclude_ids:
            continue
        if focal_code is not None and getattr(candidate, "code", None) == focal_code:
            continue
        candidates.append(candidate)
    if not candidates:
        return None

    if focal_fp is None:
        return rng.choice(candidates)

    measured = []
    for candidate in candidates:
        fingerprint = solution_fingerprint(candidate)
        if fingerprint is None:
            continue
        try:
            distance = behavior_distance(focal_fp, fingerprint)
        except ValueError:
            continue
        measured.append((candidate, fingerprint, distance))
    if not measured:
        return rng.choice(candidates)

    if target_family:
        target_pool = [
            item for item in measured
            if _family(item[0]) == target_family
            and _family(item[0]) != focal_family
        ]
        if target_pool:
            measured = target_pool

    def score(item):
        candidate, _fingerprint, distance = item
        same_family = _family(candidate) == focal_family
        novelty = max(0.0, min(1.0, _novelty(candidate)))
        # La distancia es la senal principal. El bono por cambiar de familia
        # desempata fingerprints cercanos; la novedad y el fitness solo rompen
        # empates, no dominan la eleccion.
        return (
            distance
            + (0.10 if not same_family else 0.0)
            + 0.05 * float(novelty_weight) * novelty
            + 1e-6 * max(0.0, _fitness(candidate))
        )

    return max(measured, key=score)[0]


class BehavioralLLaMEA(LLaMEA):
    """LLaMEA con seleccion conductual para el operador `combine`.

    Solo se usa cuando la busqueda incluye un operador de dos padres. Para
    `refine`/`redesign` el resto del ciclo es el original. Se mantiene
    `max_workers=1` porque la seleccion del companero necesita saber cual es el
    padre focal de la generacion actual; el job del cluster ya usa 1.
    """

    def __init__(self, *args, behavioral_parent_selection=True,
                 novelty_weight=0.5, **kwargs):
        # Estos atributos deben existir antes de que LLaMEA evalúe la población
        # inicial, porque `evaluate_fitness` consulta `_pending_target`.
        self._active_operator = None
        self._active_focal = None
        self._pending_target = None
        self.behavioral_parent_selection = bool(behavioral_parent_selection)
        self.novelty_weight = float(novelty_weight)
        super().__init__(*args, **kwargs)

    @property
    def _diversity_tracker(self):
        return getattr(self.f, "diversity", None)

    def _select_parents(self, count=None):
        # 1. El bucle de LLaMEA pide los individuos que van a evolucionar.
        #    Aqui introducimos la seleccion conductual.
        if count is None and self.behavioral_parent_selection:
            if not self.population:
                return super()._select_parents(count)
            return [
                select_behavioral_focal(
                    self.population,
                    tournament_size=getattr(self, "tournament_size", 3),
                    novelty_weight=self.novelty_weight,
                )
                for _ in range(self.n_offspring)
            ]

        # 2. Para un operador de varios padres, LLaMEA pide los companeros
        #    adicionales. Elegimos el mas distante del focal (y de la familia
        #    objetivo, si existe).
        operator = self._active_operator
        if count and operator is not None and \
                getattr(operator, "number_of_parents", 1) > 1:
            partners = []
            used = set()
            for _ in range(int(count)):
                partner = select_behavioral_partner(
                    self._active_focal,
                    self.population,
                    target_family=getattr(operator, "target_family", None),
                    novelty_weight=self.novelty_weight,
                    exclude_ids=used,
                )
                if partner is None:
                    break
                partners.append(partner)
                used.add(id(partner))
            if len(partners) < count:
                # Fallback: si la poblacion no tiene fingerprints suficientes,
                # conservar el comportamiento original de LLaMEA.
                remaining = int(count) - len(partners)
                partners.extend(super()._select_parents(remaining))
            return partners

        return super()._select_parents(count)

    def _evolve_solution(self, individual, operator):
        """Fija la familia objetivo antes de construir el prompt.

        El `Operator` de combine tiene `target_family=None` por defecto. Aqui
        se elige la familia menos representada del archivo conductual y se
        expone en `__repr__` para que el LLM sepa contra que se va a validar.
        """
        target = getattr(operator, "target_family", None)
        dynamic = getattr(operator, "dynamic_target", False)
        if getattr(operator, "name", "") == "combine" and (target is None or dynamic):
            target = least_represented_family(self._diversity_tracker)
            try:
                operator.target_family = target
            except Exception:  # noqa: BLE001 - un operador de prueba puede ser inmutable
                pass

        previous = (self._active_operator, self._active_focal,
                    self._pending_target)
        self._active_operator = operator
        self._active_focal = individual
        self._pending_target = target
        try:
            return super()._evolve_solution(individual, operator)
        finally:
            self._active_operator, self._active_focal, self._pending_target = previous

    def evaluate_fitness(self, individual):
        """Adjunta la familia objetivo antes de que el evaluador mida el hijo."""
        target = getattr(self, "_pending_target", None)
        if target:
            adder = getattr(individual, "add_metadata", None)
            if callable(adder):
                adder("target_family", target)
        return super().evaluate_fitness(individual)
