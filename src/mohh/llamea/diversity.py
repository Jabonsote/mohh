"""Diversidad **conductual** de operadores de cruce (no del texto del codigo).

Motivacion
----------
Contar hashes de codigo es una medida pobre: dos operadores distintos pueden
hacer exactamente lo mismo (p. ej. `return x.copy()` y una copia con un bucle) y
dos variantes sintacticas del mismo mecanismo cuentan como "nuevas". En las
corridas reales el fallo dominante no fue la falta de texto distinto sino el
**anclaje conductual**: 108 generaciones devolviendo el mismo OX.

Este modulo mide lo que el operador *hace* sobre pares de padres de permutacion
(tamanos reales del dominio mQAP, 11..100) y lo resume en un vector de
caracteristicas en [0, 1]:

- recombincion: distancia de Hamming al padre mas cercano;
- posiciones: fraccion de posiciones copiadas de x / de y;
- orden: similitud de orden (Kendall tau) con x / con y;
- adyacencia: fraccion de aristas (vecinos ciclicos) de x / de y;
- union de aristas: fraccion de aristas del hijo presentes en x o y;
- rachas: longitud de la racha mas larga copiada de x / de y;
- cambios de padre: cuantas veces alterna la fuente x/y.

Con ese vector se calculan metricas de diversidad y se puede guiar al LLM:

1. **Novedad conductual**: distancia al operador ya evaluado mas cercano. Un
   valor bajo significa "mismo mecanismo con otro texto".
2. **Archivo MAP-Elites conductual**: rejilla recombincion x estructura; se
   guarda el mejor HV por celda. La cobertura dice que regiones del espacio de
   comportamiento no se han explorado.
3. **Familias**: order / position / adjacency / exploratory / copy / mixed, para
   pedir en el feedback la familia menos representada.
4. **Hooks nativos de LLaMEA**: `solution_distance` y
   `solution_behavior_descriptor` leen el fingerprint que el evaluador deja en
   `Solution.metadata`. Asi `--niching novelty`, `clearing` y `map_elites`
   operan sobre comportamiento, no sobre lineas de codigo.

Evidencia
---------
- Novedad conductual y archivos de calidad-diversidad: Lehman & Stanley (2011),
  Mouret & Clune (2015, MAP-Elites).
- Reflexion con LLM sobre el resultado (no solo el score): ReEvo (Ye et al.,
  2024), EoH (Liu et al., 2024).
- LLaMEA (van Stein & Baeck, 2024) expone `distance_metric` y
  `behavior_descriptor`; aqui se conectan a esta medida.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

#: Tamanos de instancia del dominio mQAP (las instancias van de 11 a 100).
DEFAULT_SIZES = (11, 24, 50, 100)

#: Nombres y orden del vector de caracteristicas. No cambiar el orden sin
#: actualizar `FEATURE_INDEX` y las pruebas.
FEATURE_NAMES = (
    "recombination",
    "distance_x",
    "distance_y",
    "parent_bias",
    "position_x",
    "position_y",
    "order_x",
    "order_y",
    "adjacency_x",
    "adjacency_y",
    "edge_union",
    "run_x",
    "run_y",
    "switch_rate",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}

#: Familias legibles. `copy` se reserva para operadores degenerados.
FAMILIES = ("order", "position", "cycle", "adjacency", "exploratory",
            "mixed", "copy")

_FAMILY_HINTS = {
    "order": ("an order-based crossover (OX/OBX-style) that preserves the "
              "relative order of a subset of elements"),
    "position": ("a position-based crossover (PBX/PMX-style) that preserves "
                 "absolute positions or a mapped segment of one parent"),
    "cycle": ("a cycle-based crossover (CX-style) that exchanges complete cycles "
              "of the permutation"),
    "adjacency": ("an edge/adjacency-based crossover (ERX-style) that preserves "
                  "neighbours shared by the parents"),
    "exploratory": ("a high-recombination operator whose child is unlike both "
                    "parents (not a small perturbation)"),
    "mixed": ("a hybrid that deliberately combines order, position or adjacency "
              "preservation"),
}

#: Etiqueta de familia de cada operador clasico. El clasificador asigna un
#: fingerprint nuevo a la familia del operador de referencia mas cercano; no es
#: una caja negra, es "a que mecanismo clasico se parece mas".
_REFERENCE_FAMILY = {
    "ox": "order",
    "pbx": "position",
    "pmx": "position",
    "cx": "cycle",
    "erx": "adjacency",
}
_REFERENCE_CACHE = None


def _order_similarity(child, parent) -> float:
    """Kendall tau normalizado: 1 = mismo orden relativo, 0.5 = azar."""
    n = len(parent)
    if n < 2:
        return 1.0
    pos = np.empty(n, dtype=np.int64)
    pos[np.asarray(parent, dtype=np.int64)] = np.arange(n)
    sequence = pos[np.asarray(child, dtype=np.int64)]
    inversions = 0
    for i in range(n - 1):
        inversions += int(np.sum(sequence[i + 1:] < sequence[i]))
    return 1.0 - inversions / (n * (n - 1) / 2.0)


def _adjacency_preservation(child, parent) -> float:
    """Fraccion de aristas ciclicas del hijo que ya existian en `parent`."""
    n = len(child)
    if n < 2:
        return 1.0
    edges = set()
    for i in range(n):
        a, b = int(parent[i]), int(parent[(i + 1) % n])
        edges.add((a, b))
        edges.add((b, a))
    kept = sum((int(child[i]), int(child[(i + 1) % n])) in edges for i in range(n))
    return kept / n


def _edge_union_preservation(child, x, y) -> float:
    """Fraccion de aristas del hijo que aparecen en x o en y (union)."""
    n = len(child)
    if n < 2:
        return 1.0
    edges = set()
    for parent in (x, y):
        for i in range(n):
            a, b = int(parent[i]), int(parent[(i + 1) % n])
            edges.add((a, b))
            edges.add((b, a))
    kept = sum((int(child[i]), int(child[(i + 1) % n])) in edges for i in range(n))
    return kept / n


def _longest_equal_run(child, parent) -> float:
    """Racha mas larga de posiciones consecutivas copiadas, normalizada por n."""
    n = len(child)
    if n == 0:
        return 0.0
    best = current = 0
    for i in range(n):
        if child[i] == parent[i]:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best / n


def _switch_rate(child, x, y) -> float:
    """Fraccion de transiciones que alternan la fuente x/y en posiciones fijas."""
    n = len(child)
    if n < 2:
        return 0.0
    signs = []
    for i in range(n):
        if child[i] == x[i] and child[i] != y[i]:
            signs.append(1)
        elif child[i] == y[i] and child[i] != x[i]:
            signs.append(-1)
        else:
            signs.append(0)
    switches = sum(1 for a, b in zip(signs, signs[1:]) if a and b and a != b)
    return switches / (n - 1)


def _pair_features(child, x, y) -> np.ndarray:
    """Vector de caracteristicas de una sola aplicacion del operador."""
    n = len(x)
    child = np.asarray(child)
    if child.shape != x.shape:
        raise ValueError("child shape differs from parent shape")
    if not np.array_equal(np.sort(child), np.arange(n)):
        raise ValueError("child is not a permutation of range(n)")

    distance_x = float(np.mean(child != x))
    distance_y = float(np.mean(child != y))
    recombination = min(distance_x, distance_y)
    parent_bias = 0.5 + 0.5 * (distance_y - distance_x)

    values = (
        recombination,
        distance_x,
        distance_y,
        parent_bias,
        float(np.mean(child == x)),
        float(np.mean(child == y)),
        _order_similarity(child, x),
        _order_similarity(child, y),
        _adjacency_preservation(child, x),
        _adjacency_preservation(child, y),
        _edge_union_preservation(child, x, y),
        _longest_equal_run(child, x),
        _longest_equal_run(child, y),
        _switch_rate(child, x, y),
    )
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def _parent_triplets(size: int, seed: int):
    """Pares de padres que cubren azar, padres parecidos y padres invertidos."""
    rng = np.random.RandomState(seed)
    x = rng.permutation(size)
    yield x, rng.permutation(size)

    close = x.copy()
    swaps = max(1, size // 5)
    for _ in range(swaps):
        i, j = rng.choice(size, 2, replace=False)
        close[i], close[j] = close[j], close[i]
    yield x, close

    yield x, x[::-1].copy()


def behavior_fingerprint(crossover, sizes=DEFAULT_SIZES, seed=1234,
                         timeout=None):
    """Fingerprint conductual de `crossover(x, y)`, o None si no se puede medir.

    Se ejecuta el operador sobre pares de padres de los tamanos reales del
    dominio y se promedian las caracteristicas. El RNG global de numpy se guarda
    y se restaura en cada llamada (semilla fija por par), de modo que dos
    operadores se miden con los mismos numeros aleatorios y el fingerprint es
    estable entre corridas.

    `timeout` es opcional; el evaluador pasa `OPERATOR_CALL_TIMEOUT` para que un
    operador que se cuelgue solo en un par no bloquee la busqueda.
    """
    rows = []
    global_state = np.random.get_state()
    try:
        call_index = 0
        for size in sizes:
            for x, y in _parent_triplets(int(size), seed + int(size) * 101):
                call_index += 1
                np.random.seed(seed + call_index)
                try:
                    if timeout:
                        from mohh.llamea.fitness import _call_with_timeout
                        child = _call_with_timeout(crossover, timeout, x, y)
                    else:
                        child = crossover(x, y)
                except Exception:  # noqa: BLE001 - un par malo no invalida el resto
                    continue
                finally:
                    np.random.set_state(global_state)
                if isinstance(child, tuple):
                    if not child:
                        continue
                    child = child[0]
                try:
                    rows.append(_pair_features(child, x, y))
                except Exception:  # noqa: BLE001 - salida no valida en ese par
                    continue
    finally:
        np.random.set_state(global_state)

    if not rows:
        return None
    return np.clip(np.mean(np.vstack(rows), axis=0), 0.0, 1.0)


def behavior_distance(left, right) -> float:
    """Distancia euclidiana normalizada entre dos fingerprints (0..1)."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"fingerprints with different shapes: {a.shape} vs {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.linalg.norm(a - b) / math.sqrt(a.size))


def behavior_descriptor(fingerprint):
    """Descriptor 2D para MAP-Elites: (recombincion, estructura preservada)."""
    fp = np.asarray(fingerprint, dtype=float).reshape(-1)
    if fp.size != len(FEATURE_NAMES):
        raise ValueError(
            f"fingerprint has {fp.size} features; expected {len(FEATURE_NAMES)}"
        )
    recombination = float(fp[FEATURE_INDEX["recombination"]])
    structure = max(
        float(fp[FEATURE_INDEX["order_x"]]),
        float(fp[FEATURE_INDEX["order_y"]]),
        float(fp[FEATURE_INDEX["adjacency_x"]]),
        float(fp[FEATURE_INDEX["adjacency_y"]]),
    )
    return (min(max(recombination, 0.0), 1.0), min(max(structure, 0.0), 1.0))


def reference_fingerprints():
    """Fingerprints de OX/PBX/PMX/CX/ERX, calculados una sola vez."""
    global _REFERENCE_CACHE
    if _REFERENCE_CACHE is None:
        from mohh.llamea.reference import REFERENCE_OPERATORS

        _REFERENCE_CACHE = {
            name: behavior_fingerprint(operator)
            for name, operator in REFERENCE_OPERATORS.items()
        }
    return _REFERENCE_CACHE


def closest_reference(fingerprint):
    """(nombre, distancia) del operador clasico conductualmente mas cercano."""
    fp = np.asarray(fingerprint, dtype=float)
    distances = {
        name: behavior_distance(fp, reference)
        for name, reference in reference_fingerprints().items()
    }
    name = min(distances, key=distances.get)
    return name, float(distances[name])


def family_from_fingerprint(fingerprint) -> str:
    """Familia del mecanismo mas parecido (clasificador por centroides).

    Se compara el fingerprint con los cinco operadores clasicos medidos en las
    mismas condiciones. Si el operador recombina mucho y no se parece a ninguno,
    se etiqueta `exploratory`; si copia a un padre, `copy`. Es una regla
    explicita y reproducible, no un modelo entrenado.
    """
    fp = np.asarray(fingerprint, dtype=float).reshape(-1)
    if fp.size != len(FEATURE_NAMES):
        return "mixed"
    recombination = float(fp[FEATURE_INDEX["recombination"]])
    edge_union = float(fp[FEATURE_INDEX["edge_union"]])
    if recombination < 0.05:
        return "copy"
    name, distance = closest_reference(fp)
    if recombination >= 0.80 and edge_union < 0.80 and distance > 0.12:
        return "exploratory"
    return _REFERENCE_FAMILY.get(name, "mixed")


def solution_fingerprint(solution):
    """Fingerprint guardado por el evaluador en `Solution.metadata`, o None."""
    getter = getattr(solution, "get_metadata", None)
    if not callable(getter):
        return None
    try:
        raw = getter("behavior_fingerprint")
    except Exception:  # noqa: BLE001 - un candidato de prueba no debe romper LLaMEA
        return None
    if raw is None:
        return None
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        return None


def solution_distance(left, right) -> float:
    """`distance_metric` conductual para LLaMEA (fallback: distancia de codigo).

    Se pasa a `LLaMEA(distance_metric=...)`. Asi `--niching novelty`, `clearing`
    y `sharing` comparan comportamiento y no texto. Si un individuo no tiene
    fingerprint (p. ej. codigo invalido), se usa la distancia de codigo original
    de LLaMEA para no romper el contrato.
    """
    left_fp = solution_fingerprint(left)
    right_fp = solution_fingerprint(right)
    if left_fp is not None and right_fp is not None:
        try:
            return behavior_distance(left_fp, right_fp)
        except ValueError:
            pass
    from llamea.utils import code_distance

    return float(code_distance(left, right))


def solution_behavior_descriptor(solution):
    """`behavior_descriptor` conductual para MAP-Elites de LLaMEA.

    Devuelve siempre una tupla de floats: (recombincion, estructura). Si no hay
    fingerprint, cae a un descriptor de codigo para que LLaMEA no falle.
    """
    fingerprint = solution_fingerprint(solution)
    if fingerprint is not None:
        try:
            return behavior_descriptor(fingerprint)
        except (TypeError, ValueError):
            pass
    code = solution.code or ""
    return (float(len(code.splitlines())), float(len(set(code.split()))))


class BehaviorTracker:
    """Archivo incremental de fingerprints, novedad, familias y celdas.

    Es picklable (sin locks) porque el evaluador de LLaMEA puede serializarse.
    La matriz de fingerprints se preasigna para que `add` sea O(n) y no reconstruya
    `np.vstack` en cada generacion.
    """

    def __init__(self, threshold=0.08, grid_shape=(5, 5), capacity=4096):
        self.threshold = float(threshold)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self._capacity = max(int(capacity), 16)
        self._matrix = np.empty((self._capacity, len(FEATURE_NAMES)), dtype=np.float32)
        self._size = 0
        self._keys = []
        self._entries = {}
        self._clusters = []
        self._cluster_counts = []
        self._grid = {}
        self._family_counts = Counter()
        self._distance_sum = 0.0
        self._distance_count = 0
        self._novelty_sum = 0.0
        self._near_duplicates = 0

    # -- persistencia ----------------------------------------------------- #
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_matrix"] = self._matrix[: self._size].copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        capacity = max(self._capacity, self._size * 2, 16)
        matrix = np.empty((capacity, len(FEATURE_NAMES)), dtype=np.float32)
        matrix[: self._size] = self._matrix
        self._matrix = matrix
        self._capacity = capacity

    # -- consulta --------------------------------------------------------- #
    @property
    def n(self) -> int:
        return self._size

    @property
    def coverage(self) -> float:
        cells = self.grid_shape[0] * self.grid_shape[1]
        return len(self._grid) / cells if cells else 0.0

    def cell(self, fingerprint):
        recombination, structure = behavior_descriptor(fingerprint)
        return (
            min(int(recombination * self.grid_shape[0]), self.grid_shape[0] - 1),
            min(int(structure * self.grid_shape[1]), self.grid_shape[1] - 1),
        )

    def family(self, key):
        entry = self._entries.get(str(key))
        return entry["family"] if entry else None

    def nearest(self, key):
        entry = self._entries.get(str(key))
        if not entry or entry.get("nearest") is None:
            return None
        return self._entries.get(entry["nearest"])

    def hint(self) -> str:
        """Sugerencia de familia/region poco explorada, para el feedback."""
        if not self._entries:
            return ("start with an order-based operator and, in later attempts, "
                    "cover position and adjacency mechanisms")
        counts = {name: self._family_counts.get(name, 0) for name in FAMILIES
                  if name != "copy"}
        target = min(counts, key=lambda name: (counts[name], FAMILIES.index(name)))
        if counts[target] == 0:
            return f"no {target}-based mechanism yet: try {_FAMILY_HINTS[target]}"
        return (f"the least represented family is {target} "
                f"({counts[target]} operator(s)): try {_FAMILY_HINTS[target]}")

    def snapshot(self) -> dict:
        counts = np.asarray(self._cluster_counts, dtype=float)
        if counts.size:
            probabilities = counts / counts.sum()
            entropy = float(-np.sum(probabilities * np.log(probabilities)))
            effective = float(np.exp(entropy))
        else:
            entropy = 0.0
            effective = 0.0
        return {
            "n": self._size,
            "unique_behavior": int(len(self._clusters)),
            "behavior_entropy": entropy,
            "effective_behaviors": effective,
            "mean_pairwise_distance": (
                self._distance_sum / self._distance_count
                if self._distance_count else 0.0
            ),
            "mean_novelty": (self._novelty_sum / self._size) if self._size else 0.0,
            "near_duplicates": int(self._near_duplicates),
            "near_duplicate_rate": (
                self._near_duplicates / self._size if self._size else 0.0
            ),
            "archive_coverage": self.coverage,
            "occupied_cells": len(self._grid),
            "grid_shape": list(self.grid_shape),
            "family_counts": dict(self._family_counts),
            "feature_names": list(FEATURE_NAMES),
        }

    def entries_snapshot(self):
        """Entradas serializables (fingerprint incluido) para `diversity.json`."""
        result = []
        for key in self._keys:
            entry = dict(self._entries[key])
            entry["fingerprint"] = [round(float(v), 6) for v in entry["fingerprint"]]
            entry["cell"] = list(entry["cell"])
            result.append(entry)
        return result

    # -- actualizacion ---------------------------------------------------- #
    def add(self, key, fingerprint, fitness, generation=None, description=None):
        """Registra un operador y devuelve su informe de diversidad.

        Si `key` ya existe (mismo codigo), no lo duplica: actualiza el fitness si
        mejoro y devuelve `cached=True`. Asi la diversidad no se infla cuando el
        LLM repite texto.
        """
        key = str(key)
        cached = self._entries.get(key)
        if cached is not None:
            cached["fitness"] = max(float(cached["fitness"]), float(fitness))
            # `novelty=0` describe la evaluacion actual (no aporta un mecanismo
            # nuevo); la novedad original queda en la entrada del archivo.
            return {**cached, "novelty": 0.0, "cached": True,
                    "hint": self.hint(), "coverage": self.coverage}

        fp = np.asarray(fingerprint, dtype=np.float32).reshape(-1)
        if fp.size != len(FEATURE_NAMES):
            raise ValueError(
                f"fingerprint has {fp.size} features; expected {len(FEATURE_NAMES)}"
            )
        if not np.all(np.isfinite(fp)):
            raise ValueError("fingerprint contains NaN/inf")

        novelty = 1.0
        nearest = None
        if self._size:
            distances = np.linalg.norm(
                self._matrix[: self._size] - fp, axis=1
            ) / math.sqrt(len(FEATURE_NAMES))
            index = int(np.argmin(distances))
            novelty = float(distances[index])
            nearest = self._keys[index]
            self._distance_sum += float(np.sum(distances))
            self._distance_count += self._size

        if self._clusters:
            cluster_distances = [
                behavior_distance(fp, center) for center in self._clusters
            ]
            cluster_index = int(np.argmin(cluster_distances))
            if cluster_distances[cluster_index] <= self.threshold:
                count = self._cluster_counts[cluster_index]
                center = self._clusters[cluster_index]
                self._clusters[cluster_index] = (center * count + fp) / (count + 1)
                self._cluster_counts[cluster_index] = count + 1
            else:
                self._clusters.append(fp.copy())
                self._cluster_counts.append(1)
        else:
            self._clusters.append(fp.copy())
            self._cluster_counts.append(1)

        family = family_from_fingerprint(fp)
        closest_name, closest_distance = closest_reference(fp)
        cell = self.cell(fp)
        entry = {
            "key": key,
            "generation": int(generation) if generation is not None else None,
            "description": str(description or "")[:180],
            "fitness": float(fitness),
            "fingerprint": fp.copy(),
            "family": family,
            "closest_reference": closest_name,
            "closest_distance": closest_distance,
            "cell": cell,
            "novelty": novelty,
            "nearest": nearest,
        }
        self._entries[key] = entry
        self._keys.append(key)
        if self._size >= self._capacity:
            self._capacity *= 2
            matrix = np.empty((self._capacity, len(FEATURE_NAMES)), dtype=np.float32)
            matrix[: self._size] = self._matrix[: self._size]
            self._matrix = matrix
        self._matrix[self._size] = fp
        self._size += 1
        self._family_counts[family] += 1
        self._novelty_sum += novelty
        if novelty < self.threshold:
            self._near_duplicates += 1

        incumbent = self._grid.get(cell)
        if incumbent is None or entry["fitness"] > incumbent["fitness"]:
            self._grid[cell] = entry

        return {**entry, "cached": False, "coverage": self.coverage,
                "hint": self.hint()}
