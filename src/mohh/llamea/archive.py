"""Archivo de hipervolumen y utilidades de frentes del fitness LLaMEA.

Copia literal del bloque original de `mohh.llamea.fitness`; ese modulo
reexporta los nombres para no romper los importadores existentes.
"""

import threading

import numpy as np

from mohh.shared.fronts import _nondominated


def _nd_merge(left, right):
    """No dominados de `left` ∪ `right`, con `left` ya no dominado.

    Es el paso incremental de `experiment.py`: el frente no dominado del
    consolidado crece con cada operador evaluado, sin rehacer el filtro
    sobre todo el historico. `O(|left| * |right|)` en vez de `O(n^2)`.
    """
    from mohh.core.utilities.algorithm.MO import sign

    if left is None:
        return _nondominated(right) if right is not None else None
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0:
        return _nondominated(right)
    if len(right) == 0:
        return left
    adj_left = left * sign
    adj_right = right * sign
    # `dom_lr[i, j]`: el punto `i` de la izquierda domina al `j` de la derecha.
    dom_lr = (
        np.all(adj_left[:, None, :] >= adj_right[None, :, :], axis=2)
        & np.any(adj_left[:, None, :] > adj_right[None, :, :], axis=2)
    )
    # `dom_rl[j, i]`: el punto `j` de la derecha domina al `i` de la izquierda.
    dom_rl = (
        np.all(adj_right[:, None, :] >= adj_left[None, :, :], axis=2)
        & np.any(adj_right[:, None, :] > adj_left[None, :, :], axis=2)
    )
    dom_rr = (
        np.all(adj_right[:, None, :] >= adj_right[None, :, :], axis=2)
        & np.any(adj_right[:, None, :] > adj_right[None, :, :], axis=2)
    )
    np.fill_diagonal(dom_rr, False)
    keep_left = ~dom_rl.any(axis=0)                          # (|left|,)
    keep_right = ~dom_lr.any(axis=0) & ~dom_rr.any(axis=0)  # (|right|,)
    return np.vstack([left[keep_left], right[keep_right]])


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
