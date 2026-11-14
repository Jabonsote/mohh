"""Priors simples y correctos: cruzamientos clasicos para permutaciones.

Los priors de GE son utiles para el ranking, pero demasiado profundos para un
modelo pequeno: el LLM los copia y suele romper el contrato. Estos priors son
los cruces clasicos (OX, PMX, CX, PBX, ERX) en version compacta y validada; se
usan como ``reference code`` en la plantilla BAG.
"""
from __future__ import annotations

OX = """import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]
    used = set(x[a:b].tolist())
    fill = [v for v in y if v not in used]
    positions = list(range(b, n)) + list(range(0, a))
    for pos, value in zip(positions, fill):
        child[pos] = value
    return child
"""

PMX = """import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]
    pos_y = {int(v): i for i, v in enumerate(y)}
    pos_x = {int(v): i for i, v in enumerate(x)}
    for i in range(a, b):
        value = int(y[i])
        if value in set(x[a:b].tolist()):
            continue
        pos = i
        while child[pos] != -1:
            pos = pos_y[int(x[pos])]
        child[pos] = value
    for i in range(n):
        if child[i] == -1:
            child[i] = y[i] if y[i] not in child else x[i]
    # Reparar duplicados dejados por el mapeo parcial.
    seen = set()
    missing = [v for v in range(n) if v not in child]
    for i in range(n):
        value = int(child[i])
        if value in seen or value < 0 or value >= n:
            child[i] = missing.pop(0)
        seen.add(int(child[i]))
    return child
"""

CX = """import numpy as np

def crossover(x, y):
    n = len(x)
    child = np.full(n, -1, dtype=x.dtype)
    pos_y = {int(v): i for i, v in enumerate(y)}
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = pos_y[int(x[i])]
        take_x = bool(np.random.randint(2))
        for i in cycle:
            child[i] = x[i] if take_x else y[i]
    missing = [v for v in range(n) if v not in child]
    for i in range(n):
        if child[i] == -1:
            child[i] = missing.pop(0)
    return child
"""

PBX = """import numpy as np

def crossover(x, y):
    n = len(x)
    k = np.random.randint(1, n)
    keep = set(np.random.choice(n, k, replace=False).tolist())
    child = np.full(n, -1, dtype=x.dtype)
    for i in keep:
        child[i] = x[i]
    used = set(child[child != -1].tolist())
    fill = [v for v in y if v not in used]
    positions = [i for i in range(n) if child[i] == -1]
    for pos, value in zip(positions, fill):
        child[pos] = value
    return child
"""

ERX = """import numpy as np

def crossover(x, y):
    n = len(x)
    neighbours = {i: set() for i in range(n)}
    for parent in (x, y):
        for i in range(n):
            value = int(parent[i])
            neighbours[value].add(int(parent[(i - 1) % n]))
            neighbours[value].add(int(parent[(i + 1) % n]))
    child = np.empty(n, dtype=x.dtype)
    current = int(x[0])
    used = set()
    for k in range(n):
        child[k] = current
        used.add(current)
        for value in neighbours:
            neighbours[value].discard(current)
        candidates = [v for v in neighbours[current] if v not in used]
        if candidates:
            current = min(candidates,
                          key=lambda v: (len(neighbours[v]), np.random.rand()))
        else:
            rest = [v for v in range(n) if v not in used]
            current = int(np.random.choice(rest)) if rest else current
    return child
"""

SIMPLE_PRIORS = [
    ("OX", OX),
    ("PMX", PMX),
    ("CX", CX),
    ("PBX", PBX),
    ("ERX", ERX),
]


def load_simple_priors():
    """Devuelve [(nombre, codigo)] de cruces clasicos validos."""
    from mohh.llamea.fitness import compile_operator
    valid = []
    for name, code in SIMPLE_PRIORS:
        try:
            compile_operator(code)
        except Exception:  # noqa: BLE001 - un prior roto no debe romper la busqueda
            continue
        valid.append((name, code))
    return valid
