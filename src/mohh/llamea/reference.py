"""Operadores de cruce de referencia para normalizar el hipervolumen.

El proyecto original calcula el hipervolumen **normalizando** cada frente con los
limites (min/max por objetivo) del frente consolidado de *todos* los operadores
evaluados, y usa un punto de nadir fijo ``[1.1]*K`` (ver
``mohh/evaluation/experiment.py``). Durante una busqueda con LLaMEA no existe
todavia ese frente consolidado, asi que aqui se construye una **referencia
estacionaria**: se evaluan una vez estos operadores clasicos sobre las mismas
instancias y con el mismo MOEA, se consolida su frente no dominado y de ahi
salen los limites de normalizacion.

Asi el HV de todos los candidatos es comparable entre si y con el proyecto
original (misma normalizacion, mismo nadir), en lugar de normalizar cada frente
contra si mismo.

Los cinco operadores cubren las familias que el prompt describe (orden,
posicion, adyacencia y ciclos) y son deterministas en su contrato: devuelven una
permutacion valida de ``range(n)``.
"""

import numpy as np


def ox(x, y):
    """Order Crossover: preserva el orden relativo de un segmento."""
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]
    used = set(x[a:b].tolist())
    fill = [v for v in y if v not in used]
    positions = list(range(b, n)) + list(range(0, a))
    for pos, val in zip(positions, fill):
        child[pos] = val
    return child


def pbx(x, y):
    """Position-Based Crossover: preserva posiciones absolutas de x."""
    n = len(x)
    k = int(np.random.randint(1, n)) if n > 1 else 1
    idx = np.random.choice(n, size=k, replace=False)
    child = np.full(n, -1, dtype=x.dtype)
    child[idx] = x[idx]
    used = set(x[idx].tolist())
    fill = [v for v in y if v not in used]
    free = [i for i in range(n) if child[i] == -1]
    for pos, val in zip(free, fill):
        child[pos] = val
    return child


def pmx(x, y):
    """Partially Mapped Crossover: preserva un segmento y su mapeo."""
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    b += 1  # intervalo inclusivo [a, b)
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]
    segment = set(x[a:b].tolist())

    for i in range(a, b):
        if y[i] in segment:
            continue
        j = i
        while child[j] != -1:
            j = int(np.where(y == x[j])[0][0])
        child[j] = y[i]

    for i in range(n):
        if child[i] == -1:
            child[i] = y[i]
    return child


def cx(x, y):
    """Cycle Crossover: intercambia ciclos completos de la permutacion."""
    n = len(x)
    pos_in_x = {int(v): i for i, v in enumerate(x)}
    child = np.full(n, -1, dtype=x.dtype)

    start = int(np.random.randint(n))
    idx = start
    while True:
        child[idx] = x[idx]
        idx = pos_in_x[int(y[idx])]
        if idx == start:
            break

    for i in range(n):
        if child[i] == -1:
            child[i] = y[i]
    return child


def erx(x, y):
    """Edge Recombination: preserva vecinos (pares adyacentes) de ambos padres."""
    n = len(x)
    adjacency = {i: set() for i in range(n)}
    for parent in (x, y):
        for i in range(n):
            a = int(parent[i])
            b = int(parent[(i + 1) % n])
            adjacency[a].add(b)
            adjacency[b].add(a)

    used = set()
    current = int(np.random.choice(x))
    child = []
    for _ in range(n):
        child.append(current)
        used.add(current)
        for neighbours in adjacency.values():
            neighbours.discard(current)
        candidates = adjacency[current] - used
        if candidates:
            current = min(candidates, key=lambda c: len(adjacency[c]))
        else:
            remaining = [v for v in range(n) if v not in used]
            if remaining:
                current = int(np.random.choice(remaining))

    return np.array(child, dtype=x.dtype)


# Panel por defecto: cubre orden (OX), posicion (PBX), mapeo (PMX),
# ciclos (CX) y adyacencia (ERX).
REFERENCE_OPERATORS = {
    "ox": ox,
    "pbx": pbx,
    "pmx": pmx,
    "cx": cx,
    "erx": erx,
}


def namespace_for(func):
    """Namespace minimo compatible con `LLMOperator` (solo necesita `crossover`)."""
    return {"crossover": func, "np": np}
