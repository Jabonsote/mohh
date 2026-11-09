# Operadores Best

## GE — Experiment_001

### Fenotipo

```text
order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,move_right(order_based(fill_first_occurring,preserve_elements,x,map_list(order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,reverse(order_based(fill_first_occurring,preserve_segments,x,move_right(y)))))))),order_based(fill_first_occurring,preserve_elements,x,x)))))))))
```

### Operaciones

| Operación | Significado |
|---|---|
| `order_based(fill, preserve, x, y)` | Aplica `preserve` a `x` e `y`; rellena cada parte preservada con el otro padre usando `fill`. Devuelve dos hijos. |
| `preserve_elements(x)` | Elige `k` posiciones aleatorias y conserva sus valores; el resto queda a `-1`. |
| `preserve_segments(x)` | Elige un segmento aleatorio `[i:j]` y lo conserva; el resto queda a `-1`. |
| `fill_first_occurring(preserved, filler)` | Rellena cada `-1` con el primer elemento de `filler` que no esté ya presente. |
| `move_right(x)` | Rotación a la derecha: el último elemento pasa al principio. |
| `reverse(x)` | Invierte el orden de los elementos. |
| `map_list(x, y)` | `new_x = x[y]`, `new_y = y[x]`; requiere índices válidos. |
| `alternate_elements(x, y)` | Alterna elementos de `x` e `y` y elimina duplicados. |
| `verify_inputs(z)` | Si `z` es una tupla, toma el primer elemento; normaliza la entrada. |

### Python

```python
import numpy as np

def crossover(x, y):
    x = verify_inputs(x)
    y = verify_inputs(y)
    return order_based(
        fill_first_occurring, preserve_elements, y,
        order_based(
            fill_first_occurring, preserve_elements, x,
            order_based(
                fill_first_occurring, preserve_elements, x,
                order_based(
                    fill_first_occurring, preserve_segments, x,
                    order_based(
                        fill_first_occurring, preserve_elements, x,
                        move_right(
                            order_based(
                                fill_first_occurring, preserve_elements, x,
                                map_list(
                                    order_based(
                                        fill_first_occurring, preserve_segments, y,
                                        order_based(
                                            fill_first_occurring, preserve_segments, y,
                                            order_based(
                                                fill_first_occurring, preserve_segments, x,
                                                order_based(
                                                    fill_first_occurring, preserve_segments, x,
                                                    order_based(
                                                        fill_first_occurring, preserve_elements, x,
                                                        reverse(
                                                            order_based(
                                                                fill_first_occurring,
                                                                preserve_segments, x,
                                                                move_right(y),
                                                            )
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                    order_based(
                                        fill_first_occurring, preserve_elements, x, x
                                    ),
                                ),
                            )
                        ),
                    ),
                ),
            ),
        ),
    )
```

Módulo completo con las funciones auxiliares: [`operadores_ge1.py`](operadores_ge1.py).

## GE — Experiment_002

### Fenotipo

```text
order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,y,alternate_elements(order_based(fill_first_occurring,preserve_segments,x,x),order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,reverse(map_list(y,reverse(y)))))))))))
```

### Operaciones

| Operación | Significado |
|---|---|
| `order_based(fill, preserve, x, y)` | Aplica `preserve` a `x` e `y`; rellena cada parte preservada con el otro padre usando `fill`. Devuelve dos hijos. |
| `preserve_elements(x)` | Elige `k` posiciones aleatorias y conserva sus valores; el resto queda a `-1`. |
| `preserve_segments(x)` | Elige un segmento aleatorio `[i:j]` y lo conserva; el resto queda a `-1`. |
| `fill_first_occurring(preserved, filler)` | Rellena cada `-1` con el primer elemento de `filler` que no esté ya presente. |
| `move_right(x)` | Rotación a la derecha: el último elemento pasa al principio. |
| `reverse(x)` | Invierte el orden de los elementos. |
| `map_list(x, y)` | `new_x = x[y]`, `new_y = y[x]`; requiere índices válidos. |
| `alternate_elements(x, y)` | Alterna elementos de `x` e `y` y elimina duplicados. |
| `verify_inputs(z)` | Si `z` es una tupla, toma el primer elemento; normaliza la entrada. |

### Python

```python
import numpy as np

def crossover(x, y):
    x = verify_inputs(x)
    y = verify_inputs(y)
    return order_based(
        fill_first_occurring, preserve_elements, x,
        order_based(
            fill_first_occurring, preserve_elements, y,
            order_based(
                fill_first_occurring, preserve_elements, y,
                alternate_elements(
                    order_based(
                        fill_first_occurring, preserve_segments, x, x
                    ),
                    order_based(
                        fill_first_occurring, preserve_elements, x,
                        order_based(
                            fill_first_occurring, preserve_segments, y,
                            order_based(
                                fill_first_occurring, preserve_elements, x,
                                order_based(
                                    fill_first_occurring, preserve_segments, x,
                                    reverse(map_list(y, reverse(y))),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
```

Módulo completo con las funciones auxiliares: [`operadores_ge2.py`](operadores_ge2.py).

## LLaMEA — best_so_far.py

```python
import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))

    child1 = np.full(n, -1, dtype=x.dtype)
    child2 = np.full(n, -1, dtype=x.dtype)

    child1[a:b] = x[a:b]
    child2[a:b] = x[a:b]

    pos_x = np.empty(n, dtype=int)
    pos_y = np.empty(n, dtype=int)
    for i in range(n):
        pos_x[x[i]] = i
        pos_y[y[i]] = i

    not_in_child1 = [v for v in y if v not in child1[a:b]]
    used1 = set(child1[a:b])

    i = 0
    idx = 0
    while i < n and idx < len(not_in_child1):
        if child1[i] == -1:
            if i > 0 and child1[i-1] != -1:
                prev = child1[i-1]
                candidates = []
                for v in not_in_child1[idx:]:
                    if pos_x[v] == pos_x[prev] + 1 or pos_x[v] == pos_x[prev] - 1:
                        candidates.append(v)
                if candidates:
                    chosen = candidates[0]
                    child1[i] = chosen
                    not_in_child1.remove(chosen)
                else:
                    child1[i] = not_in_child1[idx]
                    idx += 1
            else:
                child1[i] = not_in_child1[idx]
                idx += 1
        i += 1

    not_in_child2 = [v for v in x if v not in child2[a:b]]
    used2 = set(child2[a:b])

    i = 0
    idx = 0
    while i < n and idx < len(not_in_child2):
        if child2[i] == -1:
            if i > 0 and child2[i-1] != -1:
                prev = child2[i-1]
                candidates = []
                for v in not_in_child2[idx:]:
                    if pos_y[v] == pos_y[prev] + 1 or pos_y[v] == pos_y[prev] - 1:
                        candidates.append(v)
                if candidates:
                    chosen = candidates[0]
                    child2[i] = chosen
                    not_in_child2.remove(chosen)
                else:
                    child2[i] = not_in_child2[idx]
                    idx += 1
            else:
                child2[i] = not_in_child2[idx]
                idx += 1
        i += 1

    def apply_cycle_reuse(child, parent):
        visited = np.zeros(n, dtype=bool)
        for start in range(n):
            if not visited[start]:
                cycle = []
                current = start
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    pos = np.where(parent == child[current])[0]
                    if len(pos) > 0:
                        current = pos[0]
                    else:
                        current = -1
                    if current == start:
                        break
                if len(cycle) > 1:
                    for i in range(len(cycle)):
                        child[cycle[i]] = parent[cycle[i % len(cycle)]]

    apply_cycle_reuse(child1, x)
    apply_cycle_reuse(child2, x)

    assert len(set(child1)) == n and all(v in child1 for v in range(n)), "child1 not a permutation"
    assert len(set(child2)) == n and all(v in child2 for v in range(n)), "child2 not a permutation"

    return (child1, child2)
```
