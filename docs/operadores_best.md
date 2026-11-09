# Operadores Best

## Operaciones Python

### order_based

```python
def order_based(fill_method, preserve_method, x, y):
    x = verify_inputs(x)
    y = verify_inputs(y)

    preserved_data_x = preserve_method(x)
    preserved_data_y = preserve_method(y)

    new_x = fill_method(preserved_data_x, y)
    new_y = fill_method(preserved_data_y, x)

    return new_x, new_y
```

### preserve_elements

```python
def preserve_elements(x):
    x = verify_inputs(x)

    k = np.random.randint(1, len(x))
    x = np.array(x)

    selected = np.random.choice(len(x), size=k, replace=False)

    result = -np.ones_like(x)
    result[selected] = x[selected]

    return result
```

### preserve_segments

```python
def preserve_segments(x):
    x = verify_inputs(x)

    i = 0
    j = len(x) - 1
    while (i == 0 and j == len(x) - 1) or (i == j):
        selected = np.sort(np.random.choice(len(x), size=2, replace=False))
        i, j = selected
        j -= 1

    result = -np.ones_like(x)
    result[i:j] = x[i:j]

    return result
```

### fill_first_occurring

```python
def fill_first_occurring(preserved, filler):
    preserved = verify_inputs(preserved)
    filler = verify_inputs(filler)

    existing_values = set(preserved[preserved != -1])

    j = 0
    for i in range(len(preserved)):
        while preserved[i] == -1 and j < len(filler):
            if filler[j] not in existing_values:
                preserved[i] = filler[j]
            j += 1

    return preserved
```

### move_right

```python
def move_right(x):
    x = verify_inputs(x)
    return np.concatenate(([x[-1]], x[:-1]))
```

### reverse

```python
def reverse(x):
    x = verify_inputs(x)
    return x[::-1]
```

### map_list

```python
def map_list(x, y):
    x = verify_inputs(x)
    y = verify_inputs(y)

    if not np.all((y >= 0) & (y < len(x))):
        raise ValueError("All elements of y must be valid indices for x and y.")

    if not np.all((x >= 0) & (x < len(y))):
        raise ValueError("All elements of x must be valid indices for x and y.")

    new_x = x[y]
    new_y = y[x]

    return new_x, new_y
```

### alternate_elements

```python
def alternate_elements(x, y):
    x = verify_inputs(x)
    y = verify_inputs(y)

    new_x = []
    new_y = []

    for i in range(len(x)):
        if np.random.rand() < 0.5:
            new_x.extend([x[i], y[i]])
            new_y.extend([y[i], x[i]])
        else:
            new_x.extend([y[i], x[i]])
            new_y.extend([x[i], y[i]])

    new_x, new_y = np.array(new_x), np.array(new_y)

    _, indx = np.unique(new_x, return_index=True)
    new_x = new_x[np.sort(indx)]

    _, indx = np.unique(new_y, return_index=True)
    new_y = new_y[np.sort(indx)]

    return new_x, new_y
```

### verify_inputs

```python
def collapse(x, y=None):
    return x

def verify_inputs(z):
    if isinstance(z, tuple):
        z = collapse(z[0], z[1])
    return z
```

## GE — Experiment_001

### Fenotipo

```text
order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,move_right(order_based(fill_first_occurring,preserve_elements,x,map_list(order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_segments,x,order_based(fill_first_occurring,preserve_elements,x,reverse(order_based(fill_first_occurring,preserve_segments,x,move_right(y)))))))),order_based(fill_first_occurring,preserve_elements,x,x)))))))))
```

### Python

```python
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

Módulo completo: [`operadores_ge1.py`](operadores_ge1.py).

## GE — Experiment_002

### Fenotipo

```text
order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_elements,y,order_based(fill_first_occurring,preserve_elements,y,alternate_elements(order_based(fill_first_occurring,preserve_segments,x,x),order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,y,order_based(fill_first_occurring,preserve_elements,x,order_based(fill_first_occurring,preserve_segments,x,reverse(map_list(y,reverse(y)))))))))))
```

### Python

```python
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

Módulo completo: [`operadores_ge2.py`](operadores_ge2.py).

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
