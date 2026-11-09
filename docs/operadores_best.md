# Operadores Best: evolución gramatical y LLaMEA

Documento de referencia con los operadores `Best` obtenidos en los
experimentos de evolución gramatical (GE) y en la búsqueda LLaMEA.

- GE: `experimentos_001_002/Experiment_001` y `Experiment_002`.
- LLaMEA: `results/searches/llamea_final_20261104_104217`.

---

## 1. GE — `Experiment_001` (Best)

Representación: fenotipo de evolución gramatical (`grammars/naturals.bnf`).
Evaluación: `HH_Operator` dentro de `mohh/evaluation/experiment.py`.

```text
order_based(fill_first_occurring,preserve_elements,y,
  order_based(fill_first_occurring,preserve_elements,x,
    order_based(fill_first_occurring,preserve_elements,x,
      order_based(fill_first_occurring,preserve_segments,x,
        order_based(fill_first_occurring,preserve_elements,x,
          move_right(order_based(fill_first_occurring,preserve_elements,x,
            map_list(order_based(fill_first_occurring,preserve_segments,y,
              order_based(fill_first_occurring,preserve_segments,y,
                order_based(fill_first_occurring,preserve_segments,x,
                  order_based(fill_first_occurring,preserve_segments,x,
                    order_based(fill_first_occurring,preserve_elements,x,
                      reverse(order_based(fill_first_occurring,preserve_segments,x,
                        move_right(y)))))))),
            order_based(fill_first_occurring,preserve_elements,x,x)))))))))
```

Configuración de la búsqueda GE asociada:

```text
MO_MODEL:            MOEAD
MO_GENERATIONS:      300
MO_POPULATION_SIZE:  105
MO_MUTATION_BOOL:    False
GENERATIONS:         30
POPULATION_SIZE:     50
ELITE_SIZE:          15
Instancias train:    80
```

---

## 2. GE — `Experiment_002` (Best)

Representación: fenotipo de evolución gramatical.
Evaluación: `HH_Operator`.

```text
order_based(fill_first_occurring,preserve_elements,x,
  order_based(fill_first_occurring,preserve_elements,y,
    order_based(fill_first_occurring,preserve_elements,y,
      alternate_elements(
        order_based(fill_first_occurring,preserve_segments,x,x),
        order_based(fill_first_occurring,preserve_elements,x,
          order_based(fill_first_occurring,preserve_segments,y,
            order_based(fill_first_occurring,preserve_elements,x,
              order_based(fill_first_occurring,preserve_segments,x,
                reverse(map_list(y,reverse(y))))))))))
```

Los operadores `Middle` y `Worst` de cada experimento están en el mismo
archivo `MOEAD_operators.txt`.

---

## 3. LLaMEA — `best_so_far.py`

Corrida: `llamea_final_20261104_104217`
Generación: 1000
Fitness: `-6.0`
Descripción registrada:

```text
A hybrid crossover combining order-based segment inheritance,
adjacency-aware edge recycling, and cycle-based structure reuse
for robust mQAP performance.
```

Código:

```python
import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))

    # Initialize children
    child1 = np.full(n, -1, dtype=x.dtype)
    child2 = np.full(n, -1, dtype=x.dtype)

    # Inherit ordered segment from x to both children (order preservation)
    child1[a:b] = x[a:b]
    child2[a:b] = x[a:b]

    # Build position maps for efficient lookup
    pos_x = np.empty(n, dtype=int)
    pos_y = np.empty(n, dtype=int)
    for i in range(n):
        pos_x[x[i]] = i
        pos_y[y[i]] = i

    # Fill child1 with elements from y using adjacency-aware edge recycling
    not_in_child1 = [v for v in y if v not in child1[a:b]]
    used1 = set(child1[a:b])

    # Use adjacency-aware filling: prefer elements adjacent to previous in child1
    i = 0
    idx = 0
    while i < n and idx < len(not_in_child1):
        if child1[i] == -1:
            # Prefer elements adjacent in x to previously placed elements
            if i > 0 and child1[i-1] != -1:
                prev = child1[i-1]
                # Look for neighbors in x (preserve order)
                candidates = []
                for v in not_in_child1[idx:]:
                    if pos_x[v] == pos_x[prev] + 1 or pos_x[v] == pos_x[prev] - 1:
                        candidates.append(v)
                if candidates:
                    chosen = candidates[0]
                    child1[i] = chosen
                    not_in_child1.remove(chosen)
                else:
                    # Fallback: place next available element
                    child1[i] = not_in_child1[idx]
                    idx += 1
            else:
                child1[i] = not_in_child1[idx]
                idx += 1
        i += 1

    # Fill child2 with elements from x using adjacency-aware edge recycling
    not_in_child2 = [v for v in x if v not in child2[a:b]]
    used2 = set(child2[a:b])

    i = 0
    idx = 0
    while i < n and idx < len(not_in_child2):
        if child2[i] == -1:
            # Prefer elements adjacent in y to previously placed elements
            if i > 0 and child2[i-1] != -1:
                prev = child2[i-1]
                # Look for neighbors in y (preserve order)
                candidates = []
                for v in not_in_child2[idx:]:
                    if pos_y[v] == pos_y[prev] + 1 or pos_y[v] == pos_y[prev] - 1:
                        candidates.append(v)
                if candidates:
                    chosen = candidates[0]
                    child2[i] = chosen
                    not_in_child2.remove(chosen)
                else:
                    # Fallback: place next available element
                    child2[i] = not_in_child2[idx]
                    idx += 1
            else:
                child2[i] = not_in_child2[idx]
                idx += 1
        i += 1

    # Apply cycle-based structure reuse to improve permutation structure
    def apply_cycle_reuse(child, parent):
        visited = np.zeros(n, dtype=bool)
        for start in range(n):
            if not visited[start]:
                cycle = []
                current = start
                while not visited[current]:
                    visited[current] = True
                    cycle.append(current)
                    # Find where child[current] appears in parent
                    pos = np.where(parent == child[current])[0]
                    if len(pos) > 0:
                        current = pos[0]
                    else:
                        current = -1
                    if current == start:
                        break
                # If we have a valid cycle, try to reuse it
                if len(cycle) > 1:
                    # Reorder the cycle to match parent structure
                    for i in range(len(cycle)):
                        child[cycle[i]] = parent[cycle[i % len(cycle)]]

    # Apply cycle reuse to improve structure
    apply_cycle_reuse(child1, x)
    apply_cycle_reuse(child2, x)

    # Final validation
    assert len(set(child1)) == n and all(v in child1 for v in range(n)), "child1 not a permutation"
    assert len(set(child2)) == n and all(v in child2 for v in range(n)), "child2 not a permutation"

    return (child1, child2)
```

---

## 4. LLaMEA — `llamea_best_operator.py`

Diferencia respecto al anterior: añade un bloque de refinamiento local
aleatorio. El resto del operador es idéntico.

```python
    # Local refinement using position-based swaps to improve QAP-specific neighborhood structures
    def refine_offspring(child, parent_x, parent_y):
        for _ in range(n // 4):
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i != j and np.random.rand() < 0.3:
                # Swap and evaluate improvement
                old_val_i = child[i]
                old_val_j = child[j]
                child[i], child[j] = child[j], child[i]

                # Revert if it degrades position mapping to x or y
                if np.random.rand() < 0.5:
                    child[i], child[j] = child[j], child[i]

    # Apply refinement
    refine_offspring(child1, x, y)
    refine_offspring(child2, x, y)
```

Observaciones:

- `old_val_i` y `old_val_j` no se utilizan.
- El intercambio se revierte con probabilidad 0.5.
- No se evalúa si el cambio mejora la función objetivo.
- El efecto neto es una mutación aleatoria, no un refinamiento guiado.

---

## 5. Comparación

| Aspecto | GE Best | LLaMEA Best |
|---|---|---|
| Representación | fenotipo GE | función Python |
| Evaluación | `HH_Operator` | `LLMOperator` |
| Estructura | composición de `order_based` | segmento + relleno por adyacencia + ciclos |
| Mutación adicional | no | `refine_offspring` aleatorio |
| MOEA de búsqueda | MOEAD 300/105 | MOEAD 300/105 |
| Mutación del MOEA | no | no |
| Instancias train | 80 | 80 |
| Indicador de búsqueda | HV + rango medio | HV + rango medio |

Evaluación final común:

```text
solvers:            MOEAD, NSGAII, SMSEMOA
mutaciones:         con y sin Swap_Mutation
presupuestos:       10k, 30k, 50k
réplicas:           31
instancias test:    20
MO generations:     500
MO population size: 105
marco:              experiment.py (normalizado, nadir [1.1] * K)
```
