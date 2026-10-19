# Prompts finales (generados)

> Generado por `python -m mohh.llamea.export_prompts`. Edita `src/mohh/llamea/prompts.py`, no este archivo.

Los MOEA disponibles son MOEAD, NSGAII y SMSEMOA.

### ROLE_PROMPT

```
You are an expert in evolutionary computation and combinatorial optimization, specialised in designing variation (recombination) operators for permutation-encoded multi-objective problems. You write original, correct and efficient Python.
```

### TASK_PROMPT (MOEAD)

```
Your job is to design a crossover (recombination) operator for MOEA/D, a multi-objective evolutionary algorithm, applied to the multi-objective Quadratic Assignment Problem (mQAP).

# The problem
A candidate solution is a permutation p of the n facilities over the n locations. For each objective k the instance provides a flow matrix W_k (n x n) and a global distance matrix D (n x n). The cost of p under objective k is the quadratic sum
    f_k(p) = sum_i sum_j W_k[i, j] * D[p[i], p[j]].
The algorithm minimises K objectives (in this benchmark 2 <= K <= 5) at the same time and must return a set of mutually non-dominated permutations that approximates the Pareto front. There is no single optimum: a good operator must help the search spread along and converge to the front, so its offspring should inherit useful structure from BOTH parents while remaining diverse.

# Where your operator is used
Your operator replaces the crossover of MOEA/D (decomposition-based: it decomposes the problem into scalarised subproblems and performs mating within neighbourhoods). This is the same algorithm and the same problem family used by the MOHH-R operator-design pipeline. The operator is called at every generation, on instances of DIFFERENT sizes, and its quality is measured by the hypervolume of the final Pareto front. It must therefore be general, not tuned to one instance or to one fixed length.

# Exact interface
Write a Python function:

    def crossover(x, y):
        # x, y : 1-D numpy integer arrays, each a permutation of range(n),
        #        with n = len(x) = len(y).
        # return: a NEW permutation of range(n), OR a tuple (child1, child2)
        #         of two NEW permutations of range(n).

Hard requirements (any violation scores 0):
- Every returned child is a permutation of range(n): each value 0..n-1 appears exactly once and the length equals n, for ANY n >= 2.
- NEVER modify x or y in place. They are views into the algorithm's internal population: `x[i] = ...`, `x.sort()`, `x[idx] = ...` or `np.random.shuffle(x)` corrupt the search. Build the result in a fresh array (`np.empty_like`, `np.full`, `np.array(x, copy=True)`, ...).
- The operator must actually recombine the two parents. Returning a parent unchanged (or a view of a parent) is detected by the novelty check and scores 0. Returning `np.random.permutation(n)` while ignoring x and y is not a crossover either and will rank last.
- Use randomness only through numpy's global RNG (np.random.*); do NOT import or call Python's `random`, and keep no global mutable state.
- Use only numpy and the Python standard library. No file, network or console access.
- Must be safe and efficient for any n >= 2 (the benchmark uses n = 11..100; support at least up to a few hundred), so prefer O(n) or O(n log n).
- Before finishing, verify your logic on paper: the output must contain every value 0..n-1 exactly once (a permutation), never a duplicate or a missing value.

# How your operator is evaluated (and what feedback you get)
Each candidate is run inside the target algorithm on several mQAP instances. Before the expensive run, a cheap check applies your operator to random parent pairs: if it returns one parent unchanged (or a view of a parent, e.g. `np.array(x, copy=False)`), it scores 0 immediately. The score is the mean hypervolume of the final Pareto fronts, normalised like the MOHH-R pipeline (0 = poor, higher = better). After every attempt you receive: (i) your RANK among all crossovers evaluated so far (1 = best), (ii) the best value so far and (iii) how much your operator actually recombines (0 = copies a parent, 1 = child unlike both parents); hypervolume ties are broken in favour of operators that recombine more. Use the rank as the search signal: if your change lowers the rank, revert it; if it improves the rank, build on it. Your goal is to reach rank 1.

# Domain knowledge: what permutation crossovers preserve
Recombination operators for permutations trade off three kinds of structure, and mQAP rewards recombination that respects them:
- Order / precedence: Order Crossover (OX) and Order-Based Crossover (OBX) keep the relative order of a subset of elements.
- Position: Position-Based Crossover (PBX) and Partially Mapped Crossover (PMX) keep elements in their absolute positions.
- Adjacency / linkage: Edge Recombination (ERX) keeps consecutive neighbours together. In mQAP this matters because f_k is a sum of pairwise W_k[i, j] * D[p[i], p[j]] terms, so preserving good neighbour pairs can lower cost directly.
- Cycles: Cycle Crossover (CX) exchanges whole permutation cycles.
Empirically no single family dominates across QAP instances, and operators that adaptively combine two of these ideas tend to generalise better than a rigid single-family rule.

# Design guidance
1. Reason first, then code: begin with a short "# Design idea:" that names the family or families you combine (order / position / adjacency / cycle), what structure from each parent it preserves, and its expected time complexity in n.
2. Combine information from both parents, or choose deliberately between order-, position-, adjacency- and cycle-based ideas (ideally two of them).
3. Do not just copy the reference example; produce a novel, well-motivated operator.
4. Keep it size-agnostic and robust: handle any n, avoid hard-coded constants and avoid unnecessary O(n^2) work.
5. Write a self-check into the code when it is cheap: assert (or construct by design) that the result is a permutation before returning it.

Write one novel crossover operator for the mQAP.
```

### TASK_PROMPT (NSGAII)

```
Your job is to design a crossover (recombination) operator for NSGA-II, a multi-objective evolutionary algorithm, applied to the multi-objective Quadratic Assignment Problem (mQAP).

# The problem
A candidate solution is a permutation p of the n facilities over the n locations. For each objective k the instance provides a flow matrix W_k (n x n) and a global distance matrix D (n x n). The cost of p under objective k is the quadratic sum
    f_k(p) = sum_i sum_j W_k[i, j] * D[p[i], p[j]].
The algorithm minimises K objectives (in this benchmark 2 <= K <= 5) at the same time and must return a set of mutually non-dominated permutations that approximates the Pareto front. There is no single optimum: a good operator must help the search spread along and converge to the front, so its offspring should inherit useful structure from BOTH parents while remaining diverse.

# Where your operator is used
Your operator replaces the crossover of NSGA-II (non-dominated sorting with crowding-distance diversity). This is the same algorithm and the same problem family used by the MOHH-R operator-design pipeline. The operator is called at every generation, on instances of DIFFERENT sizes, and its quality is measured by the hypervolume of the final Pareto front. It must therefore be general, not tuned to one instance or to one fixed length.

# Exact interface
Write a Python function:

    def crossover(x, y):
        # x, y : 1-D numpy integer arrays, each a permutation of range(n),
        #        with n = len(x) = len(y).
        # return: a NEW permutation of range(n), OR a tuple (child1, child2)
        #         of two NEW permutations of range(n).

Hard requirements (any violation scores 0):
- Every returned child is a permutation of range(n): each value 0..n-1 appears exactly once and the length equals n, for ANY n >= 2.
- NEVER modify x or y in place. They are views into the algorithm's internal population: `x[i] = ...`, `x.sort()`, `x[idx] = ...` or `np.random.shuffle(x)` corrupt the search. Build the result in a fresh array (`np.empty_like`, `np.full`, `np.array(x, copy=True)`, ...).
- The operator must actually recombine the two parents. Returning a parent unchanged (or a view of a parent) is detected by the novelty check and scores 0. Returning `np.random.permutation(n)` while ignoring x and y is not a crossover either and will rank last.
- Use randomness only through numpy's global RNG (np.random.*); do NOT import or call Python's `random`, and keep no global mutable state.
- Use only numpy and the Python standard library. No file, network or console access.
- Must be safe and efficient for any n >= 2 (the benchmark uses n = 11..100; support at least up to a few hundred), so prefer O(n) or O(n log n).
- Before finishing, verify your logic on paper: the output must contain every value 0..n-1 exactly once (a permutation), never a duplicate or a missing value.

# How your operator is evaluated (and what feedback you get)
Each candidate is run inside the target algorithm on several mQAP instances. Before the expensive run, a cheap check applies your operator to random parent pairs: if it returns one parent unchanged (or a view of a parent, e.g. `np.array(x, copy=False)`), it scores 0 immediately. The score is the mean hypervolume of the final Pareto fronts, normalised like the MOHH-R pipeline (0 = poor, higher = better). After every attempt you receive: (i) your RANK among all crossovers evaluated so far (1 = best), (ii) the best value so far and (iii) how much your operator actually recombines (0 = copies a parent, 1 = child unlike both parents); hypervolume ties are broken in favour of operators that recombine more. Use the rank as the search signal: if your change lowers the rank, revert it; if it improves the rank, build on it. Your goal is to reach rank 1.

# Domain knowledge: what permutation crossovers preserve
Recombination operators for permutations trade off three kinds of structure, and mQAP rewards recombination that respects them:
- Order / precedence: Order Crossover (OX) and Order-Based Crossover (OBX) keep the relative order of a subset of elements.
- Position: Position-Based Crossover (PBX) and Partially Mapped Crossover (PMX) keep elements in their absolute positions.
- Adjacency / linkage: Edge Recombination (ERX) keeps consecutive neighbours together. In mQAP this matters because f_k is a sum of pairwise W_k[i, j] * D[p[i], p[j]] terms, so preserving good neighbour pairs can lower cost directly.
- Cycles: Cycle Crossover (CX) exchanges whole permutation cycles.
Empirically no single family dominates across QAP instances, and operators that adaptively combine two of these ideas tend to generalise better than a rigid single-family rule.

# Design guidance
1. Reason first, then code: begin with a short "# Design idea:" that names the family or families you combine (order / position / adjacency / cycle), what structure from each parent it preserves, and its expected time complexity in n.
2. Combine information from both parents, or choose deliberately between order-, position-, adjacency- and cycle-based ideas (ideally two of them).
3. Do not just copy the reference example; produce a novel, well-motivated operator.
4. Keep it size-agnostic and robust: handle any n, avoid hard-coded constants and avoid unnecessary O(n^2) work.
5. Write a self-check into the code when it is cheap: assert (or construct by design) that the result is a permutation before returning it.

Write one novel crossover operator for the mQAP.
```

### TASK_PROMPT (SMSEMOA)

```
Your job is to design a crossover (recombination) operator for SMS-EMOA, a multi-objective evolutionary algorithm, applied to the multi-objective Quadratic Assignment Problem (mQAP).

# The problem
A candidate solution is a permutation p of the n facilities over the n locations. For each objective k the instance provides a flow matrix W_k (n x n) and a global distance matrix D (n x n). The cost of p under objective k is the quadratic sum
    f_k(p) = sum_i sum_j W_k[i, j] * D[p[i], p[j]].
The algorithm minimises K objectives (in this benchmark 2 <= K <= 5) at the same time and must return a set of mutually non-dominated permutations that approximates the Pareto front. There is no single optimum: a good operator must help the search spread along and converge to the front, so its offspring should inherit useful structure from BOTH parents while remaining diverse.

# Where your operator is used
Your operator replaces the crossover of SMS-EMOA (steady-state selection driven by hypervolume contribution). This is the same algorithm and the same problem family used by the MOHH-R operator-design pipeline. The operator is called at every generation, on instances of DIFFERENT sizes, and its quality is measured by the hypervolume of the final Pareto front. It must therefore be general, not tuned to one instance or to one fixed length.

# Exact interface
Write a Python function:

    def crossover(x, y):
        # x, y : 1-D numpy integer arrays, each a permutation of range(n),
        #        with n = len(x) = len(y).
        # return: a NEW permutation of range(n), OR a tuple (child1, child2)
        #         of two NEW permutations of range(n).

Hard requirements (any violation scores 0):
- Every returned child is a permutation of range(n): each value 0..n-1 appears exactly once and the length equals n, for ANY n >= 2.
- NEVER modify x or y in place. They are views into the algorithm's internal population: `x[i] = ...`, `x.sort()`, `x[idx] = ...` or `np.random.shuffle(x)` corrupt the search. Build the result in a fresh array (`np.empty_like`, `np.full`, `np.array(x, copy=True)`, ...).
- The operator must actually recombine the two parents. Returning a parent unchanged (or a view of a parent) is detected by the novelty check and scores 0. Returning `np.random.permutation(n)` while ignoring x and y is not a crossover either and will rank last.
- Use randomness only through numpy's global RNG (np.random.*); do NOT import or call Python's `random`, and keep no global mutable state.
- Use only numpy and the Python standard library. No file, network or console access.
- Must be safe and efficient for any n >= 2 (the benchmark uses n = 11..100; support at least up to a few hundred), so prefer O(n) or O(n log n).
- Before finishing, verify your logic on paper: the output must contain every value 0..n-1 exactly once (a permutation), never a duplicate or a missing value.

# How your operator is evaluated (and what feedback you get)
Each candidate is run inside the target algorithm on several mQAP instances. Before the expensive run, a cheap check applies your operator to random parent pairs: if it returns one parent unchanged (or a view of a parent, e.g. `np.array(x, copy=False)`), it scores 0 immediately. The score is the mean hypervolume of the final Pareto fronts, normalised like the MOHH-R pipeline (0 = poor, higher = better). After every attempt you receive: (i) your RANK among all crossovers evaluated so far (1 = best), (ii) the best value so far and (iii) how much your operator actually recombines (0 = copies a parent, 1 = child unlike both parents); hypervolume ties are broken in favour of operators that recombine more. Use the rank as the search signal: if your change lowers the rank, revert it; if it improves the rank, build on it. Your goal is to reach rank 1.

# Domain knowledge: what permutation crossovers preserve
Recombination operators for permutations trade off three kinds of structure, and mQAP rewards recombination that respects them:
- Order / precedence: Order Crossover (OX) and Order-Based Crossover (OBX) keep the relative order of a subset of elements.
- Position: Position-Based Crossover (PBX) and Partially Mapped Crossover (PMX) keep elements in their absolute positions.
- Adjacency / linkage: Edge Recombination (ERX) keeps consecutive neighbours together. In mQAP this matters because f_k is a sum of pairwise W_k[i, j] * D[p[i], p[j]] terms, so preserving good neighbour pairs can lower cost directly.
- Cycles: Cycle Crossover (CX) exchanges whole permutation cycles.
Empirically no single family dominates across QAP instances, and operators that adaptively combine two of these ideas tend to generalise better than a rigid single-family rule.

# Design guidance
1. Reason first, then code: begin with a short "# Design idea:" that names the family or families you combine (order / position / adjacency / cycle), what structure from each parent it preserves, and its expected time complexity in n.
2. Combine information from both parents, or choose deliberately between order-, position-, adjacency- and cycle-based ideas (ideally two of them).
3. Do not just copy the reference example; produce a novel, well-motivated operator.
4. Keep it size-agnostic and robust: handle any n, avoid hard-coded constants and avoid unnecessary O(n^2) work.
5. Write a self-check into the code when it is cheap: assert (or construct by design) that the result is a permutation before returning it.

Write one novel crossover operator for the mQAP.
```

### EXAMPLE_PROMPT

````text
A minimal reference implementation (a simple order-based crossover) is shown below to fix the interface. Do NOT simply copy it; design something different and better suited to mQAP:

```python
import numpy as np

def crossover(x, y):
    n = len(x)
    a, b = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1, dtype=x.dtype)
    child[a:b] = x[a:b]              # preserve a segment of parent x
    used = set(x[a:b].tolist())
    fill = [v for v in y if v not in used]   # keep y's order for the rest
    positions = list(range(b, n)) + list(range(0, a))
    for pos, val in zip(positions, fill):
        child[pos] = val
    return child
```

````

### OUTPUT_FORMAT_PROMPT

````text
Answer in exactly this format:

# Design idea: <2-3 short sentences on the mechanism and why it helps mQAP>
# Description: <one line, no line breaks>
# Code:
```python
<complete code: it MUST start with `import numpy as np` and define a single
MODULE-LEVEL function `def crossover(x, y):` (NOT a class, NOT a method) that
returns one permutation or a tuple of two permutations>
```

````

### OPERATOR_REFINE

```text
Refine the selected crossover operator: keep its core mechanism but improve its effectiveness and robustness on mQAP (better transfer of useful structure, or better diversity). Change only what is needed. Use the reported rank and best-so-far: keep changes that improve the rank and revert those that lower it.
```

### OPERATOR_REDESIGN

```text
Redesign the crossover operator: propose a mechanism based on a different combination of order-, position-, adjacency- or cycle-based ideas, and explain the change. Use the reported rank and best-so-far to avoid repeating mechanisms that already ranked poorly.
```
