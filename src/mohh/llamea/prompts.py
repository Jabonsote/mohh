"""Prompts de LLaMEA para disenar operadores de cruce sobre mQAP.

Diseno **basado en evidencia** (no derivado de la gramatica de MOHH-R). Cada
bloque retoma una practica reportada en la literatura de "automatic heuristic
design" con LLMs; el detalle y las citas estan en
`docs/llamea-prompts/diseno-de-prompts.md` y `docs/llamea-prompts/referencias.md`.

Principios aplicados
--------------------
1. Rol explicito de experto (``ROLE_PROMPT``) [LLaMEA 2024; ReEvo 2024].
2. Tarea precisa: problema, contrato de la funcion, objetivo y restricciones
   [LLaMEA 2024; FunSearch 2024].
3. Conocimiento de dominio: familias de operadores de permutacion y que
   preserva cada una (orden, posicion, adyacencia, ciclos) [ReEvo 2024;
   revisiones de operadores de cruce; Knowles & Corne 2003].
4. "Pensar y luego codificar": primero una idea corta, despues el codigo
   [EoH 2024; CoT 2022].
5. Ejemplo de referencia simple para acotar el sesgo del in-context learning
   [LLaMEA 2024; FunSearch 2024].
6. Formato de salida parseable (descripcion + bloque de codigo) [LLaMEA 2024;
   EoH 2024].
7. La refinacion/rediseno entre generaciones la aporta LLaMEA (feedback +
   instruccion de operador) [LLaMEA 2024; ReEvo 2024].
8. Restricciones negativas derivadas de fallos observados en corridas locales
   (no mutar los padres in-place, no devolver un padre sin cambios, no ignorar
   a los padres) [pruebas-locales.md].
9. Feedback interpretable: el texto reporta el ranking por hipervolumen y el
   mejor valor hasta ahora, para que el modelo pueda decidir que conservar
   [LLaMEA 2024; ReEvo 2024].

El prompt del problema se parametriza con el MOEA interno
(``make_task_prompt(solver)``) para que coincida con el algoritmo en el que el
operador se evaluara (los mismos MOEAD/NSGAII/SMSEMOA de MOHH-R).
"""

from llamea import Operator

# --------------------------------------------------------------------------- #
# Rol (LLaMEA: role prompt; ReEvo: system prompt)
# --------------------------------------------------------------------------- #
ROLE_PROMPT = (
    "You are an expert in evolutionary computation and combinatorial "
    "optimization, specialised in designing variation (recombination) "
    "operators for permutation-encoded multi-objective problems. "
    "You write original, correct and efficient Python."
)

# Nombre + descripcion breve de cada MOEA interno (los mismos de MOHH-R).
_SOLVERS = {
    "MOEAD": (
        "MOEA/D",
        "decomposition-based: it decomposes the problem into scalarised "
        "subproblems and performs mating within neighbourhoods",
    ),
    "NSGAII": (
        "NSGA-II",
        "non-dominated sorting with crowding-distance diversity",
    ),
    "SMSEMOA": (
        "SMS-EMOA",
        "steady-state selection driven by hypervolume contribution",
    ),
}

# --------------------------------------------------------------------------- #
# Tarea (LLaMEA: task prompt). Se parametriza con el MOEA interno.
# --------------------------------------------------------------------------- #
_TASK_TEMPLATE = """Your job is to design a crossover (recombination) operator for \
{solver_name}, a multi-objective evolutionary algorithm, applied to the \
multi-objective Quadratic Assignment Problem (mQAP).

# The problem
A candidate solution is a permutation p of the n facilities over the n \
locations. For each objective k the instance provides a flow matrix W_k \
(n x n) and a global distance matrix D (n x n). The cost of p under objective \
k is the quadratic sum
    f_k(p) = sum_i sum_j W_k[i, j] * D[p[i], p[j]].
The algorithm minimises K objectives (in this benchmark 2 <= K <= 5) at the \
same time and must return a set of mutually non-dominated permutations that \
approximates the Pareto front. There is no single optimum: a good operator must \
help the search spread along and converge to the front, so its offspring should \
inherit useful structure from BOTH parents while remaining diverse.

# Where your operator is used
Your operator replaces the crossover of {solver_name} ({solver_desc}). This is \
the same algorithm and the same problem family used by the MOHH-R \
operator-design pipeline. The operator is called at every generation, on \
instances of DIFFERENT sizes, and its quality is measured by the hypervolume \
of the final Pareto front. It must therefore be general, not tuned to one \
instance or to one fixed length.

# Exact interface
Write a Python function:

    def crossover(x, y):
        # x, y : 1-D numpy integer arrays, each a permutation of range(n),
        #        with n = len(x) = len(y).
        # return: a NEW permutation of range(n), OR a tuple (child1, child2)
        #         of two NEW permutations of range(n).

Hard requirements (any violation scores 0):
- Every returned child is a permutation of range(n): each value 0..n-1 \
appears exactly once and the length equals n, for ANY n >= 2.
- NEVER modify x or y in place. They are views into the algorithm's internal \
population: `x[i] = ...`, `x.sort()`, `x[idx] = ...` or `np.random.shuffle(x)` \
corrupt the search. Build the result in a fresh array (`np.empty_like`, \
`np.full`, `np.array(x, copy=True)`, ...).
- The operator must actually recombine the two parents. Returning a parent \
unchanged (or a view of a parent) is detected by the novelty check and scores \
0. Returning `np.random.permutation(n)` while ignoring x and y is not a \
crossover either and will rank last.
- Use randomness only through numpy's global RNG (np.random.*); do NOT import \
or call Python's `random`, and keep no global mutable state.
- Use only numpy and the Python standard library. No file, network or console \
access.
- Must be safe and efficient for any n >= 2 (the benchmark uses n = 11..100; \
support at least up to a few hundred), so prefer O(n) or O(n log n).
- Before finishing, verify your logic on paper: the output must contain every \
value 0..n-1 exactly once (a permutation), never a duplicate or a missing \
value.

# How your operator is evaluated (and what feedback you get)
Each candidate is run inside the target algorithm on several mQAP instances. \
Before the expensive run, a cheap check applies your operator to random parent \
pairs: if it returns one parent unchanged (or a view of a parent, e.g. \
`np.array(x, copy=False)`), it scores 0 immediately. \
The score is the mean hypervolume of the final Pareto fronts, normalised like \
the MOHH-R pipeline (0 = poor, higher = better). After every attempt you \
receive: (i) your RANK among all crossovers evaluated so far (1 = best), \
(ii) the best value so far and (iii) how much your operator actually recombines \
(0 = copies a parent, 1 = child unlike both parents); hypervolume ties are \
broken in favour of operators that recombine more. Use the rank as the search \
signal: if your change lowers the rank, revert it; if it improves the rank, \
build on it. Your goal is to reach rank 1.

# Domain knowledge: what permutation crossovers preserve
Recombination operators for permutations trade off three kinds of structure, \
and mQAP rewards recombination that respects them:
- Order / precedence: Order Crossover (OX) and Order-Based Crossover (OBX) keep \
the relative order of a subset of elements.
- Position: Position-Based Crossover (PBX) and Partially Mapped Crossover (PMX) \
keep elements in their absolute positions.
- Adjacency / linkage: Edge Recombination (ERX) keeps consecutive neighbours \
together. In mQAP this matters because f_k is a sum of pairwise \
W_k[i, j] * D[p[i], p[j]] terms, so preserving good neighbour pairs can lower \
cost directly.
- Cycles: Cycle Crossover (CX) exchanges whole permutation cycles.
Empirically no single family dominates across QAP instances, and operators that \
adaptively combine two of these ideas tend to generalise better than a rigid \
single-family rule.

# Design guidance
1. Reason first, then code: begin with a short "# Design idea:" that names the \
family or families you combine (order / position / adjacency / cycle), what \
structure from each parent it preserves, and its expected time complexity in n.
2. Combine information from both parents, or choose deliberately between \
order-, position-, adjacency- and cycle-based ideas (ideally two of them).
3. Do not just copy the reference example; produce a novel, well-motivated \
operator.
4. Keep it size-agnostic and robust: handle any n, avoid hard-coded constants \
and avoid unnecessary O(n^2) work.
5. Write a self-check into the code when it is cheap: assert (or construct by \
design) that the result is a permutation before returning it.

Write one novel crossover operator for the mQAP."""

# --------------------------------------------------------------------------- #
# Ejemplo (LLaMEA: example prompt). Un solo ejemplo simple reduce el sesgo y aun
# asi fija el contrato y el estilo (LLaMEA 2024). Se pide explicitamente ir mas
# alla del ejemplo.
# --------------------------------------------------------------------------- #
EXAMPLE_PROMPT = """A minimal reference implementation (a simple order-based crossover) \
is shown below to fix the interface. Do NOT simply copy it; design something \
different and better suited to mQAP:

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
"""

# --------------------------------------------------------------------------- #
# Esqueleto alternativo (ablacion): fija el contrato sin anclar una familia.
# --------------------------------------------------------------------------- #
SKELETON_PROMPT = """The interface is fixed by this skeleton. Fill in the mechanism
yourself; do NOT copy a known operator verbatim:

```python
import numpy as np

def crossover(x, y):
    n = len(x)
    child = np.empty_like(x)   # build a NEW permutation of range(n)
    # ... your mechanism ...
    return child
```
"""


# --------------------------------------------------------------------------- #
# Formato de salida (LLaMEA: output format prompt; EoH: idea y luego codigo).
# Se mantiene "# Description:" de una linea porque es lo que parsea LLaMEA.
# --------------------------------------------------------------------------- #
OUTPUT_FORMAT_PROMPT = """Answer in exactly this format:

# Design idea: <2-3 short sentences on the mechanism and why it helps mQAP>
# Description: <one line, no line breaks>
# Code:
```python
<complete code: it MUST start with `import numpy as np` and define a single
MODULE-LEVEL function `def crossover(x, y):` (NOT a class, NOT a method) that
returns one permutation or a tuple of two permutations>
```
"""


def make_task_prompt(solver: str = "MOEAD") -> str:
    """Prompt del problema adaptado al MOEA interno (mismos que MOHH-R)."""
    name, desc = _SOLVERS.get(solver.upper(), _SOLVERS["MOEAD"])
    return _TASK_TEMPLATE.format(solver_name=name, solver_desc=desc)


# Prompt por defecto (MOEA/D), por compatibilidad.
TASK_PROMPT = make_task_prompt("MOEAD")

# --------------------------------------------------------------------------- #
# Operadores evolutivos (LLaMEA: instrucciones de mutacion/cruce). Se inspiran en
# la dicotomia explorar/explotar de EoH (E1/E2 vs M1/M2/M3) y en el
# "refine or redesign" de LLaMEA / mutacion elitista de ReEvo.
# --------------------------------------------------------------------------- #
OPERATOR_REFINE = (
    "Refine the selected crossover operator: keep its core mechanism but improve "
    "its effectiveness and robustness on mQAP (better transfer of useful structure, "
    "or better diversity). Change only what is needed. Use the reported rank and "
    "best-so-far: keep changes that improve the rank and revert those that lower it."
)

OPERATOR_REDESIGN = (
    "Redesign the crossover operator: propose a mechanism based on a different "
    "combination of order-, position-, adjacency- or cycle-based ideas, and explain "
    "the change. Use the reported rank and best-so-far to avoid repeating mechanisms "
    "that already ranked poorly."
)

#: Familias que puede pedir el operador de combinacion conductual. `mixed` y
#: `copy` no son objetivos utiles: `copy` es un fallo y `mixed` no describe una
#: direccion concreta.
COMBINE_TARGET_FAMILIES = ("order", "position", "cycle", "adjacency",
                           "exploratory")

OPERATOR_COMBINE_TEMPLATE = """Combine the TWO selected crossover operators into \
ONE new crossover operator. This is a semantic recombination, not a syntactic \
merge of the code: take the useful mechanism from each parent and design a child \
that behaves better than either one on mQAP.

The two parents were selected because their measured behaviors are different; \
their descriptions and code are above. Do not merely refine or copy one parent: \
the child must use the mechanism of BOTH parents.

TARGET BEHAVIOR FAMILY: {target_family}.
After the child is generated, its behavior is measured on fixed parent pairs and \
classified into a behavioral family (order, position, cycle, adjacency, \
exploratory, mixed, or copy). The recombination is considered successful only if \
the observed family matches the target. Use the target as a design constraint:
- order: preserve relative order (OX/OBX-style);
- position: preserve absolute positions or mapped segments (PBX/PMX-style);
- cycle: exchange complete permutation cycles (CX-style);
- adjacency: preserve parent neighbours/edges (ERX-style);
- exploratory: produce a child unlike both parents (high recombination, not a \
small perturbation).
If the feedback says the target family was missed, change the mechanism, not just \
the syntax."""


class TargetedCombineOperator(Operator):
    """Operador LLaMEA de dos padres con una familia conductual objetivo.

    Es una subclase de `llamea.Operator`, no una modificacion del paquete: LLaMEA
    lo acepta igual (`prompt`, `name`, `weight`, `number_of_parents`, `id`,
    `rewards`). `target_family` puede ser None: `BehavioralLLaMEA` elige en cada
    generacion la familia menos representada del archivo conductual y la fija
    antes de construir el prompt. El `__repr__` es lo que LLaMEA concatena.
    """

    def __init__(self, target_family=None, weight=0.30):
        super().__init__(
            prompt=OPERATOR_COMBINE_TEMPLATE,
            name="combine",
            weight=weight,
            number_of_parents=2,
        )
        self.target_family = target_family
        # Si no se fijo una familia concreta, BehavioralLLaMEA la recalcula en
        # cada generacion (la menos representada). Si se fijo, es estatica.
        self.dynamic_target = target_family is None

    def __repr__(self):
        target = self.target_family or "the least represented behavioral family"
        return self.prompt.format(target_family=target)


def make_operators(include_combine=False):
    """Lista de operadores LLaMEA (exploracion/explotacion) para la busqueda.

    Sin `include_combine` devuelve los dos operadores historicos de un solo
    padre (compatibles con n_parents=1): refinamiento (explotacion) y rediseno
    (exploracion).

    Con `include_combine=True` agrega un tercer operador de dos padres que pide
    recombinar dos operadores conductualmente distintos y apunta a la familia
    menos representada. Los pesos suman 1: 0.35 refine, 0.35 redesign y 0.30
    combine; asi la extension no desplaza por completo la busqueda local.
    """
    operators = [
        Operator(prompt=OPERATOR_REFINE, name="refine", weight=0.5,
                 number_of_parents=1),
        Operator(prompt=OPERATOR_REDESIGN, name="redesign", weight=0.5,
                 number_of_parents=1),
    ]
    if not include_combine:
        return operators
    return [
        Operator(prompt=OPERATOR_REFINE, name="refine", weight=0.35,
                 number_of_parents=1),
        Operator(prompt=OPERATOR_REDESIGN, name="redesign", weight=0.35,
                 number_of_parents=1),
        TargetedCombineOperator(weight=0.30),
    ]
