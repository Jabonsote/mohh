
ROLE_PROMPT = (
    "You are a highly skilled computer scientist and Python expert specializing "
    "in evolutionary computation and multi-objective optimization."
)

TASK_PROMPT = """You are designing a crossover operator for a multi-objective
evolutionary algorithm (MOEA/D) solving the multi-objective Quadratic Assignment
Problem (mQAP).

The operator must be a Python function with this exact interface:

def crossover(x, y):
    # x and y are 1-D numpy integer arrays: valid permutations of range(n)
    # Return: a 1-D numpy integer array: a valid permutation of range(n)

The returned child permutation is used as offspring. Quality is measured by the
hypervolume of the Pareto front found on training instances: higher hypervolume
is better. Operators should preserve valid permutations (each location appears
exactly once), run fast, and produce diverse, high-quality children.

Write a novel, effective crossover operator for permutation-encoded mQAP."""

EXAMPLE_PROMPT = """An example of such code (a simple order-based crossover), is as follows:
```python
import numpy as np

def crossover(x, y):
    n = len(x)
    start, end = sorted(np.random.choice(n, 2, replace=False))
    child = np.full(n, -1)
    child[start:end] = x[start:end]
    taken = set(child[start:end])
    fill = [v for v in y if v not in taken]
    positions = list(range(end, n)) + list(range(0, start))
    for pos, val in zip(positions, fill):
        child[pos] = val
    return child
```
"""

OUTPUT_FORMAT_PROMPT = """Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
```python
<crossover function code>
```
"""
