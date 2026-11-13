"""Strong priors para LLaMEA: operadores de GE como código Python.

Los priors son los ``Best`` de los experimentos de evolución gramatical,
traducidos a Python por ``mohh.analysis.phenotype``. Se usan para inicializar
la búsqueda (1+1) y para el refinamiento periódico, siguiendo la idea de
"strong priors" de Huang et al. (2026).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

_CANDIDATES = [
    ("GE-Best-001", ROOT / "docs" / "operadores_ge1.py"),
    ("GE-Best-002", ROOT / "docs" / "operadores_ge2.py"),
]

_FALLBACK = """import numpy as np

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


def load_priors():
    priors = []
    for label, path in _CANDIDATES:
        try:
            code = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if "def crossover" in code:
            priors.append((label, code))
    if not priors:
        priors.append(("order-based", _FALLBACK))
    return priors
