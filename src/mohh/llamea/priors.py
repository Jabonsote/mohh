"""Strong priors para LLaMEA: operadores de GE como código Python.

Los priors son los ``Best`` de los experimentos de evolución gramatical,
traducidos a Python por ``mohh.analysis.phenotype``. Se usan para inicializar
la búsqueda (1+1) y para el refinamiento periódico, siguiendo la idea de
"strong priors" de Huang et al. (2026).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

_CANDIDATES = [
    ("GE1-Best", ROOT / "docs" / "operadores_ge1.py"),
    ("GE1-Middle", ROOT / "docs" / "operadores_ge1_middle.py"),
    ("GE1-Worst", ROOT / "docs" / "operadores_ge1_worst.py"),
    ("GE2-Best", ROOT / "docs" / "operadores_ge2.py"),
    ("GE2-Middle", ROOT / "docs" / "operadores_ge2_middle.py"),
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


def load_ge_priors():
    """Priors de GE (codigo profundo); utiles como referencia, no como inicio.

    Un prior de GE suele obtener el maximo del panel clasico (6.0). Si se usa
    como mejor inicial, el bucle (1+1) queda congelado: ningun candidato puede
    superar el techo. Por eso la busqueda arranca con los clasicos simples.
    """
    priors = []
    for label, path in _CANDIDATES:
        try:
            code = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if "def crossover" in code:
            priors.append((label, code))
    priors.append(("order-based", _FALLBACK))
    return priors


def load_priors():
    """Priors de arranque: cruces clasicos simples y validados.

    BAG inyecta algoritmos fuertes de un conjunto de referencia. Para modelos
    pequenos, OX/PMX/CX/PBX/ERX son priors fuertes y cortos; el codigo de GE
    se reserva para el panel de comparacion.
    """
    from mohh.llamea.simple_priors import load_simple_priors

    priors = load_simple_priors()
    if priors:
        return priors
    return load_ge_priors()
