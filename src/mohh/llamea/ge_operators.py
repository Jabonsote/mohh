"""Operadores de la evolucion gramatical (GE) como fuente comparable.

El proyecto original guarda los operadores que encontro la GE en
``<experimento>/<MODELO>_operators.txt``, con una linea por operador:

.. code-block:: text

    Best: order_based(fill_first_occurring,preserve_elements,y,...)
    Middle: ...
    Worst: ...

Cada expresion es un **fenotipo de la gramatica**: se evalua con las
primitivas de :mod:`mohh.core.utilities.algorithm.HH_functions`, las mismas que
usa ``HH_Operator`` (la ruta GE queda intacta).

Aqui la expresion se expone como una funcion ``crossover(x, y)`` para que el
mismo arnes de evaluacion sirva para operadores GE, LLaMEA y el panel
clasico: mismo MOEA, misma semilla, misma poblacion inicial y mismo marco de
normalizacion. Solo cambia quien produce el hijo.
"""

import os

import numpy as np

OPERATORS_SUFFIX = "_operators.txt"


def discover_operator_files(experiment_path):
    """Archivos `*_operators.txt` de un experimento GE.

    Mismo criterio que el original (`discover_models`): los modelos que
    produjeron operadores sededucen de los archivos que escribe `mohh-generate`.
    """
    if not os.path.isdir(experiment_path):
        return []
    return sorted(
        os.path.join(experiment_path, name)
        for name in os.listdir(experiment_path)
        if name.endswith(OPERATORS_SUFFIX)
    )


def load_ge_operators(experiment_path, model=None):
    """Operadores GE de un experimento: ``{etiqueta: expresion}``.

    `model` filtra por un modelo concreto (el nombre del archivo sin el
    sufijo). Sin `model` se toman todos los modelos del experimento; las
    etiquetas se prefijan con el modelo para no repetir nombres.
    """
    operators = {}
    for path in discover_operator_files(experiment_path):
        name = os.path.basename(path)[: -len(OPERATORS_SUFFIX)]
        if model is not None and name != model:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                label, expression = line.split(":", 1)
                operators[f"{name}_{label.strip()}"] = expression.strip()
    return operators


def ge_callable(expression, solution_type="Natural"):
    """Convierte una expresion de gramatica en `crossover(x, y)`.

    La evaluacion usa el mismo namespace que `HH_Operator` (las primitivas de
    `HH_functions`), con `x` e `y` explicitos, de modo que el operador GE se
    comporta en el MOEA igual que en la evaluacion original.
    """
    from mohh.core.utilities.algorithm import HH_functions

    namespace = {
        name: getattr(HH_functions, name)
        for name in dir(HH_functions)
        if not name.startswith("_")
    }
    namespace["np"] = np

    def crossover(x, y):
        local = dict(namespace)
        local["x"] = x
        local["y"] = y
        return eval(expression, local)  # noqa: S307 - expresion de la gramatica

    crossover.__name__ = "crossover"
    crossover.ge_expression = expression
    crossover.ge_solution_type = solution_type
    return crossover


def ge_namespace(expression, solution_type="Natural"):
    """Namespace de LLaMEA con el operador GE como `crossover(x, y)`."""
    return {"crossover": ge_callable(expression, solution_type), "np": np}
