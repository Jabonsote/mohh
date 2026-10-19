"""Adaptador entre el codigo generado por el LLM y el MOEA de MOHH-R.

Esta es la **unica** pieza que conecta LLaMEA con el proyecto de evolucion
gramatical. No modifica el nucleo GE: construye un `Crossover` de pymoo que
evalua la funcion `crossover(x, y)` escrita por el LLM y lo inyecta en el
`MOSolver` existente (el mismo que usa la busqueda gramatical).

Al ser un adaptador aislado, la ruta GE (`HH_Operator`) queda intacta.
"""

import numpy as np
from pymoo.core.crossover import Crossover

DEFAULT_EXPRESSION = "crossover(x, y)"


class LLMOperator(Crossover):
    """Envoltorio pymoo de una funcion `crossover(x, y)` generada por el LLM.

    Espeja exactamente la semantica de `HH_Operator` (mismo shape de entrada
    y salida y mismo segundo descendiente `child[::-1]`), pero en lugar de
    evaluar un fenotipo de gramatica evalua codigo Python definido por el LLM
    a partir de un namespace explicito.
    """

    def __init__(self, namespace: dict, expression: str = DEFAULT_EXPRESSION,
                 n_parents: int = 2, n_offsprings: int = 2, prob: float = 1.0,
                 **kwargs):
        """
        @param namespace: namespace con la funcion `crossover` ya definida
            (resultado de `exec(code, namespace)`).
        @param expression: expresion a evaluar; por defecto `crossover(x, y)`.
        """
        super().__init__(n_parents, n_offsprings, prob, **kwargs)
        self.namespace = namespace
        self.expression = expression

    def _do(self, problem, X, **kwargs):
        # X tiene shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape
        Y = np.full_like(X, None, dtype=object)

        for k in range(n_matings):
            x = X[0, k]
            y = X[1, k] if X.shape[0] > 1 else X[0, k]

            # Namespace limpio por mating: x, y explicitos. Evita fugas entre
            # llamadas y da al codigo del LLM el mismo contrato que HH_Operator.
            eval_ns = dict(self.namespace)
            eval_ns["x"] = x
            eval_ns["y"] = y

            child = eval(self.expression, eval_ns)

            if isinstance(child, tuple):
                Y[0, k], Y[1, k] = child[0], child[1]
            else:
                Y[0, k], Y[1, k] = child, child[::-1]

        return Y
