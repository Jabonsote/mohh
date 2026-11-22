"""Operadores de variacion compartidos por GE y LLaMEA.

`LLMOperator` envuelve una funcion `crossover(x, y)` generada por el LLM con el
mismo contrato pymoo que `HH_Operator` (core), sin depender de `mohh.llamea`.
`Swap_Mutation` es la mutacion de permutacion usada por la evaluacion GE.
"""

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation

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
            # Copias: un operador no debe corromper la poblacion si modifica
            # x/y in-place. La validacion minima no lo penaliza.
            x = np.array(X[0, k], copy=True)
            y = np.array(X[1, k], copy=True) if X.shape[0] > 1 else np.array(X[0, k], copy=True)

            # Namespace limpio por mating: x, y explicitos. Evita fugas entre
            # llamadas y da al codigo del LLM el mismo contrato que HH_Operator.
            eval_ns = dict(self.namespace)
            eval_ns["x"] = x
            eval_ns["y"] = y

            child = eval(self.expression, eval_ns)

            if isinstance(child, tuple) and len(child) == 2:
                first, second = child
            else:
                first, second = child, child[::-1]

            # Un operador valido en el filtro puede fallar en un par concreto
            # (p.ej. deja -1 o duplica valores). Nunca se inyecta un hijo
            # invalido al MOEA: se usa el padre como fallback y el algoritmo
            # sigue siendo comparable. Esto evita frentes con objetivos 0 y
            # HV absurdos en la evaluacion final.
            first = self._valid_child(first, n_var, x)
            second = self._valid_child(second, n_var, y)
            Y[0, k], Y[1, k] = first, second

        return Y

    @staticmethod
    def _valid_child(child, n_var, fallback):
        try:
            array = np.asarray(child)
        except Exception:  # noqa: BLE001
            return np.asarray(fallback, copy=True)
        if array.shape != (n_var,):
            return np.asarray(fallback, copy=True)
        if set(array.tolist()) != set(range(n_var)):
            return np.asarray(fallback, copy=True)
        return array


class Swap_Mutation(Mutation):

    def __init__(self, prob, **kwargs) -> None:
        super().__init__(prob, **kwargs)

        self.probability = prob

    def _do(self, problem, X, **kwargs):

        # X -> Input shape (n_individuals, n_var)
        n_individuals, n_var = X.shape

        # Create a copy of X to mutate individuals
        mutated_X = np.copy(X)

        if self.probability > 0:

            # For each individual
            for i in range(n_individuals):

                # Apply mutation based on the mutation rate
                if np.random.rand() < self.probability:

                    # Select two positions to swap
                    pos1, pos2 = np.random.choice(n_var, 2, replace=False)

                    # Swap
                    mutated_X[i, pos1], mutated_X[i, pos2] = mutated_X[i, pos2], mutated_X[i, pos1]

        return mutated_X
