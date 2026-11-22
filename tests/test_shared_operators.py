"""Contrato de los operadores compartidos por GE y LLaMEA."""

import numpy as np

from mohh.evaluation.operators.mutation import Swap_Mutation as LegacySwapMutation
from mohh.llamea.operator import LLMOperator as LegacyLLMOperator
from mohh.llamea.reference import REFERENCE_OPERATORS as LegacyReferenceOperators
from mohh.shared.operators import DEFAULT_EXPRESSION, LLMOperator, Swap_Mutation
from mohh.shared.reference import REFERENCE_OPERATORS, namespace_for

# El adaptador ya tenia un test que cubre el cruce generado y los dos hijos
# validos; se reutiliza en vez de duplicarlo.
from test_llamea import test_llm_operator_applies_generated_crossover  # noqa: F401


def test_alias_de_compatibilidad():
    assert LegacyLLMOperator is LLMOperator
    assert LegacySwapMutation is Swap_Mutation
    assert LegacyReferenceOperators is REFERENCE_OPERATORS
    assert DEFAULT_EXPRESSION == "crossover(x, y)"


def test_namespace_for_expone_cruce_y_numpy():
    ns = namespace_for(REFERENCE_OPERATORS["ox"])
    assert ns["crossover"] is REFERENCE_OPERATORS["ox"]
    assert ns["np"] is np


def test_swap_mutation_produce_permutaciones():
    op = Swap_Mutation(prob=1.0)

    class FakeProblem:
        n_var = 6

    X = np.tile(np.arange(6), (8, 1))
    Y = op._do(FakeProblem(), X)

    assert Y.shape == X.shape
    for original, mutated in zip(X, Y):
        assert set(mutated.tolist()) == set(range(6))
        assert not np.array_equal(mutated, original)


def test_swap_mutation_probabilidad_cero_no_cambia():
    op = Swap_Mutation(prob=0.0)

    class FakeProblem:
        n_var = 6

    X = np.tile(np.arange(6), (4, 1))
    np.testing.assert_array_equal(op._do(FakeProblem(), X), X)
