"""Puente de evaluation hacia llamea."""

import numpy as np

from mohh.shared.bridge import load_llm_operator
from mohh.shared.operators import LLMOperator

CROSSOVER = (
    "import numpy as np\n"
    "def crossover(x, y):\n"
    "    c = np.array(x, copy=True)\n"
    "    c[0], c[1] = c[1], c[0]\n"
    "    return c\n"
)


def test_load_llm_operator_cruza_permutaciones(tmp_path):
    path = tmp_path / "op.py"
    path.write_text(CROSSOVER, encoding="utf-8")

    operator = load_llm_operator(str(path), prob=0.5)
    assert isinstance(operator, LLMOperator)
    assert operator.prob.get() == 0.5

    parents = np.array([
        [[0, 1, 2, 3, 4, 5]],
        [[3, 0, 5, 1, 4, 2]],
    ])
    children = operator._do(None, parents)

    assert children.shape == parents.shape
    first, second = children[0, 0], children[1, 0]
    np.testing.assert_array_equal(first, [1, 0, 2, 3, 4, 5])
    np.testing.assert_array_equal(second, first[::-1])
    assert set(second.tolist()) == set(range(6))
