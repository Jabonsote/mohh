"""Alias de compatibilidad: los operadores GE viven en `mohh.evaluation.ge_operators`."""

from mohh.evaluation.ge_operators import (
    OPERATORS_SUFFIX,
    discover_operator_files,
    ge_callable,
    ge_namespace,
    load_ge_operators,
)

__all__ = [
    "OPERATORS_SUFFIX",
    "discover_operator_files",
    "load_ge_operators",
    "ge_callable",
    "ge_namespace",
]
