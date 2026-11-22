"""Alias de compatibilidad: el adaptador vive en `mohh.shared.operators`."""

from mohh.shared.operators import DEFAULT_EXPRESSION, LLMOperator

__all__ = ["LLMOperator", "DEFAULT_EXPRESSION"]
