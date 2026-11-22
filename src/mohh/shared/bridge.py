"""Puente de evaluation hacia llamea."""

from mohh.shared.operators import LLMOperator

__all__ = ["load_llm_operator"]


def load_llm_operator(path: str, prob: float = 1.0):
    """Unico punto donde evaluation depende de llamea. Import lazy para evitar ciclo."""
    from mohh.llamea.fitness import compile_operator

    with open(path, encoding="utf-8") as handle:
        code = handle.read()
    return LLMOperator(compile_operator(code), prob=prob)
