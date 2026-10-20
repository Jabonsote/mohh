"""Integracion LLaMEA + MOHH-R.

Permite buscar operadores de variacion (cruce) para mQAP con un LLM en lugar
de con la gramatica, reutilizando la logica actual del proyecto de evolucion
gramatical (mismo `MOSolver` interno y misma metrica de hipervolumen) y sin
modificar su nucleo.
"""

from mohh.llamea.vllm import VLLM_LLM, ensure_model, model_is_available, wait_for_server
from mohh.llamea.parsing import parse_llm_response, strip_thinking, repair_code
from mohh.llamea.operator import LLMOperator
from mohh.llamea.fitness import build_evaluator, compile_operator, init_params, primitive_namespace
from mohh.llamea.prompts import (
    ROLE_PROMPT,
    TASK_PROMPT,
    COMBINE_TARGET_FAMILIES,
    make_task_prompt,
    make_operators,
    TargetedCombineOperator,
    OPERATOR_REFINE,
    OPERATOR_REDESIGN,
    OPERATOR_COMBINE_TEMPLATE,
    EXAMPLE_PROMPT,
    OUTPUT_FORMAT_PROMPT,
)

__all__ = [
    "VLLM_LLM",
    "ensure_model",
    "model_is_available",
    "wait_for_server",
    "parse_llm_response",
    "strip_thinking",
    "repair_code",
    "LLMOperator",
    "build_evaluator",
    "compile_operator",
    "init_params",
    "primitive_namespace",
    "ROLE_PROMPT",
    "TASK_PROMPT",
    "COMBINE_TARGET_FAMILIES",
    "make_task_prompt",
    "make_operators",
    "TargetedCombineOperator",
    "OPERATOR_REFINE",
    "OPERATOR_REDESIGN",
    "OPERATOR_COMBINE_TEMPLATE",
    "EXAMPLE_PROMPT",
    "OUTPUT_FORMAT_PROMPT",
]
