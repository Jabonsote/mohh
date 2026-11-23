"""Razonamiento por modelo y modulation del pensamiento."""
import os
from pathlib import Path

import pytest

from mohh.llamea.reasoning import (
    REGISTRY,
    ReasoningSpec,
    resolve,
    thinking_payload,
)


def test_registro_carga_modelos_grandes():
    data = REGISTRY and __import__("yaml").safe_load(
        open(REGISTRY, encoding="utf-8"))
    nombres = [r["match"] for r in data["models"]]
    for nombre in ("qwen3-coder", "deepseek-r1", "qwq", "glm-4.5", "gpt-oss"):
        assert nombre in nombres


@pytest.mark.parametrize("modelo,parser,toggle,tokens,codigo_en_razon", [
    ("openai/gpt-oss-20b", "openai_gptoss", False, 8192, True),
    ("QuantTrio/GLM-4.7-Flash-AWQ", "glm45", False, 8192, True),
    ("Qwen/Qwen3-Coder-30B-AWQ", "", True, 16384, False),
    ("deepseek-ai/DeepSeek-R1", "deepseek_r1", False, 16384, True),
    ("Qwen/QwQ-32B", "qwen3", False, 16384, True),
    ("zai-org/GLM-4.5", "glm45", False, 16384, True),
])
def test_specs_modelos_grandes(modelo, parser, toggle, tokens, codigo_en_razon):
    from mohh.llamea import reasoning as r
    r.resolve.cache_clear()
    s = r.resolve(modelo)
    assert (s.parser, s.toggle, s.max_tokens, s.code_in_reasoning) == (
        parser, toggle, tokens, codigo_en_razon)


def test_thinking_solo_en_modelos_con_toggle():
    qwen = resolve("Qwen/Qwen3-Coder-30B-AWQ")
    assert thinking_payload(qwen, True) == {
        "chat_template_kwargs": {"enable_thinking": True}}
    assert thinking_payload(qwen, False) == {
        "chat_template_kwargs": {"enable_thinking": False}}
    for modelo in ("openai/gpt-oss-20b", "deepseek-ai/DeepSeek-R1",
                   "Qwen/QwQ-32B", "Qwen/Qwen2.5-0.5B-Instruct"):
        s = resolve(modelo)
        assert thinking_payload(s, True) is None


def test_modelo_desconocido_usa_defaults():
    s = resolve("org/modelo-no-registrado")
    assert s.parser == "" and s.toggle is False
    assert s.max_tokens == 4096 and not s.code_in_reasoning


def test_registro_externo_sin_tocar_codigo(tmp_path):
    extra = tmp_path / "registry.yaml"
    extra.write_text(
        "defaults: {parser: '', toggle: false, code_in_reasoning: false,"
        " max_tokens: 4096, vllm_args: '', env: {}}\n"
        "models:\n  - match: 'nuevo-grandote'\n    parser: 'nuevo_parser'\n"
        "    max_tokens: 32768\n", encoding="utf-8")
    os.environ["MOHH_REASONING_REGISTRY"] = str(extra)
    try:
        import importlib
        import mohh.llamea.reasoning as r
        importlib.reload(r)
        s = r.resolve("org/nuevo-grandote-70b")
        assert s.parser == "nuevo_parser" and s.max_tokens == 32768
    finally:
        os.environ.pop("MOHH_REASONING_REGISTRY", None)
        importlib.reload(r)
