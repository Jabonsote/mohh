"""Modo pensamiento por modelo con registro externo (config/reasoning.yaml)."""
import os
import subprocess
from pathlib import Path

from mohh.llamea.reasoning import (
    REGISTRY,
    resolve,
    thinking_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_registro_externo():
    assert REGISTRY.name == "reasoning.yaml"
    assert REGISTRY.exists()
    assert resolve("openai/gpt-oss-20b").parser == "openai_gptoss"
    assert resolve("openai/gpt-oss-20b").max_tokens == 8192
    assert resolve("QuantTrio/GLM-4.7-Flash-AWQ").parser == "glm45"
    assert resolve("QuantTrio/GLM-4.7-Flash-AWQ").env["VLLM_USE_DEEP_GEMM"] == "0"
    assert resolve("Qwen/Qwen3-8B-AWQ").toggle is True
    assert resolve("Qwen/Qwen2.5-Coder-32B-Instruct").parser == ""


def test_modelo_nuevo_sin_tocar_codigo(tmp_path):
    """Anadir un modelo al YAML basta: no hace falta editar Python."""
    yaml = (ROOT / "config" / "reasoning.yaml").read_text(encoding="utf-8")
    yaml += '''
  - match: "mi-modelo-futuro"
    parser: "nuevo_parser"
    max_tokens: 16384
    vllm_args: "--trust-remote-code"
'''
    custom = tmp_path / "reasoning.yaml"
    custom.write_text(yaml, encoding="utf-8")
    env = os.environ.copy()
    env["MOHH_REASONING_REGISTRY"] = str(custom)
    result = subprocess.run(
        ["python", "scripts/model_profile.py", "parser",
         "org/mi-modelo-futuro-7b"], cwd=ROOT, env=env,
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "--reasoning-parser nuevo_parser"


def test_toggle_solo_en_modelos_que_lo_aceptan():
    qwen = resolve("Qwen/Qwen3-8B-AWQ")
    assert thinking_payload(qwen, None) is None
    assert thinking_payload(qwen, True) == {
        "chat_template_kwargs": {"enable_thinking": True}}
    assert thinking_payload(qwen, False) == {
        "chat_template_kwargs": {"enable_thinking": False}}
    for model in ("openai/gpt-oss-20b", "QuantTrio/GLM-4.7-Flash-AWQ",
                  "Qwen/Qwen2.5-Coder-32B-Instruct"):
        spec = resolve(model)
        assert thinking_payload(spec, True) is None
        assert thinking_payload(spec, False) is None


def test_cli():
    for action, model, expected in (
            ("tokens", "openai/gpt-oss-20b", "8192"),
            ("toggle", "QuantTrio/GLM-4.7-Flash-AWQ", "no"),
            ("env", "QuantTrio/GLM-4.7-Flash-AWQ",
             "VLLM_USE_DEEP_GEMM=0")):
        result = subprocess.run(
            ["python", "scripts/model_profile.py", action, model],
            cwd=ROOT, capture_output=True, text=True, check=True)
        assert expected in result.stdout


def test_perfil_toma_datos_del_registro():
    script = (ROOT / "slurm" / "llamea-gpu-profile.sh").read_text(encoding="utf-8")
    assert "scripts/model_profile.py" in script
    assert "REASONING_PARSER=\"$(${MODEL_PROFILE} parser" in script
    # No debe quedar ningun parser escrito a mano en los perfiles.
    for line in script.splitlines():
        if "reasoning-parser" in line and "MODEL_PROFILE" not in line \
                and "--" not in line.split("echo")[0]:
            assert "parser=" not in line
