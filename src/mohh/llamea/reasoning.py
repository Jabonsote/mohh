"""Modo pensamiento por modelo, con registro externo (``config/reasoning.yaml``).

**Anadir un modelo no requiere tocar codigo**: se anade un bloque en el YAML
con un ``match`` (subcadena del id del modelo). Esta modulo es el unico que
lee el archivo; el cliente Python y el script de perfiles de GPU consultan
aqui, asi que no puede haber dos listas que se desincronicen.

Reglas:

- ``THINKING=auto`` -> no tocar el template; el servidor decide.
- ``THINKING=on/off`` -> solo se envia ``chat_template_kwargs.enable_thinking``
  si ``toggle: true``. Los modelos que razonan siempre (gpt-oss, GLM) no lo
  aceptan: mandarlo era el bug que rompia las respuestas con GLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = Path(os.environ.get(
    "MOHH_REASONING_REGISTRY", str(_ROOT / "config" / "reasoning.yaml")))


@dataclass(frozen=True)
class ReasoningSpec:
    parser: str
    toggle: bool
    code_in_reasoning: bool
    max_tokens: int
    vllm_args: str
    env: dict

    @property
    def parser_flag(self) -> str:
        return f"--reasoning-parser {self.parser}" if self.parser else ""


@lru_cache(maxsize=1)
def registry() -> dict:
    """Carga el YAML. Se puede recargar con ``registry.cache_clear()``."""
    with open(REGISTRY, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=256)
def resolve(model: str) -> ReasoningSpec:
    """Spec del modelo: primera regla cuyo ``match`` aparece en el id."""
    data = registry()
    base = dict(data.get("defaults") or {})
    name = str(model or "").lower()
    for rule in data.get("models") or []:
        match = str(rule.get("match") or "")
        if match and match.lower() in name:
            base.update(rule)
            break
    base.pop("match", None)
    return ReasoningSpec(
        parser=str(base.get("parser") or ""),
        toggle=bool(base.get("toggle", False)),
        code_in_reasoning=bool(base.get("code_in_reasoning", False)),
        max_tokens=int(base.get("max_tokens") or 4096),
        vllm_args=str(base.get("vllm_args") or ""),
        env=dict(base.get("env") or {}),
    )


def thinking_payload(spec: ReasoningSpec, thinking) -> dict | None:
    """Extra body para ``chat.completions.create`` (None = no enviar nada)."""
    if thinking is None or not spec.toggle:
        return None
    return {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}


def resolve_path(path: str) -> ReasoningSpec:
    """Igual que ``resolve`` pero para un id dado como texto (util en CLI)."""
    return resolve(path)
