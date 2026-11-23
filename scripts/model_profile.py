#!/usr/bin/env python3
"""CLI del registro de modelos (config/reasoning.yaml).

No importa mohh: lee el YAML directamente para funcionar sin activar el
entorno (.conda-env). Asi `slurm/llamea-gpu-profile.sh` puede llamarlo desde
el login (python del sistema) sin depender de llamea ni de src/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(os.environ.get(
    "MOHH_REASONING_REGISTRY", str(ROOT / "config" / "reasoning.yaml")))


def _resolve(model: str) -> dict:
    data = {}
    if REGISTRY.exists():
        with open(REGISTRY, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    base = dict(data.get("defaults") or {})
    name = str(model or "").lower()
    for rule in data.get("models") or []:
        match = str(rule.get("match") or "")
        if match and match.lower() in name:
            base.update(rule)
            break
    base.pop("match", None)
    return base


def _val(spec: dict, key: str, default):
    if key in spec and spec[key] is not None:
        return spec[key]
    return default


def main() -> int:
    if len(sys.argv) != 3:
        print("uso: model_profile.py {parser|args|tokens|toggle|env} <modelo>",
              file=sys.stderr)
        return 2
    action, model = sys.argv[1], sys.argv[2]
    spec = _resolve(model)
    defaults = {}
    if REGISTRY.exists():
        defaults = (yaml.safe_load(open(REGISTRY, encoding="utf-8")) or {}).get(
            "defaults", {}) or {}
    # merge defaults
    merged = dict(defaults)
    merged.update(spec)

    if action == "parser":
        p = str(merged.get("parser") or "")
        print(f"--reasoning-parser {p}" if p else "")
    elif action == "args":
        print(str(merged.get("vllm_args") or ""))
    elif action == "tokens":
        print(int(merged.get("max_tokens") or 4096))
    elif action == "toggle":
        print("yes" if merged.get("toggle") else "no")
    elif action == "env":
        for key, value in (merged.get("env") or {}).items():
            print(f"{key}={value}")
    else:
        print(f"accion desconocida: {action}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
