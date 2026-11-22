#!/usr/bin/env python3
"""CLI del registro de modelos — puente entre config/reasoning.yaml y bash.

No contiene ningun modelo escrito a mano: todo sale del YAML. Los perfiles
concretos de GPU (VRAM, constraint, longitud) siguen en llamea-gpu-profile.sh;
aqui solo va lo comun a cualquier modelo (razonamiento y servidor).

Uso:
  python scripts/model_profile.py parser  <modelo>   # --reasoning-parser ...
  python scripts/model_profile.py args    <modelo>   # flags extra de vLLM
  python scripts/model_profile.py tokens  <modelo>   # max_tokens sugerido
  python scripts/model_profile.py env     <modelo>   # VAR=valor, una por linea
  python scripts/model_profile.py toggle  <modelo>   # yes/no
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore", message="urllib3.*doesn't match")

from mohh.llamea.reasoning import resolve


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    action, model = sys.argv[1], sys.argv[2]
    spec = resolve(model)
    if action == "parser":
        print(spec.parser_flag)
    elif action == "args":
        print(spec.vllm_args)
    elif action == "tokens":
        print(spec.max_tokens)
    elif action == "toggle":
        print("yes" if spec.toggle else "no")
    elif action == "env":
        for key, value in spec.env.items():
            print(f"{key}={value}")
    else:
        print(f"accion desconocida: {action}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
