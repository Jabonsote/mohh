"""Verifica (y opcionalmente descarga) un modelo local de HuggingFace.

El modelo usado por LLaMEA es local (lo sirve vLLM); este script confirma que
ya esta en el cache de HuggingFace antes de enviar un job al cluster:

    python scripts/ensure_model.py Qwen/Qwen2.5-1.5B-Instruct
    python scripts/ensure_model.py Qwen/Qwen2.5-1.5B-Instruct --check   # no descarga
"""

import argparse

from mohh.llamea.vllm import _hf_cache_dir, ensure_model, model_is_available


def main():
    parser = argparse.ArgumentParser(description="Verifica/descarga un modelo de HuggingFace.")
    parser.add_argument("model", help="ID del modelo (ej. Qwen/Qwen2.5-1.5B-Instruct).")
    parser.add_argument("--check", action="store_true",
                        help="Solo verificar; no descargar si falta.")
    parser.add_argument("--local-dir", default=None,
                        help="Descargar a un directorio local en lugar del cache HF.")
    args = parser.parse_args()

    if args.check:
        if model_is_available(args.model):
            print(f"OK: {args.model} presente en {_hf_cache_dir()}")
        else:
            raise SystemExit(
                f"FALTA: {args.model} no esta en {_hf_cache_dir()} ni como ruta local."
            )
        return

    ensure_model(args.model, allow_download=True, local_dir=args.local_dir)
    print("Listo para servir con vLLM.")


if __name__ == "__main__":
    main()
