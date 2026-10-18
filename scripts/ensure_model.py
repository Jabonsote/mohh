
import argparse

from mohh.llamea.vllm import ensure_model


def main():
    parser = argparse.ArgumentParser(description="Verifica/descarga un modelo de HuggingFace.")
    parser.add_argument("model", help="ID del modelo (ej. Qwen/Qwen2.5-1.5B-Instruct)")
    args = parser.parse_args()
    ensure_model(args.model)
    print("Listo para servir con vLLM.")


if __name__ == "__main__":
    main()
