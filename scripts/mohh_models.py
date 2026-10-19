"""CLI de pesos persistentes de modelos (`mohh.llamea.models`).

En el cluster ixachi `$HOME` y `/tmp` no son visibles desde los nodos de
computo y se limpian; los pesos de un modelo de 14B-32B son 8-60 GB. Este CLI
los descarga/verifica una sola vez en la raiz persistente y de ahi los lee vLLM.

Subcomandos:
    root                       imprime la raiz persistente (y por que)
    list                       modelos presentes, tamano y estado
    path MODEL                 ruta local esperada de MODEL
    download MODEL [--revision R]
    verify PATH|MODEL [--hash] [--load-check]
    manifest MODEL             lee el manifiesto mohh-model.json
    smoke --base-url URL --model MODEL
    remove MODEL               borra un modelo de la raiz (liberar cuota)

Ejemplos:
    python scripts/mohh_models.py root
    python scripts/mohh_models.py download Qwen/Qwen2.5-Coder-7B-Instruct
    python scripts/mohh_models.py verify Qwen/Qwen2.5-Coder-7B-Instruct --load-check
    python scripts/mohh_models.py smoke --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

import argparse
import os
import sys

from mohh.llamea import models


def _resolve(model_or_path):
    """Acepta un id de modelo o una ruta local existente."""
    if os.path.isdir(model_or_path):
        return model_or_path
    return models.model_dir(model_or_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("root", help="Raiz persistente de modelos.")
    sub.add_parser("list", help="Modelos presentes en la raiz.")

    p_path = sub.add_parser("path", help="Ruta local esperada de un modelo.")
    p_path.add_argument("model")

    p_download = sub.add_parser("download", help="Descargar/verificar un modelo.")
    p_download.add_argument("model")
    p_download.add_argument("--revision", default=None)
    p_download.add_argument("--hash", action="store_true",
                            help="Calcular sha256 de cada archivo en el manifiesto.")
    p_download.add_argument("--no-verify", action="store_true",
                            help="No exigir que el modelo quede completo.")

    p_verify = sub.add_parser("verify", help="Verificar un modelo local.")
    p_verify.add_argument("target")
    p_verify.add_argument("--hash", action="store_true")
    p_verify.add_argument("--load-check", action="store_true",
                          help="Ademas, comprobar que el tokenizer/modelo cargan.")

    p_manifest = sub.add_parser("manifest", help="Leer el manifiesto de un modelo.")
    p_manifest.add_argument("model")

    p_smoke = sub.add_parser("smoke", help="Prueba de inferencia contra el servidor.")
    p_smoke.add_argument("--base-url", required=True)
    p_smoke.add_argument("--model", required=True)

    p_remove = sub.add_parser("remove", help="Borrar un modelo de la raiz.")
    p_remove.add_argument("model")

    args = parser.parse_args(argv)

    if args.command == "root":
        root = models.models_root()
        print(root)
        print(f"  MOHH_MODELS_DIR={os.environ.get('MOHH_MODELS_DIR', '(sin definir)')}")
        print(f"  SCRATCH={os.environ.get('SCRATCH', '(sin definir)')}")
        print(f"  LUSTRE={os.environ.get('LUSTRE', '(sin definir)')}")
        return

    if args.command == "list":
        found = models.list_models()
        if not found:
            print(f"(vacio) {models.models_root()}")
            return
        for entry in found:
            state = "OK " if entry["ok"] else "INCOMPLETO"
            print(f"{state} {entry['model_id']:<48} {entry['human_size']:>10}  "
                  f"{entry['path']}")
            if not entry["ok"]:
                print(f"      problemas: {entry['problems']}")
        return

    if args.command == "path":
        print(models.model_dir(args.model))
        return

    if args.command == "download":
        try:
            path = models.download_model(
                args.model, revision=args.revision,
                hash_files=args.hash)
        except Exception as error:  # noqa: BLE001 - mensaje accionable
            raise SystemExit(f"ERROR: {error}")
        print(path)
        return

    if args.command == "verify":
        path = _resolve(args.target)
        result = models.verify_model(path, hash_files=args.hash)
        print(f"{'OK' if result['ok'] else 'INCOMPLETO'}: {path}")
        print(f"  archivos: {result.get('files')}, pesos: {result.get('weights')}, "
              f"tamano: {models.human_size(result.get('total_bytes', 0))}")
        for problem in result.get("problems", []):
            print(f"  - {problem}")
        if args.load_check:
            check = models.load_check(path, deep=True)
            print(f"  load-check: {'OK' if check['ok'] else 'FALLO'} "
                  f"(tokenizer={check.get('tokenizer')}, modelo={check.get('model')})")
            for problem in check.get("problems", []):
                print(f"  - {problem}")
            if not check["ok"]:
                raise SystemExit(1)
        if not result["ok"]:
            raise SystemExit(1)
        return

    if args.command == "manifest":
        manifest = models.read_manifest(models.model_dir(args.model))
        if manifest is None:
            raise SystemExit(f"Sin manifiesto en {models.model_dir(args.model)}")
        import json

        print(json.dumps(manifest, indent=2))
        return

    if args.command == "smoke":
        result = models.smoke_test(args.base_url, args.model)
        print(f"OK: {result['model']} respondio en {result['seconds']:.2f}s "
              f"({result['prompt_tokens']}/{result['completion_tokens']} tokens): "
              f"{result['text']!r}")
        return

    if args.command == "remove":
        removed = models.remove_model(args.model)
        print("borrado" if removed else "no estaba")
        return


if __name__ == "__main__":
    sys.exit(main())
