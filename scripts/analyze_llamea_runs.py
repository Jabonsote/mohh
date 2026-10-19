"""Resume las corridas de LLaMEA guardadas bajo `results/`.

Cada corrida deja su registro en
`results/<categoria>/<experimento>/llamea_logs/exp-<fecha>-.../`, con:

  * `log.jsonl` (un registro por operador evaluado: fitness, feedback, error,
    codigo, operador usado);
  * `generation_summary.jsonl` (una linea por generacion);
  * `code/try-N-*.py` (el codigo de cada intento);
  * `report.md`, `evolution.csv` y `evolution.png` (generados al terminar).

Este script lee esos directorios y responde dos preguntas utiles al depurar el
bucle (sobre todo con modelos locales chicos):

  * cuantos operadores generados fueron validos (HV > 0) y con que error
    fallaron los invalidos;
  * que operadores validos *distintos* aparecieron.

Uso:
    python scripts/analyze_llamea_runs.py                 # todas las corridas
    python scripts/analyze_llamea_runs.py 'results/searches/**/exp-*'
    python scripts/analyze_llamea_runs.py --list-codes
"""

import argparse
import glob
import hashlib
import json
import os
from collections import Counter


def load_rows(dirname):
    path = os.path.join(dirname, "log.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def error_kind(row):
    err = (row.get("error") or "").strip()
    if not err:
        return None
    # La primera linea suele ser "ValueError: ..." o "NoCodeException: ...".
    return err.splitlines()[0].split(":")[0].strip() or "Unknown"


def run_frame(dirname):
    """Marco de normalizacion de la corrida (para no mezclar escalas).

    Las corridas anteriores al marco GE no traen el campo: sus HV estan en el
    marco fijo del panel y no son comparables con los del marco GE.
    """
    info_path = os.path.join(os.path.dirname(os.path.dirname(dirname)), "llamea_best_info.json")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path) as handle:
            info = json.load(handle)
    except (OSError, ValueError):
        return None
    frame = info.get("frame")
    if not frame:
        return None
    text = f"{frame}"
    degenerate = info.get("degenerate_instances") or []
    if degenerate:
        text += f" (sin escala: {', '.join(degenerate)})"
    reference = info.get("reference_hv_final_frame") or {}
    if reference:
        best = sorted(reference.items(), key=lambda kv: -kv[1])[0]
        text += f"; mejor clasico en el mismo marco: {best[0]}={best[1]:.6f}"
    return text


def token_info(dirname):
    """Tokens acumulados y media por generacion (si el log los trae)."""
    path = os.path.join(dirname, "generation_summary.jsonl")
    if not os.path.exists(path):
        return None
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records or "tokens_total" not in records[-1]:
        return None
    per_gen = [r["tokens_total_gen"] for r in records
               if r.get("tokens_total_gen") is not None]
    mean = sum(per_gen) / len(per_gen) if per_gen else 0.0
    return {"total": records[-1]["tokens_total"],
            "requests": records[-1].get("requests", 0),
            "mean_gen": mean}


def summarize(pattern, list_codes=False):
    dirs = sorted(glob.glob(pattern, recursive=True))
    # Quedarse solo con directorios que tengan log.jsonl (evita falsos positivos).
    dirs = [d for d in dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, "log.jsonl"))]
    if not dirs:
        print(f"Sin corridas con log.jsonl que coincidan con {pattern!r}")
        return

    total = valid = 0
    errors = Counter()
    unique = {}
    for dirname in dirs:
        rows = load_rows(dirname)
        tag = dirname.split("LLaMEA-", 1)[-1]
        run_valid = sum(
            1 for r in rows if not r.get("error") and float(r.get("fitness") or 0) > 0
        )
        print(f"\n### {tag}: {run_valid}/{len(rows)} validos (HV>0)")
        frame = run_frame(dirname)
        if frame:
            print(f"    HV en marco: {frame}")
        tokens = token_info(dirname)
        if tokens:
            print(f"    tokens: {tokens['total']} en {tokens['requests']} peticiones "
                  f"(~{tokens['mean_gen']:.0f} por generacion)")
        for row in rows:
            total += 1
            fit = float(row.get("fitness") or 0)
            kind = error_kind(row)
            if kind:
                errors[kind] += 1
                print(f"    fit={fit:>12.4g}  error={kind}")
            elif fit > 0:
                valid += 1
                code = (row.get("code") or "").strip()
                digest = hashlib.sha1(code.encode()).hexdigest()[:8]
                unique.setdefault(digest, (fit, row.get("description", ""), code, tag))
                print(f"    fit={fit:>12.4g}  OK   [{digest}]")
            else:
                print(f"    fit={fit:>12.4g}  sin HV (feedback: "
                      f"{(row.get('feedback') or '').splitlines()[0][:60]})")

    print(f"\n=== RESUMEN ===")
    print(f"Operadores evaluados: {total}")
    print(f"Operadores validos (HV>0): {valid}")
    print(f"Operadores validos distintos: {len(unique)}")
    if errors:
        print("Errores mas frecuentes:")
        for kind, count in errors.most_common():
            print(f"  {count:>4}  {kind}")

    if list_codes:
        print("\n=== CODIGOS VALIDOS DISTINTOS ===")
        for digest, (fit, desc, code, tag) in sorted(unique.items(), key=lambda kv: -kv[1][0]):
            print(f"\n----- [{digest}] fit={fit:.4g} ({tag}) -----")
            print(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", nargs="?", default="results/**/exp-*",
                        help="Glob (recursivo) de los directorios de experimento "
                             "(default: results/**/exp-*).")
    parser.add_argument("--list-codes", action="store_true",
                        help="Imprimir el codigo de los operadores validos distintos.")
    args = parser.parse_args()
    summarize(args.pattern, list_codes=args.list_codes)


if __name__ == "__main__":
    main()
