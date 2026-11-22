"""Recolecta los mejores operadores de R corridas LLaMEA independientes.

Por que existe: la tabla de *una* corrida reporta el maximo de ~N candidatos,
que es un estimador optimista (sesgo de maximo). La afirmacion "la busqueda con
LLM encuentra buenos operadores" es sobre el **procedimiento**, asi que hay que
replicar la busqueda: R corridas con distinta semilla y reportar la
**distribucion** del mejor-por-corrida, no un solo numero.

Salida (en `--out`):

    operators/run-<i>.py     copia del mejor operador de cada corrida
    summary.json             una fila por corrida + estadistica de la distribucion
    summary.md               la misma tabla, legible
    compare_args.txt         argumentos `nombre=ruta` para scripts/compare_operators.py

Uso:
    python scripts/collect_replicates.py 'results/multirun/qwen7b/*' \
        --out results/multirun/qwen7b_analysis
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np


def _distribution(values):
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"n": 0}
    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "sd": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _read_run(run_dir):
    info_path = os.path.join(run_dir, "llamea_best_info.json")
    operator_path = os.path.join(run_dir, "llamea_best_operator.py")
    if not (os.path.exists(info_path) and os.path.exists(operator_path)):
        return None
    with open(info_path) as handle:
        info = json.load(handle)
    cost = info.get("cost") or {}
    return {
        "run_dir": run_dir,
        "operator_path": operator_path,
        "model": info.get("model"),
        "solver": info.get("solver"),
        "budget": info.get("budget"),
        "seed": (info.get("seeds") or {}).get("search", info.get("seed")),
        "repeat_penalty": info.get("repeat_penalty"),
        "train_fitness": info.get("train_fitness"),
        "train_fitness_raw": info.get("train_fitness_raw"),
        "train_fitness_final_frame": info.get("train_fitness_final_frame"),
        "test_fitness": info.get("test_fitness"),
        "distinct_operators": info.get("distinct_operators"),
        "evaluated_valid": info.get("evaluated_valid"),
        "duplicate_evaluations": (info.get("diversity") or {}).get("duplicate_evaluations"),
        "stop_reason": info.get("stop_reason"),
        "total_tokens": cost.get("total_tokens"),
        "gpu_seconds": cost.get("gpu_seconds"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+",
                        help="Directorios de corridas (acepta globs, p. ej. 'results/multirun/x/*')")
    parser.add_argument("--out", required=True, help="Directorio de salida del resumen.")
    args = parser.parse_args()

    run_dirs = []
    for pattern in args.runs:
        matched = sorted(glob.glob(pattern))
        run_dirs.extend(matched if matched else [pattern])

    rows = []
    operators_dir = os.path.join(args.out, "operators")
    os.makedirs(operators_dir, exist_ok=True)
    compare_lines = []
    for index, run_dir in enumerate(run_dirs, start=1):
        row = _read_run(run_dir)
        if row is None:
            print(f"  (sin corrida completa) {run_dir}")
            continue
        name = f"run-{index}"
        destination = os.path.join(operators_dir, f"{name}.py")
        shutil.copyfile(row["operator_path"], destination)
        row["name"] = name
        row["saved_as"] = destination
        compare_lines.append(f"{name}={destination}")
        rows.append(row)

    if not rows:
        raise SystemExit("No se encontro ninguna corrida completa (llamea_best_info.json).")

    # La distribucion del mejor-por-corrida: esto es lo que describe al
    # procedimiento, no el maximo de una corrida.
    distribution = {
        "test_fitness": _distribution([r["test_fitness"] for r in rows]),
        "train_fitness_final_frame": _distribution(
            [r["train_fitness_final_frame"] for r in rows]),
        "train_fitness_raw": _distribution([r["train_fitness_raw"] for r in rows]),
        "distinct_operators": _distribution([r["distinct_operators"] for r in rows]),
        "duplicate_evaluations": _distribution(
            [r["duplicate_evaluations"] for r in rows]),
    }
    result = {
        "runs": rows,
        "distribution_best_per_run": distribution,
        "note": ("El valor reportado por corrida es el mejor de ~budget candidatos "
                 "(sesgo de maximo). La media/mediana de la distribucion es la "
                 "estimacion honesta del procedimiento."),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    with open(os.path.join(args.out, "compare_args.txt"), "w") as handle:
        handle.write("\n".join(compare_lines) + "\n")

    lines = [f"# Replicas de la busqueda ({len(rows)} corridas)", "",
             "| Corrida | semilla | presup. | distintos | repes. | HV train (marco) | "
             "HV test | tokens |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['seed']} | {row['budget']} | "
            f"{row['distinct_operators']} | {row['duplicate_evaluations']} | "
            f"{_fmt(row['train_fitness_final_frame'])} | {_fmt(row['test_fitness'])} | "
            f"{row['total_tokens']} |"
        )
    lines += ["", "## Distribucion del mejor-por-corrida", "",
              "| Metrica | n | media | mediana | desv. | min | max |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key, stats in distribution.items():
        if not stats.get("n"):
            continue
        lines.append(f"| {key} | {stats['n']} | {_fmt(stats['mean'])} | "
                     f"{_fmt(stats['median'])} | {_fmt(stats['sd'])} | "
                     f"{_fmt(stats['min'])} | {_fmt(stats['max'])} |")
    lines += ["", "> El HV de test por corrida es comparable (mismas instancias); el "
              "HV de train NO lo es entre corridas (marco propio). Para comparar "
              "operadores de corridas distintas usa `scripts/compare_operators.py` "
              "con `compare_args.txt` (marco compartido).", ""]
    with open(os.path.join(args.out, "summary.md"), "w") as handle:
        handle.write("\n".join(lines))
    print(f"Resumen en {args.out} ({len(rows)} corridas)")


def _fmt(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
