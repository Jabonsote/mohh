#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparacion estadistica GE vs LLaMEA, pareada por experimento.

Lee los ``run_metrics/evaluation_*.json`` generados por
``scripts/evaluate_llamea_vs_ge.py`` (o por ``experiment.py``) y compara
Best/Middle/Worst de GE contra Best/Middle/Worst de LLaMEA.

La unidad de analisis es el **experimento** (misma poblacion inicial y misma
semilla para todos los operadores). Para cada experimento se promedia el HV
sobre las instancias (y opcionalmente sobre solver/mutacion/presupuesto) y se
aplica Wilcoxon signed-rank de dos colas sobre los pares. El ajuste de Holm se
aplica dentro de cada familia (Best/Middle/Worst).

Uso:
    python scripts/compare_ge_llamea_stats.py \
        results/searches/<RUN>/evaluation_results_vs_ge
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from mohh.llamea.statistics import add_holm, paired_summary


def parse_config(config):
    parts = str(config).split("_")
    return (parts[0] if parts else "",
            parts[1] if len(parts) > 1 else "",
            parts[2] if len(parts) > 2 else "",
            parts[3] if len(parts) > 3 else "")


def load(root):
    rows = []
    for path in glob.glob(os.path.join(root, "run_metrics", "evaluation_*.json")):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("sections", {}).get("hypervolumes", []):
            solver, operator, mutation, origin = parse_config(record["config"])
            match = re.search(r"(\d+)", str(record.get("experiment", "")))
            experiment = int(match.group(1)) if match else None
            if experiment is None:
                continue
            rows.append({
                "instance": record.get("instance"),
                "budget": record.get("label"),
                "solver": solver,
                "operator": operator,
                "mutation": mutation,
                "origin": origin,
                "experiment": experiment,
                "hv": float(record.get("hypervolume", np.nan)),
            })
    return pd.DataFrame(rows)


def mean_by_experiment(frame, origin, operator, filters):
    subset = frame[(frame.origin == origin) & (frame.operator == operator)]
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    if subset.empty:
        return {}
    grouped = subset.groupby("experiment")["hv"].mean()
    return {int(k): float(v) for k, v in grouped.items() if np.isfinite(v)}


def compare(frame, filters, labels=("Best", "Middle", "Worst")):
    rows = []
    for label in labels:
        ours = mean_by_experiment(frame, "LLAMEA", label, filters)
        base = mean_by_experiment(frame, "MOEAD", label, filters)
        common = sorted(set(ours) & set(base))
        ours_values = [ours[key] for key in common]
        base_values = [base[key] for key in common]
        summary = paired_summary(ours_values, base_values)
        summary["label"] = label
        summary["pairs"] = len(common)
        if common:
            diffs = np.asarray(ours_values) - np.asarray(base_values)
            summary["median_difference"] = float(np.median(diffs))
        else:
            summary["median_difference"] = None
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    add_holm(rows)
    return pd.DataFrame(rows)


def show(title, table):
    print(f"\n=== {title} ===")
    if table.empty:
        print("sin datos")
        return
    view = table.reindex(columns=["label", "pairs", "wins", "losses", "ties",
                                  "median_difference", "cohens_dz",
                                  "wilcoxon_p", "wilcoxon_p_holm",
                                  "significant_holm"])
    print(view.to_string(index=False,
                         formatters={"median_difference": "{:.6f}".format,
                                     "cohens_dz": "{:.3f}".format,
                                     "wilcoxon_p": "{:.4f}".format,
                                     "wilcoxon_p_holm": "{:.4f}".format}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Carpeta evaluation_results_vs_ge.")
    parser.add_argument("--labels", default="Best,Middle,Worst")
    args = parser.parse_args()

    frame = load(args.root)
    if frame.empty:
        raise SystemExit("No hay hypervolumes en run_metrics.")
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())

    show("Global (todas las instancias/configuraciones)",
         compare(frame, {}, labels))

    for solver in ("MOEAD", "NSGAII", "SMSEMOA"):
        show(f"Solver {solver}", compare(frame, {"solver": solver}, labels))

    for mutation, name in (("NM", "sin mutacion"), ("WM", "Swap_Mutation")):
        show(f"Mutacion {name}", compare(frame, {"mutation": mutation}, labels))

    for budget in ("10k", "30k", "50k"):
        show(f"Presupuesto {budget}", compare(frame, {"budget": budget}, labels))


if __name__ == "__main__":
    main()
