#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagrama de diferencias criticas (Demšar) para operadores de cruce.

Construye el ranking medio por instancia (HV) y el diagrama CD con la prueba
de Nemenyi (alpha=0.05). Cada bloque es una instancia (o una combinacion
instancia x solver x mutacion x presupuesto si no se filtra).

Uso:
    python scripts/critical_difference.py --experiment <Experiment_001> \
        --solver MOEAD --budget 50k --mutation NM --out docs/cd.png
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, studentized_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OPS = ["Best", "Middle", "Worst", "CX", "PMX"]


def load_hv(root):
    rows = []
    for path in glob.glob(os.path.join(root, "evaluation_results",
                                       "run_metrics", "evaluation_*.json")):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("sections", {}).get("hypervolumes", []):
            parts = str(record["config"]).split("_")
            rows.append({
                "instance": record["instance"],
                "budget": record["label"],
                "solver": parts[0],
                "operator": parts[1],
                "mutation": parts[2],
                "origin": parts[3] if len(parts) > 3 else "",
                "experiment": record["experiment"],
                "hv": float(record["hypervolume"]),
            })
    return pd.DataFrame(rows)


def build_matrix(frame, solver=None, budget=None, mutation=None, operators=None):
    subset = frame.copy()
    if solver:
        subset = subset[subset.solver == solver]
    if budget:
        subset = subset[subset.budget == budget]
    if mutation:
        subset = subset[subset.mutation == mutation]
    # GE operators have origin MOEAD; classics have origin None
    subset = subset[
        ((subset.origin == "MOEAD") & subset.operator.isin(["Best", "Middle", "Worst"]))
        | ((subset.origin == "None") & subset.operator.isin(["CX", "PMX"]))
    ]
    operators = operators or OPS
    subset = subset[subset.operator.isin(operators)]
    # mean HV per block (instance x solver x mutation x budget) and operator
    grouped = (subset.groupby(["instance", "solver", "mutation", "budget", "operator"])
               ["hv"].mean().unstack("operator"))
    grouped = grouped.reindex(columns=operators).dropna(how="any")
    return grouped


def critical_difference(avg_ranks, n_blocks, alpha=0.05):
    k = len(avg_ranks)
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks)), q_alpha


def cliques(avg_ranks, cd):
    """Grupos maximales con diferencia de rango <= CD."""
    order = avg_ranks.sort_values().index.tolist()
    values = avg_ranks.loc[order].to_numpy()
    groups = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and values[j + 1] - values[i] <= cd:
            j += 1
        groups.append(set(order[i:j + 1]))
    maximal = []
    for group in groups:
        if not any(group < other for other in groups):
            if group not in maximal:
                maximal.append(group)
    return maximal


def plot(avg_ranks, cd, n_blocks, title, out):
    k = len(avg_ranks)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-2.0, 1.9)
    ax.axis("off")

    # Eje de rangos
    ax.plot([1, k], [0, 0], color="#444", lw=1.2, zorder=1)
    for rank in range(1, k + 1):
        ax.plot([rank, rank], [-0.06, 0.06], color="#444", lw=1.0)
        ax.text(rank, -0.18, str(rank), ha="center", va="top", fontsize=9)

    # Puntos y etiquetas con niveles para evitar solapamientos
    levels = [0.30, -0.45, 0.65, -0.80, 1.00, -1.15, 1.30, -1.45]
    used = []
    for name, rank in avg_ranks.sort_values().items():
        ax.plot(rank, 0, "o", color="#1f77b4", ms=7, zorder=3)
        level = levels[0]
        for candidate in levels:
            if all(abs(rank - other_rank) > 0.45 or candidate != other_level
                   for other_rank, other_level in used):
                level = candidate
                break
        used.append((rank, level))
        va = "bottom" if level > 0 else "top"
        ax.text(rank, level, f"{name} ({rank:.2f})", ha="center",
                va=va, fontsize=9, color="#1f3864")

    # Barra de diferencia critica
    cd_y = 1.65
    ax.plot([1, 1 + cd], [cd_y, cd_y], color="#c00000", lw=3,
            solid_capstyle="butt")
    ax.plot([1, 1], [cd_y - 0.12, cd_y + 0.12], color="#c00000", lw=2)
    ax.plot([1 + cd, 1 + cd], [cd_y - 0.12, cd_y + 0.12],
            color="#c00000", lw=2)
    ax.text(1 + cd / 2, cd_y + 0.15, f"CD = {cd:.2f}", ha="center",
            va="bottom", fontsize=9, color="#c00000")

    # Cliques (grupos no significativamente diferentes)
    groups = cliques(avg_ranks, cd)
    for index, group in enumerate(groups):
        ranks = [avg_ranks[name] for name in group]
        y = -1.05 - 0.22 * index
        ax.plot([min(ranks), max(ranks)], [y, y], color="#2e7d32", lw=3,
                solid_capstyle="round", alpha=0.85)

    ax.set_title(title, fontsize=11, pad=14)
    ax.text(1, -1.9, "1 = mejor HV", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True,
                        help="Carpeta del experimento GE (con evaluation_results).")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--budget", default=None)
    parser.add_argument("--mutation", default=None)
    parser.add_argument("--operators", default="Best,Middle,Worst,CX,PMX",
                        help="Operadores separados por coma.")
    parser.add_argument("--out", required=True, help="PNG de salida.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    frame = load_hv(args.experiment)
    operators = [name.strip() for name in args.operators.split(",") if name.strip()]
    matrix = build_matrix(frame, args.solver, args.budget, args.mutation, operators)
    if matrix.empty:
        raise SystemExit("Sin bloques completos para esa combinacion.")

    # Ranking por bloque: HV mayor = mejor = rango 1
    ranks = np.vstack([rankdata(-row, method="average")
                       for row in matrix.to_numpy()])
    avg_ranks = pd.Series(ranks.mean(axis=0), index=matrix.columns)
    cd, _ = critical_difference(avg_ranks, len(matrix))

    title = args.title or "Diferencias criticas"
    details = []
    if args.solver:
        details.append(args.solver)
    if args.budget:
        details.append(args.budget)
    if args.mutation:
        details.append(args.mutation)
    if details:
        title += " — " + " / ".join(details)
    title += f"  (N={len(matrix)})"

    plot(avg_ranks, cd, len(matrix), title, args.out)
    print(f"Guardado: {args.out}")
    print(avg_ranks.sort_values().to_string())
    print(f"CD = {cd:.4f} (N={len(matrix)})")


if __name__ == "__main__":
    main()
