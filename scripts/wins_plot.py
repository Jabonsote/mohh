#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grafica de victorias por instancia (sobre 20) frente a CX y PMX."""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OPS = ["Best"]


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
                "hv": float(record["hypervolume"]),
            })
    return pd.DataFrame(rows)


def wins_table(frame):
    subset = frame[
        ((frame.origin == "MOEAD") & frame.operator.isin(["Best"]))
        | ((frame.origin == "None") & frame.operator.isin(["CX", "PMX"]))
    ]
    mean = (subset.groupby(["solver", "budget", "mutation", "instance",
                            "operator"])["hv"].mean().unstack("operator"))
    mean = mean.dropna(how="any")
    rows = []
    for (solver, budget, mutation), group in mean.groupby(level=[0, 1, 2]):
        group = group.droplevel([0, 1, 2])
        rows.append({
            "solver": solver,
            "budget": budget,
            "mutation": mutation,
            "vs_CX": int((group["Best"] > group["CX"]).sum()),
            "vs_PMX": int((group["Best"] > group["PMX"]).sum()),
            "n": len(group),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frame = load_hv(args.experiment)
    table = wins_table(frame)
    solvers = ["MOEAD", "NSGAII", "SMSEMOA"]
    budgets = ["10k", "30k", "50k"]
    mutations = [("NM", "sin mutacion"), ("WM", "Swap_Mutation")]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    for row, base in enumerate(["vs_CX", "vs_PMX"]):
        for col, (mut, mut_label) in enumerate(mutations):
            ax = axes[row][col]
            x = np.arange(len(budgets))
            width = 0.8 / len(solvers)
            for index, solver in enumerate(solvers):
                values = []
                for budget in budgets:
                    match = table[(table.solver == solver)
                                  & (table.budget == budget)
                                  & (table.mutation == mut)]
                    values.append(int(match[base].iloc[0]) if len(match) else 0)
                positions = x + (index - (len(solvers) - 1) / 2) * width
                bars = ax.bar(positions, values, width, label=solver,
                              color=cols[index])
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3,
                            str(value), ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(budgets)
            ax.set_ylim(0, 21)
            ax.axhline(10, color="#999", lw=0.8, ls="--")
            ax.set_title(f"{base.replace('_', ' ')} — {mut_label}")
            if col == 0:
                ax.set_ylabel("victorias / 20")
            ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Experiment_001 — Best vs CX/PMX (20 instancias, media de 31 réplicas)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {args.out}")
    print(table.sort_values(["solver", "mutation", "budget"]).to_string(index=False))


if __name__ == "__main__":
    main()
