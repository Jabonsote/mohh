#!/usr/bin/env python
"""
Analiza un evaluation_results parcial de mohh-evaluate.

Entrada: results/Experiment_001/evaluation_results (MOEAD/*/*.csv).
Salida:  results/eval_partial_analysis/
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGETS = ("10k", "30k", "50k")


def _leer(base):
    filas = []
    for budget in BUDGETS:
        for ruta in sorted(glob.glob(os.path.join(base, "MOEAD", budget, "*.csv"))):
            instancia = os.path.basename(ruta)[:-4]
            tabla = pd.read_csv(ruta)
            if "Experiment" not in tabla.columns or len(tabla) != 31:
                raise ValueError(f"{ruta}: se esperaban 31 experimentos")
            valores = tabla.drop(columns=["Experiment"])
            medias = valores.mean()
            ranks = medias.rank(ascending=False, method="average")
            for config in medias.index:
                filas.append({
                    "budget": budget,
                    "instance": instancia,
                    "config": config,
                    "mean_hv": float(medias[config]),
                    "rank": float(ranks[config]),
                })
    return pd.DataFrame(filas)


def _tipo(config):
    partes = config.split("_")
    solver, operador, mutacion, fuente = partes[0], partes[1], partes[2], partes[3]
    descubierto = operador in ("Best", "Middle", "Worst")
    return solver, operador, mutacion, fuente, descubierto


def _grafica_ranks(tabla, ruta):
    orden = (tabla.groupby("config")["rank"].mean()
             .sort_values().head(15))
    figura, eje = plt.subplots(figsize=(9, 5))
    eje.barh(orden.index[::-1], orden.values[::-1], color="#1f77b4")
    eje.set_xlabel("rango promedio (menor = mejor)")
    eje.set_title("Top 15 configuraciones — 14 instancias")
    eje.grid(axis="x", alpha=0.3)
    figura.tight_layout()
    figura.savefig(ruta, dpi=180)
    plt.close(figura)


def _grafica_budget(tabla, ruta):
    medias = tabla.groupby("budget")["mean_hv"].mean().reindex(BUDGETS)
    figura, eje = plt.subplots(figsize=(6, 4))
    eje.plot(medias.index, medias.values, marker="o", linewidth=1.5)
    eje.set_xlabel("presupuesto")
    eje.set_ylabel("HV medio")
    eje.set_title("HV medio por presupuesto (14 instancias)")
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(ruta, dpi=180)
    plt.close(figura)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Análisis de evaluación parcial.")
    parser.add_argument(
        "--results",
        default="results/Experiment_001/evaluation_results")
    parser.add_argument(
        "--output", default="results/eval_partial_analysis")
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    tabla = _leer(args.results)
    tabla[["solver", "operator", "mutation", "source", "discovered"]] = (
        tabla["config"].apply(lambda c: pd.Series(_tipo(c))))
    tabla.to_csv(os.path.join(args.output, "all_values.csv"), index=False)

    medias = (tabla.pivot_table(index=["instance", "config"], columns="budget",
                                values="mean_hv").reset_index())
    medias.to_csv(os.path.join(args.output, "mean_hv_by_instance_budget.csv"),
                  index=False)

    ranks = (tabla.groupby(["config", "budget"])["rank"].mean()
             .unstack("budget").reindex(columns=BUDGETS))
    ranks["mean"] = ranks.mean(axis=1)
    ranks = ranks.sort_values("mean")
    ranks.to_csv(os.path.join(args.output, "average_ranks.csv"))

    mejores = (tabla.loc[tabla.groupby(["budget", "instance"])["mean_hv"].idxmax()]
               .sort_values(["budget", "instance"]))
    mejores.to_csv(os.path.join(args.output, "best_by_instance.csv"), index=False)

    # Descubiertos vs líneas base: mejor de cada grupo por instancia/presupuesto.
    resumen = []
    for budget in BUDGETS:
        sub = tabla[tabla["budget"] == budget]
        disc = (sub[sub["discovered"]]
                .loc[sub[sub["discovered"]].groupby("instance")["mean_hv"].idxmax()])
        base = (sub[~sub["discovered"]]
                .loc[sub[~sub["discovered"]].groupby("instance")["mean_hv"].idxmax()])
        ganadas = int((disc.set_index("instance")["mean_hv"]
                       > base.set_index("instance")["mean_hv"]).sum())
        resumen.append({
            "budget": budget,
            "discovered_mean_hv": float(disc["mean_hv"].mean()),
            "baseline_mean_hv": float(base["mean_hv"].mean()),
            "discovered_mean_rank": float(disc["rank"].mean()),
            "baseline_mean_rank": float(base["rank"].mean()),
            "discovered_wins": ganadas,
            "instances": int(len(disc)),
        })
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(os.path.join(args.output, "discovered_vs_baseline.csv"),
                   index=False)

    _grafica_ranks(tabla, os.path.join(args.output, "average_rank_top15.png"))
    _grafica_budget(tabla, os.path.join(args.output, "hv_by_budget.png"))

    instancias = sorted(tabla["instance"].unique())
    lineas = [
        "# Análisis de evaluación parcial — Experiment_001",
        "",
        f"Instancias completas: **{len(instancias)}**.",
        "",
        "Presupuestos analizados: `10k`, `30k`, `50k`.",
        "",
        "Cada valor es la media de 31 experimentos independientes. El ranking "
        "se calcula dentro de cada instancia y presupuesto; menor rango es mejor.",
        "",
        "## Cobertura",
        "",
        "```text",
    ]
    lineas.extend(instancias)
    lineas.extend([
        "```",
        "",
        "## HV medio por presupuesto",
        "",
        "| presupuesto | HV medio |",
        "|---|---:|",
    ])
    for budget in BUDGETS:
        lineas.append(
            f"| {budget} | {tabla[tabla['budget']==budget]['mean_hv'].mean():.4f} |")
    lineas.extend([
        "",
        "> La escala HV se recalcula por instancia y presupuesto; sirve para "
        "ordenar configuraciones dentro de cada instancia, no para comparar "
        "valores absolutos entre instancias.",
        "",
        "## Ranking promedio de configuraciones",
        "",
        "| configuración | 10k | 30k | 50k | media |",
        "|---|---:|---:|---:|---:|",
    ])
    for config, fila in ranks.head(12).iterrows():
        lineas.append(
            f"| {config} | {fila['10k']:.2f} | {fila['30k']:.2f} | "
            f"{fila['50k']:.2f} | {fila['mean']:.2f} |")
    lineas.extend([
        "",
        "## Descubiertos vs líneas base",
        "",
        "| presupuesto | HV descubiertos | HV baseline | rango descubiertos | rango baseline | victorias descubiertos |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for _, fila in resumen.iterrows():
        lineas.append(
            f"| {fila['budget']} | {fila['discovered_mean_hv']:.4f} | "
            f"{fila['baseline_mean_hv']:.4f} | "
            f"{fila['discovered_mean_rank']:.2f} | "
            f"{fila['baseline_mean_rank']:.2f} | "
            f"{fila['discovered_wins']}/{fila['instances']} |")
    lineas.extend([
        "",
        "## Lectura",
        "",
        "- El presupuesto 50k tiene el mayor HV medio, como se espera.",
        "- Las líneas base PMX/CX son muy fuertes en estas 14 instancias.",
        "- La mejor configuración descubierta es `MOEAD_Best_NM_MOEAD`, pero "
        "queda por detrás de `SMSEMOA_PMX_WM_None` y `SMSEMOA_PMX_NM_None`.",
        "- En 10k los descubiertos no ganan ninguna instancia; en 30k y 50k "
        "ganan 4 de 14 cada uno.",
        "- Esta lectura es parcial: faltan 6 instancias de Experiment_001 y "
        "todo Experiment_002.",
        "",
        "## Archivos",
        "",
        "```text",
        "all_values.csv",
        "mean_hv_by_instance_budget.csv",
        "average_ranks.csv",
        "best_by_instance.csv",
        "discovered_vs_baseline.csv",
        "average_rank_top15.png",
        "hv_by_budget.png",
        "```",
    ])
    with open(os.path.join(args.output, "README.md"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"Análisis escrito en {args.output}")


if __name__ == "__main__":
    main()
