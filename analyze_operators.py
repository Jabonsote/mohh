#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisis de los operadores generados (Best/Middle/Worst de GE) frente a los
operadores clasicos (CX, PMX) en los Experimentos 001 y 002.

Genera un reporte HTML autocontenido con:
  - codigo Python de los operadores
  - informacion de las corridas independientes (31 replicas)
  - victorias por operador, por problema (instancia), por MOEA
  - pruebas de Wilcoxon pareadas (no solo HV medio)
  - graficas embebidas (base64 PNG)

Uso:
  python analyze_operators.py
  python analyze_operators.py --exp-dirs experimentos_001_002/Experiment_001 experimentos_001_002/Experiment_002
"""
import argparse
import base64
import io
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mohh.analysis.phenotype import to_python_module

ROOT = Path(__file__).resolve().parent
DEFAULT_EXPS = [
    ROOT / "experimentos_001_002" / "Experiment_001",
    ROOT / "experimentos_001_002" / "Experiment_002",
]
OUT_HTML = ROOT / "results" / "reports" / "operators_report.html"
OUT_CSV = ROOT / "results" / "reports" / "operators_stats.csv"

GENERATED = ["Best", "Middle", "Worst"]
BASELINES = ["CX", "PMX"]
ALL_OPS = GENERATED + BASELINES
SOLVERS = ["MOEAD", "NSGAII", "SMSEMOA"]
MUTATIONS = ["NM", "WM"]   # NM = sin mutacion, WM = con Swap_Mutation
BUDGETS = ["10k", "30k", "50k"]


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
def parse_config(cfg):
    # "MOEAD_Best_NM_MOEAD" -> solver, operator, mutation, origin
    parts = cfg.split("_")
    solver = parts[0]
    operator = parts[1]
    mutation = parts[2]
    origin = parts[3] if len(parts) > 3 else ""
    # normalizar nombres de operador a los usados en el reporte
    op_map = {"CX": "CX", "PMX": "PMX", "Best": "Best",
              "Middle": "Middle", "Worst": "Worst"}
    operator = op_map.get(operator, operator)
    mut_map = {"NM": "NM", "WM": "WM"}
    mutation = mut_map.get(mutation, mutation)
    return solver, operator, mutation, origin


def load_hv(exp_dir):
    exp_dir = Path(exp_dir)
    folder_name = exp_dir.name
    rows = []
    rm = exp_dir / "evaluation_results" / "run_metrics"
    for f in sorted(rm.glob("evaluation_*.json")):
        d = json.load(open(f))
        s = d["sections"]
        instance = d["instance"]
        for h in s["hypervolumes"]:
            solver, operator, mutation, origin = parse_config(h["config"])
            rep = int(h["experiment"].split()[-1])
            rows.append({
                "exp_folder": folder_name,
                "instance": instance,
                "budget": h["label"],
                "solver": solver,
                "operator": operator,
                "mutation": mutation,
                "origin": origin,
                "replicate": rep,
                "hv": float(h["hypervolume"]),
            })
    return pd.DataFrame(rows)


def load_operator_code(exp_dir):
    p = Path(exp_dir) / "MOEAD_operators.txt"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        for name in ("Best", "Middle", "Worst"):
            if line.startswith(name + ":"):
                pheno = line[len(name) + 1:].strip()
                try:
                    py = to_python_module(pheno, name=f"operate_{name.lower()}")
                except Exception as e:
                    py = f"# No se pudo traducir a Python: {e}"
                out[name] = {"phenotype": pheno, "python": py}
    return out


# ---------------------------------------------------------------------------
# Estadistica pareada
# ---------------------------------------------------------------------------
def paired_compare(df, folder, instance, solver, mutation, budget,
                   gen_op, base_op):
    sub = df[(df.exp_folder == folder) & (df.instance == instance) &
             (df.solver == solver) & (df.mutation == mutation) &
             (df.budget == budget)]
    g = sub[sub.operator == gen_op].set_index("replicate")["hv"]
    b = sub[sub.operator == base_op].set_index("replicate")["hv"]
    common = g.index.intersection(b.index)
    if len(common) < 5:
        return None
    g = g.loc[common]
    b = b.loc[common]
    diff = (g - b).values
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    n = len(diff)
    # Wilcoxon (two-sided)
    try:
        if np.all(diff == 0):
            p = 1.0
        else:
            p = stats.wilcoxon(diff, alternative="two-sided").pvalue
    except Exception:
        p = float("nan")
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd and sd > 0 else float("nan")
    med = float(np.median(diff))
    mean = float(diff.mean())
    sig = (p < 0.05) and (med > 0)
    return {
        "n": n, "wins": wins, "losses": losses, "ties": ties,
        "median_diff": med, "mean_diff": mean, "p": p, "dz": dz,
        "significant_win": bool(sig),
    }


def run_analysis(df):
    records = []
    folders = sorted(df.exp_folder.unique())
    instances = sorted(df.instance.unique())
    for folder in folders:
        for instance in instances:
            for solver in SOLVERS:
                for mutation in MUTATIONS:
                    for budget in BUDGETS:
                        for gen_op in GENERATED:
                            for base_op in BASELINES:
                                r = paired_compare(df, folder, instance, solver,
                                                   mutation, budget, gen_op,
                                                   base_op)
                                if r is None:
                                    continue
                                records.append({
                                    "exp_folder": folder,
                                    "instance": instance,
                                    "solver": solver,
                                    "mutation": mutation,
                                    "budget": budget,
                                    "gen_op": gen_op,
                                    "base_op": base_op,
                                    **r,
                                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Graficas -> base64 PNG
# ---------------------------------------------------------------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def chart_wins_by_operator(stats_df):
    # victorias significativas por operador (agregado sobre todos)
    g = stats_df[stats_df.significant_win].groupby("gen_op").size()
    total = stats_df.groupby("gen_op").size()
    ops = GENERATED
    sig_vals = [int(g.get(o, 0)) for o in ops]
    tot_vals = [int(total.get(o, 0)) for o in ops]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ops))
    ax.bar(x - 0.2, tot_vals, 0.4, label="Comparaciones totales", color="#9ecae9")
    ax.bar(x + 0.2, sig_vals, 0.4, label="Victorias significativas (p<0.05)", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    ax.set_ylabel("# comparaciones")
    ax.set_title("Victorias por operador (vs CX y PMX)")
    ax.legend()
    for i, v in enumerate(sig_vals):
        ax.text(x[i] + 0.2, v + 1, str(v), ha="center", fontsize=9)
    return fig_to_b64(fig)


def chart_wins_by_solver(stats_df):
    # matriz operador x solver: victorias significativas
    pivot = (stats_df[stats_df.significant_win]
             .groupby(["gen_op", "solver"]).size()
             .unstack(fill_value=0))
    pivot = pivot.reindex(index=GENERATED, columns=SOLVERS, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(SOLVERS)))
    ax.set_xticklabels(SOLVERS, rotation=30, ha="right")
    ax.set_yticks(range(len(GENERATED)))
    ax.set_yticklabels(GENERATED)
    for i in range(len(GENERATED)):
        for j in range(len(SOLVERS)):
            ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center",
                    color="black", fontsize=11)
    ax.set_title("Victorias significativas: operador x MOEA")
    fig.colorbar(im, ax=ax, label="# victorias")
    return fig_to_b64(fig)


def chart_wins_by_instance(stats_df):
    pivot = (stats_df[stats_df.significant_win]
             .groupby(["instance", "gen_op"]).size()
             .unstack(fill_value=0))
    pivot = pivot.reindex(columns=GENERATED, fill_value=0)
    # ordenar por total de victorias
    pivot["_tot"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("_tot", ascending=True).drop(columns="_tot")
    fig, ax = plt.subplots(figsize=(9, 7))
    bottom = np.zeros(len(pivot))
    colors = {"Best": "#1f77b4", "Middle": "#ff7f0e", "Worst": "#2ca02c"}
    for op in GENERATED:
        vals = pivot[op].values
        ax.barh(pivot.index, vals, left=bottom, label=op, color=colors[op])
        bottom = bottom + vals
    ax.set_xlabel("# victorias significativas")
    ax.set_title("Victorias significativas por instancia (problema)")
    ax.legend(loc="lower right")
    return fig_to_b64(fig)


def chart_hv_boxplot(df, folder):
    sub = df[df.exp_folder == folder]
    # un boxplot por operador, agrupado por solver, para presupuesto 50k, NM
    data = []
    labels = []
    for solver in SOLVERS:
        for op in ALL_OPS:
            vals = sub[(sub.solver == solver) & (sub.operator == op) &
                       (sub.mutation == "NM") & (sub.budget == "50k")]["hv"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(f"{solver}\n{op}")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("HV (normalizado)")
    ax.set_title(f"Distribucion de HV por operador y MOEA (50k, sin mutacion) - {folder}")
    return fig_to_b64(fig)


def chart_wins_by_budget(stats_df):
    pivot = (stats_df[stats_df.significant_win]
             .groupby(["budget", "gen_op"]).size()
             .unstack(fill_value=0))
    pivot = pivot.reindex(index=BUDGETS, columns=GENERATED, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(BUDGETS))
    w = 0.25
    for i, op in enumerate(GENERATED):
        ax.bar(x + (i - 1) * w, pivot[op].values, w, label=op)
    ax.set_xticks(x)
    ax.set_xticklabels(BUDGETS)
    ax.set_ylabel("# victorias significativas")
    ax.set_title("Victorias significativas por presupuesto")
    ax.legend()
    return fig_to_b64(fig)


def chart_wins_by_mutation(stats_df):
    pivot = (stats_df[stats_df.significant_win]
             .groupby(["mutation", "gen_op"]).size()
             .unstack(fill_value=0))
    pivot = pivot.reindex(index=MUTATIONS, columns=GENERATED, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(MUTATIONS))
    w = 0.25
    for i, op in enumerate(GENERATED):
        ax.bar(x + (i - 1) * w, pivot[op].values, w, label=op)
    ax.set_xticks(x)
    ax.set_xticklabels(["Sin mutacion (NM)", "Con Swap_Mutation (WM)"])
    ax.set_ylabel("# victorias significativas")
    ax.set_title("Victorias significativas por mutacion")
    ax.legend()
    return fig_to_b64(fig)


def chart_moea_heatmap_wins(stats_df):
    sub = stats_df[stats_df.solver == "MOEAD"]
    pivot = (sub[sub.significant_win]
             .groupby(["instance", "gen_op"]).size()
             .unstack(fill_value=0))
    pivot = pivot.reindex(columns=GENERATED, fill_value=0)
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(GENERATED)))
    ax.set_xticklabels(GENERATED)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(len(GENERATED)):
            ax.text(j, i, str(pivot.values[i, j]), ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title("MOEAD: victorias significativas por instancia")
    fig.colorbar(im, ax=ax, label="# victorias")
    return fig_to_b64(fig)


def chart_moea_hv_heatmap(df):
    sub = df[(df.solver == "MOEAD") & (df.mutation == "NM") &
             (df.budget == "50k")]
    med = (sub.groupby(["instance", "operator"])["hv"]
           .median().unstack())
    med = med.reindex(columns=ALL_OPS)
    med = med.sort_index()
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(med.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ALL_OPS)))
    ax.set_xticklabels(ALL_OPS, rotation=30, ha="right")
    ax.set_yticks(range(len(med)))
    ax.set_yticklabels(med.index, fontsize=8)
    for i in range(len(med)):
        for j in range(len(ALL_OPS)):
            v = med.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=7)
    ax.set_title("MOEAD: HV mediano por instancia y operador (50k, sin mutacion)")
    fig.colorbar(im, ax=ax, label="HV mediano")
    return fig_to_b64(fig)


def chart_moea_wins_detail(stats_df):
    sub = stats_df[stats_df.solver == "MOEAD"]
    # victorias por operador x (mutation, budget)
    g = (sub[sub.significant_win]
         .groupby(["mutation", "budget", "gen_op"]).size()
         .reset_index(name="wins"))
    # pivot para barras agrupadas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mut in zip(axes, MUTATIONS):
        gg = g[g.mutation == mut]
        pivot = gg.pivot(index="budget", columns="gen_op", values="wins").fillna(0)
        pivot = pivot.reindex(index=BUDGETS, columns=GENERATED, fill_value=0)
        x = np.arange(len(BUDGETS))
        w = 0.25
        for i, op in enumerate(GENERATED):
            ax.bar(x + (i - 1) * w, pivot[op].values, w, label=op)
        ax.set_xticks(x)
        ax.set_xticklabels(BUDGETS)
        ax.set_title(f"MOEAD {'sin mutacion' if mut=='NM' else 'con Swap_Mutation'}")
        ax.set_ylabel("# victorias sig.")
        ax.legend()
    fig.suptitle("MOEAD: victorias significativas por presupuesto y mutacion")
    return fig_to_b64(fig)
# ---------------------------------------------------------------------------
def df_to_html_table(df, max_rows=None, float_cols=()):
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    out = df.copy()
    for c in float_cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return out.to_html(index=False, escape=True, border=0,
                      classes="tbl", na_rep="")


def build_html(df, stats_df, exp_dirs, operator_codes):
    folders = sorted(df.exp_folder.unique())
    n_instances = df.instance.nunique()
    n_reps = df.replicate.nunique()

    # Victorias por operador (global)
    g_sig = stats_df[stats_df.significant_win].groupby("gen_op").size()
    g_all = stats_df.groupby("gen_op").size()
    g_loss = stats_df[stats_df.significant_win == False]
    # losses significativas = comparaciones donde base gana (mediana_diff<0 y p<0.05)
    sig_loss = stats_df[(stats_df.p < 0.05) & (stats_df.median_diff < 0)]
    g_sigloss = sig_loss.groupby("gen_op").size()

    summary_rows = []
    for op in GENERATED:
        ncomp = int(g_all.get(op, 0))
        nwin = int(g_sig.get(op, 0))
        nloss = int(g_sigloss.get(op, 0))
        ntie = ncomp - nwin - nloss
        summary_rows.append({
            "Operador": op,
            "Comparaciones": ncomp,
            "Victorias sig.": nwin,
            "Derrotas sig.": nloss,
            "Sin diferencia sig.": ntie,
            "% victorias sig.": (100.0 * nwin / ncomp) if ncomp else 0.0,
        })
    summary_df = pd.DataFrame(summary_rows)

    # Por solver
    solver_rows = []
    for solver in SOLVERS:
        for op in GENERATED:
            sub = stats_df[(stats_df.solver == solver) & (stats_df.gen_op == op)]
            ncomp = len(sub)
            nwin = int(sub.significant_win.sum())
            nloss = int(((sub.p < 0.05) & (sub.median_diff < 0)).sum())
            solver_rows.append({
                "MOEA": solver, "Operador": op,
                "Comparaciones": ncomp,
                "Victorias sig.": nwin,
                "Derrotas sig.": nloss,
                "% victorias": (100.0 * nwin / ncomp) if ncomp else 0.0,
            })
    solver_df = pd.DataFrame(solver_rows)

    # Por instancia: operador con mas victorias significativas
    inst_rows = []
    for inst in sorted(df.instance.unique()):
        sub = stats_df[(stats_df.instance == inst) & (stats_df.significant_win)]
        counts = sub.groupby("gen_op").size()
        best = counts.idxmax() if len(counts) else "-"
        inst_rows.append({
            "Instancia": inst,
            "Best": int(counts.get("Best", 0)),
            "Middle": int(counts.get("Middle", 0)),
            "Worst": int(counts.get("Worst", 0)),
            "Total vict. sig.": int(counts.sum()),
            "Operador dominante": best,
        })
    inst_df = pd.DataFrame(inst_rows)

    # Instancias donde el clasico gana (global, todos los MOEAs)
    classic_global_df = stats_df[(stats_df.p < 0.05) & (stats_df.median_diff < 0)].copy()
    classic_global_df = classic_global_df.sort_values(["instance", "p"])
    classic_global_view = classic_global_df[["exp_folder", "instance", "solver", "mutation",
                                            "budget", "gen_op", "base_op", "n", "wins",
                                            "losses", "ties", "median_diff", "p", "dz"]].copy()
    classic_global_view.columns = ["Exp", "Instancia", "MOEA", "Mut", "Budget", "Op",
                                   "Base (clasico)", "n", "W", "L", "T", "mediana_diff",
                                   "p", "dz"]
    classic_inst_rows = []
    for inst in sorted(df.instance.unique()):
        sub = classic_global_df[classic_global_df.instance == inst]
        classic_inst_rows.append({
            "Instancia": inst,
            "CX gana": int((sub.base_op == "CX").sum()),
            "PMX gana": int((sub.base_op == "PMX").sum()),
            "Total clas.": len(sub),
        })
    classic_inst_df = pd.DataFrame(classic_inst_rows)

    # Por presupuesto
    bud_rows = []
    for b in BUDGETS:
        for op in GENERATED:
            sub = stats_df[(stats_df.budget == b) & (stats_df.gen_op == op)]
            bud_rows.append({
                "Presupuesto": b, "Operador": op,
                "Comparaciones": len(sub),
                "Victorias sig.": int(sub.significant_win.sum()),
            })
    bud_df = pd.DataFrame(bud_rows)

    # Por mutacion
    mut_rows = []
    for m in MUTATIONS:
        for op in GENERATED:
            sub = stats_df[(stats_df.mutation == m) & (stats_df.gen_op == op)]
            mut_rows.append({
                "Mutacion": "Sin mutacion (NM)" if m == "NM" else "Con Swap_Mutation (WM)",
                "Operador": op,
                "Comparaciones": len(sub),
                "Victorias sig.": int(sub.significant_win.sum()),
                "% victorias": (100.0 * int(sub.significant_win.sum()) / len(sub)) if len(sub) else 0.0,
            })
    mut_df = pd.DataFrame(mut_rows)

    # Por experimento (carpeta)
    exp_rows = []
    for folder in folders:
        sub = stats_df[stats_df.exp_folder == folder]
        for op in GENERATED:
            s2 = sub[sub.gen_op == op]
            exp_rows.append({
                "Experimento": folder,
                "Operador": op,
                "Comparaciones": len(s2),
                "Victorias sig.": int(s2.significant_win.sum()),
                "% victorias": (100.0 * int(s2.significant_win.sum()) / len(s2)) if len(s2) else 0.0,
            })
    exp_df = pd.DataFrame(exp_rows)

    # Top comparaciones significativas (para tabla detallada): una fila por (instancia, solver, mutation, budget, gen_op, base_op)
    detail = stats_df.copy()
    detail["sig"] = detail.apply(
        lambda r: "WIN" if r.significant_win else ("LOSS" if (r.p < 0.05 and r.median_diff < 0) else "ns"), axis=1)
    # ordenar: victorias primero
    detail = detail.sort_values(["sig", "p"], ascending=[True, True])
    detail_view = detail[["exp_folder", "instance", "solver", "mutation", "budget",
                          "gen_op", "base_op", "n", "wins", "losses", "ties",
                          "median_diff", "p", "dz", "sig"]].copy()
    detail_view.columns = ["Exp", "Instancia", "MOEA", "Mut", "Budget", "Op",
                           "Base", "n", "W", "L", "T", "mediana_diff", "p", "dz",
                           "sig"]

    # MOEAD-focused
    moea_detail = stats_df[stats_df.solver == "MOEAD"].copy()
    moea_detail = moea_detail.sort_values(["instance", "mutation", "budget", "gen_op", "base_op"])
    moea_summary_rows = []
    for m in MUTATIONS:
        for op in GENERATED:
            for base in BASELINES:
                sub = moea_detail[(moea_detail.mutation == m) & (moea_detail.gen_op == op) &
                                  (moea_detail.base_op == base)]
                ncomp = len(sub)
                nwin = int(sub.significant_win.sum())
                nloss = int(((sub.p < 0.05) & (sub.median_diff < 0)).sum())
                moea_summary_rows.append({
                    "Mutacion": "Sin mutacion (NM)" if m == "NM" else "Con Swap_Mutation (WM)",
                    "Operador": op,
                    "Base": base,
                    "Comparaciones": ncomp,
                    "Victorias sig.": nwin,
                    "Derrotas sig.": nloss,
                    "% victorias": (100.0 * nwin / ncomp) if ncomp else 0.0,
                })
    moea_summary_df = pd.DataFrame(moea_summary_rows)

    # HV mediano por instancia, MOEAD, NM, 50k
    hv_sub = df[(df.solver == "MOEAD") & (df.mutation == "NM") & (df.budget == "50k")]
    hv_med = hv_sub.groupby(["instance", "operator"])["hv"].median().unstack()
    hv_med = hv_med.reindex(columns=ALL_OPS)
    hv_med = hv_med.reset_index().rename(columns={"instance": "Instancia"})
    moea_hv_df = hv_med

    MOEA_CSV = ROOT / "results" / "reports" / "moea_stats.csv"
    moea_detail_view = moea_detail[["exp_folder", "instance", "mutation", "budget",
                                    "gen_op", "base_op", "n", "wins", "losses", "ties",
                                    "median_diff", "p", "dz"]].copy()
    moea_detail_view.columns = ["Exp", "Instancia", "Mut", "Budget", "Op", "Base",
                                "n", "W", "L", "T", "mediana_diff", "p", "dz"]
    moea_detail_view.to_csv(MOEA_CSV, index=False)

    # Victorias por instancia (MOEAD)
    inst_rows_moea = []
    for inst in sorted(df.instance.unique()):
        sub = moea_detail[moea_detail.instance == inst]
        gen_wins = sub[sub.significant_win].groupby("gen_op").size()
        classic_wins = sub[(sub.p < 0.05) & (sub.median_diff < 0)].groupby("base_op").size()
        inst_rows_moea.append({
            "Instancia": inst,
            "Best": int(gen_wins.get("Best", 0)),
            "Middle": int(gen_wins.get("Middle", 0)),
            "Worst": int(gen_wins.get("Worst", 0)),
            "CX gana": int(classic_wins.get("CX", 0)),
            "PMX gana": int(classic_wins.get("PMX", 0)),
            "Total gen.": int(gen_wins.sum()),
            "Total clas.": int(classic_wins.sum()),
        })
    moea_inst_df = pd.DataFrame(inst_rows_moea)

    # Instancias donde el clasico gana (MOEAD)
    classic_wins_df = moea_detail[(moea_detail.p < 0.05) & (moea_detail.median_diff < 0)].copy()
    classic_wins_df = classic_wins_df.sort_values(["instance", "p"])
    moea_classic_wins_df = classic_wins_df[["exp_folder", "instance", "mutation", "budget",
                                            "gen_op", "base_op", "n", "wins", "losses", "ties",
                                            "median_diff", "p", "dz"]].copy()
    moea_classic_wins_df.columns = ["Exp", "Instancia", "Mut", "Budget", "Op", "Base (clasico)",
                                    "n", "W", "L", "T", "mediana_diff", "p", "dz"]

    charts = {}
    charts["moea_heatmap_wins"] = chart_moea_heatmap_wins(stats_df)
    charts["moea_wins_detail"] = chart_moea_wins_detail(stats_df)
    charts["wins_op"] = chart_wins_by_operator(stats_df)
    charts["wins_solver"] = chart_wins_by_solver(stats_df)
    charts["wins_instance"] = chart_wins_by_instance(stats_df)
    charts["wins_budget"] = chart_wins_by_budget(stats_df)
    charts["wins_mutation"] = chart_wins_by_mutation(stats_df)
    # boxplot por experimento
    box_charts = []
    for folder in folders:
        box_charts.append((folder, chart_hv_boxplot(df, folder)))

    # Guardar CSV de detalle
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    detail_view.to_csv(OUT_CSV, index=False)

    # HTML
    def img(b64, alt=""):
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;height:auto;"/>'

    # operadores en python
    op_code_html = ""
    for folder in folders:
        codes = operator_codes.get(folder, {})
        op_code_html += f"<h4>{folder}</h4>"
        for name in ["Best", "Middle", "Worst"]:
            info = codes.get(name)
            if isinstance(info, dict):
                pheno = info.get("phenotype", "")
                py = info.get("python", "")
            else:
                pheno = str(info) if info else "(no encontrado)"
                py = ""
            op_code_html += (f"<p><b>{name}</b> — fenotipo:</p>"
                            f"<pre><code>{_escape(pheno)}</code></pre>"
                            f"<p><b>{name}</b> — codigo Python generado:</p>"
                            f"<pre><code>{_escape(py)}</code></pre>")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Analisis de operadores - Experimentos 001 y 002</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222;}}
h1{{color:#1f3864;}} h2{{color:#2e5496;margin-top:28px;border-bottom:2px solid #2e5496;padding-bottom:4px;}}
h3{{color:#404040;}} h4{{color:#595959;}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;}}
.tbl th,.tbl td{{border:1px solid #bbb;padding:4px 6px;}}
.tbl th{{background:#ddebf7;font-weight:600;}}
.tbl tr:nth-child(even){{background:#f2f2f2;}}
pre{{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:10px;overflow-x:auto;font-size:12px;white-space:pre-wrap;word-break:break-word;}}
code{{font-family:Consolas,monospace;}}
.kpi{{display:inline-block;background:#eaf1fa;border:1px solid #b4c7e7;border-radius:8px;padding:10px 14px;margin:6px;font-size:13px;}}
.kpi b{{font-size:18px;color:#1f3864;}}
.note{{background:#fff8e1;border-left:4px solid #ffc107;padding:8px 12px;margin:10px 0;font-size:13px;}}
</style>
</head>
<body>
<h1>Analisis de operadores generados vs clasicos</h1>
<p>Experimentos: {", ".join(folders)} | Instancias de test: {n_instances} | Replicas independientes: {n_reps} |
Presupuestos: {", ".join(BUDGETS)} | MOEAs: {", ".join(SOLVERS)} | Mutacion: NM (sin) / WM (Swap_Mutation).</p>

<div class="note">
<b>Marco:</b> evaluacion original de GE (<code>experiment.py</code>): frente no dominado consolidado por instancia,
<code>comprobate</code>, normalizacion <code>(f-vmin)/(vmax-vmin)</code>, nadir fijo <code>[1.1]*K</code>, sin clipping.
Cada comparacion es <b>pareada por replica</b> (misma semilla/individuos iniciales para todos los operadores dentro de
una replica): se cuenta una victoria cuando el HV del operador generado supera al del clasico en esa replica.
La significancia se evalua con Wilcoxon signed-rank (two-sided, p&lt;0.05) sobre las {n_reps} diferencias pareadas,
reportando tambien el tamano del efecto <code>dz</code>. <b>No se reporta solo el HV medio.</b>
</div>

<h2>1. Operadores generados (codigo Python)</h2>
<p>Extraidos de <code>MOEAD_operators.txt</code> de cada experimento (operadores evolucionados por GE:
Best / Middle / Worst del frente de Pareto).</p>
{op_code_html}

<h2>2. Informacion de las corridas independientes</h2>
<ul>
<li>Replicas por configuracion: <b>{n_reps}</b> (Experiment 001..{n_reps:03d}).</li>
<li>Las replicas estan <b>pareadas</b>: replica r usa los mismos individuos iniciales y la misma semilla del MOEA
para todos los operadores y MOEAs, por lo que las comparaciones son diferencias pareadas (varianza mucho menor).</li>
<li>Por instancia y presupuesto: {len(ALL_OPS)} operadores x {len(SOLVERS)} MOEAs x {len(MUTATIONS)} mutaciones
= {len(ALL_OPS)*len(SOLVERS)*len(MUTATIONS)} configuraciones, cada una con {n_reps} replicas.</li>
<li>Total de comparaciones pareadas realizadas: <b>{len(stats_df)}</b>
({len(GENERATED)} operadores generados x {len(BASELINES)} clasicos x {n_instances} instancias x
{len(SOLVERS)} MOEAs x {len(MUTATIONS)} mutaciones x {len(BUDGETS)} presupuestos x {len(folders)} experimentos).</li>
</ul>

<h2>3. Victorias por operador (global)</h2>
{img(charts["wins_op"], "Victorias por operador")}
{df_to_html_table(summary_df, float_cols=("% victorias sig.",))}

<h2>4. Por MOEA (en que solver funciona el operador)</h2>
{img(charts["wins_solver"], "Victorias por MOEA")}
{df_to_html_table(solver_df, float_cols=("% victorias",))}

<h2>5. Por instancia (problema)</h2>
{img(charts["wins_instance"], "Victorias por instancia")}
{df_to_html_table(inst_df, float_cols=())}

<h2>5b. Cuando ganan los operadores clasicos (CX, PMX)</h2>
<p>Instancias y configuraciones donde el clasico supera significativamente al operador generado
(p&lt;0.05, mediana_diff&lt;0), en todos los MOEAs.</p>
{df_to_html_table(classic_inst_df, float_cols=())}
<h3>Detalle (todas las comparaciones donde gana el clasico)</h3>
<p>Archivo: <code>results/reports/operators_stats.csv</code> filtrado (mediana_diff&lt;0, p&lt;0.05).</p>
{df_to_html_table(classic_global_view, max_rows=200, float_cols=("mediana_diff","p","dz"))}

<h2>6. Por presupuesto</h2>
{img(charts["wins_budget"], "Victorias por presupuesto")}
{df_to_html_table(bud_df, float_cols=())}

<h2>7. Por mutacion</h2>
{img(charts["wins_mutation"], "Victorias por mutacion")}
{df_to_html_table(mut_df, float_cols=("% victorias",))}

<h2>8. Por experimento independiente</h2>
{df_to_html_table(exp_df, float_cols=("% victorias",))}

<h2>9. Distribucion de HV (50k, sin mutacion)</h2>
{"".join(img(b, f"Boxplot {f}") for f, b in box_charts)}

<h2>9b. MOEAD: victorias y HV frente a los originales</h2>
<p>Seccion dedicada al MOEA para el que se evolucionaron los operadores (Best/Middle/Worst).
Compara cada operador generado contra CX y PMX, por instancia, presupuesto y mutacion.</p>
{img(charts["moea_heatmap_wins"], "MOEAD victorias por instancia")}
{img(charts["moea_wins_detail"], "MOEAD victorias por presupuesto")}

<h3>9b.1 Resumen por operador y mutacion (MOEAD)</h3>
{df_to_html_table(moea_summary_df, float_cols=("% victorias",))}

<h3>9b.2 Victorias por instancia (MOEAD): generado vs clasico</h3>
{df_to_html_table(moea_inst_df, float_cols=())}

<h3>9b.3 Instancias donde el clasico gana significativamente (MOEAD)</h3>
<p>Comparaciones donde CX o PMX superan al operador generado (p&lt;0.05, mediana_diff&lt;0).</p>
{df_to_html_table(moea_classic_wins_df, max_rows=200, float_cols=("mediana_diff","p","dz"))}
<p>Archivo: <code>{MOEA_CSV}</code> ({len(moea_detail)} filas).</p>
{df_to_html_table(moea_detail_view, max_rows=300, float_cols=("mediana_diff","p","dz"))}

<h2>10. Detalle estadistico de todas las comparaciones</h2>
<p>Columnas: Exp, Instancia, MOEA, Mut, Budget, Op (generado), Base (clasico), n (replicas),
W/L/T (victorias/derrotas/empates en replicas), mediana_diff (mediana de HV_gen - HV_base),
p (Wilcoxon two-sided), dz (tamano del efecto), sig (WIN/LOSS/ns).</p>
<p>Archivo completo: <code>{OUT_CSV}</code> ({len(detail_view)} filas).</p>
{df_to_html_table(detail_view, max_rows=200, float_cols=("mediana_diff","p","dz"))}

<hr/>
<p style="font-size:11px;color:#808080;">Generado por analyze_operators.py</p>
</body>
</html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Reporte: {OUT_HTML}")
    print(f"CSV:     {OUT_CSV}")
    print(f"Filas de detalle: {len(detail_view)}")


def _escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dirs", nargs="+", default=[str(p) for p in DEFAULT_EXPS])
    args = ap.parse_args()

    dfs = []
    operator_codes = {}
    for d in args.exp_dirs:
        p = Path(d)
        if not p.exists():
            print(f"AVISO: no existe {p}, se omite.")
            continue
        df = load_hv(p)
        dfs.append(df)
        operator_codes[p.name] = load_operator_code(p)
        print(f"Cargado {p.name}: {len(df)} filas HV, {df.instance.nunique()} instancias")

    if not dfs:
        raise SystemExit("No se cargaron datos.")
    df = pd.concat(dfs, ignore_index=True)

    stats_df = run_analysis(df)
    print(f"Comparaciones pareadas: {len(stats_df)}")

    build_html(df, stats_df, args.exp_dirs, operator_codes)


if __name__ == "__main__":
    main()
