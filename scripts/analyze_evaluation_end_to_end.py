#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analisis end-to-end de un `evaluation_results` (completo o parcial).

Lee los ``run_metrics/evaluation_*.json`` ya escritos, compara cada operador
generado contra una linea base (CX/PMX por defecto) y produce:

- resumen por operador, solver, presupuesto y mutacion;
- detalle pareado por experimento (Wilcoxon + Holm);
- medianas de HV por configuracion;
- ``report.md`` y CSV.

Funciona con resultados parciales: solo usa las instancias ya completadas.
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


def compare(frame, gen_origins, base_origins):
    generated = frame[frame.origin.isin(gen_origins)]
    baseline = frame[frame.origin.isin(base_origins)]
    keys = ["instance", "budget", "solver", "mutation"]
    rows = []
    for key, gen_group in generated.groupby(keys):
        instance, budget, solver, mutation = key
        base_group = baseline[(baseline.instance == instance) &
                              (baseline.budget == budget) &
                              (baseline.solver == solver) &
                              (baseline.mutation == mutation)]
        for gen_op, gen_rows in gen_group.groupby("operator"):
            for base_op, base_rows in base_group.groupby("operator"):
                left = gen_rows.set_index("experiment")["hv"]
                right = base_rows.set_index("experiment")["hv"]
                common = left.index.intersection(right.index)
                if len(common) < 3:
                    continue
                left = left.loc[common]
                right = right.loc[common]
                summary = paired_summary(left.tolist(), right.tolist())
                diffs = (left - right).to_numpy()
                summary.update({
                    "instance": instance,
                    "budget": budget,
                    "solver": solver,
                    "mutation": mutation,
                    "gen_origin": gen_group.origin.iloc[0],
                    "gen_op": gen_op,
                    "base_op": base_op,
                    "median_difference": float(np.median(diffs)),
                    "median_hv_gen": float(left.median()),
                    "median_hv_base": float(right.median()),
                })
                rows.append(summary)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Holm dentro de cada familia (solver, presupuesto, mutacion, operador).
    adjusted = []
    for _, group in result.groupby(["solver", "budget", "mutation",
                                    "gen_origin", "gen_op"]):
        records = group.to_dict("records")
        add_holm(records)
        adjusted.extend(records)
    return pd.DataFrame(adjusted)


def aggregate(detail, keys):
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for key, group in detail.groupby(keys):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update({
            "comparaciones": len(group),
            "wins": int(group.wins.sum()),
            "losses": int(group.losses.sum()),
            "ties": int(group.ties.sum()),
            "sig_wins": int((group.significant_holm &
                             (group.median_difference > 0)).sum()),
            "sig_losses": int((group.significant_holm &
                               (group.median_difference < 0)).sum()),
            "median_hv_gen": float(group.median_hv_gen.median()),
            "median_hv_base": float(group.median_hv_base.median()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame, float_cols=()):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in float_cols and pd.notna(value):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Carpeta evaluation_results.")
    parser.add_argument("--out", default=None,
                        help="Carpeta de salida (default <root>/analysis).")
    parser.add_argument("--gen-origin", default=None,
                        help="Origen generado (LLAMEA o MOEAD). Por defecto, todos.")
    parser.add_argument("--baseline-origin", default="None",
                        help="Origen de la linea base ('None' = CX/PMX).")
    args = parser.parse_args()

    frame = load(args.root)
    if frame.empty:
        raise SystemExit("no hay run_metrics/evaluation_*.json")
    out = args.out or os.path.join(args.root, "analysis")
    os.makedirs(out, exist_ok=True)

    origins = sorted(frame.origin.unique())
    if args.gen_origin:
        gen_origins = [args.gen_origin]
    else:
        gen_origins = [origin for origin in origins if origin != args.baseline_origin]
    base_origins = [args.baseline_origin]
    if not frame.origin.isin(base_origins).any():
        raise SystemExit(f"no hay linea base con origen {base_origins}")

    detail = compare(frame, gen_origins, base_origins)
    if detail.empty:
        raise SystemExit("sin comparaciones pareadas suficientes")

    detail.to_csv(os.path.join(out, "detail.csv"), index=False)
    by_operator = aggregate(detail, ["gen_origin", "gen_op"])
    by_solver = aggregate(detail, ["solver", "gen_origin", "gen_op"])
    by_budget = aggregate(detail, ["budget", "gen_origin", "gen_op"])
    by_mutation = aggregate(detail, ["mutation", "gen_origin", "gen_op"])
    for name, table in (("by_operator", by_operator),
                        ("by_solver", by_solver),
                        ("by_budget", by_budget),
                        ("by_mutation", by_mutation)):
        table.to_csv(os.path.join(out, f"{name}.csv"), index=False)

    instances = sorted(frame.instance.unique())
    lines = [
        "# Analisis end-to-end de evaluation_results",
        "",
        f"- Instancias con resultados: **{len(instances)}**",
        f"- Origen generado: `{gen_origins}`",
        f"- Linea base: `{base_origins}`",
        f"- Comparaciones pareadas: **{len(detail)}**",
        "",
        "## Resumen por operador",
        "",
        md_table(by_operator),
        "",
        "## Por solver",
        "",
        md_table(by_solver),
        "",
        "## Por presupuesto",
        "",
        md_table(by_budget),
        "",
        "## Por mutacion",
        "",
        md_table(by_mutation),
        "",
        "## Detalle",
        "",
        f"Archivo completo: `{os.path.join(out, 'detail.csv')}`",
    ]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"Reporte: {os.path.join(out, 'report.md')}")
    print(f"Detalle: {os.path.join(out, 'detail.csv')}")
    print(f"Instancias: {len(instances)} | comparaciones: {len(detail)}")
    print(by_operator.to_string(index=False))


if __name__ == "__main__":
    main()
