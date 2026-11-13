#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumen parcial de una evaluacion GE/LLaMEA en curso.

Lee los ``run_metrics/evaluation_*.json`` que ya existan y compara el
operador generado (LLaMEA o GE) contra los clasicos PMX/CX. Cada
``evaluation_<instancia>.json`` se escribe al terminar esa instancia, por lo
que el script puede ejecutarse repetidamente mientras el job array sigue.

Uso:
    python scripts/partial_results.py results/searches/<RUN>/evaluation_results
    python scripts/partial_results.py <evaluation_results> --gen-origin LLAMEA
    python scripts/partial_results.py <evaluation_results> --csv parcial.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def parse_config(config):
    """``MOEAD_Best_NM_LLAMEA`` -> (MOEAD, Best, NM, LLAMEA)."""
    parts = str(config).split("_")
    solver = parts[0] if parts else ""
    operator = parts[1] if len(parts) > 1 else ""
    mutation = parts[2] if len(parts) > 2 else ""
    origin = parts[3] if len(parts) > 3 else ""
    return solver, operator, mutation, origin


def load_hypervolumes(root):
    rows = []
    pattern = os.path.join(root, "run_metrics", "evaluation_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("sections", {}).get("hypervolumes", []):
            solver, operator, mutation, origin = parse_config(record["config"])
            rows.append({
                "instance": record.get("instance"),
                "budget": record.get("label"),
                "solver": solver,
                "operator": operator,
                "mutation": mutation,
                "origin": origin,
                "experiment": record.get("experiment"),
                "hv": float(record.get("hypervolume", np.nan)),
            })
    return pd.DataFrame(rows)


def paired_stats(frame, gen_origin, gen_ops, bases):
    records = []
    if frame.empty:
        return pd.DataFrame(records)
    frame = frame[frame["origin"].isin([gen_origin, "None"])]
    generated = frame[(frame["origin"] == gen_origin) &
                      (frame["operator"].isin(gen_ops))]
    classics = frame[(frame["origin"] == "None") &
                     (frame["operator"].isin(bases))]
    keys = ["instance", "budget", "solver", "mutation"]
    for key, gen_group in generated.groupby(keys):
        instance, budget, solver, mutation = key
        for gen_op, gen_rows in gen_group.groupby("operator"):
            for base_op, base_rows in classics[
                    (classics.instance == instance) &
                    (classics.budget == budget) &
                    (classics.solver == solver) &
                    (classics.mutation == mutation)].groupby("operator"):
                left = gen_rows.set_index("experiment")["hv"]
                right = base_rows.set_index("experiment")["hv"]
                common = left.index.intersection(right.index)
                if len(common) < 5:
                    continue
                diff = (left.loc[common] - right.loc[common]).to_numpy()
                wins = int((diff > 0).sum())
                losses = int((diff < 0).sum())
                ties = int((diff == 0).sum())
                try:
                    p_value = (1.0 if np.all(diff == 0)
                               else stats.wilcoxon(diff).pvalue)
                except Exception:
                    p_value = float("nan")
                sd = diff.std(ddof=1)
                records.append({
                    "instance": instance,
                    "budget": budget,
                    "solver": solver,
                    "mutation": mutation,
                    "generated": gen_op,
                    "baseline": base_op,
                    "n": len(common),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "median_diff": float(np.median(diff)),
                    "mean_diff": float(np.mean(diff)),
                    "p": p_value,
                    "dz": float(diff.mean() / sd) if sd else float("nan"),
                    "significant_win": bool(p_value < 0.05 and
                                            np.median(diff) > 0),
                })
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Resumen parcial de evaluation_results.")
    parser.add_argument("root", help="Carpeta evaluation_results.")
    parser.add_argument("--gen-origin", default="LLAMEA",
                        help="Origen del operador generado (LLAMEA o MOEAD).")
    parser.add_argument("--gen-ops", default="Best,Middle,Worst",
                        help="Operadores generados (coma).")
    parser.add_argument("--baselines", default="PMX,CX",
                        help="Clasicos (coma).")
    parser.add_argument("--csv", default=None,
                        help="Escribe el detalle por comparacion en CSV.")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"no existe: {args.root}")

    frame = load_hypervolumes(args.root)
    if frame.empty:
        print("No hay run_metrics/evaluation_*.json todavia.")
        return

    instances = sorted(frame["instance"].unique())
    print(f"Instancias con resultados: {len(instances)}")
    print("  " + ", ".join(instances))
    print()

    gen_ops = tuple(item.strip() for item in args.gen_ops.split(",") if item.strip())
    bases = tuple(item.strip() for item in args.baselines.split(",") if item.strip())
    stats_frame = paired_stats(frame, args.gen_origin, gen_ops, bases)
    if stats_frame.empty:
        print("Sin comparaciones pareadas suficientes.")
        return

    # Resumen global por operador generado
    summary = []
    for gen_op in gen_ops:
        subset = stats_frame[stats_frame["generated"] == gen_op]
        if subset.empty:
            continue
        summary.append({
            "operador": gen_op,
            "comparaciones": len(subset),
            "victorias": int(subset["wins"].sum()),
            "derrotas": int(subset["losses"].sum()),
            "empates": int(subset["ties"].sum()),
            "victorias_sig": int(subset["significant_win"].sum()),
            "p<0.05": int((subset["p"] < 0.05).sum()),
        })
    print("=== Resumen por operador ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print()

    # Resumen por solver
    print("=== Victorias significativas por solver ===")
    pivot = (stats_frame[stats_frame["significant_win"]]
             .groupby(["generated", "solver"]).size().unstack(fill_value=0))
    print(pivot.to_string())
    print()

    # Resumen por instancia
    print("=== Victorias significativas por instancia ===")
    pivot = (stats_frame[stats_frame["significant_win"]]
             .groupby(["instance", "generated"]).size().unstack(fill_value=0))
    print(pivot.to_string())
    print()

    # Resumen por presupuesto y mutacion
    print("=== Victorias significativas por presupuesto ===")
    pivot = (stats_frame[stats_frame["significant_win"]]
             .groupby(["budget", "generated"]).size().unstack(fill_value=0))
    print(pivot.to_string())
    print()

    print("=== Victorias significativas por mutacion ===")
    pivot = (stats_frame[stats_frame["significant_win"]]
             .groupby(["mutation", "generated"]).size().unstack(fill_value=0))
    print(pivot.to_string())
    print()

    if args.csv:
        stats_frame.to_csv(args.csv, index=False)
        print(f"Detalle escrito en {args.csv}")


if __name__ == "__main__":
    main()
