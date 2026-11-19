"""Escritura modular del arbol de resultados de la evaluacion GE.

Cada funcion escribe **una sola pieza** del arbol y no depende del orden en
que se llamen (salvo `write_figures`, que necesita los CSV ya escritos):

    evaluation_results/
    ├── initial_solutions/{permutation,real}/<inst>/experiment_001..N.txt
    │                                                + seeds.txt
    ├── <solver>/<budget>/<inst>.csv
    ├── <solver>/<budget>/fronts/<inst>/*.pof  (+ _norm, best_front_consolidated*)
    ├── <solver>/<budget>/Figures/{NoMutation,WithMutation}/<inst>_<NM|WM>.png
    └── run_metrics/evaluation_<inst>.json

Referencia: `mohh/evaluation/experiment.py`, `problem/instance.py` y
`make_plots.py` de AdrianMP1/MOHH-Specialized-Operators.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np

from mohh.llamea.fitness import clip_to_reference_box, normalize_front

PERMUTATION_FMT = "{:03d}"
REAL_FMT = "{:.04f}"


# --------------------------------------------------------------------------- #
# Poblaciones iniciales (random-key -> permutacion, como `problem/instance.py`)
# --------------------------------------------------------------------------- #
def random_key_population(n_variables, population_size, rng):
    """Devuelve ``(real, permutation)`` para una repeticion.

    El original genera claves reales uniformes con ``np.random.random`` y
    decodifica cada clave a una permutacion ordenando por el valor (RK/decode).
    Las dos carpetas `real/` y `permutation/` son por tanto la misma population
    en dos representaciones.
    """
    real = np.asarray(rng.random((population_size, n_variables)), dtype=float)
    permutation = np.argsort(real, axis=1, kind="stable").astype(int)
    return real, permutation


def _write_population(folder, rep, population, fmt):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"experiment_{rep:03d}.txt"), "w",
              encoding="utf-8") as handle:
        for individual in population:
            handle.write("[" + ", ".join(fmt.format(value)
                                         for value in individual) + "]\n")


def write_initial_solutions(out_dir, instance_name, real_by_rep,
                            permutation_by_rep, seeds):
    """Escribe `initial_solutions/{permutation,real}/<inst>/` + `seeds.txt`."""
    for kind, by_rep, fmt in (("permutation", permutation_by_rep, PERMUTATION_FMT),
                              ("real", real_by_rep, REAL_FMT)):
        folder = os.path.join(out_dir, "initial_solutions", kind, instance_name)
        for rep, population in enumerate(by_rep, start=1):
            _write_population(folder, rep, population, fmt)
        with open(os.path.join(folder, "seeds.txt"), "w", encoding="utf-8") as handle:
            for seed in seeds:
                handle.write(f"{seed}\n")


# --------------------------------------------------------------------------- #
# Frentes y nombres de columna (formato del original)
# --------------------------------------------------------------------------- #
def write_front(front, file_path):
    """Frente `.pof`: cabecera ``# N M`` y puntos en notacion cientifica."""
    array = np.asarray(front, dtype=float)
    if array.ndim != 2:
        return
    n_rows, n_cols = array.shape
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(f"# {n_rows} {n_cols}\n")
        for row in array:
            handle.write(" ".join(f"{value:1.6e}" for value in row) + "\n")


def ge_column_name(solver, mutation, name):
    """``<solver>_<operador>_<NM|WM>_<solver>``, como en `experiment.py`."""
    return f"{solver}_{name}_{'WM' if mutation else 'NM'}_{solver}"


# --------------------------------------------------------------------------- #
# Figuras (port de `make_plots.py` del original)
# --------------------------------------------------------------------------- #
def write_figures(results_path, solvers, budgets):
    """Boxplots por (solver, presupuesto, instancia) en `Figures/`.

    Lee los CSV que ya escribio `write_ge_layout` y guarda, dentro de cada
    `<solver>/<budget>/`, las figuras `Figures/NoMutation/<inst>_NM.png` y
    `Figures/WithMutation/<inst>_WM.png`. Los limites del eje Y se calculan
    por instancia sobre todos los solvers y presupuestos (y se ensanchan 5%),
    igual que el original.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    limits = {}
    for solver in solvers:
        for budget in budgets:
            folder = os.path.join(results_path, solver, budget)
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                if not filename.endswith(".csv"):
                    continue
                frame = pd.read_csv(os.path.join(folder, filename))
                if frame.shape[1] < 2:
                    continue
                values = frame.iloc[:, 1:].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    continue
                instance = filename[:-4]
                low, high = limits.get(instance, (np.inf, -np.inf))
                limits[instance] = (min(low, float(values.min())),
                                    max(high, float(values.max())))

    for instance, (low, high) in list(limits.items()):
        span = high - low
        limits[instance] = (low - 0.05 * span, high + 0.05 * span)

    colors = ["cadetblue", "limegreen", "orange", "red"]
    for solver in solvers:
        for budget in budgets:
            folder = os.path.join(results_path, solver, budget)
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                if not filename.endswith(".csv"):
                    continue
                instance = filename[:-4]
                if instance not in limits:
                    continue
                frame = pd.read_csv(os.path.join(folder, filename))
                columns = list(frame.columns)[1:]
                for mutation, label, subfolder in (
                        (False, "NM", "NoMutation"),
                        (True, "WM", "WithMutation")):
                    selected = [col for col in columns
                                if f"_{label}_" in col]
                    if not selected:
                        continue
                    _boxplot(frame[selected], instance, solver, budget,
                             limits[instance], label, colors,
                             os.path.join(folder, "Figures", subfolder,
                                          f"{instance}_{label}.png"), plt)


def _boxplot(frame, instance, solver, budget, ylim, label, colors, path, plt):
    """Un panel de boxplots agrupados por prefijo de solver."""
    groups = {}
    for column in frame.columns:
        groups.setdefault(str(column).split("_")[0], []).append(column)

    figure = plt.figure(figsize=(10, 6))
    positions, labels, cursor, last = [], [], 1, None
    for prefix, columns in groups.items():
        data = [frame[column].dropna().to_numpy(dtype=float)
                for column in columns]
        last = plt.boxplot(
            data, positions=list(range(cursor, cursor + len(data))),
            widths=0.6, patch_artist=True,
            medianprops=dict(color="k", linewidth=2))
        positions.extend(range(cursor, cursor + len(data)))
        cursor += len(data) + 1
        labels.extend(" ".join(str(column).split("_")[:-1])
                      for column in columns)
    if last is not None:
        for patch, color in zip(last["boxes"], colors * len(groups)):
            patch.set_facecolor(color)
    plt.xticks(positions, labels, rotation=90, ha="right")
    plt.title(f"{instance} - {budget} - {label} - ({solver}) Box plots")
    plt.ylim(ylim)
    plt.ylabel("HV")
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=400)
    plt.close(figure)


# --------------------------------------------------------------------------- #
# Layout completo: CSV + frentes + metricas por instancia
# --------------------------------------------------------------------------- #
def _protocol_label(solver, mutation):
    return f"{solver}+mut" if mutation else solver


def _median_front(fronts, values, name, index, protocols, repeats, label=None):
    """Frente de la repeticion mediana (por HV) de un operador/instancia."""
    suffix = (label,) if label is not None else ()
    scores = []
    for rep in range(repeats):
        cells = []
        for solver, mutation in protocols:
            row = values.get((_protocol_label(solver, mutation), name, rep,
                              *suffix))
            if row is not None and index < len(row) and row[index] is not None:
                cells.append(row[index])
        scores.append(float(np.mean(cells)) if cells else None)
    valid = [(rep, value) for rep, value in enumerate(scores)
             if value is not None]
    if not valid:
        return None
    valid.sort(key=lambda item: item[1])
    median = valid[len(valid) // 2][0]
    for solver, mutation in protocols:
        front = fronts.get((_protocol_label(solver, mutation), name, median,
                            *suffix, index))
        if front is not None and len(front):
            return np.asarray(front, dtype=float)
    return None


def write_evaluation_layout(out_dir, instance_paths, budgets, operators,
                            protocols, values, frames, fronts, repeats,
                            failures=None, labels=None):
    """Escribe `<solver>/<budget>/` completo para todos los solvers.

    Produce, por solver y presupuesto: el CSV por instancia, los frentes `.pof`
    (mediano y consolidado, crudos y normalizados) y las metricas por
    instancia. No escribe las figuras: `write_figures` va despues, porque lee
    los CSV que esta funcion acaba de crear.
    """
    failures = failures or {}
    labels = labels or list(budgets)
    solvers = []
    for solver, _mutation in protocols:
        if solver not in solvers:
            solvers.append(solver)

    columns = {solver: [] for solver in solvers}
    for solver, mutation in protocols:
        for name, _ns, _kind, _src in operators:
            columns[solver].append((
                solver, mutation, _protocol_label(solver, mutation), name,
                ge_column_name(solver, mutation, name)))

    for solver in solvers:
      for label in labels:
        for index, path in enumerate(instance_paths):
            instance = os.path.splitext(os.path.basename(path))[0]
            folder = os.path.join(out_dir, solver, label)
            os.makedirs(folder, exist_ok=True)

            with open(os.path.join(folder, f"{instance}.csv"), "w",
                      newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["Experiment"]
                                + [col for _s, _m, _p, _n, col in columns[solver]])
                for rep in range(repeats):
                    row = [rep + 1]
                    for _s, _m, protocol, name, _col in columns[solver]:
                        values_row = values.get((protocol, name, rep, label))
                        if values_row is not None and index < len(values_row) \
                                and values_row[index] is not None:
                            row.append(f"{values_row[index]:.10g}")
                        else:
                            row.append("")
                    writer.writerow(row)

            key_frame = ((label, index) if isinstance(frames, dict)
                         else (index if index < len(frames) else None))
            frame = frames.get(key_frame) if isinstance(frames, dict) else key_frame
            bounds = None
            if frame is not None and len(frame):
                vmin, vmax = frame.min(axis=0), frame.max(axis=0)
                if not np.any(vmax == vmin):
                    bounds = (vmin, vmax)

            front_dir = os.path.join(folder, "fronts", instance)
            for solver_p, mutation, _protocol, name, col in columns[solver]:
                selected = _median_front(fronts, values, name, index,
                                        [(solver_p, mutation)], repeats,
                                        label=label)
                if selected is None:
                    continue
                write_front(selected, os.path.join(front_dir, f"{col}.pof"))
                if bounds is not None:
                    write_front(clip_to_reference_box(
                        normalize_front(selected, bounds)),
                        os.path.join(front_dir, f"{col}_norm.pof"))
            if frame is not None and len(frame):
                write_front(frame, os.path.join(front_dir,
                                                "best_front_consolidated.pof"))
                if bounds is not None:
                    write_front(clip_to_reference_box(
                        normalize_front(frame, bounds)),
                        os.path.join(front_dir,
                                     "best_front_consolidated_norm.pof"))

            metrics = {
                "phase": "evaluation",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "instance": instance,
                "solver": solver,
                "budget": label,
                "counters": {
                    "evaluations": len(columns[solver]) * repeats,
                    "failures": sum(1 for key in failures
                                    if key[0] in (_protocol_label(solver, False),
                                                  _protocol_label(solver, True))),
                },
            }
            metrics_dir = os.path.join(out_dir, "run_metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir,
                                   f"evaluation_{instance}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=1)
