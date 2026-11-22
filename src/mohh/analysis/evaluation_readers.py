"""Lectores de los JSON de hypervolumes que deja `mohh-evaluate`.

Dos layout: `<root>/run_metrics` (evaluacion directa) y
`<root>/evaluation_results/run_metrics` (experimento GE). Los scripts de analisis
comparten aqui la lectura; ninguno calcula nada.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

__all__ = [
    "parse_config",
    "load_evaluation_summaries",
    "load_hv_by_instance",
]


def parse_config(config):
    """``MOEAD_Best_NM_LLAMEA`` -> (solver, operator, mutation, origin)."""
    parts = str(config).split("_")
    return (parts[0] if parts else "",
            parts[1] if len(parts) > 1 else "",
            parts[2] if len(parts) > 2 else "",
            parts[3] if len(parts) > 3 else "")


def _evaluation_paths(root, subdir, sort_paths):
    pattern = os.path.join(root, subdir, "run_metrics", "evaluation_*.json")
    paths = glob.glob(pattern)
    return sorted(paths) if sort_paths else paths


def _experiment_value(record, mode):
    """Valor de `experiment` segun el modo; None en "int" descarta el registro."""
    raw = record.get("experiment")
    if mode == "int":
        match = re.search(r"(\d+)", str(raw))
        return int(match.group(1)) if match else None
    return raw


def load_evaluation_summaries(root, *, experiment="int", sort_paths=False):
    """Lee `<root>/run_metrics/evaluation_*.json` en un DataFrame de HV.

    `experiment="int"` extrae el numero de "Experiment NN" y salta registros sin
    el; `"raw"` conserva el string del JSON. `sort_paths` ordena los archivos.
    """
    rows = []
    for path in _evaluation_paths(root, "", sort_paths):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("sections", {}).get("hypervolumes", []):
            value = _experiment_value(record, experiment)
            if experiment == "int" and value is None:
                continue
            solver, operator, mutation, origin = parse_config(record["config"])
            rows.append({
                "instance": record.get("instance"),
                "budget": record.get("label"),
                "solver": solver,
                "operator": operator,
                "mutation": mutation,
                "origin": origin,
                "experiment": value,
                "hv": float(record.get("hypervolume", np.nan)),
            })
    return pd.DataFrame(rows)


def load_hv_by_instance(root, *, experiment=None, exp_folder=False,
                        instance_from_file=False, sort_paths=False):
    """Lee `<root>/evaluation_results/run_metrics/evaluation_*.json`.

    `experiment`: None (sin columna), "raw" (string del JSON) o "int" (numero).
    `instance_from_file` toma la instancia del JSON en vez del registro.
    """
    rows = []
    for path in _evaluation_paths(root, "evaluation_results", sort_paths):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("sections", {}).get("hypervolumes", []):
            solver, operator, mutation, origin = parse_config(record["config"])
            row = {}
            if exp_folder:
                row["exp_folder"] = os.path.basename(os.path.normpath(root))
            row.update({
                "instance": (data.get("instance") if instance_from_file
                             else record["instance"]),
                "budget": record["label"],
                "solver": solver,
                "operator": operator,
                "mutation": mutation,
                "origin": origin,
            })
            if experiment is not None:
                row["experiment"] = _experiment_value(record, experiment)
            row["hv"] = float(record["hypervolume"])
            rows.append(row)
    return pd.DataFrame(rows)
