"""Los readers unificados devuelven lo mismo que las cargas originales."""
import glob
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mohh.analysis.evaluation_readers import (
    load_evaluation_summaries,
    load_hv_by_instance,
    parse_config,
)

ROOT = os.path.dirname(os.path.dirname(__file__))


# -- copias literales de las implementaciones previas ------------------------
def _original_summaries(root):
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


def _original_partial(root):
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


def _original_hv(root, with_experiment=True):
    rows = []
    for path in glob.glob(os.path.join(root, "evaluation_results",
                                       "run_metrics", "evaluation_*.json")):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for record in data.get("sections", {}).get("hypervolumes", []):
            parts = str(record["config"]).split("_")
            row = {
                "instance": record["instance"],
                "budget": record["label"],
                "solver": parts[0],
                "operator": parts[1],
                "mutation": parts[2],
                "origin": parts[3] if len(parts) > 3 else "",
            }
            if with_experiment:
                row["experiment"] = record["experiment"]
            row["hv"] = float(record["hypervolume"])
            rows.append(row)
    return pd.DataFrame(rows)


def _original_hv_operators(exp_dir):
    exp_dir = Path(exp_dir)
    rows = []
    rm = exp_dir / "evaluation_results" / "run_metrics"
    for f in sorted(rm.glob("evaluation_*.json")):
        d = json.load(open(f))
        for h in d["sections"]["hypervolumes"]:
            solver, operator, mutation, origin = parse_config(h["config"])
            rows.append({
                "exp_folder": exp_dir.name,
                "instance": d["instance"],
                "budget": h["label"],
                "solver": solver,
                "operator": operator,
                "mutation": mutation,
                "origin": origin,
                "replicate": int(h["experiment"].split()[-1]),
                "hv": float(h["hypervolume"]),
            })
    return pd.DataFrame(rows)


# -- datos temporales --------------------------------------------------------
def _write(path, records, instance="I1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"instance": instance,
                   "sections": {"hypervolumes": records}}, handle)


def _summary_records():
    records = []
    for experiment in (1, 2):
        for config, hv in (("MOEAD_Best_NM_MOEAD", 0.6),
                           ("MOEAD_CX_NM_None", 0.5)):
            records.append({
                "instance": "I1", "label": "10k", "config": config,
                "experiment": f"Experiment {experiment:03d}",
                "hypervolume": hv,
            })
    records.append({
        "instance": "I1", "label": "10k", "config": "MOEAD_Best_NM_MOEAD",
        "experiment": "sin numero", "hypervolume": 0.4,
    })
    records.append({
        "instance": "I1", "label": "10k", "config": "MOEAD_Best_NM_MOEAD",
        "hypervolume": 0.3,
    })
    return records


def _hv_records():
    records = []
    for experiment in (1, 2):
        for config, hv in (("MOEAD_Best_NM_MOEAD", 0.6),
                           ("MOEAD_CX_NM_None", 0.5),
                           ("MOEAD_PMX_NM_None", 0.4)):
            records.append({
                "instance": "I1", "label": "50k", "config": config,
                "experiment": f"Experiment {experiment:03d}",
                "hypervolume": hv,
            })
    return records


def _script(name):
    path = os.path.join(ROOT, "scripts", f"{name}.py")
    if not os.path.isfile(path):
        path = os.path.join(ROOT, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- pruebas -----------------------------------------------------------------
def test_evaluation_summaries(tmp_path):
    _write(tmp_path / "run_metrics" / "evaluation_I1.json", _summary_records())
    pd.testing.assert_frame_equal(
        load_evaluation_summaries(tmp_path), _original_summaries(str(tmp_path)))
    pd.testing.assert_frame_equal(
        load_evaluation_summaries(tmp_path, experiment="raw", sort_paths=True),
        _original_partial(str(tmp_path)))
    pd.testing.assert_frame_equal(
        _script("analyze_evaluation_end_to_end").load(str(tmp_path)),
        _original_summaries(str(tmp_path)))
    pd.testing.assert_frame_equal(
        _script("compare_ge_llamea_stats").load(str(tmp_path)),
        _original_summaries(str(tmp_path)))
    pd.testing.assert_frame_equal(
        _script("partial_results").load_hypervolumes(str(tmp_path)),
        _original_partial(str(tmp_path)))


def test_hv_by_instance(tmp_path):
    exp = tmp_path / "Experiment_001"
    _write(exp / "evaluation_results" / "run_metrics" / "evaluation_I1.json",
           _hv_records())
    pd.testing.assert_frame_equal(
        load_hv_by_instance(exp, experiment="raw"), _original_hv(str(exp)))
    pd.testing.assert_frame_equal(
        load_hv_by_instance(exp),
        _original_hv(str(exp), with_experiment=False))
    pd.testing.assert_frame_equal(
        load_hv_by_instance(exp, experiment="int", exp_folder=True,
                            instance_from_file=True, sort_paths=True)
        .rename(columns={"experiment": "replicate"}),
        _original_hv_operators(exp))
    pd.testing.assert_frame_equal(
        _script("critical_difference").load_hv(str(exp)), _original_hv(str(exp)))
    pd.testing.assert_frame_equal(
        _script("wins_plot").load_hv(str(exp)),
        _original_hv(str(exp), with_experiment=False))
    operators = importlib.import_module("mohh.postanalysis.analyze_operators")
    pd.testing.assert_frame_equal(operators.load_hv(exp),
                                  _original_hv_operators(exp))
    assert operators.parse_config("MOEAD_Best_NM_MOEAD") == (
        "MOEAD", "Best", "NM", "MOEAD")
