"""Pruebas del analisis end-to-end de evaluation_results."""
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
SPEC = importlib.util.spec_from_file_location(
    "analyze_e2e", os.path.join(ROOT, "scripts", "analyze_evaluation_end_to_end.py"))
e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e2e)


def _make(tmp_path):
    root = tmp_path / "run_metrics"
    root.mkdir()
    records = []
    for experiment in range(1, 5):
        for instance in ("I1", "I2"):
            records.append({
                "instance": instance, "label": "10k",
                "config": "MOEAD_Best_NM_MOEAD",
                "experiment": f"Experiment {experiment:03d}",
                "hypervolume": 0.60,
            })
            records.append({
                "instance": instance, "label": "10k",
                "config": "MOEAD_CX_NM_None",
                "experiment": f"Experiment {experiment:03d}",
                "hypervolume": 0.50,
            })
    json.dump({"sections": {"hypervolumes": records}},
              open(root / "evaluation_test.json", "w"))
    return str(tmp_path)


def test_load_and_compare(tmp_path):
    root = _make(tmp_path)
    frame = e2e.load(root)
    assert len(frame) == 16
    detail = e2e.compare(frame, ["MOEAD"], ["None"])
    assert len(detail) == 2  # dos instancias
    assert (detail.wins == 4).all()
    assert np.allclose(detail.median_difference, 0.1)


def test_aggregate(tmp_path):
    root = _make(tmp_path)
    detail = e2e.compare(e2e.load(root), ["MOEAD"], ["None"])
    summary = e2e.aggregate(detail, ["gen_origin", "gen_op"])
    assert len(summary) == 1
    assert summary.iloc[0]["wins"] == 8
    assert summary.iloc[0]["comparaciones"] == 2
