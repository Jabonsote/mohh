"""Pruebas del monitor de corridas LLaMEA (scripts/run_state.py)."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_run(folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "progress.txt"), "w", encoding="utf-8") as f:
        f.write("iter 1 best=1.500000 best_hv=0.5 candidate=1.500000 hv=0.5 "
                "valid=1 error=- parent=current-best strategy=refine wall_s=10.0\n")
        f.write("iter 2 best=1.500000 best_hv=0.5 candidate=worst valid=0 "
                "error=respuesta_truncada parent=current-best strategy=create\n")
    with open(os.path.join(folder, "llm_requests.jsonl"), "w", encoding="utf-8") as f:
        for index, finish in enumerate(("stop", "length")):
            f.write(json.dumps({
                "request": index + 1,
                "prompt": "REFERENCE CODE\n```python\ndef crossover(x, y):\n    pass\n```",
                "response": {"finish_reason": finish, "raw_content": "x"},
            }) + "\n")


def test_run_state_reports_progress(tmp_path):
    folder = str(tmp_path / "run")
    _make_run(folder)
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_state.py"), folder],
        capture_output=True, text=True, check=True)
    out = result.stdout
    assert "iteraciones: 2" in out
    assert "valid=1" in out
    assert "truncadas: 1" in out
    assert "respuesta_truncada" in out


def test_run_state_without_results(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "run_state.py"),
         str(tmp_path / "nada")],
        capture_output=True, text=True)
    assert result.returncode == 1
