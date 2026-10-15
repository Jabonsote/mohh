"""Tests for SLURM-aware parallelism (mohh.parallel) and sbatch scripts."""

import os
import re
import subprocess

import pytest

from mohh.parallel import available_cores, run_parallel

SLURM_DIR = os.path.join(os.path.dirname(__file__), "..", "slurm")

ENV_VARS = ("MOHH_WORKERS", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")


def _clear_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_mohh_workers_takes_precedence(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MOHH_WORKERS", "11")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "50")
    assert available_cores() == 11


def test_slurm_cpus_per_task_used(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "50")
    assert available_cores() == 50


def test_slurm_cpus_on_node_fallback(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "12")
    assert available_cores() == 12


def test_slurm_compact_format_parsed(monkeypatch):
    # SLURM sometimes reports counts like "50(x2)".
    _clear_env(monkeypatch)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "50(x2)")
    assert available_cores() == 50


@pytest.mark.parametrize("bad", ["0", "-3", "abc", ""])
def test_invalid_mohh_workers_ignored(monkeypatch, bad):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MOHH_WORKERS", bad)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "50")
    assert available_cores() == 50


def test_fallback_without_env(monkeypatch):
    _clear_env(monkeypatch)
    assert available_cores() >= 1


def square(x):
    return x * x


def test_run_parallel_empty():
    assert run_parallel(square, []) == []


def test_run_parallel_preserves_order():
    assert run_parallel(square, [3, 1, 2]) == [9, 1, 4]


def test_run_parallel_on_complete_called_per_task():
    seen = []
    run_parallel(square, [1, 2, 3], on_complete=seen.append)
    assert sorted(seen) == [0, 1, 2]


def _read(name):
    with open(os.path.join(SLURM_DIR, name)) as f:
        return f.read()


@pytest.mark.parametrize(
    "script", ["mohh-generate.slrm", "mohh-evaluate.slrm"]
)
def test_slurm_required_directives(script):
    content = _read(script)
    for pattern in (
        r"#SBATCH\s+-p\s+ixachi",
        r"#SBATCH\s+-J\s+\S+",
        r"#SBATCH\s+-o\s+\S+",
        r"#SBATCH\s+-e\s+\S+",
        r"#SBATCH\s+-N\s+1",
        r"#SBATCH\s+-c\s+\d+",
        r"#SBATCH\s+--time=\S+",
    ):
        assert re.search(pattern, content), f"{script}: missing {pattern}"


@pytest.mark.parametrize(
    "script", ["mohh-generate.slrm", "mohh-evaluate.slrm"]
)
def test_slurm_runtime_conventions(script):
    content = _read(script)
    assert "cd $SLURM_SUBMIT_DIR" in content or 'cd "$SLURM_SUBMIT_DIR"' in content
    assert "MOHH_WORKERS" in content
    assert "OMP_NUM_THREADS=1" in content
    assert "module load python/miniforge" in content
    # Conda env del cluster (Modules), no venv local.
    assert "conda activate" in content
    assert "conda shell.bash hook" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert ".venv" not in content


def test_setup_creates_conda_env():
    content = _read("ixachi-setup-env.sh")
    assert "module load python/miniforge" in content
    assert "conda create" in content
    assert "python=3.12" in content
    assert "pip install -e ." in content
    # SLURM abre -o/-e al arrancar: logs/ y results/ deben existir antes.
    assert "mkdir -p logs results" in content


def test_generate_requests_50_cores():
    assert re.search(r"#SBATCH\s+-c\s+50\b", _read("mohh-generate.slrm"))


def test_evaluate_requests_12_cores():
    # 11 N_EXPERIMENTS workers + 1 main process; 50 would idle ~39 cores.
    assert re.search(r"#SBATCH\s+-c\s+12\b", _read("mohh-evaluate.slrm"))


@pytest.mark.parametrize(
    "script", ["mohh-generate.slrm", "mohh-evaluate.slrm"]
)
def test_slurm_outputs_in_submit_dir(script):
    # Como holamundo.out en la metodologia: sin subcarpetas, para que el
    # job no muera si el directorio no existe al arrancar.
    for flag in ("-o", "-e"):
        match = re.search(rf"#SBATCH\s+{flag}\s+(\S+)", _read(script))
        assert match and "/" not in match.group(1), f"{script}: {flag} usa subcarpeta"


@pytest.mark.parametrize(
    "script",
    [
        "ixachi-setup-env.sh",
        "mohh-full-chain.sh",
        "mohh-generate.slrm",
        "mohh-evaluate.slrm",
    ],
)
def test_bash_syntax_ok(script):
    result = subprocess.run(
        ["bash", "-n", os.path.join(SLURM_DIR, script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
