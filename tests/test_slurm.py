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
    "script", ["mohh-generate.slrm", "mohh-generate-big.slrm", "mohh-evaluate.slrm"]
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
    "script", ["mohh-generate.slrm", "mohh-generate-big.slrm", "mohh-evaluate.slrm"]
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


def test_generate_big_requests_110_cores_exclusive():
    # 110 cores solo caben en graphos1 (112); exclusive = nodo sin vecinos.
    content = _read("mohh-generate-big.slrm")
    assert re.search(r"#SBATCH\s+-c\s+110\b", content)
    assert re.search(r"#SBATCH\s+--exclusive\b", content)


def test_evaluate_requests_32_cores():
    # 31 N_EXPERIMENTS workers + 1 main process.
    assert re.search(r"#SBATCH\s+-c\s+32\b", _read("mohh-evaluate.slrm"))


def test_evaluate_runs_31_independent_experiments():
    assert "--n-experiments 31" in _read("mohh-evaluate.slrm")


@pytest.mark.parametrize(
    "script", ["mohh-generate.slrm", "mohh-generate-big.slrm", "mohh-evaluate.slrm"]
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
        "mohh-generate-big.slrm",
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


def test_generate_supports_parallel_model_jobs():
    content = _read("mohh-generate.slrm")
    # Un modelo por job: MODELS y carpeta de resultados por entorno.
    assert 'MODELS="${MODELS:-MOEAD}"' in content
    assert "--experiment-path" in content

    cli_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "mohh", "cli.py"
    )
    with open(cli_path) as f:
        cli_src = f.read()
    assert "--experiment-path" in cli_src


# --------------------------------------------------------------------------- #
# Integracion LLaMEA en el cluster (misma metodologia que la etapa GE)
# --------------------------------------------------------------------------- #

LLAMEA_SLRM = ["llamea-vllm-server.slrm", "llamea-generate.slrm",
               "llamea-download-model.slrm"]


@pytest.mark.parametrize("script", LLAMEA_SLRM + ["llamea-full-chain.sh"])
def test_llamea_bash_syntax_ok(script):
    result = subprocess.run(
        ["bash", "-n", os.path.join(SLURM_DIR, script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("script", LLAMEA_SLRM)
def test_llamea_slurm_required_directives(script):
    content = _read(script)
    for pattern in (
        r"#SBATCH\s+-p\s+(ixachi|gpu)",
        r"#SBATCH\s+-J\s+\S+",
        r"#SBATCH\s+-o\s+\S+",
        r"#SBATCH\s+-e\s+\S+",
        r"#SBATCH\s+-N\s+1",
        r"#SBATCH\s+-c\s+\d+",
        r"#SBATCH\s+--time=\S+",
    ):
        assert re.search(pattern, content), f"{script}: missing {pattern}"
    # Como los .slrm de GE: logs en el dir de trabajo, sin subcarpetas.
    for flag in ("-o", "-e"):
        match = re.search(rf"#SBATCH\s+{flag}\s+(\S+)", content)
        assert match and "/" not in match.group(1), f"{script}: {flag} usa subcarpeta"


@pytest.mark.parametrize("script", LLAMEA_SLRM)
def test_llamea_slurm_loads_cluster_env(script):
    """Modules + conda env local, igual que la etapa GE (sin permisos de root)."""
    content = _read(script)
    assert "module load python/miniforge" in content
    assert "conda shell.bash hook" in content
    assert 'conda activate "$PWD/.conda-env"' in content or \
           'conda activate "$SLURM_SUBMIT_DIR/.conda-env"' in content
    assert ".venv" not in content


def test_llamea_generate_defaults_to_1000_generations():
    content = _read("llamea-generate.slrm")
    assert '--budget "${BUDGET:-1000}"' in content
    assert "--repeat-penalty" in content
    assert "--stall-timeout" in content


def test_llamea_vllm_server_samples_gpu_and_uses_persistent_models():
    content = _read("llamea-vllm-server.slrm")
    assert "nvidia-smi --query-gpu" in content
    assert "GPU_SAMPLE_SECONDS" in content
    assert "GPU_CSV" in content
    # Pesos en la raiz persistente via el CLI, no en el cache de $HOME.
    assert "mohh_models.py" in content
    assert "ensure_model.py" not in content
    # vLLM carga la ruta local (el id de HF intentaria descargar sin internet).
    assert "mohh_models.py path" in content
    assert '--model "$MODEL_PATH"' in content


def test_llamea_download_model_job_uses_cli():
    content = _read("llamea-download-model.slrm")
    assert "mohh_models.py" in content
    assert "download" in content
    # Debe poder correr en el login (sin $SLURM_SUBMIT_DIR).
    assert "SLURM_SUBMIT_DIR:-" in content


def test_llamea_full_chain_exports_env_and_chains_after_server():
    content = _read("llamea-full-chain.sh")
    assert "--export=ALL" in content
    assert "after:" in content
    # Pre-descarga en el login (los nodos de computo no tienen internet).
    assert "mohh_models.py" in content


def test_llamea_search_script_defaults_to_1000_generations():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "llamea_search.sh")
    with open(path) as handle:
        content = handle.read()
    assert 'BUDGET="${BUDGET:-1000}"' in content
    assert "--repeat-penalty" in content


REPO_DIR = os.path.join(os.path.dirname(__file__), "..")


def _source_profile(profile, extra_env=None):
    """Sourcea el perfil y devuelve las variables MODEL/CONSTRAINT resultantes."""
    env = dict(os.environ)
    if profile:
        env["PROFILE"] = profile
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        ["bash", "-c",
         "source slurm/llamea-gpu-profile.sh; echo \"$MODEL\"; echo \"$CONSTRAINT\""],
        cwd=REPO_DIR, capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    return lines[0], lines[1]


def test_llamea_gpu_profile_defaults_to_big_model():
    # El default del cluster no puede ser un modelo de juguete: 32B AWQ.
    model, constraint = _source_profile(None)
    assert "32B" in model
    assert "A100" in constraint or "H100" in constraint


@pytest.mark.parametrize("profile,needle", [
    ("h100-fp16", "32B"),
    ("a100-fp16", "14B"),
    ("a10-quant", "14B"),
    ("coder30b-awq", "Qwen3-Coder-30B"),
    ("coder30b-fp8", "FP8"),
    ("coder8b", "Seed-Coder-8B"),
    ("smoke", "1.5B"),
])
def test_llamea_gpu_profile_variants(profile, needle):
    model, _constraint = _source_profile(profile)
    assert needle in model


def test_llamea_gpu_profile_respects_model_override():
    model, _constraint = _source_profile("h100-fp16", {"MODEL": "org/custom"})
    assert model == "org/custom"


def _profile_vllm_args(profile):
    result = subprocess.run(
        ["bash", "-c",
         "source slurm/llamea-gpu-profile.sh; echo \"$VLLM_ARGS\""],
        cwd=REPO_DIR, capture_output=True, text=True,
        env={**os.environ, "PROFILE": profile}, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize("profile", ["coder30b-awq", "coder30b-awq-a10"])
def test_community_awq_profiles_select_marlin_explicitly(profile):
    # Para el AWQ de comunidad no dependemos solo del autodetect de vLLM.
    assert "awq_marlin" in _profile_vllm_args(profile)


def test_proven_profiles_do_not_add_extra_vllm_flags():
    # big-quant/a10-quant se dejan como estaban (autodetect de vLLM).
    assert _profile_vllm_args("big-quant") == ""
    assert _profile_vllm_args("a10-quant") == ""


def test_fp8_profile_uses_fp8_kv_cache_on_h100():
    args = _profile_vllm_args("coder30b-fp8")
    assert "--kv-cache-dtype fp8" in args


def test_llamea_profile_list_prints_catalog():
    result = subprocess.run(
        ["bash", "-c", "source slurm/llamea-gpu-profile.sh"],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PROFILE": "list"},
    )
    assert result.returncode == 0, result.stderr
    assert "coder30b-fp8" in result.stdout
    assert "Seed-Coder-8B" in result.stdout


@pytest.mark.parametrize("script", ["llamea-single.slrm", "llamea-all.sh"])
def test_llamea_single_command_syntax_ok(script):
    result = subprocess.run(
        ["bash", "-n", os.path.join(SLURM_DIR, script)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_llamea_single_job_serves_and_searches_in_one_job():
    content = _read("llamea-single.slrm")
    # Un unico job: servidor vLLM + busqueda contra localhost.
    assert "vllm.entrypoints.openai.api_server" in content
    assert "mohh.llamea.run" in content
    assert "--wait-server" in content
    assert "127.0.0.1" in content
    assert "llamea-gpu-profile.sh" in content
    # Cabecera GPU de la documentacion del cluster.
    for pattern in (r"#SBATCH\s+-p\s+gpu", r"#SBATCH\s+--gres=gpu:1",
                    r"#SBATCH\s+--constraint=", r"#SBATCH\s+--time=\S+"):
        assert re.search(pattern, content), f"llamea-single.slrm: missing {pattern}"
    # Dimensionamiento: el job es GPU-bound; 16 CPUs bastan y no se piden 24.
    assert re.search(r"#SBATCH\s+-c\s+16", content)
    assert "OMP_NUM_THREADS:-4" in content
    # Parametros de servidor para una peticion secuencial con prefijo comun.
    for flag in ("--max-num-seqs 4", "--enable-prefix-caching",
                 "--disable-log-requests"):
        assert flag in content
    # Poblacion recomendada: 5 padres / 2 hijos y presupuesto derivado para
    # que GENERATIONS=1000 sean 1000 iteraciones (no 1000 evaluaciones).
    assert '--n-parents "$NPARENTS"' in content
    assert '--n-offspring "$NOFFSPRING"' in content
    assert 'NPARENTS="${NPARENTS:-5}"' in content
    assert 'NOFFSPRING="${NOFFSPRING:-2}"' in content
    assert 'BUDGET="${BUDGET:-$((GENERATIONS * NOFFSPRING + NPARENTS))}"' in content
    # Combine conductual propio, con familia objetivo y validacion.
    assert 'COMBINE="${COMBINE:-1}"' in content
    assert "--combine" in content
    assert "--target-penalty" in content
    assert "--require-target-family" in content
    # Todo el pipeline en el mismo job: comparacion pareada opcional.
    assert "compare_operators.py" in content
    assert "classic:ox" in content
    assert "COMPARE" in content
    # vLLM debe cargar la RUTA local de la raiz persistente (no el id de HF,
    # que intentaria descargar en un nodo sin internet).
    assert "mohh_models.py path" in content
    assert '--model "$MODEL_PATH"' in content
    # Argumentos del perfil (cuantizacion/KV cache) y muestreo recomendado.
    assert "${VLLM_ARGS:-}" in content
    assert "--operator-memory" in content
    assert "--top-p" in content
    assert "--sampling-repetition-penalty" in content


def test_llamea_all_is_one_command():
    path = os.path.join(SLURM_DIR, "llamea-all.sh")
    with open(path) as handle:
        content = handle.read()
    # Descarga en el login y envia el job unico.
    assert "mohh_models.py" in content
    assert "download" in content
    assert "llamea-single.slrm" in content
    assert "--export=ALL" in content
    # Puede correr en el login (modulo opcional).
    assert "command -v module" in content
    # Un solo comando: prepara el ambiente si falta y no encadena jobs.
    assert "ixachi-setup-env.sh" in content
    assert "WITH_VLLM=1" in content
    assert "SLURM_JOB_ID" in content  # no permite sbatch dentro de un job
    assert "DRY_RUN" in content
    assert "WAIT" in content


def test_llamea_short_alias_delegates_to_the_one_command():
    content = _read("llamea.sh")
    assert "llamea-all.sh" in content
    assert "exec bash" in content


@pytest.mark.parametrize("script", ["llamea-single.slrm",
                                    "llamea-vllm-server.slrm"])
def test_gpu_jobs_never_download_on_a_compute_node(script):
    """El job GPU debe fallar rapido si falta el modelo, no intentar bajar 20 GB."""
    content = _read(script)
    # Puede aparecer en el mensaje de ayuda ("en el login: ..."), pero no como
    # comando ejecutado por el job.
    assert not re.search(r"^\s*python scripts/mohh_models\.py download",
                         content, re.MULTILINE)
    assert "mohh_models.py verify" in content
    assert "llamea-all.sh" in content  # instruccion accionable


def test_llamea_full_chain_is_deprecated_by_default():
    content = _read("llamea-full-chain.sh")
    assert "ALLOW_TWO_JOBS" in content
    assert "exec bash slurm/llamea-all.sh" in content


def test_setup_env_does_not_send_users_to_a_separate_download_job():
    content = _read("ixachi-setup-env.sh")
    assert "llamea-download-model.slrm" not in content
    assert "bash slurm/llamea-all.sh" in content


@pytest.mark.parametrize("script", ["llamea.sh", "llamea-all.sh",
                                    "llamea-single.slrm"])
def test_one_command_scripts_have_valid_bash_syntax(script):
    result = subprocess.run(
        ["bash", "-n", os.path.join(SLURM_DIR, script)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_llamea_generate_passes_diversity_flags_to_the_search():
    content = _read("llamea-generate.slrm")
    for flag in ("--patience", "--parent-selection", "--niching", "--eval-repeats",
                 "--operator-memory"):
        assert flag in content, f"llamea-generate.slrm: falta {flag}"


