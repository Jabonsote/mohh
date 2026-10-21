# Specialized Operators via a Multi-Objective Hyper-Heuristic

Source code for the paper:<br>
**Automatic Design of Specialized Variation Operators for the Multi-Objective Quadratic Assignment Problem** <br>
[GECCO, 2025]

> **[Authors]**<br>
> Adrián Isaí Morales-Paredes<br>
> Julio Juárez<br>
> Jesús Guillermo Falcón-Cardona<br>
> Hugo Terashima-Marín<br>
> Carlos A. Coello Coello<br>

A Grammatical-Evolution hyper-heuristic that generates variation operators
for multi-objective evolutionary algorithms (MOEAs), then benchmarks the
discovered operators against instances unseen during generation, on the
multi-objective Quadratic Assignment Problem (mQAP).

> **Note on reproducibility:** Running the commands below
> with the default (published) parameters reproduces the *methodology* and
> should land in the same range of operators, but will not reproduce the exact same
> discovered operators.

---

## Table of Contents

- [Background](#background)
- [Requirements](#requirements)
- [Installation](#installation)
- [Reproducing the experiments](#reproducing-the-experiments)
- [LLaMEA search (LLM-guided operators)](#llamea-search-llm-guided-operators)
- [Repository structure](#repository-structure)
- [Data](#data)
- [Configuration reference](#configuration-reference)
- [Algorithms](#algorithms)
- [Output format](#output-format)

---

## Background

**Grammatical Evolution (GE)** evolves programs indirectly: an individual is
a *genome* (a list of integer codons), which is decoded into a *phenotype*
(here, a crossover-operator expression) by walking a BNF grammar and using
each codon, modulo the number of production choices, to pick the next
production rule. Two genomes that differ can decode to the same phenotype,
and small genome edits can produce structurally different phenotypes - this
indirection is what lets a standard genetic algorithm (selection, crossover,
mutation over integer genomes) search a space of *programs* and in this case
heuristics.

This repository uses GE to search a space of **variation (crossover) operators** for
permutation-encoded multi-objective evolutionary algorithms. Each candidate
operator is an expression built from primitives such as `move_right`,
`reverse`, `order_based(...)` (see [Algorithms](#algorithms)); a candidate is
scored by embedding it as the crossover operator inside an inner MOEA and
solving the **multi-objective Quadratic Assignment Problem (mQAP)**:

$$f_k(\pi) = \sum_{i} \sum_{j} w_k(i, j) \cdot d(\pi(i), \pi(j)), \quad k = 1, \dots, m$$

where $\pi$ is a permutation of facilities to locations, $w_k(i,j)$ is the
flow between facilities $i,j$ under objective $k$, and $d(\cdot,\cdot)$ is
the distance between locations. The goal is to find operators that help an
MOEA approximate the Pareto front of $(f_1, \dots, f_m)$ well, and that it
*generalize* on the problem, hence a strict split between instances seen
during generation and instances held out for evaluation.

- **`generation`**: the outer GE loop evolves a population of candidate
  operators (encoded as genomes), each scored by running an inner MOEA
  (MOEA/D, NSGA-II, or SMS-EMOA, via [pymoo](https://pymoo.org)) with that
  operator plugged in as crossover, on the training instances. Selection
  pressure favors operators with better hypervolume across instances.
- **`evaluation`**: takes the Best/Middle/Worst operators from a generation
  run and benchmarks them against held-out mQAP instances, alongside
  standard baseline operators (PMX, CX), across multiple MOEAs and multiple
  evaluation budgets (to see whether more generations meaningfully improves
  the discovered operators' performance).

---

## Requirements

- Python >= 3.10
- NumPy >= 2.0, pymoo >= 0.6.1, Matplotlib >= 3.9, pandas >= 2.3, tqdm >= 4.67

All dependencies are declared in `pyproject.toml` and installed automatically
by following the `Installation` steps. 

---

## Installation

```bash
# Clone the repository
git clone https://github.com/AdrianMP1/Generation-HyperHeuristic.git MOHH
cd MOHH
```

```bash
# Create and activate the environment
conda create -n mohh-dev python=3.12
conda activate mohh-dev
```

```bash
# Install the mohh package in editable mode
pip install -e .
```

Verify the installation:

```bash
python -c "import mohh; print('OK')"
```

---

## Running experiments

All commands must be run from the `MOHH/` directory. Three console commands
are installed by `pip install -e .`:

### Run the GE search

```bash
mohh-generate
```

Evolves candidate operators against the training instances in
`datasets/mqap/train/` and writes `results/Experiment_NNN/` (auto-numbered).
At the end, it extracts the Best/Middle/Worst operators from the final
generation and writes them to `<model>_operators.txt` in the experiment
folder, one file per `--models` entry.

### Benchmark a generation run

```bash
mohh-evaluate results/Experiment_001
```

Reads the operators produced by a `mohh-generate` run, benchmarks them
against the held-out instances in `datasets/mqap/test/`, alongside PMX/CX
baselines, across every solver in `--solvers` and every budget in
`--eval-budgets`, and writes CSVs, `.pof` front files, and box-plot figures
under `results/Experiment_001/evaluation_results/`.

### Both stages in one command

```bash
mohh-run --full
```

`--full` is a deliberate confirmation flag, since evaluation is by far the
more expensive stage - the command refuses to run without it.

### Overriding parameters

All three commands accept CLI overrides instead of editing `params.py` by
hand; anything not exposed as a flag can still be changed directly in
`generation/params.py` / `evaluation/params.py`. Run any command with `-h`
for the full flag list.

```bash
mohh-generate --models MOEAD --population-size 6 --generations 2 --elite-size 3 \
              --mo-population-size 10 --mo-generations 5

mohh-evaluate results/Experiment_001 --models MOEAD --solvers MOEAD,NSGAII \
              --mo-population-size 6 --mo-generations 4 --n-experiments 1 \
              --eval-budgets 10k:6,30k:18,50k:24
```

`mohh-evaluate` takes two related but distinct flags:

| Flag | Meaning | Default |
|------|---------|---------|
| `--models` | Which generation model(s)' operators to pull from `experiment_path` (reads `<model>_operators.txt`) | auto-detected from the experiment folder |
| `--solvers` | Which MOEA(s) to actually benchmark those operators with | `evaluation/params.py`'s `SOLVERS` (`MOEAD`, `NSGAII`, `SMSEMOA`) |

**Useful flags:**

| Flag | Effect |
|------|--------|
| `--models MOEAD,NSGAII,SMSEMOA` | Run generation's inner MOEA / pull evaluation operators for more than one model |
| `--solvers MOEAD,NSGAII` | Restrict which MOEAs evaluation benchmarks against |
| `--eval-budgets 10k:6,30k:18` | Comma-separated `label:evaluations` pairs; a budget unreachable at the configured `--mo-generations` is skipped rather than erroring |
| `--n-experiments N` | Repeated experiments per instance/operator combination |
| `--full` | Required on `mohh-run` to confirm running both stages |

### A quick, fast sanity run

To confirm the pipeline works end-to-end without waiting for a full-scale
run (the published-scale defaults - 50/30 generations, 105/300-500 MOEA
population/generations, 11 repeated experiments - can take a long time):

```bash
mohh-run --full --models MOEAD --population-size 8 --generations 2 \
         --elite-size 2 --mo-population-size 10 --mo-generations 3 \
         --n-experiments 3 --eval-budgets 10k:6
```

---

## LLaMEA search (LLM-guided operators)

`mohh-llamea` reuses the **exact fitness of the GE search** (same instances,
same inner MOEA, same normalized-hypervolume frame from
`mohh/evaluation/experiment.py`, nadir fixed at `[1.1] * K`) but replaces the
grammar with a local LLM. [LLaMEA](https://github.com/XAI-liacs/LLaMEA) prompts
the model to write a Python `crossover(x, y)` operator, evaluates it as a pymoo
`Crossover`, and feeds the hypervolume ranking back to refine the next prompt.
The GE core is **not modified**: the LLM is injected through `solver.crossover`
(`src/mohh/llamea/`).

Because the HV is measured in the same consolidated frame (reference panel plus
every evaluated candidate), the numbers are **comparable with the GE table**.

Full background and design notes: [`docs/llamea-prompts/`](docs/llamea-prompts/)
(`comparacion-justa.md`, `areas-de-mejora.md`, `pruebas-locales.md`).

### Install

```bash
pip install -e ".[llamea]"      # llamea==1.4.0 + openai + huggingface_hub
pip install -e ".[vllm]"        # only on the node that will SERVE the model (GPU)
```

The model must be exposed through an **OpenAI-compatible** endpoint: vLLM
(cluster/GPU), Ollama, or llama.cpp (`scripts/local_openai_server.py`).

### A. Local — normal command

```bash
# 1. Serve a model. Example: Ollama, using already-downloaded models (no GPU).
ollama serve &
ollama pull qwen2.5-coder:7b-instruct

# 2. Run the search against the server (wrapper with sensible defaults).
MODEL=qwen2.5-coder:7b-instruct \
BASE_URL=http://127.0.0.1:11434/v1 \
NO_DOWNLOAD=1 \
CATEGORY=searches \
BUDGET=200 \
TRAIN_INSTANCES=8 \
EVAL_TEST=1 \
NAME=qwen25coder7b_moead_b200 \
PYTHON=python3 \
./scripts/llamea_search.sh
```

The same thing calling the entry point directly:

```bash
python -m mohh.llamea.run \
    --model qwen2.5-coder:7b-instruct \
    --base-url http://127.0.0.1:11434/v1 --no-download \
    --budget 1000 --solver MOEAD --train-instances 8 \
    --repeat-penalty 0.5 --stall-timeout 1800 \
    --eval-test --results-dir results/searches \
    --experiment-name qwen25coder7b_moead_b1000
```

For a local model with explicit reasoning, the 12 GB GPU option is
`GLM-Z1-9B-0414` Q4_K_M (~6.2 GB). With llama.cpp:

```bash
hf download unsloth/GLM-Z1-9B-0414-GGUF GLM-Z1-9B-0414-Q4_K_M.gguf \
  --local-dir "$HOME/models/glm-z1-9b"
llama-server -m "$HOME/models/glm-z1-9b/GLM-Z1-9B-0414-Q4_K_M.gguf" \
  -c 8192 -ngl 99 --port 8003
MODEL=glm-z1-9b BASE_URL=http://127.0.0.1:8003/v1 \
  NO_DOWNLOAD=1 BUDGET=30 NAME=glm_z1_9b_local ./scripts/llamea_search.sh
```

The request log (`llm_requests.jsonl`) keeps the prompt, raw response,
`reasoning_content`, sampling parameters, token usage and latency, so the
thinking is auditable even when the final code is extracted from the response.

Defaults and the flags that matter:

| Flag | Meaning | Default |
|------|---------|---------|
| `--budget N` | LLM candidate evaluations, not generations: `G ~= (B - parents) / offspring` | `1000` |
| `--n-parents` / `--n-offspring` | population size / children per generation | `1` / `1` |
| `--combine` | own behavioral two-parent recombination (requires `--n-parents >= 2`) | off |
| `--target-penalty F` | ranking multiplier when the combine child misses its target family (`1.0` = feedback only) | `1.0` |
| `--require-target-family` | reject a combine child before the inner MOEA if its family does not match | off |
| `--solver MOEAD` | inner MOEA (`MOEAD`, `NSGAII`, `SMSEMOA`) | `MOEAD` |
| `--mo-generations` / `--mo-population-size` | inner MOEA budget | `30` / `30` |
| `--repeat-penalty F` | diversity pressure: ranking score is multiplied by `F` when the code was already evaluated (`1.0` disables) | `0.5` |
| `--operator-memory N` | inject the last `N` mechanism descriptions into the prompt | `0` (CLI), `12` (cluster job) |
| `--thinking` | `auto` leaves the server template untouched; `on`/`off` sets `chat_template_kwargs.enable_thinking` (Qwen3 and similar) and logs `reasoning_content` | `auto` |
| `--no-diversity-feedback` | stop adding the behavioral diagnosis to the prompt (measurement/logging continue) | feedback on |
| `--patience N` | early stop after `N` evaluations without improving the best HV (`0` = off) | `0` |
| `--parent-selection` | LLaMEA parent selection: `random`, `roulette`, `tournament` | `tournament` |
| `--adaptive-mutation` / `--no-adaptive-mutation` | LLaMEA adaptive mutation | on |
| `--feature-guided-mutation` | guide mutation with AST features | off |
| `--niching` | LLaMEA diversity archive: `none`, `sharing`, `clearing`, `novelty`, `map_elites` | `none` |
| `--stall-timeout S` | watchdog: abort and log if no new generation in `S` seconds | `0` (off) |
| `--eval-test` | evaluate the best operator on the held-out test set | off |
| `--eval-repeats R` | repeat the final test evaluation `R` times (reports mean/sd and a paired comparison against every reference classic; frame fixed to the reference panel) | `11` |
| `--mutation` | add `Swap_Mutation` to the inner MOEA | off |
| `--dry-run` | validate server + config, write them, exit without spending GPU | off |
| `--stationary-frame` | normalize with the reference panel only (HV fixed); default is the GE frame | off |

Before a long run, validate everything cheaply:

```bash
python -m mohh.llamea.run --model <model> --base-url <url> --dry-run
```

### B. Cluster (ixachi) — one command

Same Modules + local-conda methodology as the GE stage. Everything lives in
LUSTRE; `$HOME` is wiped, compute nodes cannot see it and have **no internet**.

The recommended path is a **single SLURM job** that serves the model with vLLM
on `127.0.0.1` and runs the search against it, so there is nothing to
coordinate. A login wrapper downloads the weights (the login has internet) and
submits that job.

```bash
# 0. On the LOGIN node: get the code where the compute nodes can see it.
cd /LUSTRE/$USER
git clone -b cluster https://github.com/Jabonsote/mohh.git
cd mohh
git pull origin cluster

# 1. ONE COMMAND. On the first run it creates ./.conda-env (with vLLM) if it
#    does not exist, downloads/verifies the model in the persistent root, and
#    submits ONE GPU job that serves vLLM + runs the search. There is no
#    separate model job and no second queue wait.
PROFILE=big-quant GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh

# Shorter alias with exactly the same behavior.
GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea.sh

# Full pipeline in the SAME job: search + test evaluation + paired comparison
# of the best operator against OX/PBX/PMX/CX/ERX (11 repetitions, same frame).
PROFILE=coder30b-awq GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh

# Ablation: original refine/redesign without the behavioral combine.
COMBINE=0 GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh

# Optional: wait on the login until the job ends and print the tail.
WAIT=1 PROFILE=smoke BUDGET=20 EVAL_TEST=1 bash slurm/llamea-all.sh

# 2. Monitor (the command above already prints the JobID).
squeue --user=$USER
tail -f llamea-single-*.out             # server load + per-generation search
cat llamea-single-*.err                 # empty = clean
tail -f results/searches/LLaMEA_single/llamea.log     # per-generation log
tail -f results/searches/LLaMEA_single/error.log      # warnings/tracebacks only
cat results/searches/LLaMEA_single/progress.txt       # best-so-far
```

If the model is missing, the login wrapper downloads it **before** submitting.
The GPU job never downloads: it fails fast with the exact command to run on the
login. This avoids spending a GPU allocation (and queue time) on a node that has
no internet. `DRY_RUN=1` prints the `sbatch` command without submitting.

#### Defaults and sizing for a 1000-generation run

`GENERATIONS` is **1000**; the wrapper derives
`BUDGET = GENERATIONS * NOFFSPRING + NPARENTS` (2005 with the defaults), so the
job really runs 1000 LLaMEA iterations rather than 1000 candidate evaluations.
The single job requests one GPU and **16 CPUs** (not 24): inference is GPU-bound
and the internal MOEA is sequential, so extra CPUs only make the queue wait longer.
It keeps 4 OpenMP threads for vLLM/tokenization and one BLAS thread for the
search.

| Parameter | Cluster default | Why |
|---|---|---|
| `GENERATIONS` | `1000` | Real LLaMEA iterations |
| `NPARENTS` / `NOFFSPRING` | `5` / `2` | Population large enough for behavioral combine; two children per generation |
| `BUDGET` | `2005` (derived) | Evaluation budget; `BUDGET` overrides the derivation |
| `COMBINE` | `1` | Own behavioral two-parent recombination (see below) |
| `BEHAVIORAL_PARENTS` / `NOVELTY_WEIGHT` | `1` / `0.5` | Focal parent chosen by HV + behavioral novelty |
| `TARGET_PENALTY` / `REQUIRE_TARGET_FAMILY` | `1.0` / `0` | Validate child against target family; `1.0` is feedback-only |
| `PROFILE` | `big-quant` (Qwen2.5-Coder-32B-AWQ) | Proven fallback; on A100/H100 prefer `glm47-flash-awq` for reasoning/math/code or `coder30b-awq` / `coder30b-fp8`; `coder-next-awq` for the 80B/3B on H100 |
| `CPUS` / `MEM` | `16` / `64G` | GPU-bound job; enough for vLLM and the sequential MOEA |
| `TEMPERATURE` / `TOP_P` / `TOP_K` | `0.7` / `0.8` / `20` | Exploration without chaotic sampling |
| `SAMPLING_REPETITION_PENALTY` | `1.05` | Mild anti-anchoring |
| `MAX_TOKENS` / `OPERATOR_MEMORY` | `4096` / `12` | Full operator code; bounded recent-mechanism memory |
| `MAX_MODEL_LEN` / `GPU_MEM_UTIL` | `16384` / `0.90–0.92` | Prompt + 4k response with KV headroom |
| `MO_GENERATIONS` / `MO_POPULATION_SIZE` | `30` / `30` | Internal MOEA effort per candidate |
| `TRAIN_INSTANCES` | `8` | Training set used by every candidate |
| `PATIENCE` | `0` | Keep running all 1000 generations |
| `EVAL_TEST` / `COMPARE` | off / off | Add them for the final held-out evaluation and paired baselines |

#### Behavioral combine (own extension, not original LLaMEA)

LLaMEA 1.4 lets an `Operator` declare two parents, but it selects both by
fitness. That tends to combine similar individuals. When `COMBINE=1`, this
repository adds three things without modifying LLaMEA or the GE core:

1. the focal parent is chosen by a tournament on HV **plus behavioral
   novelty**, not fitness alone;
2. the second parent is the most behaviorally distant one (and, if available,
   from the target family), so the pair actually carries different mechanisms;
3. the child is measured on fixed parent pairs and checked against the target
   family (the least-represented family). The result is logged as
   `target_family_hit`; `TARGET_PENALTY=0.9` applies a ranking-only penalty and
   `REQUIRE_TARGET_FAMILY=1` rejects a miss before spending the internal MOEA.

The raw HV is never modified by this check. The target-family evidence is in
`diversity.jsonl` and `diversity.json`; the aggregate hit rate is in
`llamea_best_info.json`. Use `COMBINE=0` for the original refine/redesign
baseline, or `BEHAVIORAL_PARENTS=0` to isolate the target-family check.

The server adds `--max-num-seqs 4`, `--enable-prefix-caching` and disables
vLLM request logs (`--no-enable-log-requests` in vLLM 0.32; the old
`--disable-log-requests` is used only if that version supports it): LLaMEA sends
one request at a time, prompts share a long preamble, and the full prompts are
already saved under `prompts/`.

For a 1000-generation run with the final evidence, use one of:

```bash
# A100 40 GB
PROFILE=coder30b-awq GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh

# A100 40 GB / H100 80 GB: GLM-4.7-Flash, reasoning + math + code
PROFILE=glm47-flash-awq GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh

# H100 80 GB: the larger 80B/3B coder
PROFILE=coder-next-awq GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh

# H100 80 GB
PROFILE=coder30b-fp8 GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh

# Fallback known-good on A100/H100
PROFILE=big-quant GENERATIONS=1000 EVAL_TEST=1 COMPARE=1 bash slurm/llamea-all.sh
```

`COMPARE=1` runs 11 paired repetitions against OX/PBX/PMX/CX/ERX in the same
job. If you only want the search, omit it. `CPUS=24` and `MEM=...` remain
available as overrides, but are not needed.

#### GPU/model profiles (big models)

`slurm/llamea-gpu-profile.sh` picks the model and, for some profiles, its vLLM
arguments (`coder30b-awq` forces `awq_marlin`; `coder30b-fp8` uses
`--kv-cache-dtype fp8`; `glm47-flash-awq` adds
`--reasoning-parser glm45`). Override with
`MODEL=...`, `CONSTRAINT=...`, `MAX_MODEL_LEN=...`, `GPU_MEM_UTIL=...`,
`VLLM_ARGS=...`; `PROFILE=list` prints the catalog.

| `PROFILE` | Model | VRAM | Fits on |
|---|---|---|---|
| `big-quant` (default) | `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | ~19–21 GB | A100 40 GB, H100 80 GB |
| `coder30b-awq` | `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` | ~18 GB | A100 40 GB, H100 80 GB |
| `coder30b-fp8` | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | ~31 GB | H100 80 GB |
| `glm47-flash-awq` | `QuantTrio/GLM-4.7-Flash-AWQ` | ~20 GB | A100 40 GB, H100 80 GB (30B/3B; vLLM main) |
| `coder-next-awq` | `qtum/Qwen3-Coder-Next-AWQ` | ~42 GB | H100 80 GB (80B/3B; ~30 GB KV) |
| `coder30b-awq-a10` | `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` | ~18 GB | A10 24 GB, 8k context |
| `h100-fp16` | `Qwen/Qwen2.5-Coder-32B-Instruct` | ~65 GB | H100 80 GB |
| `a100-fp16` | `Qwen/Qwen2.5-Coder-14B-Instruct` | ~29 GB | A100 40 GB |
| `a10-quant` | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | ~10 GB | A10 24 GB |
| `coder8b` | `ByteDance-Seed/Seed-Coder-8B-Instruct` | ~16 GB | A10 24 GB |
| `smoke` | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | any (smoke tests only) |

Evidence for the choice (HumanEval+/BigCodeBench and quantization studies) and
the per-GPU recommendation are in [`modelos.md`](modelos.md) §1. In short:
Qwen3-Coder-30B-A3B has 32B-class HumanEval+ (89.0) with only 3.3B active
parameters, so it is much faster for 1000-generation runs; `big-quant` remains
the proven fallback. `glm47-flash-awq` is the reasoning/math/code option:
GLM-4.7-Flash reports AIME25 91.6, GPQA 75.2, LiveCodeBench v6 64.0 and
SWE-bench Verified 59.2 with only 3B active parameters. It needs a recent vLLM
(`glm4_moe_lite` support; the AWQ card was tested with `vllm==0.14.0rc2`) and
the profile passes `--reasoning-parser glm45`, so the thinking is stored in
`llm_requests.jsonl`. For the local RTX 3060 12 GB, use
`unsloth/GLM-Z1-9B-0414-GGUF` Q4_K_M (~6.2 GB) with llama.cpp/Ollama; it is a
9B reasoning model, not a replacement for the cluster model.
`coder30b-awq` is a community AWQ — verify it before a long run and fall back to
`big-quant` if vLLM fails to load it. `coder-next-awq` is
the larger 80B/3B option (LiveCodeBench v6 58.9, EvalPlus 86.6): its W4A16 AWQ
is ~42 GB and only fits an H100 80 GB; it is a candidate to test, not the
default, because quantization raises its wikitext-2 perplexity from 7.74 to
12.53.

Examples:

```bash
PROFILE=coder30b-awq GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh
PROFILE=glm47-flash-awq GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh # A100/H100, reasoning
PROFILE=coder30b-fp8 GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh  # H100
PROFILE=coder-next-awq GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh # H100, 80B/3B
PROFILE=coder8b GENERATIONS=1000 EVAL_TEST=1 bash slurm/llamea-all.sh       # A10
PROFILE=smoke BUDGET=20 EVAL_TEST=1 bash slurm/llamea-all.sh                # sanity
PROFILE=list bash slurm/llamea-all.sh                                       # catalog
```

#### Alternative (advanced): a persistent server + separate searches

The normal path is the single GPU job above. Only use two jobs when you will
run **many** searches against the same loaded model (avoids reloading the 32B
weights for every run). It requires two queue waits, so it is not the default.

```bash
# Server (long-lived GPU job).
sbatch slurm/llamea-vllm-server.slrm
tail -f llamea-vllm-*.out               # wait for "Uvicorn running on ...:8000"

# Each search points at the server node (it can autodetect it by JobID).
BUDGET=1000 EVAL_TEST=1 LLAMEA_SERVER_JOBID=<jobid-vllm> \
  sbatch slurm/llamea-generate.slrm
```

`slurm/llamea-full-chain.sh` now redirects to the single-job command unless you
set `ALLOW_TWO_JOBS=1`; this prevents accidentally submitting two chained jobs
and waiting twice in the queue.

Extra environment knobs for the search jobs: `SOLVER`, `MO_GENERATIONS`,
`MO_POPULATION_SIZE`, `TEMPERATURE` (0.7 in the single job), `TOP_P` (0.8),
`TOP_K` (20), `SAMPLING_REPETITION_PENALTY` (1.05), `MAX_TOKENS` (4096),
`OPERATOR_MEMORY` (12), `MUTATION=1`, `COMMA=1`, `REFINE_ONLY=1`,
`NO_SAVE_FRONTS=1`, `REPEAT_PENALTY`, `PATIENCE`, `PARENT_SELECTION`,
`NICHING`, `EVAL_REPEATS` (11), `CATEGORY`, `EXPERIMENT_NAME`, `SEED`. Add
`COMPARE=1` to run the paired comparison against the classics in the same job
(`COMPARE_REPEATS=11`, `COMPARE_SPLIT=test`). The comparison includes
one-child and two-child classic variants and treats a failed operator/instance
as HV 0; use `COMPARE_FAIL_POLICY=exclude|abort` to change that. For the full GE
protocol (3 solvers × 2 mutation modes × 300/105), set
`COMPARE_GE_PROTOCOL=1`; this is expensive and intended for the final table.

`--operator-memory N` appends the last `N` generated mechanisms to the prompt
and asks the model not to repeat them; `--repeat-penalty` still penalizes
repeated code in the ranking. `--sampling-repetition-penalty` is the vLLM
sampler's `repetition_penalty`, a different knob.

### C. Compare discovered operators against the classics

The HV of a single search is **not** comparable across runs (its frame contains
that run's candidates), and one seed cannot separate signal from inner-MOEA
noise. This harness re-evaluates every operator under **one shared frame**, with
**paired seeds** and **N independent repetitions**:

```bash
python scripts/compare_operators.py \
    --operator discovered=results/examples/llamea_found_operators/qwen25coder7b_moead_b100_ge.py \
    --operator ox=classic:ox --operator pbx=classic:pbx --operator pmx=classic:pmx \
    --operator cx=classic:cx --operator erx=classic:erx \
    --operator ox2=classic2:ox --operator pbx2=classic2:pbx \
    --operator pmx2=classic2:pmx --operator cx2=classic2:cx \
    --operator erx2=classic2:erx \
    --split test --repeats 11 --name b100_vs_clasicos --save-fronts
```

Use `--ge-protocol` to evaluate the same comparison with the GE evaluation
protocol (MOEA/D, NSGA-II and SMS-EMOA; with and without `Swap_Mutation`;
300 generations and population 105). The short default (one solver, 30/30, no
mutation) is cheaper and mirrors the search loop; the conclusion should be
confirmed with `--ge-protocol` for the final table.

Writes `results/evaluations/<name>/report.md` and `evaluation.json` with the
paired Wilcoxon test, effect size (`dz`), 95% CI of the mean difference, Holm
adjustment for the family of comparisons, **rank stability** across repetitions
and the **minimum detectable difference** (how much of the ranking is noise).
A failed operator/instance is counted as HV 0 by default (`--fail-policy
zero`) instead of dropping the pair; `exclude` and `abort` are also available.
`--save-fronts` / `--load-fronts` reuse the inner-MOEA fronts when adding
operators or reanalyzing.

> The shared frame fixes `vmin`/`vmax`. If an operator is better than the best
> point of the reference panel, its normalized coordinates are clipped to `0`
> before the HV integral; otherwise a fixed reference box would let an
> arbitrarily good point produce an HV above the theoretical maximum
> `1.1**K`. Clipping only affects points outside the reference box and keeps
> the comparison bounded. In the GE frame the candidate already belongs to the
> frame, so normal values are unchanged.

### D. Replicate the search (not just the operator)

The best of ~1000 candidates of one run is a max-biased estimate. To make a
claim about the *procedure*, run R independent searches and report the
distribution of the best-per-run:

```bash
for i in 0 1 2 3 4; do
  MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ BASE_URL=http://<node>:8000/v1 NO_DOWNLOAD=1 \
  SEED=$((1000+i)) NAME=proc_r$i CATEGORY=searches BUDGET=1000 \
    ./scripts/llamea_search.sh
done
python scripts/collect_replicates.py 'results/searches/proc_r*' \
    --out results/multirun/proc_analysis
```

`summary.md` gives the mean/median/sd of the best-per-run (the honest number),
and `compare_args.txt` feeds those operators into `compare_operators.py`.

### E. Choose a model with evidence from *this* loop

Code benchmarks do not predict anchoring or the valid-operator rate. Run the
same short budget/seed with each profile and compare:

```bash
PROFILE=coder30b-awq BUDGET=50 SEED=42 EXPERIMENT_NAME=bakeoff_30b \
  bash slurm/llamea-all.sh
PROFILE=big-quant BUDGET=50 SEED=42 EXPERIMENT_NAME=bakeoff_32b \
  bash slurm/llamea-all.sh
python scripts/analyze_llamea_runs.py 'results/searches/bakeoff_*' \
  --summary --csv results/bakeoff.csv
```

The table reports valid operators, distinct mechanisms, best HV, tokens per
evaluation and seconds per request — the numbers that decide which model is
worth a 1000-generation run.

### Output artifacts (per search)

Everything lands under `results/<category>/<name>/`:

| File / dir | Content |
|---|---|
| `run.json` | full effective config (incl. vLLM version and model revision) |
| `llamea_best_operator.py`, `llamea_best_info.json` | best operator and its scores |
| `generation_summary.jsonl` | per generation: seed, tokens, time, validity, best-so-far |
| `individuals.jsonl` | one record per candidate (name, fitness, code sha1, seed) |
| `prompts/` | exact prompt sent + clean response + raw response/reasoning + per-turn tokens, sampling params and `finish_reason` |
| `llm_requests.jsonl` | one raw JSONL record per request: full messages, effective sampling params, raw response, `reasoning_content`, usage and errors |
| `fronts/` | raw Pareto fronts per candidate (evidence; `--no-save-fronts` to skip) |
| `diversity.json`, `diversity.jsonl` | behavioral fingerprint, family, novelty, MAP-Elites cell and coverage per operator |
| `checkpoint.json`, `best_so_far.py` | durable state — a crash never loses the best operator |
| `seeds.json` | every seed used and the policy |
| `tokens.json`, `metrics.json` | tokens, inference time, GPU seconds (vLLM `/metrics`), energy |
| `progress.txt`, `llamea.log`, `error.log` | monitoring + traceback-only log |
| `llamea_logs/exp-*/` | LLaMEA's own logs (`report.md`, `evolution.png`, `code/`) |

### Behavioral diversity (not code hashes)

Counting distinct code hashes overestimates diversity: a renamed OX is a new
hash but the same mechanism. The evaluator therefore probes each valid operator
on fixed parent pairs (sizes 11, 24, 50, 100) and computes a 14-feature
**behavioral fingerprint**: recombination strength, distance/bias to each
parent, position/order/adjacency preservation, edge-union preservation, runs
and parent switches. From it:

- **Behavioral novelty** = distance to the closest operator already tried
  (low = syntactic rewrite of the same mechanism).
- **Families** = closest classic mechanism (`order`, `position`, `cycle`,
  `adjacency`) or `exploratory`/`copy`/`mixed`.
- **MAP-Elites archive** = best HV per cell of recombination x structure; its
  coverage shows which mechanism regions remain unexplored.
- The feedback given to the LLM includes the family, novelty, closest previous
  operator and the least-explored direction (a light ReEvo-style reflection).
- LLaMEA's native `distance_metric` and `behavior_descriptor` hooks use the
  fingerprint, so `--niching novelty`, `clearing` and `map_elites` operate on
  behavior instead of code length/tokens.

Inspect a run after it finishes (or after a crash, from `diversity.jsonl`):

```bash
python scripts/analyze_diversity.py results/searches/LLaMEA_single \
    --plot results/searches/LLaMEA_single/diversity.png
python scripts/analyze_llamea_runs.py 'results/searches/*' \
    --summary --csv results/bakeoff.csv
```

The summary now reports `distinct` (code) vs `mecanismos` (behavior), novelty,
archive coverage and family counts. `--no-diversity-feedback` keeps measuring
and logging but stops adding the diagnosis to the prompt.

### Persistent model weights

`scripts/mohh_models.py` keeps weights in `$MOHH_MODELS_DIR`, else
`$SCRATCH/$USER/mohh-models`, else `$LUSTRE/$USER/mohh-models` — a disk the
compute nodes can see, never `$HOME`. Subcommands: `root`, `list`, `path`,
`download`, `verify [--hash] [--load-check]`, `manifest`, `smoke`, `remove`.

```bash
python scripts/mohh_models.py root
python scripts/mohh_models.py download Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
python scripts/mohh_models.py verify Qwen/Qwen2.5-Coder-32B-Instruct-AWQ --load-check
```

---

## Repository structure

```
MOHH/
├── pyproject.toml
├── README.md
├── grammars/
│   ├── naturals.bnf                 <- active grammar (permutation operators, used by default)
│   └── reals.bnf                    <- earlier real-valued grammar.
│                                        predates the project's focus on permutation-encoded
│                                        problems; not wired into any current entry point
├── datasets/
│   └── mqap/
│       ├── train/                   <- instances used during generation
│       └── test/                    <- held-out instances used during evaluation
│
├── results/                         <- created on first run; one Experiment_NNN/ per mohh-generate call
└── src/mohh/
    ├── cli.py                       <- mohh-generate / mohh-evaluate / mohh-run
    ├── main.py                      <- run_generation: orchestrates one or more generation models
    ├── run_evaluation.py            <- discover_models / extract_operators (reads *_operators.txt)
    ├── parallel.py                  <- simple process-pool parallelization (run_parallel)
    ├── core/                        <- shared by both stages
    │   ├── params.py                <- Params singleton (dict-like global config)
    │   ├── representation/
    │   │   ├── grammar.py           <- BNF parser (adapted from PonyGE2)
    │   │   ├── mapper.py            <- genome -> phenotype mapping (adapted from PonyGE2)
    │   │   └── population.py        <- Population / Individual
    │   ├── problem/instance.py      <- QAP problem definition (pymoo Problem), Instance loader
    │   ├── operators/operator_template.py <- HH_Operator: wraps a decoded phenotype as a pymoo Crossover
    │   └── utilities/
    │       ├── algorithm/MO.py      <- non-dominated sorting, hypervolume, nadir point
    │       ├── algorithm/HH_functions.py <- grammar primitive functions (swap, reverse, order_based, ...)
    │       ├── instance_utils.py    <- lists train/ or test/ instance file paths
    │       ├── load_modules.py      <- string-name -> operator class dynamic dispatch
    │       ├── paths.py             <- project_root()
    │       └── print_utils.py       <- in-place CLI progress line updates
    │ 
    ├── generation/                  <- the GE search stage
    │   ├── params.py                <- generation defaults + set_params(experiment_path, model, overrides)
    │   ├── saver.py                 <- PopulationSaver / MyLogger singletons
    │   ├── algorithm/
    │   │   ├── hyperheuristic.py    <- HyperHeuristic: the outer GE loop (init, evaluate, evolve, replace)
    │   │   └── multiobjective.py    <- MOSolver: inner MOEA wrapper (MOEAD/NSGAII/SMSEMOA), per candidate
    │   ├── operators/                <- GE-level operators (on genomes)
    │   │   ├── initialization.py    <- Rvd: random valid population, discarding invalid/duplicate genomes
    │   │   ├── selection.py         <- Tournament
    │   │   ├── crossover.py         <- KPointCrossover, UniformCrossover
    │   │   └── mutation.py          <- BitFlipMutation
    │   ├── problem/instance.py      <- re-export shim over core.problem.instance
    │   ├── representation/population.py <- re-export shim + get_genome_usage()
    │   └── utilities/algorithm/HH_auxiliars.py <- compute_rank, test_individual (phenotype sanity eval)
    │
    └── evaluation/                   <- the benchmarking stage
        ├── params.py                <- evaluation defaults + set_params(overrides)
        ├── main.py                  <- execute_experiments: run_experiments() then make_figures()
        ├── experiment.py            <- run_experiments: the full sweep over instances x operators x solvers
        ├── algorithms.py            <- MOSolver (ABC), MOEA_Decomposition/NSGAII/SMS_MOEA, MyCallback
        ├── make_plots.py            <- reads saved CSVs, produces the box-plot figures
        ├── problem/instance.py      <- Instance subclass adding N-experiments initial-solution handling
        └── operators/
            ├── operator_template.py <- re-export shim over core.operators.operator_template
            ├── crossover.py         <- SBX_Cross, PMX_Cross, CX_Cross
            └── mutation.py          <- PM_Mutation, Swap_Mutation
```

**Two separate `MOSolver` classes exist by design**: `generation/algorithm/multiobjective.py::MOSolver`
dispatches on a `model_name` string and is built fresh per candidate operator (many short MOEA runs);
`evaluation/algorithms.py::MOSolver` is an `ABC` with one subclass per MOEA and adds eval-budget
snapshotting (`MyCallback`) for the benchmarking sweep. They share a name and a role but not a
contract, do not assume one's API from the other.

> **IMPORTANT — project rule:** Modifying the logic of the MOEAs is **forbidden**
> (`Prohibido modificar`). This fork only *parallelizes* the independent MOEA runs.
> No algorithm logic was touched (mating, decomposition, ask/tell, problem evaluation,
> operators); the results/state updates applied after each run are identical to the
> sequential version.

### Parallelization (process pool, all cores)

The independent MOEA runs are executed on all available cores via a simple
process pool — parallelism comes from running *separate MOEA instances*
concurrently, not from changing how a single MOEA advances.

- `mohh/parallel.py` — `run_parallel(fn, tasks, on_complete=None)`: creates a
  `ProcessPoolExecutor` with one worker per CPU (dynamic, capped by the number
  of tasks) and returns the results in task order. Falls back to a plain
  sequential loop for a single task or single core. The optional `on_complete(i)`
  callback is invoked in the main process as each task finishes so the original
  `tqdm` progress bars stay live (they advance while solving, exactly as in the
  sequential version) instead of jumping at the end.
- **Generation (search)** — `HyperHeuristic.evaluation_step` builds one task per
  (candidate operator × training instance) and each worker runs that inner MOEA
  (`generation/algorithm/hyperheuristic.py::_solve_individual_instance`). The
  main process then applies the exact same updates as before (pareto sets/fronts,
  consolidated fronts, metrics).
- **Evaluation** — `run_experiments` builds one task per repeated experiment
  (`evaluation/experiment.py::_solve_experiment_task`); this applies to **all**
  solvers benchmarked (MOEAD, NSGAII, SMSEMOA). Each worker reproduces the same
  global seeds (`random.seed`, `np.random.seed`) as the sequential loop, so the
  evaluation results are comparable run-to-run. Generation seeds
  `np.random.seed(seed)` per independent run for reproducibility (the RNG stream
  is no longer shared across serial candidates).

---

## Data

`datasets/mqap/{train,test}/` hold mQAP instances in the Knowles-Corne
(`KC`) benchmark format. Filename encodes the instance: `KC{n}-{k}fl-{tag}`,
e.g. `KC20-2fl-3rl` = 20 facilities, 2 flow matrices (objectives), variant
`3rl`. Each file starts with a header line, followed by the distance matrix
and one flow matrix per objective:

```
facilities= 20 objectives= 2 max_distances= ... max_flows= ... overlap= ... seed= ...
<N x N distance matrix>
<N x N flow matrix, objective 1>
<N x N flow matrix, objective 2>
...
```

Training instances (used only by `mohh-generate`) are smaller (`KC10`,
`KC20`, `KC30`); test instances (used only by `mohh-evaluate`) are held out
to check that discovered operators generalize rather than overfit the
training set.

---

## Configuration reference

Both `generation/params.py` and `evaluation/params.py` are plain Python
dicts merged into a `Params()` singleton at `set_params(...)` time, then
patched by any CLI overrides. Representative excerpts:

```python
# generation/params.py
hh_params = {
    'POPULATION_SIZE': 50,       # outer GE population
    'GENERATIONS': 30,           # outer GE generations
    'ELITE_SIZE': 15,            # kept unconditionally each generation

    'INITIALIZATION': "Rvd",
    'SELECTION': "Tournament", 'TOURNAMENT_SIZE': 3,
    'CROSSOVER': 'KPointCrossover', 'CROSSOVER_K_POINTS': 2, 'CROSSOVER_PARENTS': 2,
    'MUTATION': 'BitFlipMutation', 'MUTATION_PROBABILITY': 0.01,
}

mo_params = {
    'MO_MODEL': "MOEAD",         # inert default - overwritten by the CLI's --models value
    'MO_POPULATION_SIZE': 105,   # inner MOEA population, per candidate operator
    'MO_GENERATIONS': 300,
    'MO_MUTATION_BOOL': False,   # inner MOEA runs crossover-only by default
}
```

```python
# evaluation/params.py
mo_params = {
    'SOLVERS': ["MOEAD", "NSGAII", "SMSEMOA"],   # benchmarked by default
    'N_EXPERIMENTS': 11,                          # repeats per instance/operator/solver
    # 10500/31500/52500 = 105 (MO_POPULATION_SIZE) * 100/300/500 generations -
    # these are the published evaluation numbers.
    'EVAL_BUDGETS': {"10k": 10500, "30k": 31500, "50k": 52500},
    'MO_POPULATION_SIZE': 105,
    'MO_GENERATIONS': 500,
}
```

Both share `problem_params` (`PROBLEM_NAME: "QAP"`, `DATASET: "mqap"`,
`OPTIMIZATION_KIND: False` for minimization) and `grammar_params`
(`GRAMMAR_FILE: "naturals.bnf"`).

---

## Algorithms

### MOEAs (inner loop and benchmarking)

| Key | pymoo class | Used in |
|-----|-------------|---------|
| `MOEAD` | `pymoo.algorithms.moo.moead.MOEAD` | generation (inner loop, default) and evaluation |
| `NSGAII` | `pymoo.algorithms.moo.nsga2.NSGA2` | generation and evaluation |
| `SMSEMOA` | `pymoo.algorithms.moo.sms.SMSEMOA` | generation and evaluation |

### Grammar primitives (`grammars/naturals.bnf`)

The active grammar builds operator expressions from these primitives
(implemented in `core/utilities/algorithm/HH_functions.py`):

| Primitive | Arity | Description |
|-----------|-------|-------------|
| `move_right` | 1 | Rotate the permutation one position to the right |
| `reverse` | 1 | Reverse the permutation |
| `map_list` | 2 | Use each list as an index permutation of the other |
| `alternate_elements` | 2 | Interleave elements from both parents, then de-duplicate |
| `order_based(fill, preserve, x, y)` | 2 | `preserve` selects elements/segments to keep from `x`, `fill_first_occurring` fills the rest with the first not-yet-used value from `y`, in order |
| `preserve_elements` / `preserve_segments` | - | Two selection strategies for `order_based`'s `preserve` slot |

`grammars/reals.bnf` additionally defines real-valued primitives
(`sin`, `cos`, `convolution`, `one_point`, `masked_cross`, ...) but is not
wired into any current entry point.

**The grammar can be extended and modified once its non-terminals are programmed accordingly**

---

## Output format

After `mohh-generate`, then `mohh-evaluate <path>`, the results tree is:

```
results/Experiment_NNN/
├── <MODEL>_operators.txt                     <- "Best: <phenotype>\nMiddle: ...\nWorst: ..."
├── <MODEL>_<hostname>_<timestamp>/
│   └── generation_results/
│       ├── parameters.txt                    <- every param value used for this run
│       ├── CLI_output.log
│       ├── individuals/
│       │   └── Individual_NNNNNN/
│       │       ├── general_info.json         <- name, phenotype, nodes, depth, genome, codons_usage
│       │       └── instance_<name>.json      <- per-instance pareto set/front/solutions for this individual
│       ├── generations/
│       │   └── generation_NNNN/
│       │       ├── population.json           <- individual IDs in this generation
│       │       ├── offspring.json            <- (only present when offspring were just created)
│       │       └── Instance_Fronts_<name>.json <- per-instance consolidated non-dominated fronts
│       └── initial_solutions/permutation/<instance>.txt
│
└── evaluation_results/
    ├── initial_solutions/{permutation,real}/<instance>/{seeds.txt, experiment_NNN.txt}
    └── <SOLVER>/<budget>/
        ├── <instance>.csv                    <- one row per experiment, one column per operator x mutation
        ├── fronts/<instance>/
        │   ├── <label>_<mutation>_<solver>.pof[,_norm.pof]   <- median front per operator combination
        │   └── best_front_consolidated[_norm].pof
        └── Figures/{NoMutation,WithMutation}/<instance>_{NM,WM}.png
```

**`<instance>.csv` columns** are named `<label>_<combination>_<NM|WM>_<solver>`
(e.g. `MOEAD_Best_NM_MOEAD`, `None_PMX_WM_NSGAII`), where `label` is `MOEAD`
(the generation model) for discovered operators or `None` for baselines,
`combination` is `Best`/`Middle`/`Worst`/`Standard`/`PMX`/`CX`, and each row
is one repeated experiment's hypervolume.

A budget directory is only created if at least one solver/instance actually
reached it; if `--mo-generations` is too small for a given `--eval-budgets`
entry, that budget is silently skipped for that run rather than erroring.
