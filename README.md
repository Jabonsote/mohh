# Specialized Operators via a Multi-Objective Hyper-Heuristic

Source code for the paper **Automatic Design of Specialized Variation Operators for the Multi-Objective Quadratic Assignment Problem** (GECCO 2025).

Adrián Isaí Morales-Paredes, Julio Juárez, Jesús Guillermo Falcón-Cardona, Hugo Terashima-Marín, Carlos A. Coello Coello.

A grammatical-evolution (GE) hyper-heuristic evolves crossover operators for permutation-encoded multi-objective evolutionary algorithms (MOEAs). A genome is a list of integer codons decoded through a BNF grammar into an operator expression; the candidate is embedded as the crossover of an inner MOEA and scored by hypervolume on mQAP training instances. The best operators are benchmarked on held-out instances against classic crossovers.

## Background

GE decodes indirectly: each codon selects a production rule modulo the number of alternatives at the current non-terminal. Two genomes can decode to the same expression and a small edit can change it altogether, so a genetic algorithm over integer genomes searches a space of programs.

The target is a crossover operator for permutation-encoded MOEAs, composed of primitives such as `move_right`, `reverse` and `order_based(...)` (see [Algorithms](#algorithms)). Its fitness comes from the multi-objective Quadratic Assignment Problem:

$$f_k(\pi) = \sum_i \sum_j w_k(i,j)\, d(\pi(i), \pi(j)), \quad k = 1,\dots,m$$

where $\pi$ assigns facilities to locations, $w_k$ is objective $k$'s flow matrix and $d$ the location distance matrix. Generation uses training instances only; evaluation uses a disjoint test set. The split is enforced by the dataset layout: `mohh-generate` reads `train/` and `mohh-evaluate` reads `test/`.

The **generation** stage evolves genomes, each fitness being an inner MOEA run (MOEA/D, NSGA-II or SMS-EMOA via pymoo) with the candidate as crossover. Selection favors the mean hypervolume rank across instances. The **evaluation** stage benchmarks the Best/Middle/Worst operators on the test set against PMX and CX over all configured solvers and budgets.

## Requirements and installation

Python >3.10. Dependencies: NumPy >=2.0, pymoo >=0.6.2, Matplotlib >=3.9, pandas >=2.3, tqdm >=4.67 and PyYAML >=6.0. The optional `llamea`, `vllm`, `analysis`, `local-server` and `dev` extras are declared in `pyproject.toml`.

```bash
git clone https://github.com/Jabonsote/mohh.git MOOHH-R
cd MOOHH-R
conda create -n mohh-dev python=3.12
conda activate mohh-dev
python -m pip install -e .
python -c "import mohh; print('OK')"
```

## Running experiments

`pip install -e .` installs five entry points; all expect the repository root as current directory. The `results/Experiment_001` path in the examples below is a folder produced by `mohh-generate`.

| Command | Role |
|---|---|
| `mohh-generate` | Outer GE search on the training instances. |
| `mohh-evaluate <experiment>` | Benchmark one generation run on the test set. |
| `mohh-run --full` | Both stages in one process; `--full` is required. |
| `mohh-llamea` | LLM-guided operator search (see [LLaMEA search](#llamea-search)). |
| `mohh-post` | Analysis of finished runs (see [Post-analysis](#post-analysis)). |

```bash
mohh-generate
mohh-evaluate results/Experiment_001
mohh-run --full
```

`mohh-generate` writes `results/Experiment_NNN/` and extracts `<model>_operators.txt` (Best/Middle/Worst phenotypes) per `--models` entry. `mohh-evaluate` reads those files, evaluates on `datasets/mqap/test/` and writes CSVs, fronts and figures under `evaluation_results/`. `mohh-run --full` chains both stages; evaluation dominates the cost, hence the flag. All stages accept CLI overrides listed by `-h`.

```bash
mohh-generate --models MOEAD --population-size 6 --generations 2 \
              --elite-size 3 --mo-population-size 10 --mo-generations 5
mohh-evaluate results/Experiment_001 --models MOEAD --solvers MOEAD,NSGAII \
              --mo-population-size 6 --mo-generations 4 --n-experiments 31 \
              --eval-budgets 10k:6,30k:18,50k:24
mohh-evaluate results/Experiment_001 --dataset mqap --instances \
              KC100-5fl-1un KC31-3fl-1un
```

`--models` selects whose operators are read; `--solvers` selects the MOEAs that run the benchmark. These are independent: the first names the generation runs to pull operators from, the second the solvers that execute them. `--eval-budgets` takes `label:evaluations` pairs and skips unreachable budgets. `--dataset` names a directory under `datasets/`, `--dataset-path` an external one, and `--generation N` evaluates a past generation.

A short end-to-end check before a production run:

```bash
mohh-run --full --models MOEAD --population-size 8 --generations 2 \
         --elite-size 2 --mo-population-size 10 --mo-generations 3 \
         --n-experiments 3 --eval-budgets 10k:6
```

## Datasets

`datasets/mqap/{train,test}/` holds mQAP instances in the Knowles-Corne (KC) format. The name encodes size, number of objectives and variant, e.g. `KC20-2fl-3rl` (20 facilities, two flow matrices, variant `3rl`). Each file has a header, an `N x N` distance matrix, then one `N x N` flow matrix per objective:

```
facilities= 20 objectives= 2 max_distances= ... max_flows= ... overlap= ... seed= ...
<N x N distance matrix>
<N x N flow matrix, objective 1>
<N x N flow matrix, objective 2>
```

Training instances are smaller (`KC10`, `KC20`, `KC30`); test instances are held out and used only by evaluation.

## Configuration

Defaults live in three Python files and two YAML files, plus two environment overrides. Values not exposed as CLI flags are edited there before a run.

| Source | Scope | Contents |
|---|---|---|
| `src/mohh/generation/params.py` | GE search | `hh_params`: population 50, generations 30, elite 15, initialization/selection/crossover/mutation; `mo_params`: inner population 105, generations 300, crossover-only by default. |
| `src/mohh/evaluation/params.py` | Benchmarking | `SOLVERS` (MOEAD, NSGAII, SMSEMOA), `N_EXPERIMENTS` (31 repeats per cell), `EVAL_BUDGETS` (10k/30k/50k), inner MOEA budget. |
| `src/mohh/constants.py` | Shared protocol | `NADIR`, `EVAL_BUDGETS`, `MO_POPULATION_SIZE`, `MO_GENERATIONS`, `N_EXPERIMENTS = 31`, probabilities and timeouts. |
| `config/prompts.yaml` | LLaMEA prompts | `ROLE`, `TASK`, `SKELETON`, `FORMAT`, `STRATEGY_REFINE`, `STRATEGY_CREATE`; `MOHH_PROMPTS_REGISTRY` replaces this path. |
| `config/reasoning.yaml` | Reasoning models | Per-model `match`, vLLM `parser`, thinking `toggle`, `code_in_reasoning`, `max_tokens`, extra `vllm_args` and `env`; `MOHH_REASONING_REGISTRY` replaces this path. |

Both `params.py` files are dicts merged into a `Params()` singleton at `set_params(...)`, then patched by CLI overrides. The YAML files are read by `mohh.llamea.prompt_config` and `mohh.llamea.reasoning`; the environment variables above point to alternative registries.

## Post-analysis

`mohh-post` is the CLI for finished runs; `mohh-post --help` prints the subcommands and `mohh-post <subcommand> --help` their flags. The registry is `src/mohh/postanalysis/__init__.py`; analyses read the `run_metrics/` JSON of an experiment.

```bash
mohh-post --help
mohh-post gen --help
```

| Subcommand | Module | Purpose |
|---|---|---|
| `gen` | `analyze_gen.py` | Per-generation metrics and saturation diagnostics. |
| `operators` | `analyze_operators.py` | Discovered-operator statistics. |
| `search` | `analyze_search.py` | LLaMEA search summaries. |
| `ast-diversity` | `ast_diversity.py` | Diversity of operator source code. |
| `eval31` | `evaluate_31.py` | Analysis of 31-repetition evaluations. |
| `generations` | `evaluate_generations.py` | Results of evaluation by generation. |
| `figures` | `export_figure_values.py` | Values behind plotted figures. |
| `reference-sets` | `generate_reference_sets_pop.py` | Reference-set populations. |
| `generation-comparison` | `generation_comparison.py` | Compare generations of a run. |
| `hv-scale` | `hv_scale_from_evaluation.py` | Hypervolume scale of an evaluation. |
| `indicators` | `indicator_quality.py` | Quality of computed indicators. |
| `reference-report` | `reference_sets_report.py` | Report on generated reference sets. |

Standalone comparison scripts are not part of `mohh-post`: `scripts/compare_operators.py` re-evaluates discovered and classic operators under one shared frame with paired seeds and reports Wilcoxon/Holm statistics; `scripts/compare_ge_frame.py` uses the GE evaluation harness for a direct comparison with a generation experiment.

## LLaMEA search

`mohh-llamea` replaces the grammar with an LLM that writes a Python `crossover(x, y)` operator. The inner MOEA and fitness are the same as in GE: population nadir as reference point, hypervolume per instance, mean rank across instances. The GE core is unchanged; the operator is injected through `solver.crossover` (`src/mohh/llamea/`). Prompts and reasoning settings come from the files in [Configuration](#configuration).

```bash
python -m pip install -e ".[llamea]"
python -m pip install -e ".[vllm]"   # only on the GPU node serving the model
```

The model is reached through an OpenAI-compatible endpoint (vLLM, Ollama or `scripts/local_openai_server.py`).

```bash
MODEL=qwen2.5-coder:7b-instruct BASE_URL=http://127.0.0.1:11434/v1 \
NO_DOWNLOAD=1 CATEGORY=searches BUDGET=200 TRAIN_INSTANCES=8 \
EVAL_TEST=1 NAME=qwen25coder7b_moead_b200 PYTHON=python3 \
./scripts/llamea_search.sh
```

Relevant flags of `mohh-llamea`:

| Flag | Meaning | Default |
|---|---|---|
| `--budget N` | Candidate evaluations, not generations. | `1000` |
| `--n-parents` / `--n-offspring` | Population size / children per step. | `1` / `1` |
| `--solver` | Inner MOEA (`MOEAD`, `NSGAII`, `SMSEMOA`). | `MOEAD` |
| `--mo-generations` / `--mo-population-size` | Inner MOEA budget. | `30` / `30` |
| `--repeat-penalty F` | Score multiplier for repeated code; `1.0` disables. | `0.5` |
| `--operator-memory N` | Previous mechanism descriptions in the prompt. | `0` (CLI), `6` (cluster job) |
| `--eval-test` / `--eval-repeats R` | Test-set evaluation and its repeat count. | off / `11` |
| `--dry-run` | Validate endpoint and config, write them, exit. | off |

`refine` shows the parent operator and its rank; `create` shows only the interface skeleton. Reasoning models keep the raw response, including `reasoning_content`, in `llm_requests.jsonl`.

An operator must use numpy's global RNG (`np.random.*`); a private `np.random.default_rng()` ignores the seed and measures noise. Reasoning models need `MAX_TOKENS=8192` to avoid truncation before the code is written. Searches write `run.json`, `search_summary.json`, checkpoints and the best operator to `results/searches/<EXPERIMENT_NAME>/` (generated at run time).

## Cluster

The scripts below live in `slurm/` and run from the repository root on the login node. Paths are passed through environment variables; no checkout path is fixed in the scripts. Outputs go to `results/` and `logs/` (generated on the cluster).

| Script | Purpose | Main environment variables |
|---|---|---|
| `slurm/llamea-all.sh` | Downloads/verifies the model, submits the vLLM + LLaMEA GPU job. | `PROFILE`, `GENERATIONS`, `BUDGET`, `EXPERIMENT_NAME`, `EVAL_TEST`, `COMPARE`, `OPERATOR_MEMORY`, `CPUS`, `MEM`, `TIME`, `WAIT`, `DRY_RUN`, `MOHH_MODELS_DIR` |
| `slurm/compare-operators.sh` | Shared-frame comparison of discovered and classic operators. | `OPERATORS`, `GROUPS`, `CONTRASTS`, `SPLIT`, `INSTANCES`, `REPEATS`, `GE_PROTOCOL`, `SOLVER`, `SEED`, `NAME`, `LOAD_FRONTS` |
| `slurm/mohh-full-chain.sh` | Submits generation and evaluation with `--dependency=afterok`. | `USER` (from the environment) |
| `slurm/mohh-generate-big.slrm` | GE search on a full compute node; extra flags from variables. | `MODELS`, `EXPERIMENT_ARGS`, `GENERATE_ARGS`, `EXPERIMENT_PATH` |
| `slurm/mohh-evaluate.slrm` | Evaluation of one experiment, GE or LLaMEA operator. | `EXPERIMENT`, `DATASET_PATH`, `EVALUATION_DIR`, `SOLVERS`, `LLAMEA_OPERATOR`, `SEED`, `FORCE`, `MOEA_TIMEOUT`, `CONDA_ENV` |
| `slurm/submit_evaluate_array.sh` | Array of one task per test instance plus the figures job. | `EXPERIMENT`, `DATASET_PATH`, `SOLVERS`, `N_EXPERIMENTS`, `MAX_CONCURRENT`, `DEPENDENCY`, `EVALUATION_DIR` |

```bash
bash slurm/mohh-full-chain.sh
EXPERIMENT=<path-to-experiment> DATASET_PATH=<path-to-dataset> \
    bash slurm/submit_evaluate_array.sh
PROFILE=smoke BUDGET=20 EVAL_TEST=1 WAIT=1 bash slurm/llamea-all.sh
```

`EXPERIMENT` and `DATASET_PATH` are resolved on the cluster: the first points to an `Experiment_NNN/` folder from generation, the second to a dataset directory with `train/` and `test/`. Use an explicit `PROFILE` (for example `coder30b-awq` or `smoke`); the default is a fallback. Monitor jobs with `squeue --user=$USER` and the files under `logs/`.

## Parallelism

Independent MOEA runs are dispatched by `run_parallel` (`src/mohh/parallel.py`) through a process pool over separate solver instances; no algorithm logic changes. Workers come from `MOHH_WORKERS`, then `SLURM_CPUS_PER_TASK` (or `SLURM_CPUS_ON_NODE`), then `os.cpu_count()`. Results keep task order and an `on_complete` callback advances the progress bars in the main process. Tasks are per candidate and training instance in generation, per repeated experiment in evaluation; a timeout or `on_error="record"` yields an error record instead of aborting the batch.

## Repository layout

```
MOOHH-R/
├── pyproject.toml
├── config/{prompts,reasoning}.yaml
├── grammars/{naturals,reals}.bnf
├── datasets/mqap/{train,test}/
├── src/mohh/
│   ├── cli.py, main.py, run_evaluation.py
│   ├── constants.py                 shared protocol values
│   ├── parallel.py                  run_parallel, worker resolution
│   ├── core/
│   │   ├── representation/          grammar, mapper, population
│   │   ├── problem/                 QAP problem and instance loader
│   │   ├── indicators/              HV, IGD, R2, delta-plus, reference sets
│   │   ├── operators/               HH_Operator crossover wrapper
│   │   └── utilities/               algorithm and instance helpers
│   ├── shared/                      fronts, MO, instances, operators, statistics
│   ├── generation/                  outer GE loop, inner MOEA, GE operators
│   ├── evaluation/                  sweep, solver classes, plots, result layout
│   ├── llamea/                      LLM search
│   │   ├── fitness.py               fitness (compatibility alias)
│   │   ├── operator.py              LLMOperator (compatibility alias)
│   │   └── statistics.py            paired statistics (compatibility alias)
│   ├── analysis/                    readers, indicators, diversity, plots
│   └── postanalysis/                mohh-post CLI and 12 analyses
├── scripts/                         comparisons, search wrappers, model tools
├── slurm/                           cluster scripts
├── results/                         generated by runs
└── logs/                            generated on the cluster
```

`src/mohh/evaluation/operators/mutation.py` also re-exports `Swap_Mutation` as a compatibility alias. `grammars/reals.bnf` is not wired into an entry point. The two `MOSolver` classes (`generation/algorithm/multiobjective.py` and `evaluation/algorithms.py`) share a name, not an API.

## Algorithms

| Key | pymoo class | Used in |
|---|---|---|
| `MOEAD` | `pymoo.algorithms.moo.moead.MOEAD` | inner loop and evaluation |
| `NSGAII` | `pymoo.algorithms.moo.nsga2.NSGA2` | inner loop and evaluation |
| `SMSEMOA` | `pymoo.algorithms.moo.sms.SMSEMOA` | inner loop and evaluation |

The active grammar, `grammars/naturals.bnf`, builds expressions from the primitives in `core/utilities/algorithm/HH_functions.py`:

| Primitive | Arity | Effect |
|---|---|---|
| `move_right` | 1 | Rotate the permutation one position right. |
| `reverse` | 1 | Reverse the permutation. |
| `map_list` | 2 | Apply one list as an index permutation of the other. |
| `alternate_elements` | 2 | Interleave both parents and de-duplicate. |
| `order_based(fill, preserve, x, y)` | 2 | Keep elements from `x` chosen by `preserve`, fill the rest from `y`. |

`preserve_elements` and `preserve_segments` fill the `preserve` slot; the grammar extends by adding non-terminals and their implementations.

## Output format

A generation run followed by `mohh-evaluate` writes:

```
results/Experiment_NNN/
├── <MODEL>_operators.txt
├── <MODEL>_<host>_<timestamp>/
│   ├── generation_results/
│   │   ├── parameters.txt, CLI_output.log
│   │   ├── individuals/Individual_NNNNNN/{general_info.json, instance_*.json}
│   │   └── generations/generation_NNNN/{population.json, offspring.json, Instance_Fronts_*.json}
│   └── run_metrics/generation_NNNN.json
└── evaluation_results/
    ├── initial_solutions/{permutation,real}/<instance>/{seeds.txt, experiment_NNN.txt}
    ├── <SOLVER>/<budget>/<instance>.csv
    ├── <SOLVER>/<budget>/fronts/<instance>/*.pof
    ├── <SOLVER>/<budget>/Figures/{NoMutation,WithMutation}/*.png
    └── run_metrics/evaluation_<instance>.json
```

`run_metrics/` holds one JSON per generation or evaluation instance with timers, counters and phase data, plus consolidated fronts under `fronts/`; these feed the analyses in [Post-analysis](#post-analysis).

CSV columns are `<label>_<combination>_<NM|WM>_<solver>` (label = model or `None`, combination = Best/Middle/Worst/Standard/PMX/CX, NM/WM = mutation).

A budget directory appears only when a solver reached it; unreachable budgets are skipped.
