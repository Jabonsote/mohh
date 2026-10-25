
import argparse
import glob
import os

from mohh.main import make_experiment_path, run_generation
from mohh.run_evaluation import discover_models, extract_operators
from mohh.evaluation.main import execute_experiments
from mohh.evaluation.params import params_dict as evaluation_params_dict

def _generation_overrides(args) -> dict:

    overrides = {}

    if args.population_size is not None:
        overrides["POPULATION_SIZE"] = args.population_size
    if args.generations is not None:
        overrides["GENERATIONS"] = args.generations
    if args.elite_size is not None:
        overrides["ELITE_SIZE"] = args.elite_size
    if args.mo_population_size is not None:
        overrides["MO_POPULATION_SIZE"] = args.mo_population_size
    if args.mo_generations is not None:
        overrides["MO_GENERATIONS"] = args.mo_generations

    return overrides

def _parse_eval_budgets(value: str) -> dict:

    budgets = {}

    for pair in value.split(","):
        label, budget = pair.split(":")
        budgets[label] = int(budget)

    return budgets

def _evaluation_overrides(args, solvers: list, models: list) -> dict:

    overrides = {"SOLVERS": solvers, "MODELS": models}

    if args.mo_population_size is not None:
        overrides["MO_POPULATION_SIZE"] = args.mo_population_size
    if args.mo_generations is not None:
        overrides["MO_GENERATIONS"] = args.mo_generations
    if args.n_experiments is not None:
        overrides["N_EXPERIMENTS"] = args.n_experiments
    if args.eval_budgets is not None:
        overrides["EVAL_BUDGETS"] = _parse_eval_budgets(args.eval_budgets)
    if getattr(args, "dataset_path", None):
        overrides["DATASET_PATH"] = os.path.abspath(
            os.path.expanduser(args.dataset_path))
    if getattr(args, "dataset", None):
        overrides["DATASET"] = args.dataset
        if not getattr(args, "dataset_path", None):
            overrides["DATASET_PATH"] = None

    return overrides


def _resolve_test_instances(args):
    """
    Resuelve las instancias de test que debe ejecutar este proceso.

    Por defecto usa todas las `.txt` del test del dataset. Con `--instances` se
    pueden elegir nombres concretos y con `--shard-index/--shard-count` se
    reparte el conjunto entre varios jobs.
    """
    from mohh.core.utilities.paths import project_root

    if getattr(args, "dataset_path", None):
        dataset_root = os.path.abspath(
            os.path.expanduser(args.dataset_path))
    else:
        dataset = (getattr(args, "dataset", None)
                   or evaluation_params_dict["DATASET"])
        dataset_root = os.path.join(project_root(), "datasets", dataset)

    test_dir = os.path.join(dataset_root, "test")
    todas = sorted(glob.glob(os.path.join(test_dir, "*.txt")))
    if not todas:
        raise FileNotFoundError(
            f"no hay instancias .txt en {test_dir}")

    pedidas = getattr(args, "instances", None)
    if pedidas:
        elegidas = []
        for nombre in pedidas:
            ruta = nombre
            if not os.path.isfile(ruta):
                archivo = nombre if nombre.endswith(".txt") else nombre + ".txt"
                ruta = os.path.join(test_dir, archivo)
            if not os.path.isfile(ruta):
                raise FileNotFoundError(
                    f"no existe la instancia de test {nombre!r} "
                    f"(buscada en {test_dir})")
            elegidas.append(os.path.abspath(ruta))
        todas = elegidas

    count = int(getattr(args, "shard_count", 1) or 1)
    index = int(getattr(args, "shard_index", 0) or 0)
    if count < 1:
        raise ValueError("--shard-count debe ser >= 1")
    if not 0 <= index < count:
        raise ValueError("--shard-index debe estar en [0, shard-count)")

    return todas[index::count]

def _add_mo_args(parser: argparse.ArgumentParser):

    parser.add_argument("--mo-population-size", type=int, default=None)
    parser.add_argument("--mo-generations", type=int, default=None)

def _add_eval_budgets_arg(parser: argparse.ArgumentParser):

    parser.add_argument("--eval-budgets", default=None,
                        help="Comma-separated label:evaluations pairs, e.g. 10k:10500,30k:31500,50k:52500.")

def _add_solvers_arg(parser: argparse.ArgumentParser):

    parser.add_argument("--solvers", default=None,
                        help="Comma-separated MOEAs to benchmark with (default: from params).")

def generate():

    parser = argparse.ArgumentParser(description="Run the GE hyper-heuristic to generate variation operators.")
    parser.add_argument("--models", default="MOEAD", help="Comma-separated MOEA models to try (default: MOEAD).")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--elite-size", type=int, default=None)
    parser.add_argument("--experiment-path", default=None,
                        help="Carpeta de resultados fija (default: auto-numerar results/Experiment_NNN). "
                             "Necesaria para correr varios modelos en jobs SLURM paralelos sin colision.")
    _add_mo_args(parser)
    args = parser.parse_args()

    models = args.models.split(",")
    overrides = _generation_overrides(args)

    if args.experiment_path:
        experiment_path = args.experiment_path
        os.makedirs(experiment_path, exist_ok=True)
    else:
        experiment_path = make_experiment_path()
    run_generation(experiment_path, models, overrides)

    print(f"\nGeneration complete: {experiment_path}")

def evaluate():

    parser = argparse.ArgumentParser(description="Benchmark discovered operators against unseen instances.")
    parser.add_argument("experiment_path", help="Path to an experiment folder produced by mohh-generate.")
    parser.add_argument("--models", default=None,
                        help="Comma-separated generation models to pull operators from (default: auto-detect from experiment_path).")
    _add_solvers_arg(parser)
    parser.add_argument("--n-experiments", type=int, default=None)
    _add_mo_args(parser)
    _add_eval_budgets_arg(parser)
    parser.add_argument("--dataset-path", default=None,
                        help="Ruta al dataset con train/ y test/ (por defecto "
                             "<repo>/datasets/<DATASET>).")
    parser.add_argument("--dataset", default=None,
                        help="Nombre del dataset bajo <repo>/datasets/ "
                             "(por defecto el de params).")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="Instancias de test concretas (nombres o rutas). "
                             "Por defecto todas las .txt del test.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Índice de este shard (para job arrays).")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Número total de shards (para job arrays).")
    parser.add_argument("--no-figures", action="store_true",
                        help="No generar figuras (útil en shards paralelos).")
    parser.add_argument("--figures-only", action="store_true",
                        help="Sólo generar figuras de un evaluation_results ya calculado.")
    args = parser.parse_args()

    if args.no_figures and args.figures_only:
        parser.error("--no-figures y --figures-only son incompatibles")

    models = args.models.split(",") if args.models else discover_models(args.experiment_path)
    solvers = args.solvers.split(",") if args.solvers else evaluation_params_dict["SOLVERS"]
    overrides = _evaluation_overrides(args, solvers, models)

    if args.figures_only:
        from mohh.evaluation.params import set_params
        from mohh.evaluation.make_plots import make_figures
        set_params(overrides)
        ruta = os.path.join(args.experiment_path, "evaluation_results")
        print(f"Generando figuras de {ruta}")
        make_figures(ruta)
        return

    instances = _resolve_test_instances(args)
    if not instances:
        print("No hay instancias de test para este shard; nada que hacer.")
        return
    print(f"Instancias de test en este job: {len(instances)}")
    for ruta in instances:
        print(f"  - {os.path.basename(ruta)}")

    operators = extract_operators(args.experiment_path, models)
    execute_experiments(args.experiment_path, "", operators, overrides,
                        instances=instances, figures=not args.no_figures)

def run_full():

    parser = argparse.ArgumentParser(description="Run generation followed by evaluation end-to-end.")
    parser.add_argument("--full", action="store_true",
                        help="Confirm you want to run both stages (evaluation is expensive).")
    parser.add_argument("--models", default="MOEAD", help="Comma-separated MOEA models to try (default: MOEAD).")
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--elite-size", type=int, default=None)
    parser.add_argument("--n-experiments", type=int, default=None)
    _add_mo_args(parser)
    _add_eval_budgets_arg(parser)
    _add_solvers_arg(parser)
    args = parser.parse_args()

    if not args.full:
        parser.error("Pass --full to confirm you want to run generation followed by evaluation end-to-end (evaluation is expensive).")

    models = args.models.split(",")
    gen_overrides = _generation_overrides(args)
    solvers = args.solvers.split(",") if args.solvers else evaluation_params_dict["SOLVERS"]
    eval_overrides = _evaluation_overrides(args, solvers, models)

    experiment_path = make_experiment_path()
    phenotypes = run_generation(experiment_path, models, gen_overrides)

    execute_experiments(experiment_path, "", phenotypes, eval_overrides)
