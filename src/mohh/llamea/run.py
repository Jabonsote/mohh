"""CLI de la busqueda LLaMEA de operadores de cruce para mQAP.

Conecta LLaMEA con MOHH-R: el LLM propone `crossover(x, y)` y el fitness
reutiliza el MOEA interno y la metrica de hipervolumen del proyecto de
evolucion gramatical (mismos conjuntos `datasets/mqap/{train,test}`).

Un unico algoritmo por generacion por defecto ((1+1)-EA); para busqueda
poblacional usar `--n-parents/--n-offspring`.

Todos los artefactos quedan bajo ``results/<experimento>/``:

    results/<experimento>/
        llamea_best_operator.py      mejor operador encontrado
        llamea_best_info.json        resumen de la corrida
        run.json                     configuracion exacta + comando
        bounds.json                  limites de normalizacion (si --save-bounds)
        tokens.json                  tokens consumidos (acumulado + historico)
        llamea_logs/exp-*/           logs completos de LLaMEA
            log.jsonl                un registro por individuo evaluado
            conversationlog.jsonl    prompts y respuestas completas
            generation_summary.jsonl una linea por generacion
            code/                    codigo de cada candidato
            evolution.csv            tabla de la evolucion
            generations.csv          una fila por generacion (+tokens)
            report.md                reporte legible
            evolution.png            HV por generacion y mejor acumulado

Ejemplos:
    # contra un servidor vLLM local
    python -m mohh.llamea.run --model Qwen/Qwen2.5-1.5B-Instruct \
        --base-url http://localhost:8000/v1 --budget 100

    # en el cluster, contra el nodo del job vLLM
    python -m mohh.llamea.run --model Qwen/Qwen2.5-7B-Instruct \
        --base-url http://gpu-nodo:8000/v1 --wait-server --eval-test
"""

import argparse
import json
import logging
import os
import random
import shlex
import sys
import time
import traceback

import numpy as np

from mohh.core.utilities.instance_utils import instance_paths
from mohh.llamea import metrics as model_metrics
from mohh.llamea.fitness import EarlyStop, build_evaluator, init_params
from mohh.llamea.prompts import (
    ROLE_PROMPT,
    make_task_prompt,
    make_operators,
    EXAMPLE_PROMPT,
    SKELETON_PROMPT,
    OUTPUT_FORMAT_PROMPT,
)
from mohh.llamea.diversity import (
    solution_behavior_descriptor,
    solution_distance,
)
from mohh.llamea.reporting import (
    ResultsExperimentLogger,
    _atomic_write_json,
    write_report,
    write_seeds,
)
from mohh.llamea.statistics import add_holm, paired_summary
from mohh.llamea.vllm import VLLM_LLM, model_is_available, wait_for_server


class _EvalSolution:
    """Objeto minimo compatible con el contrato del fitness de LLaMEA."""

    def __init__(self, code):
        self.code = code
        self.fitness = float("nan")
        self.feedback = ""
        self.error = None

    def set_scores(self, fitness, feedback="", error=None):
        self.fitness = fitness
        self.feedback = feedback
        self.error = error


def _select_instances(paths, amount, seed):
    # `os.listdir` no garantiza orden entre sistemas de ficheros. Ordenar antes
    # de muestrear hace que el mismo `--seed` seleccione las mismas instancias en
    # el cluster y en local (la comparacion pareada depende de ello).
    ordered = sorted(str(path) for path in paths)
    if amount == "all":
        return ordered
    amount = int(amount)
    rng = random.Random(seed)
    return rng.sample(ordered, min(amount, len(ordered)))


def _merge_reference_scores(history, scores, rep):
    """Alinea los HV de referencia con la repeticion `rep` (incluye fallos).

    Un clasico puede faltar en una repeticion (p. ej. su MOEA falla). Se
    rellena con `None` hasta `rep`, se agrega el valor actual y se conserva la
    alineacion para el Wilcoxon pareado. No se rellena de mas: cada llamada
    agrega exactamente una posicion por clasico.
    """
    for name in scores:
        rows = history.setdefault(name, [])
        if len(rows) < rep:
            rows.extend([None] * (rep - len(rows)))
    for name in history:
        history[name].append(scores.get(name))
    return history


def _load_bounds(path):
    """Lee limites de normalizacion: ``{instancia: {vmin, vmax}}``.

    Acepta el formato plano o envuelto en ``{"instances": {...}}`` (el que
    escribe :func:`_save_bounds`).
    """
    with open(path) as file:
        data = json.load(file)
    data = data.get("instances", data)
    return {
        key: (np.asarray(value["vmin"], dtype=float),
              np.asarray(value["vmax"], dtype=float))
        for key, value in data.items()
    }


def _save_bounds(path, bounds, paths):
    payload = {
        "instances": {
            os.path.basename(instance): {
                "vmin": np.asarray(pair[0], dtype=float).tolist(),
                "vmax": np.asarray(pair[1], dtype=float).tolist(),
            }
            for instance, pair in zip(paths, bounds)
        }
    }
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def _reference_panel(names):
    """Subconjunto del panel de referencia de `mohh.llamea.reference`."""
    from mohh.llamea.reference import REFERENCE_OPERATORS

    if not names:
        return None
    selected = {}
    for name in names.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name not in REFERENCE_OPERATORS:
            raise ValueError(
                f"Operador de referencia desconocido: {name!r}. "
                f"Disponibles: {', '.join(REFERENCE_OPERATORS)}"
            )
        selected[name] = REFERENCE_OPERATORS[name]
    return selected or None


def _model_revision(model):
    """Revision local del modelo desde el manifiesto de `mohh.llamea.models`.

    Devuelve `None` si no hay manifiesto (p. ej. modelo servido por Ollama en
    otro nodo); no es un error.
    """
    try:
        from mohh.llamea.models import model_dir, read_manifest

        path = model_dir(model)
        manifest = read_manifest(path)
        if manifest:
            return {"revision": manifest.get("revision"),
                    "verified_at": manifest.get("verified_at"),
                    "path": path}
    except Exception:  # noqa: BLE001 - la procedencia no debe tumbar la corrida
        return None
    return None


def _make_evaluator(paths, args, bounds=None, archive=None, fronts_dir=None,
                    diversity_log=None):
    return build_evaluator(
        paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
        model_name=args.solver,
        mutation=args.mutation,
        instance_timeout=args.instance_timeout,
        normalize=args.normalize,
        bounds=bounds,
        reference_operators=_reference_panel(args.reference_operators),
        archive=archive,
        frame="stationary" if getattr(args, "stationary_frame", False) else "ge",
        fronts_dir=fronts_dir,
        repeat_penalty=getattr(args, "repeat_penalty", 0.5),
        patience=getattr(args, "patience", 0),
        diversity_feedback=getattr(args, "diversity_feedback", True),
        diversity_log=diversity_log,
        target_penalty=getattr(args, "target_penalty", 1.0),
        require_target_family=getattr(args, "require_target_family", False),
    )


def _llamea_kwargs(args):
    """Mecanismos de diversidad que LLaMEA 1.4 trae y la busqueda base no usaba.

    Atacan directamente el anclaje observado (el modelo repite el mismo
    operador): seleccion por torneo (LLaMEA usa `random` por defecto), mutacion
    adaptativa, guia por caracteristicas del AST, niching (clearing/novelty/
    map_elites con el descriptor conductual del evaluador) y tamano del archivo
    de novedad. No se toca el nucleo GE ni el motor de LLaMEA.
    """
    kwargs = {
        "parent_selection": args.parent_selection,
        "tournament_size": args.tournament_size,
        "adaptive_mutation": args.adaptive_mutation,
        "feature_guided_mutation": args.feature_guided_mutation,
        "novelty_k": args.novelty_k,
        # Hooks nativos de LLaMEA: la distancia y el descriptor leen la huella
        # conductual que el evaluador deja en `Solution.metadata`. Asi
        # `--niching novelty`, `clearing` y `map_elites` comparan lo que hace el
        # operador, no cuantas lineas o tokens tiene el codigo. Si un candidato
        # no tiene fingerprint se cae a la distancia de codigo original.
        "distance_metric": solution_distance,
        "behavior_descriptor": solution_behavior_descriptor,
    }
    if args.novelty_archive_size is not None:
        kwargs["novelty_archive_size"] = args.novelty_archive_size
    if args.niching and args.niching != "none":
        kwargs["niching"] = args.niching
        if args.niche_radius is not None:
            kwargs["niche_radius"] = args.niche_radius
        kwargs["adaptive_niche_radius"] = args.adaptive_niche_radius
        if args.clearing_interval is not None:
            kwargs["clearing_interval"] = args.clearing_interval
    return kwargs


def _setup_logging(exp_dir):
    """Log de la corrida: `llamea.log` (todo) y `error.log` (solo avisos/errores).

    En el cluster los `.out`/`.err` de SLURM se buffean y un job puede morir sin
    dejar rastro. Estos dos archivos quedan dentro del experimento y `error.log`
    sobrevive incluso a un fallo temprano (servidor caido, modelo ausente), que
    es justo lo que hay que diagnosticar cuando "el cluster no funciona".
    """
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger("mohh.llamea")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    full = logging.FileHandler(os.path.join(exp_dir, "llamea.log"), encoding="utf-8")
    full.setLevel(logging.INFO)
    full.setFormatter(formatter)
    errors = logging.FileHandler(os.path.join(exp_dir, "error.log"), encoding="utf-8")
    errors.setLevel(logging.WARNING)
    errors.setFormatter(formatter)
    logger.addHandler(full)
    logger.addHandler(errors)
    return logger


def _start_stall_watchdog(exp_dir, logger, timeout):
    """Aborta la corrida si pasan `timeout` s sin una generacion nueva.

    Vigila la fecha de `checkpoint.json` (reescrito cada generacion). Si el
    proceso se cuelga (ya paso una vez: una generacion quedo 17 min en estado R;
    ver `areas-de-mejora.md` §14), el checkpoint queda viejo y este hilo lo
    detecta, lo deja en `error.log` y termina el proceso. El mejor operador ya
    esta guardado en `best_so_far.py`, asi que un cuelgue no pierde la corrida.
    """
    if not timeout or timeout <= 0:
        return None
    import threading

    checkpoint = os.path.join(exp_dir, "checkpoint.json")

    def watch():
        while True:
            time.sleep(min(max(timeout / 4.0, 5.0), 120.0))
            if not os.path.exists(checkpoint):
                continue
            stalled = time.time() - os.path.getmtime(checkpoint)
            if stalled > timeout:
                logger.error(
                    "Sin progreso desde hace %.0f s (limite %s s); se aborta. "
                    "El mejor operador esta en best_so_far.py.", stalled, timeout)
                logging.shutdown()
                os._exit(2)

    thread = threading.Thread(target=watch, name="stall-watchdog", daemon=True)
    thread.start()
    return thread


def run_llamea(args):
    # 0. Inicializar los defaults de MOHH-R (DATASET, PROBLEM_NAME, ...) para
    #    poder listar los mismos conjuntos train/test que la busqueda GE.
    init_params(RANDOM_SEED=args.seed)

    run_start = time.time()
    exp_dir = os.path.join(args.results_dir, args.experiment_name)
    log_base = os.path.join(exp_dir, "llamea_logs")
    logger = _setup_logging(exp_dir)
    os.makedirs(log_base, exist_ok=True)
    _start_stall_watchdog(exp_dir, logger, getattr(args, "stall_timeout", 0.0))
    logger.info("Inicio de la corrida: model=%s base_url=%s budget=%s",
                args.model, args.base_url, args.budget)

    # 1. Modelo local: verificar/descargar antes de conectar. Con
    #    --no-download (uso cliente en el cluster) solo se informa: el modelo
    #    lo garantiza el job vLLM y puede no estar en el cache de este nodo.
    if args.no_download:
        if model_is_available(args.model):
            print(f"Modelo presente en este nodo: {args.model}")
            logger.info("Modelo presente en este nodo: %s", args.model)
        else:
            message = (f"'{args.model}' no esta en el cache de HuggingFace de "
                       "este nodo; se asume que lo sirve el endpoint de --base-url "
                       "(vLLM en el cluster, Ollama o llama.cpp en local).")
            print(f"AVISO: {message}")
            logger.warning("AVISO: %s", message)
    else:
        try:
            from mohh.llamea import models as model_store

            # Pesos en la raiz persistente (LUSTRE/$SCRATCH), no en $HOME:
            # el ambiente del cluster borra el home y los nodos no lo ven.
            model_store.configure_hf_env()
            model_store.download_model(args.model)
        except Exception:
            logger.exception("Fallo al verificar/descargar el modelo %s", args.model)
            raise

    # 2. Servidor vLLM: opcionalmente esperar a que cargue el modelo.
    if args.wait_server:
        print(f"Esperando servidor en {args.base_url} (max {args.server_timeout}s)...")
        if not wait_for_server(args.base_url, timeout=args.server_timeout):
            logger.error("El servidor no respondio en %s tras %ss",
                         args.base_url, args.server_timeout)
            raise RuntimeError(f"El servidor no respondio en {args.base_url}")
        logger.info("Servidor disponible en %s", args.base_url)

    # 2b. Procedencia del servidor y del modelo: requisito de reproducibilidad
    #     (el mismo modelo en vLLM vs Ollama puede dar resultados distintos).
    server_info = model_metrics.server_version(args.base_url)
    model_info = _model_revision(args.model)
    print(f"Servidor: {server_info or 'sin endpoint /version'}")
    print(f"Modelo local: {model_info or 'sin manifiesto local'}")

    # 3. Fitness con la logica actual de MOHH-R (train).
    train_paths = _select_instances(instance_paths(train=True), args.train_instances, args.seed)
    print(f"Instancias de train: {len(train_paths)}")

    bounds = _load_bounds(args.bounds_file) if args.bounds_file else None
    fronts_dir = None if args.no_save_fronts else os.path.join(exp_dir, "fronts")
    fitness = _make_evaluator(
        train_paths, args, bounds=bounds, fronts_dir=fronts_dir,
        diversity_log=os.path.join(exp_dir, "diversity.jsonl"),
    )

    config = {
        "command": shlex.join([sys.executable] + sys.argv),
        "model": args.model,
        "base_url": args.base_url,
        "budget": args.budget,
        "n_parents": args.n_parents,
        "n_offspring": args.n_offspring,
        "combine": args.combine,
        "behavioral_parent_selection": args.behavioral_parent_selection,
        "novelty_weight": args.novelty_weight,
        "target_penalty": args.target_penalty,
        "require_target_family": args.require_target_family,
        "solver": args.solver,
        "mo_generations": args.mo_generations,
        "mo_population_size": args.mo_population_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "sampling_repetition_penalty": args.sampling_repetition_penalty,
        "thinking": args.thinking,
        "operator_memory": args.operator_memory,
        "max_tokens": args.max_tokens,
        "request_log": os.path.join(exp_dir, "llm_requests.jsonl"),
        "prompt_log_root": log_base,
        "prompt_log": None,
        "train_instances": len(train_paths),
        "normalize": args.normalize,
        "bounds_file": args.bounds_file,
        "reference_operators": args.reference_operators,
        "refine_only": args.refine_only,
        "example": ("none" if args.no_example else
                    "skeleton" if args.skeleton_example else "ox"),
        "comma": args.comma,
        "repeat_penalty": args.repeat_penalty,
        "diversity_feedback": args.diversity_feedback,
        "patience": args.patience,
        "parent_selection": args.parent_selection,
        "tournament_size": args.tournament_size,
        "adaptive_mutation": args.adaptive_mutation,
        "feature_guided_mutation": args.feature_guided_mutation,
        "niching": args.niching,
        "niche_radius": args.niche_radius,
        "novelty_k": args.novelty_k,
        "novelty_archive_size": args.novelty_archive_size,
        "eval_repeats": args.eval_repeats,
        "max_workers": args.max_workers,
        "results_dir": args.results_dir,
        "log_dir": log_base,
        # Procedencia (reproducibilidad entre local y cluster).
        "server_info": server_info,
        "model_info": model_info,
        "stall_timeout": args.stall_timeout,
    }
    with open(os.path.join(exp_dir, "run.json"), "w") as file:
        json.dump(config, file, indent=2)

    # 3c. Semillas y trazabilidad del proceso. Se guardan siempre, en su propio
    #     archivo y dentro de cada registro (individuos y generaciones).
    seeds = {
        "search": args.seed,
        "instance_selection": args.seed,
        "moea_internal": args.seed,
        "moea_seed_policy": ("np.random.seed(search_seed) antes de construir las "
                             "soluciones iniciales y antes de cada corrida del MOEA "
                             "interno; la misma semilla para todas las instancias y "
                             "candidatos, como el arnes de evaluacion"),
        "reference_panel": args.seed,
        "eval_test": args.seed,
        "llm_sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.sampling_repetition_penalty,
            "max_tokens": args.max_tokens,
            "thinking": args.thinking,
            "operator_memory": args.operator_memory,
            "note": ("el servidor muestrea; sin semilla fija, asi que la busqueda no "
                     "es reproducible bit a bit aunque el MOEA si lo sea"),
        },
        "purpose": {
            "search": "MOEA interno, soluciones iniciales, seleccion de instancias y panel",
            "llm_sampling": "muestreo del LLM (no determinista)",
        },
    }
    write_seeds(os.path.join(exp_dir, "seeds.json"), seeds)
    logger.info("Semillas: %s", seeds)

    # 3d. `--dry-run`: validar servidor + configuracion y salir sin gastar GPU.
    if args.dry_run:
        print("DRY RUN: configuracion y procedencia escritas en run.json/seeds.json; "
              "no se ejecuta la busqueda.")
        if server_info is None:
            print("AVISO: el servidor no respondio a /version; verifica --base-url "
                  "antes de lanzar una corrida larga.")
            logger.warning("dry-run: el servidor no respondio a /version")
        logger.info("dry-run: fin")
        return

    # 4. Redirigir el logger de LLaMEA a results/<exp>/llamea_logs/.
    import llamea.llamea as llamea_module

    ResultsExperimentLogger.results_dir = log_base
    ResultsExperimentLogger.verbose = not args.quiet
    # Semilla y configuracion del proceso: cada linea de individuals.jsonl y
    # generation_summary.jsonl la lleva, para que el log sea autosuficiente.
    ResultsExperimentLogger.seed = args.seed
    ResultsExperimentLogger.solver = args.solver
    ResultsExperimentLogger.mo_generations = args.mo_generations
    ResultsExperimentLogger.mo_population_size = args.mo_population_size
    ResultsExperimentLogger.mutation = args.mutation
    llamea_module.ExperimentLogger = ResultsExperimentLogger

    llm = VLLM_LLM(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.request_timeout,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        sampling_repetition_penalty=args.sampling_repetition_penalty,
        operator_memory=args.operator_memory,
        thinking=(None if args.thinking == "auto" else args.thinking == "on"),
        request_log_path=os.path.join(exp_dir, "llm_requests.jsonl"),
        request_context={
            "experiment": args.experiment_name,
            "solver": args.solver,
            "mutation": bool(args.mutation),
            "combine": bool(args.combine),
            "seed": args.seed,
        },
    )
    ResultsExperimentLogger.token_source = llm
    # Diversidad conductual por generacion (mecanismos distintos, entropia,
    # novedad y cobertura del archivo). El evaluador ya la acumula.
    ResultsExperimentLogger.diversity_source = fitness

    from llamea import LLaMEA

    if args.no_example:
        # Ojo: LLaMEA interpreta example_prompt=None como *su* ejemplo por
        # defecto (una busqueda aleatoria, de otro dominio). Para no anclar el
        # prompt a nada hay que pasar cadena vacia.
        example_prompt = ""
    elif args.skeleton_example:
        example_prompt = SKELETON_PROMPT
    else:
        example_prompt = EXAMPLE_PROMPT

    search_class = LLaMEA
    search_kwargs = {}
    if args.combine:
        # Extension propia: seleccion conductual de los dos padres y
        # validacion de la familia objetivo. No se toca el LLaMEA instalado.
        from mohh.llamea.behavioral import BehavioralLLaMEA

        search_class = BehavioralLLaMEA
        search_kwargs = {
            "behavioral_parent_selection": args.behavioral_parent_selection,
            "novelty_weight": args.novelty_weight,
        }

    llamea = search_class(
        f=fitness,
        llm=llm,
        n_parents=args.n_parents,
        n_offspring=args.n_offspring,
        role_prompt=ROLE_PROMPT,
        task_prompt=make_task_prompt(args.solver),
        example_prompt=example_prompt,
        output_format_prompt=OUTPUT_FORMAT_PROMPT,
        operators=([] if args.refine_only
                   else make_operators(include_combine=args.combine)),
        budget=args.budget,
        elitism=not args.comma,
        minimization=False,
        max_workers=args.max_workers,
        experiment_name=args.experiment_name,
        **_llamea_kwargs(args),
        **search_kwargs,
    )

    # 3b. Costo del modelo: foto de `/metrics` (vLLM) antes de la corrida; los
    #     deltas contra la foto final dan el tiempo de GPU y los tokens reales.
    metrics_before = model_metrics.scrape_prometheus(args.base_url)
    if metrics_before is not None:
        print("Metricas Prometheus del servidor: disponibles (vLLM).")
    else:
        print("Metricas Prometheus no disponibles; se usaran solo las del cliente "
              "(tokens y latencia).")

    if args.patience:
        print(f"Parada temprana: se corta tras {args.patience} evaluaciones sin "
              "mejora del mejor HV.")
    stop_reason = "budget reached"
    try:
        best = llamea.run()
    except EarlyStop as error:
        # Parada temprana: LLaMEA no llego a devolver nada, pero el evaluador ya
        # tiene el mejor operador guardado y el checkpoint esta en disco.
        stop_reason = f"early stop (patience={args.patience})"
        best = fitness.best_solution()
        logger.warning("Parada temprana: %s", error)
        if best is None:
            raise
        print(f"Parada temprana: {error}")
    train_fitness = best.fitness
    print(f"Best train fitness (HV): {train_fitness}")
    print(f"Best code:\n{best.code}")

    log_dir = llamea.logger.dirname if llamea.logger is not None else log_base

    # 4b. Costo del modelo: tokens y latencia (cliente) + tiempo de GPU y KV
    #     cache (servidor vLLM, `/metrics`) + GPU local (`nvidia-smi`).
    metrics_after = model_metrics.scrape_prometheus(args.base_url)
    gpu_snapshot = model_metrics.nvidia_smi()
    usage = llm.usage_snapshot() if hasattr(llm, "usage_snapshot") else {}
    cost = model_metrics.cost_summary(
        usage, before=metrics_before, after=metrics_after,
        gpu=gpu_snapshot, wall_seconds=time.time() - run_start,
    )
    if metrics_before is not None or metrics_after is not None:
        print(f"Costo del modelo (vLLM): {cost}")
    else:
        print(f"Costo del modelo (solo cliente): {cost}")

    # 5. Guardar resultado.
    with open(os.path.join(exp_dir, "llamea_best_operator.py"), "w") as file:
        file.write(best.code + "\n")

    info = dict(config)
    counters = (llamea.logger.counters()
                if llamea.logger is not None and hasattr(llamea.logger, "counters")
                else {})
    diversity_stats = fitness.duplicate_stats()
    info.update({
        "train_fitness": train_fitness,
        # Aclaracion para quien lea el JSON: `train_fitness` es el valor que
        # tenia el mejor operador cuando se evaluo (el marco GE crece despues),
        # asi que NO es comparable entre corridas. El numero comparable es
        # `train_fitness_final_frame`, con la linea base en el mismo marco.
        "train_fitness_frame": (
            "frame at the time of evaluation (not comparable across runs)"
            if fitness.normalize else "raw hypervolume"
        ),
        "description": best.description,
        "log_dir": log_dir,
        "request_log": os.path.join(exp_dir, "llm_requests.jsonl"),
        "prompt_log": os.path.join(log_dir, "prompts"),
        "evaluations": len(fitness.archive),
        "evaluated_total": diversity_stats.get("evaluations"),
        "best_hv_ranking": fitness.archive.best(),
        "diversity": diversity_stats,
        # Trazabilidad del proceso: semillas, diversidad y costo.
        "seeds": seeds,
        "seeds_file": os.path.join(exp_dir, "seeds.json"),
        "distinct_operators": counters.get("distinct_operators"),
        "evaluated_valid": diversity_stats.get("valid_evaluations"),
        # Tasa de validez real: evaluaciones con HV definido / evaluaciones
        # totales. Antes se dividia por `budget` y podia superar 100%.
        "validity_rate": diversity_stats.get("validity_rate"),
        "stop_reason": stop_reason,
        "cost": cost,
    })
    # Traza completa de diversidad (fingerprints incluidos) para auditoria y
    # para regenerar el mapa conductual sin volver a ejecutar el LLM/MOEA.
    _atomic_write_json(os.path.join(exp_dir, "diversity.json"), {
        "summary": diversity_stats,
        "entries": fitness.diversity_entries(),
    })

    # 5b. Marco de normalizacion final. El HV del mejor candidato y el de la
    #     linea base clasica se miden en el MISMO marco (el de la evaluacion GE:
    #     frente consolidado del panel y de todos los candidatos evaluados), que
    #     es lo que los hace comparables. Si el marco cambio tras evaluar al
    #     mejor, el argmax en el marco final puede ser otro candidato: se guarda
    #     aparte para no perder esa informacion.
    if fitness.normalize:
        final_scores = fitness.final_scores()
        info.update({
            "frame": fitness.frame,
            "degenerate_instances": fitness.frame_degenerate_instances(),
            "reference_hv_final_frame": fitness.reference_scores(),
            "train_fitness_final_frame": final_scores.get(best.code.strip(), (None,))[0],
            # HV crudo del operador entregado (el ranking pudo penalizarlo si se
            # repitio; el HV de la tabla no se penaliza).
            "train_fitness_raw": final_scores.get(best.code.strip(), (None,))[0],
        })
        best_key = fitness.best_key_in_final_frame()
        if best_key is not None and best_key != (best.code or "").strip():
            info["best_operator_final_frame"] = {
                "train_fitness": final_scores[best_key][0],
                "saved_as": "llamea_best_operator_final_frame.py",
            }
            with open(os.path.join(exp_dir, "llamea_best_operator_final_frame.py"), "w") as file:
                file.write(best_key + "\n")
    else:
        info["frame"] = "none"

    # 5c. Marco final y frentes crudos: evidencia para recalcular el HV.
    if fronts_dir:
        try:
            frame_path = fitness.save_frame(os.path.join(fronts_dir, "frame.npz"))
            if frame_path:
                info["frame_file"] = frame_path
            info["fronts_dir"] = fronts_dir
        except Exception as error:  # noqa: BLE001
            logger.exception("No se pudo guardar el marco final: %s", error)

    if args.save_bounds:
        try:
            computed = fitness.ensure_reference_bounds()
            if computed is not None:
                _save_bounds(os.path.join(exp_dir, args.save_bounds), computed, train_paths)
                info["bounds_file_written"] = os.path.join(exp_dir, args.save_bounds)
        except Exception as error:  # noqa: BLE001 - no debe tumbar la corrida
            print(f"AVISO: no se pudieron guardar los limites: {error}")

    # 6. Evaluacion opcional sobre el conjunto de TEST (el mismo split que
    #    usa la evaluacion de MOHH-R; nunca se usa durante la busqueda).
    if args.eval_test:
        test_paths = _select_instances(instance_paths(train=False), args.test_instances, args.seed)
        repeats = max(1, args.eval_repeats)
        print(f"Evaluando el mejor operador en {len(test_paths)} instancias de test "
              f"({repeats} repeticion(es), marco fijo del panel)...")
        test_fronts_dir = (None if args.no_save_fronts
                           else os.path.join(exp_dir, "fronts_test"))
        hvs, rep_seeds = [], []
        candidate_by_rep = []          # alineado con reference_by_rep, incluye fallos
        reference_by_rep = {}          # clasico -> [HV por repeticion]
        test_failures = []
        degenerate = None
        test_frame = ("stationary (panel de referencia)" if args.normalize
                      else "none (HV crudo)")
        for rep in range(repeats):
            # El marco final se fija al panel de referencia (`stationary`) en
            # TODAS las repeticiones, no solo cuando hay varias: un marco que
            # incluye al propio candidato lo favorece y no es un metro fijo.
            # El marco GE completo (todas las configuraciones) lo reconstruye
            # `scripts/compare_operators.py`; aqui solo se reporta candidato vs
            # clasicos con el mismo criterio y las mismas semillas.
            rep_args = argparse.Namespace(**vars(args))
            rep_args.seed = args.seed + rep * 7919
            rep_args.patience = 0
            rep_args.repeat_penalty = 1.0
            rep_args.stationary_frame = True
            rep_dir = (test_fronts_dir if (test_fronts_dir is None or repeats == 1)
                       else os.path.join(test_fronts_dir, f"rep-{rep}"))
            test_fitness = _make_evaluator(test_paths, rep_args, fronts_dir=rep_dir)
            evaluated = None
            failure_reason = None
            try:
                evaluated = test_fitness(_EvalSolution(best.code))
            except Exception as error:  # noqa: BLE001 - se reporta y sigue
                logger.exception("Fallo la evaluacion en test (rep %d)", rep)
                failure_reason = f"{type(error).__name__}: {error}"
            if evaluated is None:
                failure_reason = failure_reason or "sin resultado"
            elif not np.isfinite(evaluated.fitness):
                failure_reason = "fitness no finito"
            if failure_reason is None:
                value = float(evaluated.fitness)
            else:
                # Mismo criterio que la busqueda y que compare_operators.py:
                # un fallo es HV 0, no un par perdido (descartarlo sesgaria).
                value = 0.0
                test_failures.append({"repeat": rep, "seed": rep_args.seed,
                                      "reason": failure_reason})
            hvs.append(value)
            candidate_by_rep.append(value)
            rep_seeds.append(rep_args.seed)

            if test_fitness.normalize:
                try:
                    reference = test_fitness.reference_scores()
                except Exception:  # noqa: BLE001 - no perder el informe final
                    logger.exception("No se pudieron calcular los HV de referencia "
                                     "en test (rep %d)", rep)
                    reference = {}
                # Si un clasico aparece por primera vez en una repeticion
                # posterior, las anteriores cuentan como no pareadas.
                _merge_reference_scores(reference_by_rep, reference, rep)
                degenerate = test_fitness.frame_degenerate_instances()
            if rep_dir:
                try:
                    test_fitness.save_frame(os.path.join(rep_dir, "frame.npz"))
                except Exception:  # noqa: BLE001
                    logger.exception("No se pudo guardar el marco de test")
        if hvs:
            info["test_instances"] = len(test_paths)
            info["test_fitness"] = float(np.mean(hvs))
            info["test_fitness_repeats"] = {
                "n": len(hvs),
                "attempts": len(candidate_by_rep),
                "failed": len(test_failures),
                "values": [round(v, 6) for v in hvs],
                "mean": float(np.mean(hvs)),
                "sd": float(np.std(hvs, ddof=1)) if len(hvs) > 1 else 0.0,
                "min": float(np.min(hvs)),
                "max": float(np.max(hvs)),
                "frame": test_frame,
                "seeds": rep_seeds,
            }
            info["test_frame"] = test_frame
            if degenerate is not None:
                info["test_degenerate_instances"] = sorted(set(degenerate))
            info["test_seeds"] = {
                "moea_internal": rep_seeds,
                "selection": args.seed,
                "policy": ("seeds = base + rep*7919; el candidato y todos los "
                           "clasicos usan la misma poblacion inicial en cada rep"),
            }
            if test_failures:
                info["test_failures"] = test_failures
                print(f"AVISO: {len(test_failures)} evaluacion(es) de test "
                      "fallaron y se contaron como HV 0.")
            if reference_by_rep:
                reference_mean = {
                    name: float(np.mean([v for v in values if v is not None]))
                    for name, values in reference_by_rep.items()
                    if any(v is not None for v in values)
                }
                info["test_reference_hv"] = reference_mean
                info["test_reference_hv_repeats"] = reference_by_rep
                test_comparison = []
                for name, values in reference_by_rep.items():
                    summary = paired_summary(candidate_by_rep, values)
                    if summary["pairs"] == 0:
                        continue
                    summary.update({"operator": "llamea_best", "baseline": name})
                    test_comparison.append(summary)
                add_holm(test_comparison)
                info["test_comparison"] = test_comparison
            print(f"Test fitness (HV, {len(hvs)} rep): {info['test_fitness']:.6f} "
                  f"(sd {info['test_fitness_repeats']['sd']:.6f})")
            if info.get("test_reference_hv"):
                best_name, best_value = max(
                    info["test_reference_hv"].items(), key=lambda item: item[1]
                )
                print(f"Mejor clasico en test, mismo marco: "
                      f"{best_name.upper()} = {best_value:.6f}")
                rows = info.get("test_comparison") or []
                for row in rows:
                    if row["baseline"] == best_name:
                        p = row.get("wilcoxon_p_holm")
                        p_text = "-" if p is None else f"{p:.4f}"
                        print(f"Candidato vs {best_name.upper()}: "
                              f"{row['wins']}/{row['pairs']} victorias, "
                              f"dif {row['mean_difference']:+.6f}, "
                              f"p Holm={p_text}")

    with open(os.path.join(exp_dir, "llamea_best_info.json"), "w") as file:
        json.dump(info, file, indent=2)

    # 7. Tokens consumidos por el LLM (acumulado y por peticion).
    if hasattr(llm, "usage_snapshot"):
        usage = llm.usage_snapshot()
        usage["history"] = getattr(llm, "usage_history", [])
        usage["max_tokens"] = args.max_tokens
        usage["temperature"] = args.temperature
        usage["top_p"] = args.top_p
        usage["top_k"] = args.top_k
        usage["sampling_repetition_penalty"] = args.sampling_repetition_penalty
        usage["thinking"] = args.thinking
        usage["request_log"] = os.path.join(exp_dir, "llm_requests.jsonl")
        usage["operator_memory"] = (
            llm.operator_memory_snapshot()
            if hasattr(llm, "operator_memory_snapshot") else {}
        )
        usage["seeds"] = seeds
        with open(os.path.join(exp_dir, "tokens.json"), "w") as file:
            json.dump(usage, file, indent=2)
        print(f"tokens: {llm.usage_snapshot()}")

    # 7b. Metricas de costo (tokens, tiempo de GPU, energia) + fotos crudas.
    try:
        model_metrics.write_metrics(
            os.path.join(exp_dir, "metrics.json"),
            summary=cost,
            before=metrics_before,
            after=metrics_after,
            history=getattr(llm, "usage_history", []),
            extra={"seeds": seeds, "gpu_snapshot": gpu_snapshot,
                   "request_log": os.path.join(exp_dir, "llm_requests.jsonl"),
                   "thinking": args.thinking},
        )
        info["metrics_file"] = os.path.join(exp_dir, "metrics.json")
    except Exception:  # noqa: BLE001
        logger.exception("No se pudo guardar metrics.json")

    # El info final incluye el costo ya calculado; se reescribe para que
    # llamea_best_info.json sea autosuficiente.
    with open(os.path.join(exp_dir, "llamea_best_info.json"), "w") as file:
        json.dump(info, file, indent=2)

    # 8. Reporte de la evolucion (CSV, Markdown y grafica).
    best_final = (info.get("best_operator_final_frame") or {}).get("train_fitness")
    if best_final is None:
        best_final = info.get("train_fitness_final_frame")
    frame_info = {
        "frame": info.get("frame"),
        "degenerate_instances": info.get("degenerate_instances"),
        "reference_hv_final_frame": info.get("reference_hv_final_frame"),
        "best_hv_final_frame": best_final,
        "test_fitness": info.get("test_fitness"),
        "test_reference_hv": info.get("test_reference_hv"),
        "test_comparison": info.get("test_comparison"),
        "test_failures": info.get("test_failures"),
        "seeds": seeds,
        "stop_reason": info.get("stop_reason"),
        "distinct_operators": info.get("distinct_operators"),
        "validity_rate": info.get("validity_rate"),
    }
    paths = write_report(log_dir, title=f"LLaMEA {args.experiment_name}",
                         frame_info=frame_info,
                         cost=cost,
                         usage_history=getattr(llm, "usage_history", []),
                         request_log=os.path.join(exp_dir, "llm_requests.jsonl"))
    for kind, path in paths.items():
        print(f"{kind}: {path}")

    print(f"Guardado en {exp_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Busqueda LLaMEA de operadores de cruce para mQAP (conectada a MOHH-R)."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="ID del modelo local (HuggingFace) o ruta.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1",
                        help="Endpoint OpenAI-compatible del servidor (vLLM).")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling (top_p). Qwen recomienda 0.8 para "
                             "sus modelos Coder; None = default del servidor.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="top_k (extension de vLLM; va en extra_body). Qwen "
                             "recomienda 20 para sus modelos Coder.")
    parser.add_argument("--sampling-repetition-penalty", type=float, default=None,
                        help="repetition_penalty del muestreo (extension de vLLM). "
                             "No confundir con --repeat-penalty, que penaliza en el "
                             "ranking los operadores repetidos. Qwen recomienda 1.05.")
    parser.add_argument("--operator-memory", type=int, default=0,
                        help="Numero de mecanismos recientes que se recuerdan en el "
                             "prompt para evitar el anclaje (0 = desactivado). El job "
                             "del cluster usa 12.")
    parser.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                        help="Modo de razonamiento del modelo. auto no toca el "
                             "template del servidor; on/off fuerza "
                             "chat_template_kwargs.enable_thinking (Qwen3 y "
                             "similares) y registra reasoning_content. Con on, "
                             "sube --max-tokens para no gastar el presupuesto "
                             "razonando.")
    parser.add_argument("--request-timeout", type=float, default=600,
                        help="Timeout por peticion al LLM (segundos).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximo de tokens generados por peticion "
                             "(None = default del servidor).")
    parser.add_argument("--budget", type=int, default=1000,
                        help="Numero de EVALUACIONES/llamadas al LLM, no de "
                             "generaciones: generaciones ~= (budget - n_parents) / "
                             "n_offspring. Default 1000; para humo usar 20.")
    parser.add_argument("--n-parents", type=int, default=1,
                        help="Tamano de la poblacion de padres (default 1). Un "
                             "combine de dos padres requiere >= 2.")
    parser.add_argument("--n-offspring", type=int, default=1,
                        help="Descendientes/evaluaciones por generacion (default 1).")
    parser.add_argument("--combine", action="store_true",
                        help="Agregar el operador propio de recombinacion conductual "
                             "de dos padres (requiere --n-parents >= 2 y "
                             "--max-workers 1). Los dos padres se eligen por "
                             "diversidad conductual y el hijo se valida contra la "
                             "familia objetivo.")
    parser.add_argument("--no-behavioral-parent-selection",
                        dest="behavioral_parent_selection", action="store_false",
                        default=True,
                        help="Con --combine, elegir el padre focal solo por fitness "
                             "(comportamiento de LLaMEA original).")
    parser.add_argument("--novelty-weight", type=float, default=0.5,
                        help="Peso del bono de novedad en el torneo conductual "
                             "(default 0.5).")
    parser.add_argument("--target-penalty", type=float, default=1.0,
                        help="Multiplicador del ranking cuando el hijo no alcanza la "
                             "familia objetivo (default 1.0 = solo feedback; 0.0 "
                             "hace que un fallo no pueda ganar). El HV crudo no "
                             "cambia.")
    parser.add_argument("--require-target-family", action="store_true",
                        help="Rechazar un hijo de combine que no pertenezca a la "
                             "familia objetivo antes de gastar el MOEA interno.")
    parser.add_argument("--comma", action="store_true",
                        help="Estrategia (1,1) comma en lugar de (1+1) con elitismo.")
    parser.add_argument("--refine-only", action="store_true",
                        help="Usar solo el operador por defecto de LLaMEA (refinar); "
                             "sin el operador de rediseno (exploracion).")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--no-example", action="store_true",
                              help="Ablacion: omitir el ejemplo del prompt.")
    prompt_group.add_argument("--skeleton-example", action="store_true",
                              help="Ablacion: usar un esqueleto de interfaz en vez "
                                   "del ejemplo OX completo.")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Paralelismo interno de LLaMEA para evaluar individuos.")
    parser.add_argument("--solver", default="MOEAD",
                        help="MOEA interno: MOEAD, NSGAII o SMSEMOA.")
    parser.add_argument("--mutation", action="store_true",
                        help="Agregar mutacion al MOEA interno (default: solo cruce). "
                             "Usa Swap_Mutation, la mutacion de permutacion de "
                             "experiment.py; el nucleo GE no se modifica.")
    parser.add_argument("--train-instances", default="8",
                        help="Numero de instancias train, o 'all'.")
    parser.add_argument("--test-instances", default="8",
                        help="Numero de instancias test (con --eval-test), o 'all'.")
    parser.add_argument("--mo-generations", type=int, default=30)
    parser.add_argument("--mo-population-size", type=int, default=30)
    parser.add_argument("--instance-timeout", type=float, default=300.0,
                        help="Segundos maximos por instancia del MOEA interno "
                             "(corta operadores que no terminan). Subirlo si se "
                             "usan MO_GENERATIONS/MO_POPULATION_SIZE grandes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="LLaMEA_Experiment")
    parser.add_argument("--results-dir", default="results",
                        help="Carpeta raiz de todos los artefactos generados.")
    parser.add_argument("--normalize", dest="normalize", action="store_true",
                        default=True,
                        help="HV normalizado como el proyecto original (default).")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="HV con nadir max(frente)*1.1 (no comparable con GE).")
    parser.add_argument("--stationary-frame", action="store_true",
                        help="Normaliza solo con el panel de referencia (HV fijos). "
                             "Por defecto el marco es el de la evaluacion GE: frente "
                             "consolidado del panel y de todos los candidatos ya "
                             "evaluados, que es lo que hace comparables los HV.")
    parser.add_argument("--bounds-file", default=None,
                        help="JSON con limites de normalizacion por instancia "
                             "(util para reutilizar los del proyecto GE).")
    parser.add_argument("--save-bounds", default=None, const="bounds.json", nargs="?",
                        help="Guardar los limites calculados (nombre dentro de la "
                             "carpeta del experimento).")
    parser.add_argument("--no-save-fronts", action="store_true",
                        help="No guardar los frentes crudos por candidato "
                             "(por defecto se guardan en <exp>/fronts/).")
    parser.add_argument("--repeat-penalty", type=float, default=0.5,
                        help="Presion de diversidad: multiplicador del HV (solo para "
                             "el ranking) cuando el codigo ya se habia evaluado. 1.0 "
                             "lo desactiva. Ataca el anclaje del modelo (repeticion "
                             "literal del mismo operador).")
    parser.add_argument("--no-diversity-feedback", dest="diversity_feedback",
                        action="store_false", default=True,
                        help="No agregar el diagnostico conductual (familia, novedad, "
                             "cobertura) al feedback del LLM. La medicion y el "
                             "archivo se siguen guardando.")
    # Diversidad y parada temprana (mecanismos de LLaMEA 1.4).
    parser.add_argument("--patience", type=int, default=0,
                        help="Parada temprana: corta tras N evaluaciones sin mejora "
                             "del mejor HV (0 = sin parada; requiere --max-workers 1).")
    parser.add_argument("--parent-selection",
                        choices=["random", "roulette", "tournament"],
                        default="tournament",
                        help="Seleccion de padres de LLaMEA (default tournament; "
                             "LLaMEA usa random).")
    parser.add_argument("--tournament-size", type=int, default=3,
                        help="Candidatos muestreados por torneo.")
    parser.add_argument("--adaptive-mutation", action="store_true", default=True,
                        help="Mutacion adaptativa de LLaMEA (default activada).")
    parser.add_argument("--no-adaptive-mutation", dest="adaptive_mutation",
                        action="store_false",
                        help="Desactiva la mutacion adaptativa.")
    parser.add_argument("--feature-guided-mutation", action="store_true",
                        help="Guia la mutacion con caracteristicas del AST (LLaMEA).")
    parser.add_argument("--niching",
                        choices=["none", "sharing", "clearing", "novelty", "map_elites"],
                        default="none",
                        help="Niching de LLaMEA (presion de diversidad sobre el "
                             "archivo). novelty/clearing/map_elites usan el "
                             "fingerprint conductual del operador, no el texto.")
    parser.add_argument("--niche-radius", type=float, default=None)
    parser.add_argument("--adaptive-niche-radius", action="store_true")
    parser.add_argument("--clearing-interval", type=int, default=None)
    parser.add_argument("--novelty-k", type=int, default=5,
                        help="Vecinos para el puntaje de novedad (niching=novelty).")
    parser.add_argument("--novelty-archive-size", type=int, default=None,
                        help="Tamano del archivo de novedad (default de LLaMEA: 100).")
    parser.add_argument("--eval-repeats", type=int, default=11,
                        help="Repeticiones de la evaluacion final en test (default 11, "
                             "como N_EXPERIMENTS; reporta media, desviacion y "
                             "comparacion pareada contra los clasicos). 1 = solo una.")
    parser.add_argument("--reference-operators", default="ox,pbx,pmx,cx,erx",
                        help="Panel de operadores de referencia para los limites, "
                             "separados por coma (subconjunto de ox,pbx,pmx,cx,erx).")
    parser.add_argument("--quiet", action="store_true",
                        help="No imprimir una linea por generacion.")
    parser.add_argument("--wait-server", action="store_true",
                        help="Esperar a que el servidor responda antes de arrancar.")
    parser.add_argument("--server-timeout", type=float, default=1800)
    parser.add_argument("--eval-test", action="store_true",
                        help="Evaluar el mejor operador en el conjunto de test al final.")
    parser.add_argument("--no-download", action="store_true",
                        help="No descargar el modelo si falta (solo verificar).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validar el servidor y la configuracion, escribirlos y "
                             "salir sin gastar GPU. Requiere que el servidor responda.")
    parser.add_argument("--stall-timeout", type=float, default=0.0,
                        help="Segundos sin una generacion nueva antes de abortar la "
                             "corrida (watchdog del cluster). 0 = desactivado.")
    args = parser.parse_args(argv)

    if args.combine and args.n_parents < 2:
        parser.error("--combine requiere --n-parents >= 2")
    if args.combine and args.refine_only:
        parser.error("--combine y --refine-only son incompatibles")
    if args.combine and args.max_workers != 1:
        parser.error("--combine usa estado de seleccion por generacion; "
                     "requiere --max-workers 1")
    if not 0.0 <= args.target_penalty <= 1.0:
        parser.error("--target-penalty debe estar en [0, 1]")

    if args.train_instances != "all":
        args.train_instances = int(args.train_instances)
    if args.test_instances != "all":
        args.test_instances = int(args.test_instances)

    run_llamea(args)


if __name__ == "__main__":
    main()
