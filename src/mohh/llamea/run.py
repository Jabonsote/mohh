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
from mohh.llamea.fitness import build_evaluator, init_params
from mohh.llamea.prompts import (
    ROLE_PROMPT,
    make_task_prompt,
    make_operators,
    EXAMPLE_PROMPT,
    SKELETON_PROMPT,
    OUTPUT_FORMAT_PROMPT,
)
from mohh.llamea.reporting import ResultsExperimentLogger, write_report, write_seeds
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
    if amount == "all":
        return list(paths)
    amount = int(amount)
    rng = random.Random(seed)
    return rng.sample(list(paths), min(amount, len(paths)))


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


def _make_evaluator(paths, args, bounds=None, archive=None, fronts_dir=None):
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
    )


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
    fitness = _make_evaluator(train_paths, args, bounds=bounds, fronts_dir=fronts_dir)

    config = {
        "command": shlex.join([sys.executable] + sys.argv),
        "model": args.model,
        "base_url": args.base_url,
        "budget": args.budget,
        "n_parents": args.n_parents,
        "n_offspring": args.n_offspring,
        "solver": args.solver,
        "mo_generations": args.mo_generations,
        "mo_population_size": args.mo_population_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "train_instances": len(train_paths),
        "normalize": args.normalize,
        "bounds_file": args.bounds_file,
        "reference_operators": args.reference_operators,
        "refine_only": args.refine_only,
        "example": ("none" if args.no_example else
                    "skeleton" if args.skeleton_example else "ox"),
        "comma": args.comma,
        "repeat_penalty": args.repeat_penalty,
        "max_workers": args.max_workers,
        "results_dir": args.results_dir,
        "log_dir": log_base,
        # Procedencia (reproducibilidad entre local y cluster).
        "server_info": server_info,
        "model_info": model_info,
        "repeat_penalty": args.repeat_penalty,
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
            "max_tokens": args.max_tokens,
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
    )
    ResultsExperimentLogger.token_source = llm

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

    llamea = LLaMEA(
        f=fitness,
        llm=llm,
        n_parents=args.n_parents,
        n_offspring=args.n_offspring,
        role_prompt=ROLE_PROMPT,
        task_prompt=make_task_prompt(args.solver),
        example_prompt=example_prompt,
        output_format_prompt=OUTPUT_FORMAT_PROMPT,
        operators=[] if args.refine_only else make_operators(),
        budget=args.budget,
        elitism=not args.comma,
        minimization=False,
        max_workers=args.max_workers,
        experiment_name=args.experiment_name,
    )

    # 3b. Costo del modelo: foto de `/metrics` (vLLM) antes de la corrida; los
    #     deltas contra la foto final dan el tiempo de GPU y los tokens reales.
    metrics_before = model_metrics.scrape_prometheus(args.base_url)
    if metrics_before is not None:
        print("Metricas Prometheus del servidor: disponibles (vLLM).")
    else:
        print("Metricas Prometheus no disponibles; se usaran solo las del cliente "
              "(tokens y latencia).")

    best = llamea.run()
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
        "evaluations": len(fitness.archive),
        "best_hv_ranking": fitness.archive.best(),
        "diversity": fitness.duplicate_stats(),
        # Trazabilidad del proceso: semillas, diversidad y costo.
        "seeds": seeds,
        "seeds_file": os.path.join(exp_dir, "seeds.json"),
        "distinct_operators": counters.get("distinct_operators"),
        "evaluated_valid": counters.get("evaluated_valid"),
        "validity_rate": (
            counters["evaluated_valid"] / args.budget
            if counters.get("evaluated_valid") is not None and args.budget else None
        ),
        "stop_reason": "budget reached",
        "cost": cost,
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
        print(f"Evaluando el mejor operador en {len(test_paths)} instancias de test...")
        test_fronts_dir = (None if args.no_save_fronts
                           else os.path.join(exp_dir, "fronts_test"))
        test_fitness = _make_evaluator(test_paths, args, fronts_dir=test_fronts_dir)
        try:
            evaluated = test_fitness(_EvalSolution(best.code))
        except Exception:
            logger.exception("Fallo la evaluacion en test")
            evaluated = None
        if evaluated is not None:
            info["test_instances"] = len(test_paths)
            info["test_fitness"] = evaluated.fitness
            info["test_seeds"] = {"moea_internal": args.seed, "selection": args.seed}
            if test_fitness.normalize:
                info["test_reference_hv"] = test_fitness.reference_scores()
                info["test_degenerate_instances"] = test_fitness.frame_degenerate_instances()
            if test_fronts_dir:
                try:
                    test_fitness.save_frame(os.path.join(test_fronts_dir, "frame.npz"))
                except Exception:  # noqa: BLE001
                    logger.exception("No se pudo guardar el marco de test")
            print(f"Test fitness (HV): {evaluated.fitness}")
            if info.get("test_reference_hv"):
                print(f"Test HV de la linea base en el mismo marco: {info['test_reference_hv']}")

    with open(os.path.join(exp_dir, "llamea_best_info.json"), "w") as file:
        json.dump(info, file, indent=2)

    # 7. Tokens consumidos por el LLM (acumulado y por peticion).
    if hasattr(llm, "usage_snapshot"):
        usage = llm.usage_snapshot()
        usage["history"] = getattr(llm, "usage_history", [])
        usage["max_tokens"] = args.max_tokens
        usage["temperature"] = args.temperature
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
            extra={"seeds": seeds, "gpu_snapshot": gpu_snapshot},
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
        "seeds": seeds,
        "stop_reason": info.get("stop_reason"),
        "distinct_operators": info.get("distinct_operators"),
        "validity_rate": info.get("validity_rate"),
    }
    paths = write_report(log_dir, title=f"LLaMEA {args.experiment_name}",
                         frame_info=frame_info,
                         cost=cost,
                         usage_history=getattr(llm, "usage_history", []))
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
    parser.add_argument("--request-timeout", type=float, default=600,
                        help="Timeout por peticion al LLM (segundos).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximo de tokens generados por peticion "
                             "(None = default del servidor).")
    parser.add_argument("--budget", type=int, default=1000,
                        help="Numero de generaciones (llamadas al LLM). Default 1000; "
                             "para pruebas de humo usar un valor bajo (p. ej. 20).")
    parser.add_argument("--n-parents", type=int, default=1,
                        help="Padres por generacion (1 = un algoritmo por generacion).")
    parser.add_argument("--n-offspring", type=int, default=1,
                        help="Descendientes por generacion (1 = un algoritmo por generacion).")
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

    if args.train_instances != "all":
        args.train_instances = int(args.train_instances)
    if args.test_instances != "all":
        args.test_instances = int(args.test_instances)

    run_llamea(args)


if __name__ == "__main__":
    main()
