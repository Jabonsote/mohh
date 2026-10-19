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
import os
import random
import shlex
import sys

import numpy as np

from mohh.core.utilities.instance_utils import instance_paths
from mohh.llamea.fitness import build_evaluator, init_params
from mohh.llamea.prompts import (
    ROLE_PROMPT,
    make_task_prompt,
    make_operators,
    EXAMPLE_PROMPT,
    SKELETON_PROMPT,
    OUTPUT_FORMAT_PROMPT,
)
from mohh.llamea.reporting import ResultsExperimentLogger, write_report
from mohh.llamea.vllm import VLLM_LLM, ensure_model, model_is_available, wait_for_server


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


def _make_evaluator(paths, args, bounds=None, archive=None):
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
    )


def run_llamea(args):
    # 0. Inicializar los defaults de MOHH-R (DATASET, PROBLEM_NAME, ...) para
    #    poder listar los mismos conjuntos train/test que la busqueda GE.
    init_params(RANDOM_SEED=args.seed)

    exp_dir = os.path.join(args.results_dir, args.experiment_name)
    log_base = os.path.join(exp_dir, "llamea_logs")
    os.makedirs(log_base, exist_ok=True)

    # 1. Modelo local: verificar/descargar antes de conectar. Con
    #    --no-download (uso cliente en el cluster) solo se informa: el modelo
    #    lo garantiza el job vLLM y puede no estar en el cache de este nodo.
    if args.no_download:
        if model_is_available(args.model):
            print(f"Modelo presente en este nodo: {args.model}")
        else:
            print(f"AVISO: '{args.model}' no esta en el cache de HuggingFace de "
                  "este nodo; se asume que lo sirve el endpoint de --base-url "
                  "(vLLM en el cluster, Ollama o llama.cpp en local).")
    else:
        ensure_model(args.model, allow_download=True)

    # 2. Servidor vLLM: opcionalmente esperar a que cargue el modelo.
    if args.wait_server:
        print(f"Esperando servidor en {args.base_url} (max {args.server_timeout}s)...")
        if not wait_for_server(args.base_url, timeout=args.server_timeout):
            raise RuntimeError(f"El servidor no respondio en {args.base_url}")

    # 3. Fitness con la logica actual de MOHH-R (train).
    train_paths = _select_instances(instance_paths(train=True), args.train_instances, args.seed)
    print(f"Instancias de train: {len(train_paths)}")

    bounds = _load_bounds(args.bounds_file) if args.bounds_file else None
    fitness = _make_evaluator(train_paths, args, bounds=bounds)

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
        "max_workers": args.max_workers,
        "results_dir": args.results_dir,
        "log_dir": log_base,
    }
    with open(os.path.join(exp_dir, "run.json"), "w") as file:
        json.dump(config, file, indent=2)

    # 4. Redirigir el logger de LLaMEA a results/<exp>/llamea_logs/.
    import llamea.llamea as llamea_module

    ResultsExperimentLogger.results_dir = log_base
    ResultsExperimentLogger.verbose = not args.quiet
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

    best = llamea.run()
    train_fitness = best.fitness
    print(f"Best train fitness (HV): {train_fitness}")
    print(f"Best code:\n{best.code}")

    log_dir = llamea.logger.dirname if llamea.logger is not None else log_base

    # 5. Guardar resultado.
    with open(os.path.join(exp_dir, "llamea_best_operator.py"), "w") as file:
        file.write(best.code + "\n")

    info = dict(config)
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
        "best_hv": fitness.archive.best(),
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
        test_fitness = _make_evaluator(test_paths, args)
        evaluated = test_fitness(_EvalSolution(best.code))
        info["test_instances"] = len(test_paths)
        info["test_fitness"] = evaluated.fitness
        if test_fitness.normalize:
            info["test_reference_hv"] = test_fitness.reference_scores()
            info["test_degenerate_instances"] = test_fitness.frame_degenerate_instances()
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
        with open(os.path.join(exp_dir, "tokens.json"), "w") as file:
            json.dump(usage, file, indent=2)
        print(f"tokens: {llm.usage_snapshot()}")

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
    }
    paths = write_report(log_dir, title=f"LLaMEA {args.experiment_name}",
                         frame_info=frame_info)
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
    parser.add_argument("--budget", type=int, default=100,
                        help="Numero de generaciones (llamadas al LLM).")
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
    args = parser.parse_args(argv)

    if args.train_instances != "all":
        args.train_instances = int(args.train_instances)
    if args.test_instances != "all":
        args.test_instances = int(args.test_instances)

    run_llamea(args)


if __name__ == "__main__":
    main()
