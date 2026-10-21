r"""Compara operadores con el marco de evaluacion de la evolucion gramatical.

La evaluacion final de LLaMEA usa un marco fijo (panel de referencias) y un
protocolo distinto al de GE. Este script hace lo que si es comparable:

  * re-ejecuta todos los operadores con el mismo arnes de evaluacion de GE
    (mismos solvers, mutaciones, poblaciones iniciales y semillas por
    repeticion);
  * construye un solo marco por instancia con el frente no dominado
    consolidado de TODAS las configuraciones y repeticiones;
  * normaliza con (f - vmin) / (vmax - vmin) y calcula el HV con nadir fijo
    [1.1] * K, igual que ``mohh/evaluation/experiment.py``;
  * compara los HV por repeticion (Wilcoxon pareado).

No modifica el nucleo de GE ni ``mohh-evaluate``. Solo usa sus clases.

Ejemplos:

    # LLaMEA contra los operadores (Best/Middle/Worst) de un experimento GE
    python scripts/compare_ge_frame.py \
        --operator llamea=results/final_a10/LLaMEA_single_.../llamea_best_operator.py \
        --operator ge1=ge:results/Experiment_001/MOEAD_operators.txt \
        --repeats 31 --instances all --name llamea_vs_ge1

    # Tabla final (caro): 31 repeticiones, protocolo GE completo
    python scripts/compare_ge_frame.py \
        --operator llamea=.../llamea_best_operator.py \
        --operator ge1=ge:.../Experiment_001/MOEAD_operators.txt \
        --operator pmx=classic:pmx --operator cx=classic:cx \
        --repeats 31 --mo-generations 500 --mo-population-size 105 \
        --solver MOEAD,NSGAII,SMSEMOA --mutation both --budget 50k \
        --instances all --name final_ge_frame

Salida: ``results/evaluations_ge_frame/<name>/evaluation.json`` y ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np

from mohh.core.utilities.algorithm.MO import (
    compute_hypervolume,
    non_dominated_sorting_vectorized,
)
from mohh.core.utilities.instance_utils import instance_paths
from mohh.evaluation.algorithms import MOEA_Decomposition, NSGAII, SMS_MOEA
from mohh.evaluation.operators.mutation import Swap_Mutation
from mohh.evaluation.operators.operator_template import NullMutation
from mohh.evaluation.params import Params, set_params
from mohh.evaluation.problem.instance import Instance
from mohh.llamea.fitness import (
    NADIR_FACTOR,
    compile_operator,
    normalize_front,
)
from mohh.llamea.operator import LLMOperator
from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for
from mohh.llamea.statistics import add_holm, paired_summary
from mohh.parallel import available_cores

SOLVERS = {
    "MOEAD": MOEA_Decomposition,
    "NSGAII": NSGAII,
    "SMSEMOA": SMS_MOEA,
}

MUTATION_PROBABILITY = {
    "MOEAD": "MOEAD_MUTATION_PROBABILITY",
    "NSGAII": "NSGA_MUTATION_PROBABILITY",
    "SMSEMOA": "SMS_MUTATION_PROBABILITY",
}


def _select_instances(paths, amount, seed):
    """Selecciona instancias de forma reproducible (mismo criterio que LLaMEA)."""
    ordered = sorted(str(path) for path in paths)
    if amount == "all":
        return ordered
    amount = int(amount)
    return random.Random(seed).sample(ordered, min(amount, len(ordered)))


def _read_ge_operators(path: str, labels: list[str]) -> dict[str, str]:
    """Lee ``<model>_operators.txt`` (``Best: fenotipo``) y filtra etiquetas."""
    found = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, phenotype = line.split(":", 1)
            found[label.strip()] = phenotype.strip()
    missing = [label for label in labels if label not in found]
    if missing:
        raise SystemExit(
            f"{path}: faltan las etiquetas {missing}; hay {sorted(found)}"
        )
    return {label: found[label] for label in labels}


def load_specs(specs: list[str], ge_labels: list[str]) -> list[dict]:
    """Convierte ``nombre=fuente`` en especificaciones serializables."""
    operators = []
    for spec in specs:
        name, separator, source = spec.partition("=")
        name, source = name.strip(), source.strip()
        if not separator or not name or not source:
            raise SystemExit(f"--operator debe ser nombre=fuente, no {spec!r}")

        if source.startswith("ge:"):
            path = source[3:]
            phenotypes = _read_ge_operators(path, ge_labels)
            for label, phenotype in phenotypes.items():
                operator_name = name if len(phenotypes) == 1 else f"{name}_{label}"
                operators.append({
                    "name": operator_name,
                    "kind": "ge",
                    "source": path,
                    "label": label,
                    "payload": phenotype,
                })
        elif source.startswith("classic:"):
            key = source.split(":", 1)[1].strip()
            if key not in REFERENCE_OPERATORS:
                raise SystemExit(f"clasico desconocido: {key!r}")
            operators.append({
                "name": name,
                "kind": "classic",
                "source": source,
                "label": key,
                "payload": key,
            })
        else:
            with open(source, encoding="utf-8") as handle:
                code = handle.read()
            # Validacion temprana: el error se ve antes de lanzar el MOEA.
            compile_operator(code)
            operators.append({
                "name": name,
                "kind": "llamea",
                "source": source,
                "label": None,
                "payload": code,
            })

    if not operators:
        raise SystemExit("no hay operadores para evaluar")
    names = [operator["name"] for operator in operators]
    duplicated = {name for name in names if names.count(name) > 1}
    if duplicated:
        raise SystemExit(f"nombres de operador repetidos: {sorted(duplicated)}")
    return operators


@lru_cache(maxsize=8)
def _compile_cached(code: str) -> dict:
    """Compila una vez por proceso el codigo LLaMEA (no es picklable)."""
    return compile_operator(code)


def _make_crossover(operator: dict):
    """Construye el cruce pymoo para un operador (sin tocar el nucleo)."""
    if operator["kind"] == "ge":
        from mohh.core.operators.operator_template import HH_Operator

        return HH_Operator(operator["payload"], solution_type="Natural", prob=1.0)
    if operator["kind"] == "classic":
        key = operator["payload"]
        # PMX y CX son los dos cruces que usa la evaluacion GE. Usar sus
        # clases exactas evita comparar contra una reimplementacion distinta.
        if key in ("pmx", "cx"):
            from mohh.evaluation.operators.crossover import CX_Cross, PMX_Cross

            return (PMX_Cross if key == "pmx" else CX_Cross)(prob=1.0)
        return LLMOperator(
            namespace_for(REFERENCE_OPERATORS[key]), prob=1.0
        )
    return LLMOperator(_compile_cached(operator["payload"]), prob=1.0)


# Las instancias se envian una sola vez a cada worker (initializer), no en
# cada tarea: con 31 repeticiones x 20 instancias, picklear la instancia en
# cada una de las ~25k tareas seria inviable.
_WORKER_INSTANCES = []


def _init_worker(instances):
    global _WORKER_INSTANCES
    _WORKER_INSTANCES = instances


def _run_tasks(tasks, instances, on_complete=None):
    """Ejecuta tareas con un pool persistente y las instancias precargadas."""
    if not tasks:
        return []
    n_workers = min(available_cores(), len(tasks))
    if n_workers <= 1:
        _init_worker(instances)
        results = []
        for index, task in enumerate(tasks):
            results.append(_run_task(task))
            if on_complete is not None:
                on_complete(index)
        return results

    results = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(instances,),
    ) as pool:
        futures = [pool.submit(_run_task, task) for task in tasks]
        index_by_future = {future: index for index, future in enumerate(futures)}
        for future in as_completed(futures):
            index = index_by_future[future]
            results[index] = future.result()
            if on_complete is not None:
                on_complete(index)
    return results


def _run_task(task: dict) -> dict:
    """Una corrida MOEA: operador x instancia x repeticion x solver x mutacion."""
    try:
        params = task["params"]
        Params().update_params(params)

        seed = task["seed"]
        random.seed(seed)
        np.random.seed(seed)

        instance = _WORKER_INSTANCES[task["instance_index"]]
        instance.set_initial_solutions(experiment=task["experiment"], kind="permutation")

        solver_class = SOLVERS[task["solver"]]
        solver = solver_class()
        solver.crossover = _make_crossover(task["operator"])

        probability = float(params[MUTATION_PROBABILITY[task["solver"]]])
        solver.mutation = Swap_Mutation(probability) if task["mutation"] else NullMutation()
        solver.load_instance(instance)
        solver.start_model(seed)
        snapshots = solver.solve_instance()

        fronts = snapshots.get(task["budget"])
        fallback = False
        if fronts is None:
            if not task["allow_budget_fallback"]:
                return {
                    "front": None,
                    "fallback": False,
                    "error": f"budget {task['budget']!r} no alcanzado",
                }
            # Solo para smokes con MO_GENERATIONS demasiado bajo para alcanzar
            # el budget pedido. En la tabla final no deberia ocurrir.
            result = solver.algorithm.result()
            fronts = None if result is None else result.F
            fallback = True

        return {
            "front": None if fronts is None else np.asarray(fronts, dtype=float),
            "fallback": fallback,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - un operador no debe tumbar el resto
        return {
            "front": None,
            "fallback": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _protocols(solver_arg: str, mutation: str) -> list[tuple[str, bool]]:
    solvers = list(dict.fromkeys(
        value.strip() for value in solver_arg.split(",") if value.strip()
    ))
    if not solvers:
        raise SystemExit("--solver no puede estar vacio")
    for solver in solvers:
        if solver not in SOLVERS:
            raise SystemExit(f"solver desconocido: {solver!r}")
    if mutation == "none":
        return [(solver, False) for solver in solvers]
    if mutation == "swap":
        return [(solver, True) for solver in solvers]
    return [(solver, False) for solver in solvers] + [
        (solver, True) for solver in solvers
    ]


def _protocol_name(solver: str, mutation: bool) -> str:
    return f"{solver}+mut" if mutation else solver


def _collect_fronts(operators, paths, args, protocols, params_snapshot,
                    instances):
    """Reejecuta todo y devuelve frentes por (protocolo, operador, rep, instancia)."""
    tasks = []
    keys = []
    for rep in range(args.repeats):
        for solver, mutation in protocols:
            for operator in operators:
                for index, _path in enumerate(paths):
                    # Todas las configuraciones de la repeticion comparten la
                    # misma instancia/poblacion inicial. La instancia vive en
                    # el worker (initializer), no en cada tarea.
                    keys.append((_protocol_name(solver, mutation), operator["name"], rep, index))
                    tasks.append({
                        "params": params_snapshot,
                        "operator": operator,
                        "instance_index": index,
                        "experiment": rep + 1,
                        "seed": args.seed + rep,
                        "solver": solver,
                        "mutation": mutation,
                        "budget": args.budget,
                        "allow_budget_fallback": args.allow_budget_fallback,
                    })

    print(
        f"Corriendo {len(tasks)} MOEAs: {len(operators)} operadores x "
        f"{len(protocols)} protocolos x {len(paths)} instancias x "
        f"{args.repeats} repeticiones",
        flush=True,
    )
    start = time.time()
    done = {"count": 0}

    def on_complete(_index):
        done["count"] += 1
        if done["count"] % max(1, len(tasks) // 20) == 0 or done["count"] == len(tasks):
            print(f"  {done['count']}/{len(tasks)}", flush=True)

    results = _run_tasks(tasks, instances, on_complete=on_complete)
    print(f"MOEAs terminados en {time.time() - start:.1f}s", flush=True)

    fronts = {}
    failures = {}
    fallbacks = 0
    for key, result in zip(keys, results):
        front = result["front"]
        if front is None or len(front) == 0:
            failures[key] = result["error"] or "sin frente"
            front = np.empty((0, 0), dtype=float)
        fronts[key] = front
        fallbacks += int(result["fallback"])
    if fallbacks:
        print(
            f"AVISO: {fallbacks} corridas no alcanzaron '{args.budget}' y usaron "
            "el frente final; sube --mo-generations para el protocolo exacto.",
            flush=True,
        )
    return fronts, failures, fallbacks


def _ge_frame(*point_sets):
    """Reproduce el ``comprobate`` de ``evaluation/experiment.py``.

    GE no usa directamente el primer frente no dominado si tiene una o dos
    soluciones: las elimina y vuelve a calcular el frente. Eso evita fijar el
    nadir con un frente que domina a todos los demas. No modifica la
    evaluacion final; solo replica su criterio para el marco compartido.
    """
    points = [np.asarray(points, dtype=float) for points in point_sets
              if points is not None and len(points)]
    if not points:
        return None
    consolidated = set(map(tuple, np.vstack(points)))
    best = non_dominated_sorting_vectorized(consolidated)[0]
    while best and len(best) <= 2 and len(consolidated) > len(best):
        for point in best:
            consolidated.remove(point)
        if not consolidated:
            return None
        best = non_dominated_sorting_vectorized(consolidated)[0]
    return np.asarray(best, dtype=float) if best else None


def _shared_frames(fronts, operators, protocols, paths, repeats):
    """Marco GE por instancia: frente consolidado de TODAS las configuraciones."""
    frames = []
    for index in range(len(paths)):
        pieces = [
            fronts[(_protocol_name(solver, mutation), operator["name"], rep, index)]
            for solver, mutation in protocols
            for operator in operators
            for rep in range(repeats)
        ]
        pieces = [piece for piece in pieces if piece is not None and len(piece)]
        frames.append(_ge_frame(*pieces) if pieces else None)
    return frames


def _hv_values(fronts, operators, protocols, frames, paths, repeats):
    """HV de cada (protocolo, operador, repeticion, instancia) en el marco GE."""
    values = {}
    failures = {}
    for solver, mutation in protocols:
        protocol = _protocol_name(solver, mutation)
        for operator in operators:
            for rep in range(repeats):
                row = []
                for index, frame in enumerate(frames):
                    key = (protocol, operator["name"], rep, index)
                    front = fronts[key]
                    if frame is None or len(frame) == 0:
                        row.append(None)
                        continue
                    vmin, vmax = frame.min(axis=0), frame.max(axis=0)
                    if np.any(vmax == vmin):
                        row.append(None)
                        continue
                    if front is None or len(front) == 0:
                        failures[key] = "sin frente"
                        row.append(0.0)
                        continue
                    normalized = normalize_front(front, (vmin, vmax))
                    if not np.all(np.isfinite(normalized)):
                        failures[key] = "normalizacion no finita"
                        row.append(0.0)
                        continue
                    # Exactamente GE: no hay clip_to_reference_box.
                    try:
                        hv = compute_hypervolume(
                            np.full(normalized.shape[1], NADIR_FACTOR),
                            normalized,
                        )
                    except Exception as error:  # noqa: BLE001
                        failures[key] = f"HV: {type(error).__name__}: {error}"
                        row.append(0.0)
                        continue
                    row.append(float(hv))
                values[(protocol, operator["name"], rep)] = row
    return values, failures


def _per_run_scores(values, operators, protocols, repeats):
    scores = {}
    for operator in operators:
        scores[operator["name"]] = []
        for rep in range(repeats):
            cells = [
                value
                for solver, mutation in protocols
                for value in values[(_protocol_name(solver, mutation), operator["name"], rep)]
                if value is not None
            ]
            scores[operator["name"]].append(float(np.mean(cells)) if cells else None)
    return scores


def _summarize(scores, operators):
    rows = []
    for operator in operators:
        values = [value for value in scores[operator["name"]] if value is not None]
        rows.append({
            "operator": operator["name"],
            "kind": operator["kind"],
            "label": operator["label"],
            "source": operator["source"],
            "runs": len(values),
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        })
    return rows


def _comparisons(scores, rows, baseline):
    candidates = [row for row in rows if row.get("runs")]
    if baseline:
        base = next((row for row in candidates if row["operator"] == baseline), None)
        if base is None:
            raise SystemExit(f"baseline {baseline!r} no tiene HV valido")
        pairs = [(row, base) for row in candidates if row is not base]
    else:
        # Todos los pares: asi la comparacion LLaMEA-vs-GE siempre aparece,
        # aunque tambien se incluyan clasicos. Con --baseline se enfoca.
        pairs = [
            (left, right)
            for index, left in enumerate(candidates)
            for right in candidates[index + 1:]
        ]

    results = []
    for row, base in pairs:
        summary = paired_summary(
            scores[row["operator"]], scores[base["operator"]]
        )
        if summary["pairs"] == 0:
            continue
        summary.update({"operator": row["operator"], "baseline": base["operator"]})
        results.append(summary)
    return add_holm(results)


def _fmt(value, digits=6, signed=False):
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _write_report(path, args, rows, comparisons, frames, operators, protocols,
                  failures, fallbacks):
    degenerate = [
        os.path.basename(args.paths[index])
        for index, frame in enumerate(frames)
        if frame is None or np.any(frame.max(axis=0) == frame.min(axis=0))
    ]
    lines = [
        f"# Comparacion en marco GE: {args.name}", "",
        "## Diseno", "",
        f"- Instancias: `{args.split}` ({len(args.paths)}), "
        f"repeticiones: {args.repeats}, semilla base: {args.seed}.",
        f"- Protocolos: {', '.join(_protocol_name(s, m) for s, m in protocols)}.",
        f"- MOEA interno: {args.mo_generations} generaciones, "
        f"poblacion {args.mo_population_size}, budget `{args.budget}`.",
        "- Operadores: " + ", ".join(f"`{o['name']}` ({o['kind']})" for o in operators),
        "- Marco: por instancia, frente no dominado consolidado de todas las "
        "configuraciones y repeticiones (con el peeling de GE cuando el frente "
        "tiene <=2 puntos); normalizacion `(f - vmin)/(vmax - vmin)`; nadir "
        "fijo `[1.1] * K`; sin clipping extra.",
        "- Semillas: `base + repeticion`; la misma poblacion inicial por "
        "repeticion para todos los operadores y protocolos.",
        "",
        "- Instancias sin escala (excluidas): "
        + (", ".join(degenerate) if degenerate else "ninguna"),
        f"- Fallos de evaluacion contabilizados: {len(failures)} "
        "(se cuentan como HV 0).",
    ]
    if fallbacks:
        lines.append(
            f"- AVISO: {fallbacks} corridas no alcanzaron `{args.budget}` y "
            "usaron el frente final."
        )
    lines += [
        "", "## HV por operador", "",
        "Cada repeticion promedia sus celdas instancia x protocolo; la tabla "
        "resume esas repeticiones.", "",
        "| Operador | Tipo | Reps | Media | Desv. | Mediana | Min | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: -(item["mean"] if item["mean"] is not None else -1)):
        lines.append(
            f"| {row['operator']} | {row['kind']} | {row['runs']} | "
            f"{_fmt(row['mean'])} | {_fmt(row['sd'])} | {_fmt(row['median'])} | "
            f"{_fmt(row['min'])} | {_fmt(row['max'])} |"
        )
    lines += [
        "", "## Comparaciones pareadas (marco GE compartido)", "",
        "| Operador | vs | Pares | Falta | Gana | Pierde | Empata | "
        "Dif. media [IC95] | Relativo | dz | p | p Holm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        rel = "-" if row["relative"] is None else f"{100 * row['relative']:+.1f}%"
        dz = "-" if row["cohens_dz"] is None else f"{row['cohens_dz']:.2f}"
        low, high = row.get("ci95_low"), row.get("ci95_high")
        interval = "-" if low is None or high is None else f"[{low:+.6f}, {high:+.6f}]"
        lines.append(
            f"| {row['operator']} | {row['baseline']} | {row['pairs']} | "
            f"{row['missing']} | {row['wins']} | {row['losses']} | {row['ties']} | "
            f"{_fmt(row['mean_difference'], signed=True)} {interval} | {rel} | "
            f"{dz} | {_fmt(row['wilcoxon_p'], 4)} | {_fmt(row['wilcoxon_p_holm'], 4)} |"
        )
    lines += ["", "p de Wilcoxon pareado de dos colas; Holm sobre la familia de comparaciones."]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--operator", action="append", required=True,
                        help="nombre=ruta.py, nombre=ge:ruta_operadores.txt o "
                             "nombre=classic:ox (repetible); classic:pmx/cx "
                             "usan las clases exactas de GE, classic:ox/pbx/erx "
                             "el panel de LLaMEA")
    parser.add_argument("--ge-labels", default="Best,Middle,Worst",
                        help="etiquetas del .txt de GE separadas por comas "
                             "(default: Best,Middle,Worst)")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--instances", default="all",
                        help="numero de instancias o 'all'")
    parser.add_argument("--repeats", type=int, default=31,
                        help="repeticiones independientes (default 31, protocolo "
                             "final; usa 2-3 para un smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", default="MOEAD,NSGAII,SMSEMOA",
                        help="solvers separados por comas")
    parser.add_argument("--mutation", choices=["none", "swap", "both"], default="both")
    parser.add_argument("--mo-generations", type=int, default=500)
    parser.add_argument("--mo-population-size", type=int, default=105)
    parser.add_argument("--budget", default="50k",
                        help="etiqueta de EVAL_BUDGETS (10k/30k/50k); para "
                             "10k/30k puedes bajar --mo-generations a 100/300")
    parser.add_argument("--allow-budget-fallback", action="store_true",
                        help="si el budget no se alcanza, usar el frente final "
                             "(solo para smokes; invalido para la tabla final)")
    parser.add_argument("--baseline", default=None,
                        help="operador de referencia; si se omite, se comparan "
                             "todos los pares")
    parser.add_argument("--out", default="results/evaluations_ge_frame")
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=None,
                        help="workers; default: MOHH_WORKERS/CPUs disponibles")
    parser.add_argument("--save-fronts", action="store_true",
                        help="guardar fronts.npz")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats debe ser >= 1")
    if args.mo_generations < 1 or args.mo_population_size < 1:
        raise SystemExit("--mo-generations y --mo-population-size deben ser >= 1")
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers debe ser >= 1")
    if args.workers:
        os.environ["MOHH_WORKERS"] = str(args.workers)

    args.name = args.name or f"ge_frame_{time.strftime('%Y%m%d_%H%M%S')}"
    operators = load_specs(args.operator, [
        label.strip() for label in args.ge_labels.split(",") if label.strip()
    ])
    protocols = _protocols(args.solver, args.mutation)

    # `instance_paths()` lee DATASET de Params, asi que set_params va primero.
    init_dir = tempfile.mkdtemp(prefix="compare_ge_frame_init_")
    set_params({
        "SOLVERS": sorted({solver for solver, _mutation in protocols}),
        "MO_GENERATIONS": args.mo_generations,
        "MO_POPULATION_SIZE": args.mo_population_size,
        "FILE_PATH_INITIAL_SOLUTIONS": init_dir,
        "RANDOM_SEED": args.seed,
    })
    params = Params()
    if args.budget not in params["EVAL_BUDGETS"]:
        raise SystemExit(
            f"budget {args.budget!r} no existe; usa {sorted(params['EVAL_BUDGETS'])}"
        )
    snapshot = {
        key: value for key, value in params.get_params().items()
        if key != "BNF_GRAMMAR"
    }

    args.paths = _select_instances(
        instance_paths(train=args.split == "train"), args.instances, args.seed
    )
    if not args.paths:
        raise SystemExit("no hay instancias para evaluar")

    # Poblaciones iniciales: una por repeticion, compartida por todos los
    # operadores/protocolos. Se generan una vez por instancia.
    instances = []
    for index, path in enumerate(args.paths):
        np.random.seed(args.seed + index)
        random.seed(args.seed + index)
        instance = Instance(args.mo_population_size, params["SOLUTION_TYPE"])
        instance.load_problem("QAP", path, args.repeats)
        instances.append(instance)

    fronts, failures, fallbacks = _collect_fronts(
        operators, args.paths, args, protocols, snapshot, instances
    )
    frames = _shared_frames(fronts, operators, protocols, args.paths, args.repeats)
    values, hv_failures = _hv_values(
        fronts, operators, protocols, frames, args.paths, args.repeats
    )
    failures.update(hv_failures)
    scores = _per_run_scores(values, operators, protocols, args.repeats)
    rows = _summarize(scores, operators)
    comparisons = _comparisons(scores, rows, args.baseline)

    degenerate = [
        os.path.basename(args.paths[index])
        for index, frame in enumerate(frames)
        if frame is None or np.any(frame.max(axis=0) == frame.min(axis=0))
    ]

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "config": {
            "name": args.name,
            "argv": sys.argv,
            "split": args.split,
            "instances": [os.path.basename(path) for path in args.paths],
            "initial_population_seeds": {
                os.path.basename(path): args.seed + index
                for index, path in enumerate(args.paths)
            },
            "ge_labels": args.ge_labels,
            "repeats": args.repeats,
            "seed": args.seed,
            "solver": args.solver,
            "protocols": [
                {"solver": solver, "mutation": mutation,
                 "name": _protocol_name(solver, mutation)}
                for solver, mutation in protocols
            ],
            "mo_generations": args.mo_generations,
            "mo_population_size": args.mo_population_size,
            "budget": args.budget,
            "allow_budget_fallback": args.allow_budget_fallback,
            "frame": "consolidated over every operator, protocol and repetition",
            "nadir_factor": NADIR_FACTOR,
            "degenerate_instances": degenerate,
            "failed_runs": len(failures),
            "budget_fallbacks": fallbacks,
        },
        "operators": rows,
        "comparisons": comparisons,
        "per_run_scores": scores,
        "failures": [
            {"protocol": protocol, "operator": operator, "repeat": rep,
             "instance": os.path.basename(args.paths[index]), "reason": reason}
            for (protocol, operator, rep, index), reason in sorted(failures.items())
        ],
        "run_seeds": [
            {"operator": operator["name"], "repeat": rep, "seed": args.seed + rep}
            for operator in operators
            for rep in range(args.repeats)
        ],
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    _write_report(
        os.path.join(out_dir, "report.md"), args, rows, comparisons, frames,
        operators, protocols, failures, fallbacks,
    )
    if args.save_fronts:
        arrays = {
            f"{protocol}|{operator}|{rep}|{index}": front
            for (protocol, operator, rep, index), front in fronts.items()
        }
        np.savez_compressed(os.path.join(out_dir, "fronts.npz"), **arrays)
    print(f"Resultados en {out_dir}", flush=True)


if __name__ == "__main__":
    main()
