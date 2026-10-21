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

    # Reproducir la evaluacion GE original reutilizando sus individuos y semillas
    python scripts/compare_ge_frame.py \
        --operator ge1=ge:/ruta/Experiment_001/MOEAD_operators.txt \
        --operator pmx=classic:pmx --operator cx=classic:cx \
        --dataset mqap_original --split test --instances all --repeats 11 \
        --init-from /ruta/Experiment_001/evaluation_results/initial_solutions \
        --reuse-seeds --solver MOEAD,NSGAII,SMSEMOA --mutation both \
        --mo-generations 500 --mo-population-size 105 --budget 50k \
        --name reproduce_ge_experiment_001

Salida en ``results/evaluations_ge_frame/<name>/``:

  * ``evaluation.json``: config (con procedencia), HV por operador, HV por
    instancia/protocolo/repeticion, comparaciones pareadas y fallos;
  * ``report.md``: diseno, procedencia, tabla de HV y comparaciones;
  * ``operators/``: cada individuo (codigo LLaMEA ``.py``, fenotipo GE
    ``.phenotype.txt``) y su ``.json``, mas ``operators.json``;
  * ``initial_solutions/{real,permutation}/<instancia>/``: las poblaciones
    iniciales (``experiment_NNN.txt``) y ``seeds.txt``, con el mismo formato
    que ``mohh/evaluation/experiment.py``;
  * ``fronts/<instancia>/best_front_consolidated{,_norm}.pof``: el marco GE de
    cada instancia, tambien como en la evaluacion original;
  * ``fronts.npz``: todos los frentes crudos por (protocolo, operador,
    repeticion, instancia) para reanalizar sin repetir el MOEA
    (``--no-save-fronts`` para omitirlo).

``--init-from`` permite reutilizar poblaciones iniciales ya guardadas (por
ejemplo las de ``Experiment_001``/``Experiment_002``): los individuos son los
mismos que en la corrida anterior.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
                    instances, seeds_by_instance=None):
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
                    seed = (seeds_by_instance[index][rep]
                            if seeds_by_instance else args.seed + rep)
                    keys.append((_protocol_name(solver, mutation), operator["name"], rep, index))
                    tasks.append({
                        "params": params_snapshot,
                        "operator": operator,
                        "instance_index": index,
                        "experiment": rep + 1,
                        "seed": seed,
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
                  failures, fallbacks, frame_records=None):
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
        "- Semillas: "
        + ("las del experimento original (`initial_solutions/real/<instancia>/"
           "seeds.txt`), pareadas dentro de cada (instancia, repeticion)."
           if getattr(args, "reuse_seeds", False) else
           "`base + repeticion`; la misma poblacion inicial por repeticion "
           "para todos los operadores y protocolos."),
        "- Poblaciones iniciales: "
        + (f"reutilizadas de `{args.init_from}`." if args.init_from else
           "generadas de nuevo con semilla determinista por instancia "
           "(`base + indice`)."),
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
        "", "## Artefactos y procedencia", "",
        f"- Individuos (poblaciones iniciales): `initial_solutions/"
        "{real,permutation}/<instancia>/experiment_NNN.txt` y `seeds.txt`.",
        "- Operadores evaluados: `operators/` (codigo LLaMEA `.py`, fenotipo GE "
        "`.phenotype.txt`, metadata `.json`) y `operators.json`.",
        "- Marco GE por instancia: `fronts/<instancia>/"
        "best_front_consolidated{,_norm}.pof`.",
        "- Frentes crudos por (protocolo, operador, repeticion, instancia): "
        f"`fronts.npz`{' (omitido)' if args.no_save_fronts else ''}.",
        f"- Dataset: `{args.dataset or 'mqap (default del proyecto)'}` "
        f"({args.split}).",
    ]
    if frame_records:
        lines += ["", "| Instancia | Puntos marco | Degenerada |", "|---|---:|---|"]
        for record in frame_records:
            lines.append(
                f"| {record['instance']} | {record.get('size', 0)} | "
                f"{'si' if record.get('degenerate') else 'no'} |"
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


def _write_front(front: np.ndarray, path: str) -> None:
    """Escribe un frente con el mismo formato que ``experiment.py`` (``.pof``)."""
    front = np.asarray(front, dtype=float)
    rows, columns = front.shape
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {rows} {columns}\n")
        for row in front:
            handle.write(" ".join(f"{value:1.6e}" for value in row) + "\n")


def _save_seeds(init_dir: str, instance_name: str, seeds: list[int]) -> list[str]:
    """Guarda ``seeds.txt`` en ``real`` y ``permutation``, como la evaluacion GE."""
    written = []
    for kind in ("real", "permutation"):
        folder = os.path.join(init_dir, kind, instance_name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "seeds.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for seed in seeds:
                handle.write(f"{seed}\n")
        written.append(path)
    return written


def _save_operators(out_dir: str, operators: list[dict]) -> tuple[str, list[dict]]:
    """Guarda cada individuo evaluado (codigo/fenotipo) y su metadata.

    Mismo espiritu que ``Experiment_*/*_operators.txt``: el artefacto guarda lo
    que se corrio, no solo su nombre. Devuelve ``(carpeta, registros)``.
    """
    operators_dir = os.path.join(out_dir, "operators")
    os.makedirs(operators_dir, exist_ok=True)
    records = []
    for operator in operators:
        record = {
            "name": operator["name"],
            "kind": operator["kind"],
            "source": operator["source"],
            "label": operator["label"],
        }
        payload = operator["payload"]
        if operator["kind"] == "llamea":
            filename = f"{operator['name']}.py"
            with open(os.path.join(operators_dir, filename), "w",
                      encoding="utf-8") as handle:
                handle.write(payload.rstrip("\n") + "\n")
            record["file"] = filename
        elif operator["kind"] == "ge":
            filename = f"{operator['name']}.phenotype.txt"
            with open(os.path.join(operators_dir, filename), "w",
                      encoding="utf-8") as handle:
                handle.write(payload.strip() + "\n")
            record["file"] = filename
        else:
            record["reference"] = payload
        record["payload"] = payload
        records.append(record)
        with open(os.path.join(operators_dir, f"{operator['name']}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "operators.json"), "w",
              encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    return operators_dir, records


def _frame_record(frame) -> dict | None:
    """Metadatos serializables del marco GE de una instancia."""
    if frame is None or len(frame) == 0:
        return None
    vmin = frame.min(axis=0)
    vmax = frame.max(axis=0)
    return {
        "size": int(len(frame)),
        "vmin": [float(value) for value in vmin],
        "vmax": [float(value) for value in vmax],
        "degenerate": bool(np.any(vmax == vmin)),
    }


def _save_frames(out_dir: str, paths: list[str], frames: list) -> list[dict]:
    """Guarda el marco GE por instancia como ``best_front_consolidated{,_norm}.pof``."""
    frames_dir = os.path.join(out_dir, "fronts")
    records = []
    for index, frame in enumerate(frames):
        name = os.path.basename(paths[index])
        record = {"instance": name, **(_frame_record(frame) or {})}
        if frame is None or len(frame) == 0:
            records.append(record)
            continue
        folder = os.path.join(frames_dir, os.path.splitext(name)[0])
        os.makedirs(folder, exist_ok=True)
        _write_front(frame, os.path.join(folder, "best_front_consolidated.pof"))
        vmin, vmax = frame.min(axis=0), frame.max(axis=0)
        if np.all(vmax > vmin):
            normalized = normalize_front(frame, (vmin, vmax))
            _write_front(
                normalized,
                os.path.join(folder, "best_front_consolidated_norm.pof"),
            )
        record["front_file"] = os.path.join(
            folder, "best_front_consolidated.pof")
        records.append(record)
    return records


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
    parser.add_argument("--dataset", default=None,
                        help="dataset de Params (default: el del proyecto, "
                             "mqap); usa mqap_original para reproducir la "
                             "evaluacion original de Experiment_001/002")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--instances", default="all",
                        help="numero de instancias o 'all'")
    parser.add_argument("--instance-names", default=None,
                        help="lista de nombres (basename) separada por comas; "
                             "tiene prioridad sobre --instances")
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
    parser.add_argument("--init-from", default=None,
                        help="carpeta con poblaciones iniciales ya guardadas "
                             "(real/<instancia>/experiment_NNN.txt) para "
                             "reutilizar los mismos individuos, p. ej. el "
                             "initial_solutions de Experiment_001")
    parser.add_argument("--reuse-seeds", dest="reuse_seeds",
                        action="store_true", default=None,
                        help="usar las semillas de <init-from>/real/<instancia>/"
                             "seeds.txt (por defecto si se pasa --init-from)")
    parser.add_argument("--no-reuse-seeds", dest="reuse_seeds",
                        action="store_false",
                        help="ignorar seeds.txt y usar semillas pareadas "
                             "base + repeticion")
    parser.add_argument("--save-fronts", action="store_true",
                        help="(obsoleto) fronts.npz ya se guarda por defecto")
    parser.add_argument("--no-save-fronts", action="store_true",
                        help="no guardar fronts.npz (si solo interesan los .pof)")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats debe ser >= 1")
    if args.mo_generations < 1 or args.mo_population_size < 1:
        raise SystemExit("--mo-generations y --mo-population-size deben ser >= 1")
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers debe ser >= 1")
    if args.workers:
        os.environ["MOHH_WORKERS"] = str(args.workers)
    if args.reuse_seeds is None:
        args.reuse_seeds = bool(args.init_from)
    if args.reuse_seeds and not args.init_from:
        raise SystemExit("--reuse-seeds requiere --init-from")

    args.name = args.name or f"ge_frame_{time.strftime('%Y%m%d_%H%M%S')}"
    operators = load_specs(args.operator, [
        label.strip() for label in args.ge_labels.split(",") if label.strip()
    ])
    protocols = _protocols(args.solver, args.mutation)

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Las poblaciones iniciales ("individuos") se guardan con el mismo formato
    # que `experiment.py`: initial_solutions/{real,permutation}/<instancia>/.
    init_dir = os.path.join(out_dir, "initial_solutions")
    os.makedirs(init_dir, exist_ok=True)

    # `instance_paths()` lee DATASET de Params, asi que set_params va primero.
    overrides = {
        "SOLVERS": sorted({solver for solver, _mutation in protocols}),
        "MO_GENERATIONS": args.mo_generations,
        "MO_POPULATION_SIZE": args.mo_population_size,
        "FILE_PATH_INITIAL_SOLUTIONS": init_dir,
        "RANDOM_SEED": args.seed,
    }
    if args.dataset:
        overrides["DATASET"] = args.dataset
    set_params(overrides)
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
    if args.instance_names:
        wanted = [name.strip() for name in args.instance_names.split(",")
                  if name.strip()]
        available = {os.path.basename(path): path for path in args.paths}
        missing = [name for name in wanted if name not in available]
        if missing:
            raise SystemExit(f"instancias no encontradas: {missing}")
        args.paths = [available[name] for name in wanted]
    if not args.paths:
        raise SystemExit("no hay instancias para evaluar")

    # Poblaciones iniciales: una por repeticion, compartida por todos los
    # operadores/protocolos. Se generan (o se cargan de --init-from) una vez
    # por instancia y se guardan en disco, igual que la evaluacion GE.
    instances = []
    for index, path in enumerate(args.paths):
        np.random.seed(args.seed + index)
        random.seed(args.seed + index)
        instance = Instance(args.mo_population_size, params["SOLUTION_TYPE"])
        instance.load_problem("QAP", path, args.repeats, args.init_from or "")
        instances.append(instance)

    # Semillas del MOEA por (instancia, repeticion). Por defecto, pareadas
    # (base + repeticion, identicas en todas las instancias). Con --reuse-seeds
    # se cargan las del experimento original (seeds.txt por instancia), que
    # siguen siendo pareadas dentro de cada (instancia, repeticion).
    paired_seeds = [args.seed + rep for rep in range(args.repeats)]
    seeds_by_instance = []
    for instance in instances:
        seeds = paired_seeds
        if args.reuse_seeds:
            seed_file = os.path.join(
                args.init_from, "real", instance.file_name, "seeds.txt"
            )
            if os.path.exists(seed_file):
                with open(seed_file, encoding="utf-8") as handle:
                    seeds = [int(line) for line in handle if line.strip()]
                if len(seeds) < args.repeats:
                    raise SystemExit(
                        f"{seed_file}: {len(seeds)} semillas < --repeats "
                        f"{args.repeats}"
                    )
                seeds = seeds[:args.repeats]
            else:
                print(f"AVISO: no hay {seed_file}; semillas pareadas", flush=True)
        seeds_by_instance.append(seeds)

    # Se guardan por instancia como en `save_seeds` de `experiment.py`.
    for index, instance in enumerate(instances):
        _save_seeds(init_dir, instance.file_name, seeds_by_instance[index])
    operators_dir, operator_records = _save_operators(out_dir, operators)

    fronts, failures, fallbacks = _collect_fronts(
        operators, args.paths, args, protocols, snapshot, instances,
        seeds_by_instance,
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

    # Marco GE de cada instancia en disco (best_front_consolidated{,_norm}.pof).
    frame_records = _save_frames(out_dir, args.paths, frames)

    # HV completo por (protocolo, operador, repeticion) -> lista por instancia.
    per_instance_hv = {
        operator["name"]: {
            _protocol_name(solver, mutation): [
                values[(_protocol_name(solver, mutation), operator["name"], rep)]
                for rep in range(args.repeats)
            ]
            for solver, mutation in protocols
        }
        for operator in operators
    }

    result = {
        "config": {
            "name": args.name,
            "argv": sys.argv,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "split": args.split,
            "instances": [os.path.basename(path) for path in args.paths],
            "initial_population_seeds": {
                os.path.basename(path): args.seed + index
                for index, path in enumerate(args.paths)
            },
            "initial_solutions_dir": init_dir,
            "initial_solutions_reused_from": args.init_from,
            "operators_dir": operators_dir,
            "ge_labels": args.ge_labels,
            "repeats": args.repeats,
            "seed": args.seed,
            "reuse_seeds": bool(args.reuse_seeds),
            "run_seeds_by_instance": {
                instance.file_name: seeds_by_instance[index]
                for index, instance in enumerate(instances)
            },
            "seeds_policy": (
                "semillas del experimento original cargadas de "
                "initial_solutions/real/<instancia>/seeds.txt (pareadas dentro "
                "de cada instancia y repeticion)"
                if args.reuse_seeds else
                "semilla MOEA = base + repeticion, identica para todos los "
                "operadores y protocolos; una poblacion inicial por repeticion "
                "compartida por todos"
            ),
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
            "frame_operators": [operator["name"] for operator in operators],
            "nadir_factor": NADIR_FACTOR,
            "degenerate_instances": degenerate,
            "failed_runs": len(failures),
            "budget_fallbacks": fallbacks,
            "save_fronts": not args.no_save_fronts,
        },
        "operators": rows,
        "operator_files": operator_records,
        "comparisons": comparisons,
        "per_run_scores": scores,
        "per_instance_hv": per_instance_hv,
        "frames": frame_records,
        "failures": [
            {"protocol": protocol, "operator": operator, "repeat": rep,
             "instance": os.path.basename(args.paths[index]), "reason": reason}
            for (protocol, operator, rep, index), reason in sorted(failures.items())
        ],
        "run_seeds": [
            {"instance": instance.file_name, "repeat": rep,
             "seed": seeds_by_instance[index][rep], "operator": operator["name"]}
            for index, instance in enumerate(instances)
            for operator in operators
            for rep in range(args.repeats)
        ],
    }
    with open(os.path.join(out_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    _write_report(
        os.path.join(out_dir, "report.md"), args, rows, comparisons, frames,
        operators, protocols, failures, fallbacks, frame_records,
    )
    if not args.no_save_fronts:
        arrays = {
            f"{protocol}|{operator}|{rep}|{index}": front
            for (protocol, operator, rep, index), front in fronts.items()
        }
        np.savez_compressed(os.path.join(out_dir, "fronts.npz"), **arrays)
    print(
        f"Resultados en {out_dir} (individuos en {init_dir}, "
        f"operadores en {operators_dir})",
        flush=True,
    )


if __name__ == "__main__":
    main()
