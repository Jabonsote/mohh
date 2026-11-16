"""Comparacion pareada de operadores de cruce con el mismo arnes y el mismo marco.

Por que existe: el HV de una corrida de LLaMEA no se puede comparar con el de
otra (el marco GE incluye los candidatos de *esa* corrida) y una sola semilla no
distingue una mejora real del ruido. Este script hace la comparacion que si es
valida:

  * mismos conjuntos de instancias para todos los operadores;
  * **mismas semillas** (el par operador/instancia usa la misma semilla en todos
    los operadores: comparacion pareada, varianza reducida);
  * **un solo marco por instancia**, el frente no dominado consolidado de
    *todas* las configuraciones (todos los operadores x todas las repeticiones),
    normalizado con `(f - vmin) / (vmax - vmin)` y nadir fijo `[1.1] * K`, igual
    que `experiment.py`. Si una instancia no tiene escala se excluye y se reporta;
  * **N repeticiones independientes** (por defecto 31, como `N_EXPERIMENTS` de
    la evaluacion GE) para estimar el ruido y hacer una prueba no parametrica
    pareada (Wilcoxon), con intervalo de confianza de la diferencia y ajuste
    de Holm por comparaciones multiples;
  * **tres tipos de operador en el mismo arnes**: LLaMEA (un `.py` con
    `def crossover(x, y)`), GE (los operadores que escribio la evolucion
    gramatical, `<experimento>/<MODELO>_operators.txt`) y el panel clasico
    (`ox`, `pbx`, `pmx`, `cx`, `erx`);
  * **contrastes modulares entre grupos** (`--group` / `--contrast`): cada grupo
    promedia sus operadores por repeticion y se contrasta contra otro grupo con
    el mismo test pareado (p. ej. "OX,PMX vs GE" y "LLaMEA vs OX,PMX");
  * **protocolo configurable**: `--solver` acepta varios solvers y
    `--mutation-both` evalua con y sin `Swap_Mutation`. `--ge-protocol` activa
    el protocolo de la evaluacion GE (MOEAD, NSGAII y SMSEMOA; ambas mutaciones;
    300/105; 31 repeticiones);
  * un fallo de evaluacion (operador que lanza una excepcion en una instancia)
    se cuenta como HV 0, no se descarta del par: descartarlo haria que un
    operador roto pareciera mejor. ``--fail-policy`` permite abortar o excluir.
  * **todo se guarda** para no repetir el MOEA: `fronts.npz`, `failures.json`,
    CSV por instancia y por operador, contrastes, frentes `.pof` (formato del
    original) y `evaluation.json`. Con `--load-fronts` se reanaliza sin MOEA.

Uso:
    python scripts/compare_operators.py \
        --operator descubierto=results/examples/llamea_found_operators/qwen25coder7b_moead_b100_ge.py \
        --operator ox=classic:ox --operator pbx=classic:pbx \
        --split test --repeats 31 --name b100_vs_clasicos

    # GE vs LLaMEA vs clasicos, con contrastes entre grupos:
    python scripts/compare_operators.py \
        --operator ox=classic:ox --operator pmx=classic:pmx \
        --operator llamea=results/searches/<exp>/llamea_best_operator.py \
        --ge-operators /ruta/Experiment_001 \
        --group base=ox,pmx --group llamea=llamea \
        --contrast llamea:base --ge-protocol

    # Tabla final con el protocolo GE completo (caro):
    python scripts/compare_operators.py ... --ge-protocol

Salida: `results/evaluations/<name>/` con `evaluation.json`, `report.md`,
`hv_por_instancia/`, `fronts/`, `fronts.npz`, `failures.json` y los CSV.
"""

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.core.utilities.instance_utils import instance_paths
from mohh.llamea.fitness import (
    EVAL_BUDGETS,
    NADIR_FACTOR,
    _consolidated_frame,
    _solve_front_namespace,
    solve_front_snapshots,
    build_evaluator,
    clip_to_reference_box,
    comprobate_frame,
    compile_operator,
    init_params,
    uses_unseeded_rng,
    normalize_front,
)
from mohh.llamea.ge_operators import ge_namespace, load_ge_operators
from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for
from mohh.llamea.run import _select_instances
from mohh.llamea.statistics import (
    add_holm,
    minimum_detectable_difference,
    paired_summary,
)


def load_operator(spec):
    """`nombre=ruta.py`, `nombre=classic:ox`, `nombre=ge:Best@archivo`.

    Devuelve `(nombre, namespace, kind, source)` donde `source` es lo necesario
    para reconstruir el namespace en un worker de otro proceso (la clave del
    clasico, la expresion GE o el codigo del operador descubierto); el
    namespace en si no es picklable (contiene `np` y funciones de `exec`).

    `classic2:ox` es el clasico devolviendo **dos** hijos, `(ox(x, y), ox(y, x))`:
    el mismo contrato que un operador que devuelve una tupla. Comparar un
    operador de dos hijos contra clasicos de un solo hijo no es justo, porque el
    arnes (`LLMOperator`) usa `child[::-1]` como segundo descendiente.

    `ge:<etiqueta>@<archivo_operators.txt>` es un operador de la evolucion
    gramatical: la expresion de la linea `<etiqueta>: <expresion>` del archivo.
    Se evalua con las primitivas de `HH_functions`, igual que en la
    evaluacion original, pero a traves del mismo arnes que el resto.
    """
    name, _, source = spec.partition("=")
    name = name.strip()
    if not name or not source:
        raise SystemExit(f"--operator debe ser nombre=fuente, no {spec!r}")
    if source.startswith("classic:"):
        key = source.split(":", 1)[1].strip()
        if key not in REFERENCE_OPERATORS:
            raise SystemExit(f"operador clasico desconocido: {key!r}")
        return name, namespace_for(REFERENCE_OPERATORS[key]), "classic", key
    if source.startswith("classic2:"):
        key = source.split(":", 1)[1].strip()
        if key not in REFERENCE_OPERATORS:
            raise SystemExit(f"operador clasico desconocido: {key!r}")
        single = REFERENCE_OPERATORS[key]

        def two_children(x, y):
            return single(x, y), single(y, x)

        return name, namespace_for(two_children), "classic2", key
    if source.startswith("ge:"):
        label, _, path = source[3:].partition("@")
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise SystemExit(f"--operator GE debe ser ge:etiqueta@archivo, "
                             f"no {spec!r}")
        expression = _ge_expression(path, label)
        return name, ge_namespace(expression), "ge", expression
    with open(source) as handle:
        code = handle.read()
    # El operador se carga sin el filtro de RNG sembrado: el aviso va aparte.
    # Un `np.random.default_rng()` sin semilla hace que el operador no
    # dependa de la semilla del MOEA, asi que los resultados no son
    # reproducibles y la comparacion pareada no es valida.
    namespace = compile_operator(code, check_seeded_rng=False)
    if uses_unseeded_rng(code):
        print(f"AVISO: el operador {name!r} crea su propio generador sin "
              "semilla (np.random.default_rng()). No es reproducible: la "
              "semilla del MOEA no lo controla. Semilla el operador antes de "
              "usar estos resultados.", file=sys.stderr)
    return name, namespace, "llamea", code


def _ge_expression(path, label):
    """Expresion de un operador GE en un `*_operators.txt`."""
    if not os.path.isfile(path):
        raise SystemExit(f"archivo de operadores GE no encontrado: {path!r}")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, expression = line.split(":", 1)
            if name.strip() == label:
                return expression.strip()
    raise SystemExit(f"operador GE {label!r} no encontrado en {path!r}")


def parse_groups(specs):
    """`--group nombre=op1,op2` -> `{nombre: [operadores]}`."""
    groups = {}
    for spec in specs or []:
        name, _, members = spec.partition("=")
        name = name.strip()
        if not name or not members.strip():
            raise SystemExit(f"--group debe ser nombre=op1,op2, no {spec!r}")
        groups[name] = [item.strip() for item in members.split(",")
                        if item.strip()]
    return groups


def parse_contrasts(specs):
    """`--contrast grupoA:grupoB` -> `[(grupoA, grupoB)]`."""
    contrasts = []
    for spec in specs or []:
        left, sep, right = spec.partition(":")
        if not sep:
            raise SystemExit(f"--contrast debe ser grupoA:grupoB, no {spec!r}")
        contrasts.append((left.strip(), right.strip()))
    return contrasts



def _solve_task(task):
    """Worker picklable: evalua una instancia con un operador.

    Reconstruye el namespace a partir de `source`/`kind` (el namespace no es
    picklable) y corre el MOEA con la semilla/solver/mutacion dados. Devuelve
    `(key, front, error)`; `error` es `None` si todo salio bien.
    """
    (keys, source, kind, instance, seed, model_name, mutation,
     instance_timeout, overrides) = task
    try:
        # El worker puede arrancar con el singleton `Params` vacio (fork/
        # forkserver): se inicializa igual que en la ruta de LLaMEA para que
        # el MOEA corra con MO_GENERATIONS, MO_POPULATION_SIZE y las
        # direcciones de referencia correctos.
        init_params(**(overrides or {}))
        if kind == "classic":
            ns = namespace_for(REFERENCE_OPERATORS[source])
        elif kind == "classic2":
            single = REFERENCE_OPERATORS[source]

            def two_children(x, y):
                return single(x, y), single(y, x)

            ns = namespace_for(two_children)
        elif kind == "ge":
            ns = ge_namespace(source)
        else:
            ns = compile_operator(source)
        if isinstance(keys[0], tuple) and len(keys[0]) == 5:
            # Varios presupuestos: una corrida con snapshots devuelve los tres.
            snaps = solve_front_snapshots(
                instance, ns, seed=seed, model_name=model_name,
                mutation=mutation, instance_timeout=instance_timeout,
            )
            empty = np.empty((0, 0), dtype=float)
            return [(key, snaps.get(key[3], empty), None) for key in keys]
        front = _solve_front_namespace(
            instance, ns, seed=seed, model_name=model_name,
            mutation=mutation, instance_timeout=instance_timeout,
        )
        if front is None:
            raise RuntimeError("el MOEA no devolvio un frente")
        return keys[0], front, None
    except Exception as error:  # noqa: BLE001 - se reporta y sigue
        return [(key, np.empty((0, 0), dtype=float),
                 f"{type(error).__name__}: {error}") for key in keys]


def build_protocols(args):
    """Protocolos (solver, mutacion) que se evaluan con el mismo marco.

    La evaluacion GE usa los tres solvers y las dos variantes de mutacion
    (`NullMutation` y `Swap_Mutation`). Para reproducirla:
    ``--solver MOEAD,NSGAII,SMSEMOA --mutation-both``. El default conserva una
    sola combinacion para que una comparacion rapida siga siendo barata.
    """
    solvers = [value.strip() for value in str(args.solver).split(",") if value.strip()]
    if not solvers:
        raise SystemExit("--solver no puede estar vacio")
    if args.mutation_both:
        return [(solver, False) for solver in solvers] + \
               [(solver, True) for solver in solvers]
    return [(solver, bool(args.mutation)) for solver in solvers]


def protocol_name(solver, mutation):
    return f"{solver}+mut" if mutation else solver


def collect_fronts(operators, paths, args, protocols, existing=None,
                   stored_failures=None, labels=None):
    """Corre el MOEA para cada (repeticion, operador, instancia).

    Con `labels` (10k/30k/50k) cada tarea devuelve los frentes de los
    presupuestos de la misma corrida (como el `MyCallback` del original); sin
    `labels`, un solo frente (el final).

    Si `existing` trae frentes ya calculados (con la misma semilla por
    repeticion), esos se reutilizan: sirve para agregar operadores a una
    comparacion ya corrida sin repetir el MOEA de los demas.

    Un fallo se guarda como frente vacio. Asi ``hv_values`` puede contarlo
    como HV 0 en el par (o abortar con ``--fail-policy abort``) en vez de
    eliminar la repeticion entera.
    """
    labels = labels or [None]
    fronts = {} if existing is None else dict(existing)
    failures = {} if stored_failures is None else dict(stored_failures)
    instance_timeout = 300  # s por corrida del MOEA, como el original
    # Las instancias como objetos `Instance` (picklable para el worker) y los
    # overrides de Params: los workers arrancan con el singleton vacio
    # (forkserver) y necesitan MO_GENERATIONS/MO_POPULATION_SIZE y las
    # direcciones de referencia. Con snapshots, MO_GENERATIONS es el ultimo
    # presupuesto (500), no el presupuesto de analisis.
    evaluator = build_evaluator(
        paths, mo_generations=max(EVAL_BUDGETS[label] // 105 for label in labels),
        mo_population_size=105, seed=args.seed,
        model_name=protocols[0][0], mutation=protocols[0][1], normalize=False,
    )
    instances = evaluator._ensure_instances()
    instance_timeout = evaluator.instance_timeout
    overrides = evaluator._params_overrides()

    tasks = []
    for rep in range(args.repeats):
        seed = args.seed + rep
        for solver, mutation in protocols:
            protocol = protocol_name(solver, mutation)
            for name, _ns, kind, source in operators:
                for index, instance in enumerate(instances):
                    todo = [(protocol, name, rep, label, index)
                            for label in labels
                            if (protocol, name, rep, label, index) not in fronts]
                    if not todo:
                        continue
                    tasks.append((todo, source, kind, instance, seed, solver,
                                  mutation, instance_timeout, overrides))

    if not tasks:
        return fronts, failures

    total = sum(len(task[0]) for task in tasks)
    workers = args.fitness_workers
    print(f"  Evaluando {len(tasks)} tareas ({total} frentes) "
          f"con {workers} worker(s)...", flush=True)
    if workers > 1 and len(tasks) > 1:
        try:
            context = multiprocessing.get_context("forkserver")
        except ValueError:  # pragma: no cover - plataformas sin forkserver
            context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=context
        ) as pool:
            futures = {pool.submit(_solve_task, task): task for task in tasks}
            done_count = 0
            started = time.time()
            for future in as_completed(futures):
                for key, front, error in future.result():
                    fronts[key] = front
                    if error:
                        failures[key] = error
                done_count += 1
                if done_count % 5 == 0 or done_count == len(tasks):
                    elapsed = time.time() - started
                    pending = len(tasks) - done_count
                    eta = elapsed / done_count * pending if done_count else 0.0
                    print(f"\r  {done_count}/{len(tasks)} tareas "
                          f"({100.0 * done_count / len(tasks):4.1f} %)  "
                          f"faltan {pending}  "
                          f"ETA {eta / 60:.0f} min   ", end="", flush=True)
            print("", flush=True)
    else:
        for task in tasks:
            for key, front, error in _solve_task(task):
                fronts[key] = front
                if error:
                    failures[key] = error
    return fronts, failures


def _frame_from_points(pieces):
    """Marco de normalizacion de una instancia, como en `experiment.py`.

    El original consolida los puntos de todos los operadores evaluados, saca
    el frente no dominado (`best_front`), le aplica `comprobate` y de ahi
    salen los limites (min/max por objetivo) con los que se normaliza cada
    frente antes de integrar hasta el nadir fijo ``[1.1] * K``.
    """
    points = [np.asarray(piece, dtype=float) for piece in pieces
              if piece is not None and len(piece)]
    if not points:
        return None
    union = np.unique(np.vstack(points), axis=0)
    return comprobate_frame(union)


def shared_frame(fronts, operators, protocols, n_instances, repeats, labels):
    """Marco por instancia y presupuesto: frente consolidado de todo."""
    frames = {}
    for label in labels:
        for index in range(n_instances):
            pieces = [fronts.get((protocol_name(solver, mutation), name, rep,
                                  label, index))
                      for solver, mutation in protocols
                      for name, _ns, _kind, _src in operators
                      for rep in range(repeats)]
            pieces = [p for p in pieces if p is not None and len(p)]
            frames[(label, index)] = _frame_from_points(pieces)
    return frames


def hv_values(fronts, operators, protocols, frames, repeats, labels,
              failures=None, fail_policy="zero"):
    """HV de cada (operador, repeticion, instancia) en el marco comun.

    ``frame is None`` significa que la instancia no tiene escala (``vmin ==
    vmax``) y se excluye para todos los operadores, como en
    ``experiment.py``. Un frente vacio es un fallo del operador en esa
    instancia: con ``fail_policy="zero"`` cuenta como HV 0 (peor caso); con
    ``"exclude"`` se elimina el par; con ``"abort"`` se detiene la comparacion.
    """
    failures = failures or {}
    values = {}
    for label in labels:
      for solver, mutation in protocols:
        protocol = protocol_name(solver, mutation)
        for name, _ns, _kind, _src in operators:
            for rep in range(repeats):
                row = []
                for index in range(len(  # frames[(label, index)]
                        [k for k in frames if k[0] == label])):
                    frame = frames.get((label, index))
                    key = (protocol, name, rep, label, index)
                    front = fronts.get(key)
                    if frame is None:
                        row.append(None)
                        continue
                    if front is None or len(front) == 0:
                        reason = failures.get(key, "sin frente")
                        if fail_policy == "abort":
                            raise RuntimeError(
                                f"fallo en {protocol} {name} rep={rep} "
                                f"inst={index}: {reason}"
                            )
                        if fail_policy == "zero":
                            row.append(0.0)
                        else:
                            row.append(None)
                        continue
                    vmin, vmax = frame.min(axis=0), frame.max(axis=0)
                    if np.any(vmax == vmin):
                        row.append(None)
                        continue
                    normalized = normalize_front(front, (vmin, vmax))
                    if not np.all(np.isfinite(normalized)):
                        row.append(None)
                        continue
                    normalized = clip_to_reference_box(normalized)
                    row.append(float(compute_hypervolume(
                        np.full(normalized.shape[1], NADIR_FACTOR), normalized
                    )))
                values[(protocol, name, rep, label)] = row
    return values


def per_run_scores(values, operators, protocols, repeats):
    """HV medio por repeticion, promediando instancias y protocolos."""
    scores = {}
    for name, _ns, _kind, _src in operators:
        scores[name] = []
        for rep in range(repeats):
            cells = [value
                     for solver, mutation in protocols
                     for value in values[(protocol_name(solver, mutation), name, rep)]
                     if value is not None]
            scores[name].append(float(np.mean(cells)) if cells else None)
    return scores


def pairwise_design(fronts, rows, protocols, n_instances, repeats, failures=None,
                    fail_policy="zero"):
    """Comparacion cabeza a cabeza: marco propio de cada pareja.

    Es lo que seria un experimento GE con dos configuraciones: el marco es el
    frente consolidado de *esas dos* configuraciones (todas sus repeticiones), no
    el de todas. Evita que un tercer operador muy bueno aplaste a los demas a HV
    ~0 (el marco compartido se vuelve degenerado cuando uno domina la envolvente).
    """
    classics = [r["operator"] for r in rows if r["kind"] == "classic" and r.get("runs")]
    others = [r["operator"] for r in rows
              if r["kind"] in ("llamea", "ge") and r.get("runs")]
    results = []
    for ours in others:
        for base in classics:
            frames = []
            for index in range(n_instances):
                pieces = [fronts.get(
                              (protocol_name(solver, mutation), name, rep, index))
                          for solver, mutation in protocols
                          for name in (ours, base)
                          for rep in range(repeats)]
                pieces = [p for p in pieces if p is not None and len(p)]
                frames.append(_frame_from_points(pieces))
            subset = [(ours, None, "llamea", None), (base, None, "classic", None)]
            values = hv_values(fronts, subset, protocols, frames, repeats,
                               failures=failures, fail_policy=fail_policy)
            scores = per_run_scores(values, subset, protocols, repeats)
            if not any(value is not None for value in scores[ours]) or \
                    not any(value is not None for value in scores[base]):
                continue
            summary = paired_summary(scores[ours], scores[base])
            if summary["pairs"] == 0:
                continue
            per_instance_operator = []
            per_instance_baseline = []
            for index in range(n_instances):
                ours_rep = [values[(protocol_name(solver, mutation), ours, rep)][index]
                            for solver, mutation in protocols
                            for rep in range(repeats)
                            if values[(protocol_name(solver, mutation), ours, rep)][index]
                            is not None]
                base_rep = [values[(protocol_name(solver, mutation), base, rep)][index]
                            for solver, mutation in protocols
                            for rep in range(repeats)
                            if values[(protocol_name(solver, mutation), base, rep)][index]
                            is not None]
                per_instance_operator.append(
                    float(np.mean(ours_rep)) if ours_rep else None
                )
                per_instance_baseline.append(
                    float(np.mean(base_rep)) if base_rep else None
                )
            summary.update({
                "operator": ours,
                "baseline": base,
                "per_instance_operator": per_instance_operator,
                "per_instance_baseline": per_instance_baseline,
            })
            results.append(summary)
    return results


def _group_scores(scores, members):
    """Puntuacion de un grupo por repeticion: media de sus operadores."""
    if not members:
        return []
    lengths = [len(scores[name]) for name in members if name in scores]
    n_reps = max(lengths) if lengths else 0
    out = []
    for rep in range(n_reps):
        values = [scores[name][rep] for name in members
                  if name in scores and rep < len(scores[name])
                  and scores[name][rep] is not None]
        out.append(float(np.mean(values)) if values else None)
    return out


def group_contrasts(fronts, rows, protocols, n_instances, repeats, groups,
                     contrasts, failures=None, fail_policy="zero"):
    """Contraste pareado entre **grupos** de operadores.

    Cada grupo tiene una puntuacion por repeticion: la media de sus operadores
    (y del HV por instancia), igual que `per_run_scores`. El marco de
    normalizacion es el consolidado de los dos grupos comparados (el mismo
    criterio que el par head-to-head), de modo que el contraste no dependa de
    los operadores que quedan fuera.

    Es la forma modular de plantear "OX,PMX vs GE" y "LLaMEA vs OX,PMX": los
    grupos se declaran con `--group` y se contrastan con `--contrast`.
    """
    known = {r["operator"] for r in rows if r.get("runs")}
    results = []
    for left, right in contrasts:
        members_left = [name for name in groups.get(left, []) if name in known]
        members_right = [name for name in groups.get(right, []) if name in known]
        if not members_left or not members_right:
            continue
        subset = [(name, None, "group", None)
                  for name in members_left + members_right]
        frames = []
        for index in range(n_instances):
            pieces = [fronts.get(
                          (protocol_name(solver, mutation), name, rep, index))
                      for solver, mutation in protocols
                      for name in members_left + members_right
                      for rep in range(repeats)]
            pieces = [p for p in pieces if p is not None and len(p)]
            frames.append(_frame_from_points(pieces))
        values = hv_values(fronts, subset, protocols, frames, repeats,
                            failures=failures, fail_policy=fail_policy)
        scores = per_run_scores(values, subset, protocols, repeats)
        left_scores = _group_scores(scores, members_left)
        right_scores = _group_scores(scores, members_right)
        summary = paired_summary(left_scores, right_scores)
        if summary["pairs"] == 0:
            continue
        summary.update({
            "group": left,
            "baseline_group": right,
            "members": members_left,
            "baseline_members": members_right,
        })
        results.append(summary)
    return results


def summarize(scores, operators):
    rows = []
    for name, _ns, kind, _src in operators:
        values = [v for v in scores[name] if v is not None]
        if not values:
            rows.append({"operator": name, "kind": kind, "runs": 0})
            continue
        rows.append({
            "operator": name,
            "kind": kind,
            "runs": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return rows


def pairwise(scores, operators, rows):
    """Cada operador no clasico (LLaMEA o GE) contra cada clasico.

    El contraste es pareado por repeticion: el clasico y el evaluado comparten
    semilla, poblacion inicial y marco, asi que la diferencia es del operador.
    """
    classics = [r["operator"] for r in rows if r["kind"] == "classic" and r.get("runs")]
    others = [r["operator"] for r in rows
              if r["kind"] in ("llamea", "ge") and r.get("runs")]
    comparisons = []
    for ours in others:
        for base in classics:
            summary = paired_summary(scores[ours], scores[base])
            if summary["pairs"] == 0:
                continue
            summary.update({"operator": ours, "baseline": base})
            comparisons.append(summary)
    return comparisons


def rank_stability(scores, operators):
    """Estabilidad del ranking entre repeticiones y diferencia minima detectable.

    Con una sola semilla, el "mejor operador" puede ser ruido del MOEA interno.
    Aqui se rankean los operadores en **cada** repeticion (media sobre
    instancias): si el primero cambia mucho de una repeticion a otra, el ruido
    domina la senal y ninguna mejora por debajo de `mdd_top_two` es creible.
    """
    names = [name for name, _ns, _kind, _src in operators
             if any(v is not None for v in scores.get(name, []))]
    if len(names) < 2:
        return None
    n_repeats = len(scores[names[0]])
    wins = {name: 0 for name in names}
    ranks = {name: [] for name in names}
    for rep in range(n_repeats):
        values = {name: scores[name][rep] for name in names
                  if scores[name][rep] is not None}
        if not values:
            continue
        for position, (name, _value) in enumerate(
                sorted(values.items(), key=lambda kv: -kv[1]), start=1):
            ranks[name].append(position)
            if position == 1:
                wins[name] += 1
    ordered = sorted(
        names,
        key=lambda n: -float(np.mean([v for v in scores[n] if v is not None] or [0.0])),
    )
    mdd = None
    if len(ordered) >= 2:
        top, second = ordered[0], ordered[1]
        mdd = minimum_detectable_difference(scores[top], scores[second])
    return {
        "repeats": n_repeats,
        "wins_as_best": wins,
        "mean_rank": {name: (float(np.mean(ranks[name])) if ranks[name] else None)
                      for name in names},
        "rank1_changes": sum(1 for name in names if wins[name] > 0),
        "top_two": ordered[:2],
        "mdd_top_two": mdd,
    }


def _fmt(value, digits=6, signed=False):
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _p(value):
    return "-" if value is None else f"{value:.4f}"


def _ci(row):
    low, high = row.get("ci95_low"), row.get("ci95_high")
    if low is None or high is None:
        return "-"
    return f"[{low:+.6f}, {high:+.6f}]"


def write_front(front, file_path):
    """Escribe un frente en el formato `.pof` del original.

    Cabecera `# N M` y una fila por punto en notacion cientifica: el mismo
    formato que escribe `mohh/evaluation/experiment.py`, para que los frentes
    de esta comparacion se puedan leer con las herramientas del original.
    """
    array = np.asarray(front, dtype=float)
    if array.ndim != 2:
        return
    n_rows, n_cols = array.shape
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as handle:
        handle.write(f"# {n_rows} {n_cols}\n")
        for row in array:
            handle.write(" ".join(f"{num:1.6e}" for num in row) + "\n")


# --------------------------------------------------------------------------- #
# Estructura de la evaluacion original (`experiment.py`)
# --------------------------------------------------------------------------- #
def budget_label(mo_generations, mo_population_size):
    """Etiqueta del presupuesto como en el original.

    El original evalua tres presupuestos ("10k", "30k", "50k") que son
    100/300/500 generaciones con poblacion 105. Aqui solo se evalua el
    final, asi que la etiqueta sale del propio presupuesto.
    """
    budget = int(mo_generations) * int(mo_population_size)
    thousands = round(budget / 1050)
    if thousands >= 1:
        return f"{thousands}k"
    return f"{mo_generations}x{mo_population_size}"


def ge_column_name(solver, mutation, name):
    """`<solver>_<operador>_<NM|WM>_<solver>`, como en `experiment.py`."""
    return f"{solver}_{name}_{'WM' if mutation else 'NM'}_{solver}"


def write_operator_catalog(out_dir, operators):
    """Guarda los operadores usados: `operators.txt` (una linea por operador).

    Es el equivalente del `MOEAD_operators.txt` que deja la evolucion
    gramatical, pero con **todos** los operadores de la comparacion (los de
    la GE, los clasicos y el descubierto por LLaMEA), para que el resultado
    sea reproducible sin el comando que lo lanzo.
    """
    path = os.path.join(out_dir, "operators.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for name, _ns, kind, source in operators:
            if kind in ("classic", "classic2"):
                text = f"{kind}:{source}"
            else:
                text = source
            handle.write(f"{name}: {text}\n")
    return path


def write_ge_layout(out_dir, args, fronts, values, frames, operators, protocols,
                    failures=None, labels=None):
    """Escribe la evaluacion con la estructura del original (`experiment.py`).

    * `<solver>/<presupuesto>/<instancia>.csv`: una fila por repeticion y una
      columna por operador, con el nombre `<solver>_<operador>_<NM|WM>_<solver>`
      que usa el original. Un CSV por presupuesto (10k/30k/50k, como el
      original).
    * `<solver>/<presupuesto>/fronts/<instancia>/<columna>.pof` (y `_norm`):
      el frente de la repeticion mediana; y `best_front_consolidated.pof` con
      el marco consolidado de la instancia.
    * `run_metrics/evaluation_<instancia>.json`: metricas por instancia.
    """
    failures = failures or {}
    labels = labels or [budget_label(args.mo_generations, args.mo_population_size)]
    solvers = []
    for solver, _mutation in protocols:
        if solver not in solvers:
            solvers.append(solver)

    # Columnas por solver: (solver, mutacion, protocolo, operador, columna).
    columns = {solver: [] for solver in solvers}
    for solver, mutation in protocols:
        for name, _ns, _kind, _src in operators:
            columns[solver].append((solver, mutation,
                                    protocol_name(solver, mutation), name,
                                    ge_column_name(solver, mutation, name)))

    for solver in solvers:
      for label in labels:
        for index, path in enumerate(args.paths):
            # El original nombra los ficheros sin extension (`KC11-2fl-1rl.csv`).
            instance = os.path.splitext(os.path.basename(path))[0]
            folder = os.path.join(out_dir, solver, label)
            os.makedirs(folder, exist_ok=True)

            # -- CSV por instancia (formato del original) --------------------
            with open(os.path.join(folder, f"{instance}.csv"), "w",
                      newline="", encoding="utf-8") as handle:
                # `\n` como el original (pandas `to_csv`), no `\r\n`.
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(["Experiment"]
                                + [col for _s, _m, _p, _n, col in columns[solver]])
                for rep in range(args.repeats):
                    row = [rep + 1]
                    for _s, _m, protocol, name, _col in columns[solver]:
                        values_row = values.get((protocol, name, rep, label))
                        if values_row is not None and index < len(values_row) \
                                and values_row[index] is not None:
                            row.append(f"{values_row[index]:.10g}")
                        else:
                            row.append("")
                    writer.writerow(row)

            # -- Frentes (repeticion mediana por operador) ------------------
            frame = frames.get((label, index)) if isinstance(frames, dict) else (
                frames[index] if index < len(frames) else None)
            bounds = None
            if frame is not None and len(frame):
                vmin, vmax = frame.min(axis=0), frame.max(axis=0)
                if not np.any(vmax == vmin):
                    bounds = (vmin, vmax)
            front_dir = os.path.join(folder, "fronts", instance)
            for solver_p, mutation, _protocol, name, col in columns[solver]:
                selected = _median_front(fronts, values, name, index,
                                         [(solver_p, mutation)], args.repeats,
                                         label=label)
                if selected is None:
                    continue
                write_front(selected, os.path.join(front_dir, f"{col}.pof"))
                if bounds is not None:
                    write_front(clip_to_reference_box(
                        normalize_front(selected, bounds)),
                        os.path.join(front_dir, f"{col}_norm.pof"))
            if frame is not None and len(frame):
                write_front(frame, os.path.join(front_dir,
                                                "best_front_consolidated.pof"))
                if bounds is not None:
                    write_front(clip_to_reference_box(
                        normalize_front(frame, bounds)),
                        os.path.join(front_dir,
                                     "best_front_consolidated_norm.pof"))

            # -- Metricas por instancia -------------------------------------
            metrics = {
                "phase": "evaluation",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "instance": instance,
                "solver": solver,
                "budget": label,
                "counters": {
                    "evaluations": len(columns[solver]) * args.repeats,
                    "failures": sum(1 for key in failures
                                    if key[0] in (protocol_name(solver, False),
                                                  protocol_name(solver, True))),
                },
            }
            metrics_dir = os.path.join(out_dir, "run_metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir,
                                   f"evaluation_{instance}.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(metrics, handle, indent=1)


def write_instance_artifacts(out_dir, args, fronts, values, frames, operators,
                             protocols):
    """Artefactos por instancia, como deja la evaluacion original.

    * `hv_por_instancia/<instancia>.csv`: una fila por repeticion y una
      columna por operador (el HV de esa instancia), igual que el CSV del
      original (`<solver>/<label>/<instancia>.csv`).
    * `fronts/<instancia>/<operador>.pof` y `_norm.pof`: el frente de la
      repeticion mediana de cada operador, crudo y normalizado.
    * `fronts/<instancia>/best_front_consolidated.pof` y `_norm.pof`: el marco
      consolidado de la instancia, crudo y normalizado.
    """
    import csv

    csv_dir = os.path.join(out_dir, "hv_por_instancia")
    os.makedirs(csv_dir, exist_ok=True)
    for index, path in enumerate(args.paths):
        instance = os.path.basename(path)
        # CSV por instancia: filas = repeticion, columnas = operador.
        with open(os.path.join(csv_dir, f"{instance}.csv"), "w",
                  newline="") as handle:
            writer = csv.writer(handle)
            header = ["Experiment"]
            for name, _ns, _kind, _src in operators:
                header.append(name)
            writer.writerow(header)
            for rep in range(args.repeats):
                row = [rep + 1]
                for name, _ns, _kind, _src in operators:
                    cells = []
                    for solver, mutation in protocols:
                        values_row = values.get((protocol_name(solver, mutation),
                                                name, rep))
                        if values_row is not None and index < len(values_row) \
                                and values_row[index] is not None:
                            cells.append(values_row[index])
                    row.append(f"{np.mean(cells):.10g}" if cells else "")
                writer.writerow(row)

        # Frentes: mediana por operador y consolidado de la instancia.
        frame = frames[index] if index < len(frames) else None
        bounds = None
        if frame is not None and len(frame):
            vmin, vmax = frame.min(axis=0), frame.max(axis=0)
            if not np.any(vmax == vmin):
                bounds = (vmin, vmax)
        for name, _ns, _kind, _src in operators:
            selected = _median_front(fronts, values, name, index, protocols,
                                     args.repeats)
            if selected is None:
                continue
            write_front(selected, os.path.join(out_dir, "fronts", instance,
                                               f"{name}.pof"))
            if bounds is not None:
                write_front(clip_to_reference_box(
                    normalize_front(selected, bounds)),
                    os.path.join(out_dir, "fronts", instance, f"{name}_norm.pof"))
        if frame is not None and len(frame):
            write_front(frame, os.path.join(out_dir, "fronts", instance,
                                            "best_front_consolidated.pof"))
            if bounds is not None:
                write_front(clip_to_reference_box(normalize_front(frame, bounds)),
                            os.path.join(out_dir, "fronts", instance,
                                         "best_front_consolidated_norm.pof"))


def _median_front(fronts, values, name, index, protocols, repeats, label=None):
    """Frente de la repeticion mediana (por HV) de un operador en una instancia."""
    suffix = (label,) if label is not None else ()
    scores = []
    for rep in range(repeats):
        cells = []
        for solver, mutation in protocols:
            row = values.get((protocol_name(solver, mutation), name, rep, *suffix))
            if row is not None and index < len(row) and row[index] is not None:
                cells.append(row[index])
        scores.append(float(np.mean(cells)) if cells else None)
    valid = [(rep, value) for rep, value in enumerate(scores)
             if value is not None]
    if not valid:
        return None
    valid.sort(key=lambda item: item[1])
    median = valid[len(valid) // 2][0]
    for solver, mutation in protocols:
        front = fronts.get((protocol_name(solver, mutation), name, median,
                            *suffix, index))
        if front is not None and len(front):
            return np.asarray(front, dtype=float)
    return None


def write_csvs(out_dir, args, rows, comparisons, pairwise_results, values,
               operators, protocols, group_results=None):
    """CSV con todo el detalle: HV por instancia, resumen y contrastes.

    Se escriben aparte del `evaluation.json` para que el analisis se pueda
    re-hacer (o pegar en una hoja) sin volver a correr el MOEA interno.
    """
    import csv

    # 1. HV por (operador, protocolo, repeticion, instancia).
    path = os.path.join(out_dir, "hv_por_instancia.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operador", "tipo", "protocolo", "repeticion",
                         "instancia", "hv"])
        for name, _ns, kind, _src in operators:
            for solver, mutation in protocols:
                protocol = protocol_name(solver, mutation)
                for rep in range(args.repeats):
                    row = values.get((protocol, name, rep))
                    if row is None:
                        continue
                    for index, hv in enumerate(row):
                        writer.writerow([
                            name, kind, protocol, rep,
                            os.path.basename(args.paths[index]),
                            "" if hv is None else f"{hv:.10g}",
                        ])

    # 2. Resumen por operador (media de repeticiones).
    path = os.path.join(out_dir, "resumen_operadores.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operador", "tipo", "repeticiones", "media", "desv",
                         "mediana", "min", "max"])
        for row in sorted(rows, key=lambda r: -(r.get("mean") or -1)):
            writer.writerow([
                row["operator"], row["kind"], row.get("runs", 0),
                _csv_num(row.get("mean")), _csv_num(row.get("sd")),
                _csv_num(row.get("median")), _csv_num(row.get("min")),
                _csv_num(row.get("max")),
            ])

    # 3. Contrastes estadisticos (marco compartido, por pareja y por grupo).
    path = os.path.join(out_dir, "contrastes.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tipo", "operador", "baseline", "pares", "faltan",
                         "gana", "pierde", "empata", "media_op", "media_base",
                         "dif_media", "sd_dif", "ic95_low", "ic95_high",
                         "relativo", "dz", "wilcoxon", "p", "p_holm"])
        for tipo, items, label_a, label_b in (
                ("marco_compartido", comparisons, "operator", "baseline"),
                ("por_pareja", pairwise_results, "operator", "baseline"),
                ("por_grupo", group_results or [], "group", "baseline_group")):
            for item in items:
                writer.writerow([
                    tipo, item.get(label_a), item.get(label_b),
                    item.get("pairs"), item.get("missing"), item.get("wins"),
                    item.get("losses"), item.get("ties"),
                    _csv_num(item.get("operator_mean")),
                    _csv_num(item.get("baseline_mean")),
                    _csv_num(item.get("mean_difference")),
                    _csv_num(item.get("sd_difference")),
                    _csv_num(item.get("ci95_low")),
                    _csv_num(item.get("ci95_high")),
                    _csv_num(item.get("relative")),
                    _csv_num(item.get("cohens_dz")),
                    _csv_num(item.get("wilcoxon_statistic")),
                    _csv_num(item.get("wilcoxon_p")),
                    _csv_num(item.get("wilcoxon_p_holm")),
                ])


def _csv_num(value):
    if value is None:
        return ""
    try:
        return f"{float(value):.10g}"
    except (TypeError, ValueError):
        return str(value)


def write_report(path, name, args, rows, comparisons, frames, scores, values,
                 operators, protocols, pairwise_results=None, stability=None,
                 failures=None, group_results=None):
    lines = [f"# Comparacion de operadores: {name}", ""]
    lines += [
        "## Diseno",
        "",
        f"- Instancias: {args.split} ({len(args.paths)}), semilla base {args.seed}.",
        f"- Repeticiones independientes: {args.repeats} (semilla = base + repeticion).",
        "- Semillas por repeticion: " + ", ".join(
            f"rep{rep}={args.seed + rep}" for rep in range(args.repeats)),
        f"- MOEA interno: {args.mo_generations} generaciones, poblacion "
        f"{args.mo_population_size}.",
        "- Protocolos evaluados (solver, mutacion): " + ", ".join(
            f"`{protocol_name(solver, mutation)}`" for solver, mutation in protocols),
        "- Mismas semillas para todos los operadores (comparacion pareada) y un "
        "solo marco por instancia: frente consolidado de todas las "
        "configuraciones, normalizacion `(f - vmin)/(vmax - vmin)` y nadir "
        f"`[{NADIR_FACTOR}] * K` (como `experiment.py`).",
        "",
    ]
    degenerate = [os.path.basename(args.paths[i]) for i, f in enumerate(frames)
                  if f is None or np.any(f.max(axis=0) == f.min(axis=0))]
    lines += ["- Instancias sin escala (excluidas de la media): "
              + (", ".join(degenerate) if degenerate else "ninguna"), ""]
    if failures:
        if args.fail_policy == "zero":
            policy_note = "Un fallo cuenta como HV 0 en el par; no se descarta la instancia."
        elif args.fail_policy == "exclude":
            policy_note = "Un fallo descarta ese par (usa `zero` para el peor caso)."
        else:
            policy_note = "Un fallo aborta la comparacion."
        lines += [f"- Fallos de evaluacion contabilizados: {len(failures)} "
                  f"(politica `{args.fail_policy}`). {policy_note}", ""]
        for key, reason in sorted(failures.items()):
            index = key[-1] if len(key) == 5 else key[3]
            operator, rep = key[1], key[2]
            protocol = key[0]
            lines.append(f"  - `{protocol}` `{operator}` rep={rep} "
                         f"`{os.path.basename(args.paths[index])}`: {reason}")
        lines.append("")

    if stability:
        lines += ["## Ruido vs senal (estabilidad del ranking)", "",
                  "Ranking de cada operador en cada repeticion (media sobre instancias).",
                  "Si el primero cambia entre repeticiones, el ruido del MOEA interno",
                  "domina y toda mejora menor que la diferencia minima detectable no es",
                  "creible.", "",
                  "| Operador | Veces 1ro | Rank medio |", "|---|---:|---:|"]
        for op in sorted(stability["wins_as_best"], key=lambda n: -stability["wins_as_best"][n]):
            mean_rank = stability["mean_rank"].get(op)
            rank_txt = "-" if mean_rank is None else f"{mean_rank:.2f}"
            lines.append(f"| {op} | {stability['wins_as_best'][op]}"
                         f"/{stability['repeats']} | {rank_txt} |")
        lines.append("")
        if stability.get("mdd_top_two") is not None:
            top_two = " vs ".join(stability["top_two"])
            lines.append(f"- Diferencia minima detectable ({top_two}, 95%): "
                         f"**{stability['mdd_top_two']:.6f}**. Por debajo de eso, "
                         "dos operadores no se distinguen con estas repeticiones.")
            lines.append("")

    lines += ["## HV por operador (media de las repeticiones)", "",
              "| Operador | Tipo | Reps | Media | Desv. | Mediana | Min | Max |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(rows, key=lambda r: -r.get("mean", -1)):
        if not row.get("runs"):
            lines.append(f"| {row['operator']} | {row['kind']} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {row['operator']} | {row['kind']} | {row['runs']} | "
            f"{row['mean']:.6f} | {row['sd']:.6f} | {row['median']:.6f} | "
            f"{row['min']:.6f} | {row['max']:.6f} |"
        )
    lines.append("")

    if len(protocols) > 1:
        lines += ["## HV por protocolo (media de instancias y repeticiones)", "",
                  "El test principal promedia los protocolos; esta tabla muestra si "
                  "la conclusion depende de un solver o de la mutacion.", "",
                  "| Operador | Protocolo | Media |", "|---|---|---:|"]
        for name, _ns, _kind, _src in operators:
            for solver, mutation in protocols:
                protocol = protocol_name(solver, mutation)
                cells = [value
                         for rep in range(args.repeats)
                         for value in values[(protocol, name, rep)]
                         if value is not None]
                mean = f"{np.mean(cells):.6f}" if cells else "-"
                lines.append(f"| {name} | {protocol} | {mean} |")
        lines.append("")

    if pairwise_results:
        lines += ["## Cabeza a cabeza (marco propio de la pareja)", "",
                  "Cada pareja se renormaliza con el frente consolidado de **esas dos**",
                  "configuraciones (lo que seria un experimento GE de dos configuraciones),",
                  "para que un tercer operador no aplaste a los demas a HV ~0. Las medias",
                  "se calculan solo sobre los pares validos (misma repeticion).", "",
                  "| Operador | vs | Pares | Falta | Media op | Media clasico | "
                  "Gana | Pierde | Dif. media [IC95] | Relativo | dz | p | p Holm |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in pairwise_results:
            rel = "-" if c["relative"] is None else f"{100 * c['relative']:+.1f}%"
            dz = "-" if c["cohens_dz"] is None else f"{c['cohens_dz']:.2f}"
            lines.append(
                f"| {c['operator']} | {c['baseline']} | {c['pairs']} | "
                f"{c['missing']} | {_fmt(c['operator_mean'])} | "
                f"{_fmt(c['baseline_mean'])} | {c['wins']} | {c['losses']} | "
                f"{_fmt(c['mean_difference'], signed=True)} {_ci(c)} | {rel} | "
                f"{dz} | {_p(c['wilcoxon_p'])} | "
                f"{_p(c.get('wilcoxon_p_holm'))} |"
            )
        lines.append("")

    if group_results:
        lines += ["## Contraste entre grupos", "",
                  "Cada grupo promedia sus operadores por repeticion (misma "
                  "semilla, misma poblacion inicial). El marco de "
                  "normalizacion es el consolidado de los dos grupos "
                  "comparados.", "",
                  "| Grupo | vs | Miembros | Pares | Falta | Gana | Pierde | "
                  "Dif. media [IC95] | Relativo | dz | p | p Holm |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in group_results:
            rel = "-" if c["relative"] is None else f"{100 * c['relative']:+.1f}%"
            dz = "-" if c["cohens_dz"] is None else f"{c['cohens_dz']:.2f}"
            lines.append(
                f"| {c['group']} | {c['baseline_group']} | "
                f"{', '.join(c['members'])} vs {', '.join(c['baseline_members'])} | "
                f"{c['pairs']} | {c['missing']} | {c['wins']} | {c['losses']} | "
                f"{_fmt(c['mean_difference'], signed=True)} {_ci(c)} | {rel} | "
                f"{dz} | {_p(c['wilcoxon_p'])} | "
                f"{_p(c.get('wilcoxon_p_holm'))} |"
            )
        lines.append("")

    if comparisons:
        lines += ["## Comparaciones pareadas (marco compartido de todos)", "",
                  "| Operador | vs | Pares | Falta | Gana | Pierde | Empata | "
                  "Dif. media [IC95] | Relativo | dz | p | p Holm |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in comparisons:
            rel = "-" if c["relative"] is None else f"{100 * c['relative']:+.1f}%"
            dz = "-" if c["cohens_dz"] is None else f"{c['cohens_dz']:.2f}"
            lines.append(
                f"| {c['operator']} | {c['baseline']} | {c['pairs']} | "
                f"{c['missing']} | {c['wins']} | {c['losses']} | {c['ties']} | "
                f"{_fmt(c['mean_difference'], signed=True)} {_ci(c)} | {rel} | "
                f"{dz} | {_p(c['wilcoxon_p'])} | "
                f"{_p(c.get('wilcoxon_p_holm'))} |"
            )
        lines += ["", "p de Wilcoxon pareado de dos colas; `p Holm` controla la "
                  "familia de comparaciones de la tabla. La diferencia media y su "
                  "IC95 se calculan sobre los pares con ambos valores; `Falta` "
                  "cuenta repeticiones no pareadas. Con "
                  f"{args.repeats} pares el p minimo alcanzable es "
                  f"{2 / 2 ** args.repeats:.4f}.", ""]

    lines += ["## HV por instancia (media de protocolos y repeticiones)", ""]
    header = "| Instancia | " + " | ".join(name for name, _n, _k, _src in operators) + " |"
    lines += [header, "|---|" + "---:|" * len(operators)]
    for index in range(len(args.paths)):
        cells = []
        for name, _ns, _kind, _src in operators:
            vals = [values[(protocol_name(solver, mutation), name, rep)][index]
                    for solver, mutation in protocols
                    for rep in range(args.repeats)
                    if values[(protocol_name(solver, mutation), name, rep)][index]
                    is not None]
            cells.append("-" if not vals else f"{np.mean(vals):.6f}")
        lines.append(f"| {os.path.basename(args.paths[index])} | " + " | ".join(cells) + " |")
    lines.append("")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--operator", action="append", required=False,
                        help="nombre=ruta.py (LLaMEA), nombre=classic:ox, "
                             "nombre=ge:Etiqueta@archivo_operators.txt (GE). "
                             "Repetible.")
    parser.add_argument("--ge-operators", action="append", default=[],
                        help="Experimento GE del que tomar sus operadores "
                             "(`<modelo>_operators.txt`: Best/Middle/Worst). "
                             "Repetible.")
    parser.add_argument("--group", action="append", default=[],
                        help="grupo=nombre1,nombre2: agrupa operadores para "
                             "los contrastes entre grupos. Repetible.")
    parser.add_argument("--contrast", action="append", default=[],
                        help="grupoA:grupoB: contraste pareado entre dos "
                             "grupos (p. ej. base:ge, llamea:base). "
                             "Repetible.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--instances", default="8",
                        help="Numero de instancias del split, o 'all'. Usar el "
                             "mismo valor que TEST_INSTANCES para que la "
                             "evaluacion final y la comparacion sean pareadas.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="repeticiones independientes (default 31, como "
                             "N_EXPERIMENTS de la evaluacion GE)")
    parser.add_argument("--mo-generations", type=int, default=None,
                        help="generaciones del MOEA interno (default 30; 300 con --ge-protocol)")
    parser.add_argument("--mo-population-size", type=int, default=None,
                        help="poblacion del MOEA interno (default 30; 105 con --ge-protocol)")
    parser.add_argument("--solver", default=None,
                        help="MOEAD, NSGAII, SMSEMOA o lista separada por comas "
                             "(default MOEAD; los tres con --ge-protocol)")
    parser.add_argument("--mutation", action="store_true",
                        help="evaluar con Swap_Mutation")
    parser.add_argument("--mutation-both", action="store_true",
                        help="evaluar con y sin Swap_Mutation (como la evaluacion GE)")
    parser.add_argument("--ge-protocol", action="store_true",
                        help="atajo para la evaluacion GE: los tres solvers, ambas "
                             "mutaciones, 300/105 y 11 repeticiones. Los valores "
                             "explicitos de --solver/--mutation/--mo-* tienen prioridad.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fitness-workers", type=int,
                        default=os.cpu_count() or 1,
                        help="procesos que corren el MOEA en paralelo")
    parser.add_argument("--out", default="results/evaluations")
    parser.add_argument("--name", default=None)
    parser.add_argument("--budget", default=None,
                        help="presupuesto del analisis (resumen, contrastes, "
                             "report); por defecto 30k si se evaluo, y si no "
                             "el primero; con GE_PROTOCOL=1 se evaluan los "
                             "tres (10k/30k/50k) y se escriben los tres")
    parser.add_argument("--fail-policy", choices=["zero", "exclude", "abort"],
                        default="zero",
                        help="que hacer si un operador falla en una instancia: "
                             "zero = HV 0 (default, conserva el par); exclude = "
                             "descarta el par; abort = detiene la comparacion.")
    parser.add_argument("--save-fronts", action="store_true", default=True,
                        help="guarda los frentes en fronts.npz (por defecto: "
                             "permite reanalizar sin repetir el MOEA)")
    parser.add_argument("--no-save-fronts", dest="save_fronts",
                        action="store_false",
                        help="no guardar los frentes (solo el informe)")
    parser.add_argument("--load-fronts", action="store_true",
                        help="reutiliza fronts.npz del directorio de salida para los (operador, repeticion, instancia) ya calculados")
    args = parser.parse_args()

    # Defaults que dependen de --ge-protocol. Un valor explicito del usuario
    # siempre gana; `None` distingue "no lo especifique" de "quiero 30/30".
    if args.repeats is None:
        args.repeats = 31
    if args.mo_generations is None:
        args.mo_generations = 300 if args.ge_protocol else 30
    if args.mo_population_size is None:
        args.mo_population_size = 105 if args.ge_protocol else 30
    if args.solver is None:
        args.solver = ("MOEAD,NSGAII,SMSEMOA" if args.ge_protocol else "MOEAD")
    if args.ge_protocol and not args.mutation:
        args.mutation_both = True
    protocols = build_protocols(args)

    init_params(RANDOM_SEED=args.seed)
    all_paths = instance_paths(train=args.split == "train")
    args.paths = _select_instances(all_paths, args.instances, args.seed)
    operators = [load_operator(spec) for spec in args.operator]
    # Operadores de la evolucion gramatical: se toman del experimento tal
    # como los exporta el original (`<modelo>_operators.txt`).
    for experiment_path in args.ge_operators:
        found = load_ge_operators(experiment_path)
        if not found:
            raise SystemExit(
                f"no se encontraron operadores GE en {experiment_path!r} "
                "(se esperaba un archivo `<modelo>_operators.txt`)")
        for label, expression in found.items():
            operators.append((label, ge_namespace(expression), "ge", expression))
    if not operators:
        raise SystemExit("indica al menos un operador con --operator o "
                         "--ge-operators")
    names = [name for name, _ns, _kind, _src in operators]
    duplicated = {name for name in names if names.count(name) > 1}
    if duplicated:
        raise SystemExit(f"nombres de operador repetidos: {sorted(duplicated)}")

    name = args.name or "vs_".join(n for n, _ns, _k, _src in operators)
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Operadores: {', '.join(n for n, _a, _b, _src in operators)}")
    print(f"Instancias: {args.split} ({len(args.paths)}), repeticiones: {args.repeats}")
    print("Protocolos: " + ", ".join(
        protocol_name(solver, mutation) for solver, mutation in protocols))
    default_protocol = protocol_name(*protocols[0])
    existing = None
    existing_failures = {}
    if args.load_fronts:
        npz_path = os.path.join(out_dir, "fronts.npz")
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                existing = {}
                for key in data.files:
                    parts = key.split("|")
                    if len(parts) == 3:
                        # Formato anterior: un solo solver/mutacion.
                        op_name, rep, index = parts
                        protocol = default_protocol
                    elif len(parts) == 4:
                        protocol, op_name, rep, index = parts
                    else:
                        raise SystemExit(f"clave invalida en fronts.npz: {key!r}")
                    existing[(protocol, op_name, int(rep), int(index))] = data[key]
            print(f"Reutilizando {len(existing)} frentes de {npz_path}")
            # Los fallos se recargan con su motivo exacto: sin esto, un par
            # fallido se contaria como "fallo recuperado" sin explicacion.
            failures_path = os.path.join(out_dir, "failures.json")
            if os.path.exists(failures_path):
                with open(failures_path) as handle:
                    stored = json.load(handle)
                for key, reason in stored.items():
                    parts = key.split("|")
                    if len(parts) == 4:
                        protocol, op_name, rep, index = parts
                        existing_failures[(protocol, op_name, int(rep),
                                           int(index))] = reason
                    elif len(parts) == 5:
                        protocol, op_name, rep, label, index = parts
                        existing_failures[(protocol, op_name, int(rep), label,
                                           int(index))] = reason
                print(f"Reutilizando {len(stored)} fallos de {failures_path}")
        else:
            print(f"AVISO: no hay {npz_path}; se recalcula todo")
    # Presupuestos de la evaluacion (10k/30k/50k como el original). Con
    # GE_PROTOCOL=1 los tres salen de la misma corrida (snapshots); sin el,
    # uno solo (el final).
    labels = list(EVAL_BUDGETS) if args.ge_protocol else [budget_label(
        args.mo_generations, args.mo_population_size)]
    init_params(MO_GENERATIONS=max(EVAL_BUDGETS[label] // 105
                                   for label in labels),
                MO_POPULATION_SIZE=105)

    start = time.time()
    fronts, failures = collect_fronts(operators, args.paths, args, protocols,
                                      existing,
                                      stored_failures=existing_failures,
                                      labels=labels)
    print(f"MOEA terminado en {time.time() - start:.1f}s; calculando HV...")

    frames_all = shared_frame(fronts, operators, protocols, len(args.paths),
                              args.repeats, labels)
    values = hv_values(fronts, operators, protocols, frames_all, args.repeats,
                       labels, failures=failures, fail_policy=args.fail_policy)

    # El analisis (resumen, contrastes, report) se hace sobre el presupuesto
    # `--budget`; los tres se evaluan y se escriben en el layout del original.
    if args.budget is None:
        args.budget = "30k" if "30k" in labels else labels[-1]
    if args.budget not in labels:
        raise SystemExit(f"--budget {args.budget!r} no se evaluo "
                         f"(presupuestos: {', '.join(labels)})")
    values_all, fronts_all = values, fronts
    values_flat = {(protocol, name, rep): row
                   for (protocol, name, rep, label), row in values.items()
                   if label == args.budget}
    fronts_flat = {(protocol, name, rep, index): front
                   for (protocol, name, rep, label, index), front in fronts.items()
                   if label == args.budget}
    frames_flat = [frames_all[(args.budget, index)]
                   for index in range(len(args.paths))]
    values, fronts, frames = values_flat, fronts_flat, frames_flat
    scores = per_run_scores(values, operators, protocols, args.repeats)
    rows = summarize(scores, operators)
    comparisons = add_holm(pairwise(scores, operators, rows))
    pairwise_results = add_holm(
        pairwise_design(fronts, rows, protocols, len(args.paths), args.repeats,
                        failures=failures, fail_policy=args.fail_policy)
    )
    stability = rank_stability(scores, operators)

    groups = parse_groups(args.group)
    contrasts = parse_contrasts(args.contrast)
    for left, right in contrasts:
        for group in (left, right):
            if group not in groups:
                raise SystemExit(
                    f"el contraste {left}:{right} usa el grupo {group!r}, que no "
                    f"esta declarado (usa --group {group}=op1,op2)")
    group_results = []
    if contrasts:
        group_results = add_holm(
            group_contrasts(fronts, rows, protocols, len(args.paths),
                            args.repeats, groups, contrasts,
                            failures=failures, fail_policy=args.fail_policy)
        )

    result = {
        "config": {
            "split": args.split,
            "instances": [os.path.basename(p) for p in args.paths],
            "repeats": args.repeats,
            "seed": args.seed,
            "seeds": {
                "base": args.seed,
                "instance_selection": args.seed,
                "per_repeat": {str(rep): args.seed + rep
                               for rep in range(args.repeats)},
                "policy": ("semilla = base + repeticion; una poblacion inicial por "
                           "repeticion compartida por todos los operadores, de modo "
                           "que la comparacion es pareada"),
            },
            "solver": args.solver,
            "mo_generations": args.mo_generations,
            "mo_population_size": args.mo_population_size,
            "mutation": args.mutation,
            "mutation_both": args.mutation_both,
            "ge_protocol": args.ge_protocol,
            "protocols": [
                {"solver": solver, "mutation": mutation,
                 "name": protocol_name(solver, mutation)}
                for solver, mutation in protocols
            ],
            "fail_policy": args.fail_policy,
            "frame": "consolidated over every operator and repetition",
            "nadir_factor": NADIR_FACTOR,
            "degenerate_instances": [
                os.path.basename(args.paths[i])
                for i, frame in enumerate(frames)
                if frame is None or np.any(frame.max(axis=0) == frame.min(axis=0))
            ],
            "failed_runs": len(failures),
        },
        "operators": rows,
        "comparisons": comparisons,
        "pairwise_head_to_head": pairwise_results,
        "group_contrasts": group_results,
        "groups": groups,
        "rank_stability": stability,
        "per_run_scores": scores,
        "per_protocol_scores": {
            name: {
                protocol_name(solver, mutation): [
                    float(np.mean([
                        v for v in values[(protocol_name(solver, mutation), name, rep)]
                        if v is not None
                    ]))
                    if any(v is not None
                           for v in values[(protocol_name(solver, mutation), name, rep)])
                    else None
                    for rep in range(args.repeats)
                ]
                for solver, mutation in protocols
            }
            for name, _ns, _k, _src in operators
        },
        "per_instance_hv": {
            name: {
                protocol_name(solver, mutation): [
                    values[(protocol_name(solver, mutation), name, rep)]
                    for rep in range(args.repeats)
                ]
                for solver, mutation in protocols
            }
            for name, _ns, _k, _src in operators
        },
        "failures": [
            {"protocol": key[0], "operator": key[1], "repeat": key[2],
             "budget": (key[3] if len(key) == 5 else None),
             "instance": os.path.basename(
                 args.paths[key[-1] if len(key) == 5 else key[3]]),
             "reason": reason}
            for key, reason in sorted(failures.items())
        ],
        # Semilla efectiva de cada corrida del MOEA interno (operador x repeticion).
        "run_seeds": [
            {"operator": name, "repeat": rep, "seed": args.seed + rep}
            for name, _ns, _k, _src in operators
            for rep in range(args.repeats)
        ],
    }
    if args.save_fronts:
        np.savez_compressed(
            os.path.join(out_dir, "fronts.npz"),
            **{f"{protocol}|{name}|{rep}|{index}": np.asarray(front)
               for (protocol, name, rep, index), front in fronts.items()
               if front is not None},
        )
    # Los fallos no caben en el npz (son texto): van aparte, para que al
    # recargar los frentes se sepa que pares son HV 0 y cuales missing.
    with open(os.path.join(out_dir, "failures.json"), "w") as handle:
        json.dump({"|".join(str(part) for part in key): reason
                   for key, reason in failures.items()}, handle, indent=2)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    write_csvs(out_dir, args, rows, comparisons, pairwise_results, values,
               operators, protocols, group_results)
    write_instance_artifacts(out_dir, args, fronts, values, frames, operators,
                             protocols)
    # Estructura de la evaluacion original (experiment.py): <solver>/<presupuesto>/
    # <instancia>.csv, los frentes por solver y las metricas por instancia; mas
    # el catalogo de operadores usados.
    write_ge_layout(out_dir, args, fronts_all, values_all, frames_all,
                    operators, protocols, failures, labels=labels)
    catalog = write_operator_catalog(out_dir, operators)
    write_report(os.path.join(out_dir, "report.md"), name, args, rows,
                 comparisons, frames, scores, values, operators, protocols,
                 pairwise_results, stability, failures, group_results)
    print(f"Guardado en {out_dir}")
    print("  hv_por_instancia.csv  resumen_operadores.csv  contrastes.csv")
    print("  fronts.npz (reutilizable con --load-fronts)  evaluation.json")
    print(f"  {budget_label(args.mo_generations, args.mo_population_size)}/"
          "<solver>/<instancia>.csv  (formato del original)")
    print(f"  {os.path.relpath(catalog, out_dir)}  (operadores usados)")


if __name__ == "__main__":
    main()
