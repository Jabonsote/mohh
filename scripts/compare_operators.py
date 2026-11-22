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
    se puntua con HV 0 para conservar el pareado; descartarlo sesgaria
    la media. ``--fail-policy`` permite abortar o excluir.
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

    # Panel clasico exacto del proyecto original (dos hijos complementarios):
    python scripts/compare_operators.py \
        --operator pmx=project:PMX_Cross --operator cx=project:CX_Cross \
        --operator llamea=results/searches/<exp>/llamea_best_operator.py \
        --split test --repeats 31 --name llamea_vs_pmx_cx_original --ge-protocol

    # Tabla final con el protocolo GE completo (caro):
    python scripts/compare_operators.py ... --ge-protocol

Salida: `results/evaluations/<name>/` con `evaluation.json`, `report.md`,
`hv_por_instancia/`, `fronts/`, `fronts.npz`, `failures.json` y los CSV.
"""

import argparse
import copy
import csv
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np

from mohh.constants import EVAL_BUDGETS, N_EXPERIMENTS
from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.core.utilities.instance_utils import instance_paths
from mohh.evaluation.operators.crossover import CX_Cross, PMX_Cross
from mohh.evaluation.results_layout import (
    ge_column_name,
    random_key_population,
    write_front,
    write_figures,
    write_evaluation_layout,
    write_initial_solutions,
)
from mohh.llamea.fitness import (
    NADIR_FACTOR,
    _consolidated_frame,
    build_evaluator,
    clip_to_reference_box,
    comprobate_frame,
    compile_operator,
    uses_unseeded_rng,
    normalize_front,
)
from mohh.shared.mo import (
    _solve_front_namespace,
    init_params,
    solve_front_snapshots,
)
from mohh.llamea.ge_operators import ge_namespace, load_ge_operators
from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for
from mohh.shared.instances import select_instances as _select_instances
from mohh.shared.statistics import (
    add_holm,
    minimum_detectable_difference,
    paired_summary,
)

from mohh.comparison.engine import (
    _ci,
    _comparison_pairs,
    _fmt,
    _frame_from_points,
    _group_scores,
    _log,
    _p,
    _safe_frame,
    budget_label,
    group_contrasts,
    hv_values,
    pairwise,
    pairwise_design,
    per_run_scores,
    protocol_name,
    rank_stability,
    shared_frame,
    summarize,
)
from mohh.comparison.writers import (
    _csv_num,
    _fronts_payload,
    _median_front,
    save_failures,
    save_fronts,
    save_json,
    write_csvs,
    write_ge_layout,
    write_instance_artifacts,
    write_operator_catalog,
    write_report,
    write_run_metrics_summary,
)


#: Cruces del proyecto original, tal cual los usa `experiment.py`.
#: No son operadores nativos de pymoo: son subclases de `Crossover` definidas
#: en `mohh/evaluation/operators/crossover.py`. Producen **dos hijos
#: complementarios con el mismo corte/inicio**, a diferencia del contrato
#: `LLMOperator` (un hijo + `child[::-1]`).
PROJECT_CROSSOVERS = {
    "PMX_Cross": PMX_Cross,
    "CX_Cross": CX_Cross,
}


def project_crossover(source):
    """Instancia un cruce del proyecto por nombre (`PMX_Cross`, `CX_Cross`)."""
    try:
        cls = PROJECT_CROSSOVERS[source]
    except KeyError:
        raise SystemExit(
            f"cruce del proyecto desconocido: {source!r} "
            f"(disponibles: {', '.join(sorted(PROJECT_CROSSOVERS))})"
        ) from None
    return cls(prob=1.0)


def load_operator(spec):
    """`nombre=ruta.py`, `nombre=classic:ox`, `nombre=ge:Best@archivo`.

    Devuelve `(nombre, namespace, kind, source)` donde `source` es lo necesario
    para reconstruir el namespace en un worker de otro proceso (la clave del
    clasico, la expresion GE o el codigo del operador descubierto); el
    el namespace no es serializable (contiene `np` y funciones `exec`).

    `classic2:ox` es el clasico devolviendo **dos** hijos, `(ox(x, y), ox(y, x))`:
    el mismo contrato que un operador que devuelve una tupla. Comparar un
    operador de dos hijos contra clasicos de un solo hijo no es justo, porque el
    arnes (`LLMOperator`) usa `child[::-1]` como segundo descendiente.
    `classic2` llama dos veces al operador de un hijo, con cortes/inicios
    independientes; no reproduce el corte unico del original.

    `project:PMX_Cross` / `project:CX_Cross` carga la clase real del proyecto
    (`mohh.evaluation.operators.crossover`) y la asigna directamente al solver,
    sin `LLMOperator`. Es la unica forma de obtener el original exacto: dos
    hijos complementarios con el mismo corte/inicio. Los operadores GE tambien
    devuelven dos hijos, asi que con ellos el contrato coincide; el operador
    descubierto por LLaMEA devuelve uno solo y `LLMOperator` usa
    `child[::-1]` como segundo, asi que con LLaMEA el contrato no es identico
    (para eso esta `classic:`).

    `ge:<etiqueta>@<archivo_operators.txt>` es un operador de la evolucion
    gramatical: la expresion de la linea `<etiqueta>: <expresion>` del archivo.
    Se evalua con las primitivas de `HH_functions`, igual que en la
    evaluacion original, pero a traves del mismo arnes que el resto.
    """
    name, _, source = spec.partition("=")
    name = name.strip()
    if not name or not source:
        raise SystemExit(f"--operator debe ser nombre=fuente, no {spec!r}")
    if source.startswith("project:"):
        key = source.split(":", 1)[1].strip()
        if key not in PROJECT_CROSSOVERS:
            raise SystemExit(
                f"cruce del proyecto desconocido: {key!r} "
                f"(disponibles: {', '.join(sorted(PROJECT_CROSSOVERS))})"
            )
        return name, None, "project", key
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
    """Evalua una instancia con un operador en el proceso worker.

    Reconstruye el namespace a partir de `source`/`kind` (el namespace no es
    ) y corre el MOEA con la semilla, solver y mutacion dados. Devuelve
    `(key, front, error)`; `error` es `None` si todo salio bien.
    """
    (keys, source, kind, instance, seed, model_name, mutation,
     instance_timeout, overrides, initial_solutions) = task
    if initial_solutions is not None:
        # Poblacion inicial de ESTA repeticion (como el original, que fija
        # `set_initial_solutions(experiment=i)` antes de construir el solver).
        instance = copy.deepcopy(instance)
        instance.initial_solutions = np.asarray(initial_solutions)
    try:
        # El worker puede arrancar con el singleton `Params` vacio (fork/
        # forkserver): se inicializa igual que en la ruta de LLaMEA para que
        # el MOEA corra con MO_GENERATIONS, MO_POPULATION_SIZE y las
        # direcciones de referencia correctos.
        init_params(**(overrides or {}))
        operator = None
        if kind == "classic":
            ns = namespace_for(REFERENCE_OPERATORS[source])
        elif kind == "classic2":
            single = REFERENCE_OPERATORS[source]

            def two_children(x, y):
                return single(x, y), single(y, x)

            ns = namespace_for(two_children)
        elif kind == "ge":
            ns = ge_namespace(source)
        elif kind == "project":
            # Cruce real del proyecto: dos hijos complementarios con el mismo
            # corte/inicio. Se asigna directo, sin `LLMOperator`.
            ns = None
            operator = project_crossover(source)
        else:
            ns = compile_operator(source)
        if isinstance(keys[0], tuple) and len(keys[0]) == 5:
            # Varios presupuestos: una corrida con snapshots devuelve los tres.
            snaps = solve_front_snapshots(
                instance, ns, seed=seed, model_name=model_name,
                mutation=mutation, instance_timeout=instance_timeout,
                crossover=operator,
            )
            empty = np.empty((0, 0), dtype=float)
            return [(key, snaps.get(key[3], empty), None) for key in keys]
        front = _solve_front_namespace(
            instance, ns, seed=seed, model_name=model_name,
            mutation=mutation, instance_timeout=instance_timeout,
            crossover=operator,
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


def collect_fronts(operators, paths, args, protocols, existing=None,
                   stored_failures=None, labels=None, out_dir=None):
    """Corre el MOEA para cada (repeticion, operador, instancia).

    Con `labels` (10k/30k/50k) cada tarea devuelve los frentes de los
    presupuestos de la misma corrida (como el `MyCallback` del original); sin
    `labels`, un solo frente (el final).

    Si `existing` trae frentes ya calculados (con la misma semilla por
    repeticion), esos se reutilizan: sirve para agregar operadores a una
    comparacion ya corrida sin repetir el MOEA de los demas.

    Una excepcion se guarda como frente vacio. Asi ``hv_values`` puede contarla
    como HV 0 en el par (o abortar con ``--fail-policy abort``) en lugar de
    eliminar la repeticion entera. No hay timeout por instancia: el MOEA ya
    esta acotado por generaciones y un limite fijo mataba corridas largas en
    instancias dificiles, perdiendo el frente ya calculado.
    """
    labels = labels or [None]
    fronts = {} if existing is None else dict(existing)
    failures = {} if stored_failures is None else dict(stored_failures)
    # Las instancias como objetos `Instance` para el worker y los
    # overrides de Params: los workers arrancan con el singleton vacio
    # (forkserver) y necesitan MO_GENERATIONS/MO_POPULATION_SIZE y las
    # direcciones de referencia. Con snapshots, MO_GENERATIONS es el ultimo
    # presupuesto (500), no el presupuesto de analisis.
    # Con GE_PROTOCOL las etiquetas son 10k/30k/50k; sin el, una etiqueta
    # arbitraria (p. ej. "1k") que no esta en EVAL_BUDGETS: usar los valores
    # explicitos en vez de indexar y morir con KeyError.
    generations = max(
        EVAL_BUDGETS[label] // 105 if label in EVAL_BUDGETS else args.mo_generations
        for label in labels)
    population = 105 if getattr(args, "ge_protocol", False) else args.mo_population_size
    evaluator = build_evaluator(
        paths, mo_generations=generations,
        mo_population_size=population, seed=args.seed,
        model_name=protocols[0][0], mutation=protocols[0][1], normalize=False,
        instance_timeout=None,
    )
    instances = evaluator._ensure_instances()
    instance_timeout = evaluator.instance_timeout
    overrides = evaluator._params_overrides()
    if out_dir is None:
        out_dir = os.path.join(args.out, args.name or "evaluacion")

    # Soluciones iniciales: UNA poblacion por repeticion (como el original,
    # donde cada experimento arranca con la suya). Determinista por
    # (instancia, repeticion); se guardan en el formato del original y el
    # MOEA de esa repeticion arranca con ellas.
    population_size = instances[0].initial_solutions.shape[0]
    seeds = [args.seed + rep for rep in range(args.repeats)]
    initial = {}
    for index, instance in enumerate(instances):
        n_variables = instance.initial_solutions.shape[1]
        instance_name = os.path.splitext(os.path.basename(paths[index]))[0]
        populations = [
            random_key_population(
                n_variables, population_size,
                np.random.default_rng(args.seed + rep * 1_000 + index))
            for rep in range(args.repeats)
        ]
        real_by_rep = [real for real, _perm in populations]
        perm_by_rep = [perm for _real, perm in populations]
        initial[index] = perm_by_rep
        write_initial_solutions(out_dir, instance_name, real_by_rep,
                                perm_by_rep, seeds)

    tasks = []
    for rep in range(args.repeats):
        seed = args.seed + rep
        for solver, mutation in protocols:
            protocol = protocol_name(solver, mutation)
            for name, _ns, kind, source in operators:
                for index, instance in enumerate(instances):
                    todo = [(protocol, name, rep, label, index)
                            for label in labels
                            if (protocol, name, rep, label, index) not in fronts
                            # Un fallo registrado no se re-ejecuta (son
                            # excepciones del MOEA, no timeouts). Para
                            # reintentarlas hay que borrar `failures.json`.
                            and (protocol, name, rep, label, index) not in failures]
                    if not todo:
                        continue
                    tasks.append((todo, source, kind, instance, seed, solver,
                                  mutation, instance_timeout, overrides,
                                  initial[index][rep]))

    if not tasks:
        return fronts, failures

    save_checkpoints = bool(out_dir) and getattr(args, "save_fronts", True)

    def checkpoint():
        """Vuelca frentes + fallos sin esperar al final de la tanda."""
        if not save_checkpoints:
            return
        save_fronts(out_dir, fronts)
        save_failures(out_dir, failures)

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
                if done_count % CHECKPOINT_EVERY == 0:
                    checkpoint()
            print("", flush=True)
    else:
        for task in tasks:
            for key, front, error in _solve_task(task):
                fronts[key] = front
                if error:
                    failures[key] = error
            if len(fronts) % CHECKPOINT_EVERY == 0:
                checkpoint()
    checkpoint()
    return fronts, failures


# Cada cuantas tareas se vuelca el checkpoint de frentes. Con frentes de
# ~105 x K floats el npz comprimido es pequeño; 25 tareas acotan la perdida a
# unos minutos de MOEA en el peor caso.
CHECKPOINT_EVERY = 25


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--operator", action="append", required=False,
                        help="nombre=ruta.py (LLaMEA), nombre=classic:ox, "
                             "nombre=classic2:ox (dos hijos), "
                             "nombre=project:PMX_Cross|CX_Cross (cruce real "
                             "del proyecto), "
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
        args.repeats = N_EXPERIMENTS
    if args.mo_generations is None:
        args.mo_generations = 300 if args.ge_protocol else 30
    if args.mo_population_size is None:
        args.mo_population_size = 105 if args.ge_protocol else 30
    if args.solver is None:
        args.solver = ("MOEAD,NSGAII,SMSEMOA" if args.ge_protocol else "MOEAD")
    if args.ge_protocol and not args.mutation:
        args.mutation_both = True
    protocols = build_protocols(args)

    # Validar grupos/contrastes ANTES de lanzar el MOEA: un typo aqui hacia
    # que la corrida terminara los 7440 frentes y muriera en el analisis.
    groups = parse_groups(args.group)
    contrasts = parse_contrasts(args.contrast)
    for left, right in contrasts:
        for group in (left, right):
            if group not in groups:
                raise SystemExit(
                    f"el contraste {left}:{right} usa el grupo {group!r}, que no "
                    f"esta declarado (usa --group {group}=op1,op2)")

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
                        existing[(protocol, op_name, int(rep), int(index))] = data[key]
                    elif len(parts) == 4:
                        protocol, op_name, rep, index = parts
                        existing[(protocol, op_name, int(rep), int(index))] = data[key]
                    elif len(parts) == 5:
                        # Con snapshots: el presupuesto (10k/30k/50k) va en la clave.
                        protocol, op_name, rep, label, index = parts
                        existing[(protocol, op_name, int(rep), label,
                                  int(index))] = data[key]
                    else:
                        raise SystemExit(f"clave invalida en fronts.npz: {key!r}")
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
    if args.ge_protocol:
        labels = list(EVAL_BUDGETS)
        init_params(MO_GENERATIONS=max(EVAL_BUDGETS[label] // 105
                                       for label in labels),
                    MO_POPULATION_SIZE=105)
    else:
        labels = [budget_label(args.mo_generations, args.mo_population_size)]
        init_params(MO_GENERATIONS=args.mo_generations,
                    MO_POPULATION_SIZE=args.mo_population_size)

    start = time.time()
    fronts, failures = collect_fronts(operators, args.paths, args, protocols,
                                      existing,
                                      stored_failures=existing_failures,
                                      labels=labels, out_dir=out_dir)
    print(f"MOEA terminado en {time.time() - start:.1f}s")

    # Los frentes se guardan ANTES de calcular el HV: si el analisis falla
    # (marco degenerado, HV no finito, excepcion de pymoo), el MOEA no se
    # repite (mismo comando + --load-fronts). Con snapshots, las claves llevan
    # el presupuesto (protocolo|op|rep|label|inst).
    if args.save_fronts:
        save_fronts(out_dir, fronts)
    save_failures(out_dir, failures)
    _log("frentes guardados; calculando HV...")

    _log("FASE 1/5: marcos consolidados (comprobate + vmin/vmax)")
    frames_all = shared_frame(fronts, operators, protocols, len(args.paths),
                              args.repeats, labels, out_dir=out_dir)
    _log(f"FASE 1/5 lista: {len(frames_all)} marcos")
    _log("FASE 2/5: hipervolumenes de cada (protocolo, operador, repeticion)")
    hv_problems = {}
    values = hv_values(fronts, operators, protocols, frames_all, args.repeats,
                       labels, failures=failures, fail_policy=args.fail_policy,
                       problems=hv_problems)
    if hv_problems:
        save_json(out_dir, "hv_problems.json", hv_problems)
        _log(f"  AVISO: {len(hv_problems)} celda(s) sin HV (ver hv_problems.json)")
    _log("FASE 2/5 lista")

    # El analisis (resumen, contrastes, report) se hace sobre el presupuesto
    # `--budget`; los tres se evaluan y se escriben en el layout del original.
    # Un fallo aqui NO pierde los frentes: ya estan en fronts.npz.
    if args.budget is None:
        args.budget = "30k" if "30k" in labels else labels[-1]
    if args.budget not in labels:
        raise SystemExit(f"--budget {args.budget!r} no se evaluo "
                         f"(presupuestos: {', '.join(labels)})")
    _log("FASE 3/5: emparejamiento por parejas (marco propio de cada par)")
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
    try:
        scores = per_run_scores(values, operators, protocols, args.repeats)
        rows = summarize(scores, operators)
        comparisons = add_holm(pairwise(scores, operators, rows))
        pairwise_results = add_holm(
            pairwise_design(fronts, rows, protocols, len(args.paths), args.repeats,
                            failures=failures, fail_policy=args.fail_policy)
        )
        stability = rank_stability(scores, operators)

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
        with open(os.path.join(out_dir, "evaluation.json"), "w") as handle:
            json.dump(result, handle, indent=2)
        _log("FASE 5/5: escribiendo CSV, artefactos por instancia y evaluation.json")
        write_csvs(out_dir, args, rows, comparisons, pairwise_results, values,
                   operators, protocols, group_results)
        write_instance_artifacts(out_dir, args, fronts, values, frames, operators,
                                 protocols)
        # Estructura de la evaluacion original (experiment.py): <solver>/<presupuesto>/
        # <instancia>.csv, los frentes por solver y las metricas por instancia; mas
        # el catalogo de operadores usados.
        write_ge_layout(out_dir, args, fronts_all, values_all, frames_all,
                        operators, protocols, failures, labels=labels)
        # Figuras del original: <solver>/<budget>/Figures/{NoMutation,WithMutation}/
        write_figures(out_dir, sorted({solver for solver, _ in protocols}),
                      list(EVAL_BUDGETS) if args.ge_protocol else labels)
        catalog = write_operator_catalog(out_dir, operators)
        write_report(os.path.join(out_dir, "report.md"), name, args, rows,
                     comparisons, frames, scores, values, operators, protocols,
                     pairwise_results, stability, failures, group_results)
    except Exception as error:  # noqa: BLE001 - los frentes ya estan guardados
        import traceback
        traceback.print_exc()
        print(f"AVISO: el analisis fallo ({type(error).__name__}: {error}); "
              "los frentes estan en fronts.npz (reanuda con --load-fronts).")
    print(f"Guardado en {out_dir}")
    print("  hv_por_instancia.csv  resumen_operadores.csv  contrastes.csv")
    print("  fronts.npz (reutilizable con --load-fronts)  evaluation.json")
    print(f"  {budget_label(args.mo_generations, args.mo_population_size)}/"
          "<solver>/<instancia>.csv  (formato del original)")
    print(f"  {os.path.relpath(catalog, out_dir)}  (operadores usados)")
    with open(os.path.join(out_dir, "seeds.txt"), "w", encoding="utf-8") as handle:
        handle.write("base_seed\t" + str(args.seed) + "\n")
        for rep in range(args.repeats):
            handle.write(f"rep\t{rep + 1}\t{args.seed + rep}\n")
    write_run_metrics_summary(
        out_dir, len(args.paths),
        len(protocols) * len(operators) * args.repeats * len(args.paths),
        len(failures), time.time() - start)


if __name__ == "__main__":
    main()
