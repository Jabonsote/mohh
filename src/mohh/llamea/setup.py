"""Helpers de configuracion del evaluador y de LLaMEA (extraidos de `run.py`).

`run.py` re-exporta estos nombres para conservar la API publica del modulo.
"""

import json
import os

import numpy as np

from mohh.llamea.fitness import build_evaluator, build_population_evaluator


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
                    ):
    model_names = [name.strip() for name in
                   str(getattr(args, "solvers", "")).split(",") if name.strip()]
    if not model_names:
        model_names = [args.solver]
    mutation_names = [name.strip().lower() for name in
                      str(getattr(args, "mutations", "")).split(",") if name.strip()]
    if mutation_names:
        mutations = [name in ("swap", "wm", "true", "1", "yes")
                     for name in mutation_names]
    else:
        mutations = [bool(args.mutation)]
    builder = (build_population_evaluator
               if getattr(args, "population_fitness", True)
               else build_evaluator)
    return builder(
        paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
        model_name=model_names[0],
        mutation=mutations[0],
        instance_timeout=args.instance_timeout,
        normalize=args.normalize,
        bounds=bounds,
        reference_operators=_reference_panel(args.reference_operators),
        archive=archive,
        frame="ge" if getattr(args, "accumulated_frame", False) else "stationary",
        fronts_dir=fronts_dir,
        repeat_penalty=getattr(args, "repeat_penalty", 0.5),
        patience=getattr(args, "patience", 0),
        fitness_workers=getattr(args, "fitness_workers", 1),
        model_names=model_names,
        mutations=mutations,
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
