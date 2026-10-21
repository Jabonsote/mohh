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
  * **N repeticiones independientes** (por defecto 11) para estimar el ruido y
    hacer una prueba no parametrica pareada (Wilcoxon), con intervalo de
    confianza de la diferencia y ajuste de Holm por comparaciones multiples;
  * **protocolo configurable**: `--solver` acepta varios solvers y
    `--mutation-both` evalua con y sin `Swap_Mutation`. `--ge-protocol` activa
    el protocolo de la evaluacion GE (MOEAD, NSGAII y SMSEMOA; ambas mutaciones;
    300/105; 11 repeticiones);
  * un fallo de evaluacion (operador que lanza una excepcion en una instancia)
    se cuenta como HV 0, no se descarta del par: descartarlo haria que un
    operador roto pareciera mejor. ``--fail-policy`` permite abortar o excluir.

Uso:
    python scripts/compare_operators.py \
        --operator descubierto=results/examples/llamea_found_operators/qwen25coder7b_moead_b100_ge.py \
        --operator ox=classic:ox --operator pbx=classic:pbx \
        --split test --repeats 11 --name b100_vs_clasicos

    # Tabla final con el protocolo GE completo (caro):
    python scripts/compare_operators.py ... --ge-protocol

Salida: `results/evaluations/<name>/evaluation.json` y `report.md`.
"""

import argparse
import json
import os
import time

import numpy as np

from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.core.utilities.instance_utils import instance_paths
from mohh.llamea.fitness import (
    NADIR_FACTOR,
    _consolidated_frame,
    build_evaluator,
    clip_to_reference_box,
    compile_operator,
    init_params,
    normalize_front,
)
from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for
from mohh.llamea.run import _select_instances
from mohh.llamea.statistics import (
    add_holm,
    minimum_detectable_difference,
    paired_summary,
)


def load_operator(spec):
    """`nombre=ruta.py`, `nombre=classic:ox` o `nombre=classic2:ox`.

    `classic2:ox` es el clasico devolviendo **dos** hijos, `(ox(x, y), ox(y, x))`:
    el mismo contrato que un operador que devuelve una tupla. Comparar un
    operador de dos hijos contra clasicos de un solo hijo no es justo, porque el
    arnes (`LLMOperator`) usa `child[::-1]` como segundo descendiente.
    """
    name, _, source = spec.partition("=")
    name = name.strip()
    if not name or not source:
        raise SystemExit(f"--operator debe ser nombre=fuente, no {spec!r}")
    if source.startswith("classic:"):
        key = source.split(":", 1)[1].strip()
        if key not in REFERENCE_OPERATORS:
            raise SystemExit(f"operador clasico desconocido: {key!r}")
        return name, namespace_for(REFERENCE_OPERATORS[key]), "classic"
    if source.startswith("classic2:"):
        key = source.split(":", 1)[1].strip()
        if key not in REFERENCE_OPERATORS:
            raise SystemExit(f"operador clasico desconocido: {key!r}")
        single = REFERENCE_OPERATORS[key]

        def two_children(x, y):
            return single(x, y), single(y, x)

        return name, namespace_for(two_children), "classic"
    with open(source) as handle:
        return name, compile_operator(handle.read()), "discovered"


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


def collect_fronts(operators, paths, args, protocols, existing=None):
    """Corre el MOEA para cada (repeticion, operador, instancia).

    Si `existing` trae frentes ya calculados (con la misma semilla por
    repeticion), esos se reutilizan: sirve para agregar operadores a una
    comparacion ya corrida sin repetir el MOEA de los demas.

    Un fallo se guarda como frente vacio. Asi ``hv_values`` puede contarlo como
    HV 0 en el par (o abortar con ``--fail-policy abort``) en vez de eliminar
    silenciosamente la instancia, que sesgaria la comparacion a favor del
    operador que falla.

    Devuelve ``(fronts, failures)``; ``failures`` mapea la clave a un mensaje.
    """
    fronts = dict(existing or {})
    failures = {
        key: "fallo recuperado de fronts.npz"
        for key, front in fronts.items()
        if front is not None and len(front) == 0
    }
    total = args.repeats * len(protocols) * len(operators) * len(paths)
    done = len(fronts)
    for rep in range(args.repeats):
        for solver, mutation in protocols:
            protocol = protocol_name(solver, mutation)
            # Una instancia (y una poblacion inicial) por repeticion,
            # compartida por todos los operadores y protocolos: asi la
            # comparacion es pareada de verdad.
            evaluator = build_evaluator(
                paths,
                mo_generations=args.mo_generations,
                mo_population_size=args.mo_population_size,
                seed=args.seed + rep,
                model_name=solver,
                mutation=mutation,
                normalize=False,
            )
            instances = evaluator._ensure_instances()
            for name, namespace, _kind in operators:
                reused = 0
                for index, instance in enumerate(instances):
                    key = (protocol, name, rep, index)
                    if key in fronts:
                        reused += 1
                        continue
                    try:
                        front = evaluator._solve_front(instance, namespace)
                        if front is None:
                            raise RuntimeError("el MOEA no devolvio un frente")
                    except Exception as error:  # noqa: BLE001 - se reporta y sigue
                        print(f"  AVISO rep={rep} {protocol} {name} "
                              f"inst={index}: {error}")
                        front = np.empty((0, 0), dtype=float)
                        failures[key] = f"{type(error).__name__}: {error}"
                    fronts[key] = front
                    done += 1
                if reused:
                    print(f"  rep {rep + 1}/{args.repeats} [{protocol}]: "
                          f"{name} reutilizado ({reused}/{len(instances)})",
                          flush=True)
                else:
                    print(f"  rep {rep + 1}/{args.repeats} [{protocol}]: "
                          f"{name} listo ({done}/{total})", flush=True)
    return fronts, failures


def shared_frame(fronts, operators, protocols, n_instances, repeats):
    """Marco por instancia: frente consolidado de TODAS las configuraciones."""
    frames = []
    for index in range(n_instances):
        pieces = [fronts.get((protocol_name(solver, mutation), name, rep, index))
                  for solver, mutation in protocols
                  for name, _ns, _kind in operators
                  for rep in range(repeats)]
        pieces = [p for p in pieces if p is not None and len(p)]
        frames.append(_consolidated_frame(*pieces) if pieces else None)
    return frames


def hv_values(fronts, operators, protocols, frames, repeats, failures=None,
              fail_policy="zero"):
    """HV de cada (operador, repeticion, instancia) en el marco comun.

    ``frame is None`` significa que la instancia no tiene escala (``vmin ==
    vmax``) y se excluye para todos los operadores, como en
    ``experiment.py``. Un frente vacio es un fallo del operador en esa
    instancia: con ``fail_policy="zero"`` cuenta como HV 0 (peor caso); con
    ``"exclude"`` se elimina el par; con ``"abort"`` se detiene la comparacion.
    """
    failures = failures or {}
    values = {}
    for solver, mutation in protocols:
        protocol = protocol_name(solver, mutation)
        for name, _ns, _kind in operators:
            for rep in range(repeats):
                row = []
                for index, frame in enumerate(frames):
                    key = (protocol, name, rep, index)
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
                values[(protocol, name, rep)] = row
    return values


def per_run_scores(values, operators, protocols, repeats):
    """HV medio por repeticion, promediando instancias y protocolos."""
    scores = {}
    for name, _ns, _kind in operators:
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
    discovered = [r["operator"] for r in rows if r["kind"] == "discovered" and r.get("runs")]
    results = []
    for ours in discovered:
        for base in classics:
            frames = []
            for index in range(n_instances):
                pieces = [fronts.get(
                              (protocol_name(solver, mutation), name, rep, index))
                          for solver, mutation in protocols
                          for name in (ours, base)
                          for rep in range(repeats)]
                pieces = [p for p in pieces if p is not None and len(p)]
                frames.append(_consolidated_frame(*pieces) if pieces else None)
            subset = [(ours, None, "discovered"), (base, None, "classic")]
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


def summarize(scores, operators):
    rows = []
    for name, _ns, kind in operators:
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
    """Cada operador descubierto contra cada clasico (pareado por repeticion)."""
    classics = [r["operator"] for r in rows if r["kind"] == "classic" and r.get("runs")]
    discovered = [r["operator"] for r in rows if r["kind"] == "discovered" and r.get("runs")]
    comparisons = []
    for ours in discovered:
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
    names = [name for name, _ns, _kind in operators
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


def write_report(path, name, args, rows, comparisons, frames, scores, values,
                 operators, protocols, pairwise_results=None, stability=None,
                 failures=None):
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
        for (protocol, operator, rep, index), reason in sorted(failures.items()):
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
        for name, _ns, _kind in operators:
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
    header = "| Instancia | " + " | ".join(name for name, _n, _k in operators) + " |"
    lines += [header, "|---|" + "---:|" * len(operators)]
    for index in range(len(args.paths)):
        cells = []
        for name, _ns, _kind in operators:
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
    parser.add_argument("--operator", action="append", required=True,
                        help="nombre=ruta.py o nombre=classic:ox (repetible)")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--instances", default="8",
                        help="Numero de instancias del split, o 'all'. Usar el "
                             "mismo valor que TEST_INSTANCES para que la "
                             "evaluacion final y la comparacion sean pareadas.")
    parser.add_argument("--repeats", type=int, default=None,
                        help="repeticiones independientes (default 11)")
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
    parser.add_argument("--out", default="results/evaluations")
    parser.add_argument("--name", default=None)
    parser.add_argument("--fail-policy", choices=["zero", "exclude", "abort"],
                        default="zero",
                        help="que hacer si un operador falla en una instancia: "
                             "zero = HV 0 (default, conserva el par); exclude = "
                             "descarta el par; abort = detiene la comparacion.")
    parser.add_argument("--save-fronts", action="store_true",
                        help="guarda los frentes en fronts.npz (permite reanalizar sin repetir el MOEA)")
    parser.add_argument("--load-fronts", action="store_true",
                        help="reutiliza fronts.npz del directorio de salida para los (operador, repeticion, instancia) ya calculados")
    args = parser.parse_args()

    # Defaults que dependen de --ge-protocol. Un valor explicito del usuario
    # siempre gana; `None` distingue "no lo especifique" de "quiero 30/30".
    if args.repeats is None:
        args.repeats = 11
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
    names = [name for name, _ns, _kind in operators]
    duplicated = {name for name in names if names.count(name) > 1}
    if duplicated:
        raise SystemExit(f"nombres de operador repetidos: {sorted(duplicated)}")

    name = args.name or "vs_".join(n for n, _ns, _k in operators)
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Operadores: {', '.join(n for n, _a, _b in operators)}")
    print(f"Instancias: {args.split} ({len(args.paths)}), repeticiones: {args.repeats}")
    print("Protocolos: " + ", ".join(
        protocol_name(solver, mutation) for solver, mutation in protocols))
    default_protocol = protocol_name(*protocols[0])
    existing = None
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
        else:
            print(f"AVISO: no hay {npz_path}; se recalcula todo")
    start = time.time()
    fronts, failures = collect_fronts(operators, args.paths, args, protocols,
                                      existing)
    print(f"MOEA terminado en {time.time() - start:.1f}s; calculando HV...")

    frames = shared_frame(fronts, operators, protocols, len(args.paths),
                          args.repeats)
    values = hv_values(fronts, operators, protocols, frames, args.repeats,
                       failures=failures, fail_policy=args.fail_policy)
    scores = per_run_scores(values, operators, protocols, args.repeats)
    rows = summarize(scores, operators)
    comparisons = add_holm(pairwise(scores, operators, rows))
    pairwise_results = add_holm(
        pairwise_design(fronts, rows, protocols, len(args.paths), args.repeats,
                        failures=failures, fail_policy=args.fail_policy)
    )
    stability = rank_stability(scores, operators)

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
            for name, _ns, _k in operators
        },
        "per_instance_hv": {
            name: {
                protocol_name(solver, mutation): [
                    values[(protocol_name(solver, mutation), name, rep)]
                    for rep in range(args.repeats)
                ]
                for solver, mutation in protocols
            }
            for name, _ns, _k in operators
        },
        "failures": [
            {"protocol": protocol, "operator": operator, "repeat": rep,
             "instance": os.path.basename(args.paths[index]), "reason": reason}
            for (protocol, operator, rep, index), reason in sorted(failures.items())
        ],
        # Semilla efectiva de cada corrida del MOEA interno (operador x repeticion).
        "run_seeds": [
            {"operator": name, "repeat": rep, "seed": args.seed + rep}
            for name, _ns, _k in operators
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
    with open(os.path.join(out_dir, "evaluation.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    write_report(os.path.join(out_dir, "report.md"), name, args, rows,
                 comparisons, frames, scores, values, operators, protocols,
                 pairwise_results, stability, failures)
    print(f"Guardado en {out_dir}")


if __name__ == "__main__":
    main()
