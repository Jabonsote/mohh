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
    hacer una prueba no parametrica pareada (Wilcoxon).

Uso:
    python scripts/compare_operators.py \
        --operator descubierto=results/examples/llamea_found_operators/qwen25coder7b_moead_b100_ge.py \
        --operator ox=classic:ox --operator pbx=classic:pbx \
        --split test --repeats 11 --name b100_vs_clasicos

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
    compile_operator,
    init_params,
    normalize_front,
)
from mohh.llamea.reference import REFERENCE_OPERATORS, namespace_for
from mohh.llamea.run import _select_instances


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


def collect_fronts(operators, paths, args, existing=None):
    """Corre el MOEA para cada (repeticion, operador, instancia).

    Si `existing` trae frentes ya calculados (con la misma semilla por
    repeticion), esos se reutilizan: sirve para agregar operadores a una
    comparacion ya corrida sin repetir el MOEA de los demas.
    """
    fronts = dict(existing or {})
    total = args.repeats * len(operators) * len(paths)
    done = len(fronts)
    for rep in range(args.repeats):
        # Una instancia (y una poblacion inicial) por repeticion, compartida por
        # todos los operadores: asi la comparacion es pareada de verdad.
        evaluator = build_evaluator(
            paths,
            mo_generations=args.mo_generations,
            mo_population_size=args.mo_population_size,
            seed=args.seed + rep,
            model_name=args.solver,
            mutation=args.mutation,
            normalize=False,
        )
        instances = evaluator._ensure_instances()
        for name, namespace, _kind in operators:
            reused = 0
            for index, instance in enumerate(instances):
                if (name, rep, index) in fronts:
                    reused += 1
                    continue
                try:
                    front = evaluator._solve_front(instance, namespace)
                except Exception as error:  # noqa: BLE001 - se reporta y sigue
                    print(f"  AVISO rep={rep} {name} inst={index}: {error}")
                    front = None
                fronts[(name, rep, index)] = front
                done += 1
            if reused:
                print(f"  rep {rep + 1}/{args.repeats}: {name} reutilizado "
                      f"({reused}/{len(instances)})", flush=True)
            else:
                print(f"  rep {rep + 1}/{args.repeats}: {name} listo "
                      f"({done}/{total})", flush=True)
    return fronts


def shared_frame(fronts, operators, n_instances, repeats):
    """Marco por instancia: frente consolidado de TODAS las configuraciones."""
    frames = []
    for index in range(n_instances):
        pieces = [fronts.get((name, rep, index))
                  for name, _ns, _kind in operators
                  for rep in range(repeats)]
        pieces = [p for p in pieces if p is not None]
        frames.append(_consolidated_frame(*pieces) if pieces else None)
    return frames


def hv_values(fronts, operators, frames, repeats):
    """HV de cada (operador, repeticion, instancia) en el marco comun."""
    values = {}
    for name, _ns, _kind in operators:
        for rep in range(repeats):
            row = []
            for index, frame in enumerate(frames):
                front = fronts.get((name, rep, index))
                if frame is None or front is None or len(front) == 0:
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
                row.append(float(compute_hypervolume(
                    np.full(normalized.shape[1], NADIR_FACTOR), normalized
                )))
            values[(name, rep)] = row
    return values


def per_run_scores(values, operators, repeats):
    scores = {}
    for name, _ns, _kind in operators:
        scores[name] = [float(np.mean([v for v in values[(name, rep)] if v is not None]))
                        if any(v is not None for v in values[(name, rep)]) else None
                        for rep in range(repeats)]
    return scores


def pairwise_design(fronts, rows, n_instances, repeats):
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
                pieces = [fronts.get((name, rep, index))
                          for name in (ours, base) for rep in range(repeats)]
                pieces = [p for p in pieces if p is not None]
                frames.append(_consolidated_frame(*pieces) if pieces else None)
            subset = [(ours, None, "discovered"), (base, None, "classic")]
            values = hv_values(fronts, subset, frames, repeats)
            scores = per_run_scores(values, subset, repeats)
            diffs = [a - b for a, b in zip(scores[ours], scores[base])
                     if a is not None and b is not None]
            if not diffs:
                continue
            stat, pvalue = wilcoxon(diffs)
            ours_mean = float(np.mean([v for v in scores[ours] if v is not None]))
            base_mean = float(np.mean([v for v in scores[base] if v is not None]))
            results.append({
                "operator": ours,
                "baseline": base,
                "pairs": len(diffs),
                "wins": sum(1 for d in diffs if d > 0),
                "losses": sum(1 for d in diffs if d < 0),
                "ties": sum(1 for d in diffs if d == 0),
                "operator_mean": ours_mean,
                "baseline_mean": base_mean,
                "mean_difference": float(np.mean(diffs)),
                "relative": (ours_mean / base_mean - 1.0) if base_mean else None,
                "cohens_dz": (float(np.mean(diffs) / np.std(diffs, ddof=1))
                              if len(diffs) > 1 and np.std(diffs, ddof=1) > 0 else None),
                "wilcoxon_statistic": stat,
                "wilcoxon_p": pvalue,
                "per_instance_operator": [
                    float(np.mean([v for v in
                                   (values[(ours, rep)][i] for rep in range(repeats))
                                   if v is not None]))
                    if any(v is not None for v in
                           (values[(ours, rep)][i] for rep in range(repeats))) else None
                    for i in range(n_instances)],
                "per_instance_baseline": [
                    float(np.mean([v for v in
                                   (values[(base, rep)][i] for rep in range(repeats))
                                   if v is not None]))
                    if any(v is not None for v in
                           (values[(base, rep)][i] for rep in range(repeats))) else None
                    for i in range(n_instances)],
            })
    return results


def wilcoxon(differences):
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
    except ImportError:  # pragma: no cover - scipy es dependencia de pymoo
        return None, None
    nonzero = [d for d in differences if d != 0]
    if len(nonzero) < 2:
        return None, None
    result = scipy_wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


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
            diffs = [a - b for a, b in zip(scores[ours], scores[base])
                     if a is not None and b is not None]
            if not diffs:
                continue
            stat, pvalue = wilcoxon(diffs)
            ours_mean = float(np.mean([v for v in scores[ours] if v is not None]))
            base_mean = float(np.mean([v for v in scores[base] if v is not None]))
            comparisons.append({
                "operator": ours,
                "baseline": base,
                "pairs": len(diffs),
                "wins": sum(1 for d in diffs if d > 0),
                "losses": sum(1 for d in diffs if d < 0),
                "ties": sum(1 for d in diffs if d == 0),
                "mean_difference": float(np.mean(diffs)),
                "relative": (ours_mean / base_mean - 1.0) if base_mean else None,
                "cohens_dz": (float(np.mean(diffs) / np.std(diffs, ddof=1))
                              if len(diffs) > 1 and np.std(diffs, ddof=1) > 0 else None),
                "wilcoxon_statistic": stat,
                "wilcoxon_p": pvalue,
            })
    return comparisons


def write_report(path, name, args, rows, comparisons, frames, scores, values, operators,
                 pairwise_results=None):
    lines = [f"# Comparacion de operadores: {name}", ""]
    lines += [
        "## Diseno",
        "",
        f"- Instancias: {args.split} ({len(args.paths)}), semilla base {args.seed}.",
        f"- Repeticiones independientes: {args.repeats} (semilla = base + repeticion).",
        f"- MOEA interno: {args.solver}, {args.mo_generations} generaciones, "
        f"poblacion {args.mo_population_size}, mutacion {args.mutation}.",
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

    if pairwise_results:
        lines += ["## Cabeza a cabeza (marco propio de la pareja)", "",
                  "Cada pareja se renormaliza con el frente consolidado de **esas dos**",
                  "configuraciones (lo que seria un experimento GE de dos configuraciones),",
                  "para que un tercer operador no aplaste a los demas a HV ~0.", "",
                  "| Operador | vs | Media op | Media clasico | Gana | Pierde | Dif. | Relativo | dz | p |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for c in pairwise_results:
            p = "-" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.4f}"
            dz = "-" if c["cohens_dz"] is None else f"{c['cohens_dz']:.2f}"
            rel = "-" if c["relative"] is None else f"{100 * c['relative']:+.1f}%"
            lines.append(
                f"| {c['operator']} | {c['baseline']} | {c['operator_mean']:.6f} | "
                f"{c['baseline_mean']:.6f} | {c['wins']} | {c['losses']} | "
                f"{c['mean_difference']:+.6f} | {rel} | {dz} | {p} |"
            )
        lines.append("")

    if comparisons:
        lines += ["## Comparaciones pareadas (marco compartido de todos)", "",
                  "| Operador | vs | Gana | Pierde | Empata | Dif. media | Relativo | dz | p (Wilcoxon) |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for c in comparisons:
            p = "-" if c["wilcoxon_p"] is None else f"{c['wilcoxon_p']:.4f}"
            dz = "-" if c["cohens_dz"] is None else f"{c['cohens_dz']:.2f}"
            rel = "-" if c["relative"] is None else f"{100 * c['relative']:+.1f}%"
            lines.append(
                f"| {c['operator']} | {c['baseline']} | {c['wins']} | {c['losses']} | "
                f"{c['ties']} | {c['mean_difference']:+.6f} | {rel} | {dz} | {p} |"
            )
        lines += ["", "p de Wilcoxon pareado de dos colas; con 11 pares el minimo "
                  f"alcanzable es {2 / 2 ** args.repeats:.4f}. Con 5 clasicos conviene "
                  "leerlo con Bonferroni (0.05 / numero de comparaciones).", ""]

    lines += ["## HV por instancia (media de las repeticiones)", ""]
    header = "| Instancia | " + " | ".join(name for name, _n, _k in operators) + " |"
    lines += [header, "|---|" + "---:|" * len(operators)]
    for index in range(len(args.paths)):
        cells = []
        for name, _ns, _kind in operators:
            vals = [values[(name, rep)][index] for rep in range(args.repeats)
                    if values.get((name, rep), [None] * len(args.paths))[index] is not None]
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
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--mo-generations", type=int, default=30)
    parser.add_argument("--mo-population-size", type=int, default=30)
    parser.add_argument("--solver", default="MOEAD")
    parser.add_argument("--mutation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/evaluations")
    parser.add_argument("--name", default=None)
    parser.add_argument("--save-fronts", action="store_true",
                        help="guarda los frentes en fronts.npz (permite reanalizar sin repetir el MOEA)")
    parser.add_argument("--load-fronts", action="store_true",
                        help="reutiliza fronts.npz del directorio de salida para los (operador, repeticion, instancia) ya calculados")
    args = parser.parse_args()

    init_params(RANDOM_SEED=args.seed)
    all_paths = instance_paths(train=args.split == "train")
    args.paths = _select_instances(all_paths, args.instances, args.seed)
    operators = [load_operator(spec) for spec in args.operator]

    name = args.name or "vs_".join(n for n, _ns, _k in operators)
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Operadores: {', '.join(n for n, _a, _b in operators)}")
    print(f"Instancias: {args.split} ({len(args.paths)}), repeticiones: {args.repeats}")
    existing = None
    if args.load_fronts:
        npz_path = os.path.join(out_dir, "fronts.npz")
        if os.path.exists(npz_path):
            with np.load(npz_path) as data:
                existing = {}
                for key in data.files:
                    op_name, rep, index = key.split("|")
                    existing[(op_name, int(rep), int(index))] = data[key]
            print(f"Reutilizando {len(existing)} frentes de {npz_path}")
        else:
            print(f"AVISO: no hay {npz_path}; se recalcula todo")
    start = time.time()
    fronts = collect_fronts(operators, args.paths, args, existing)
    print(f"MOEA terminado en {time.time() - start:.1f}s; calculando HV...")

    frames = shared_frame(fronts, operators, len(args.paths), args.repeats)
    values = hv_values(fronts, operators, frames, args.repeats)
    scores = per_run_scores(values, operators, args.repeats)
    rows = summarize(scores, operators)
    comparisons = pairwise(scores, operators, rows)
    pairwise_results = pairwise_design(fronts, rows, len(args.paths), args.repeats)

    result = {
        "config": {
            "split": args.split,
            "instances": [os.path.basename(p) for p in args.paths],
            "repeats": args.repeats,
            "seed": args.seed,
            "solver": args.solver,
            "mo_generations": args.mo_generations,
            "mo_population_size": args.mo_population_size,
            "mutation": args.mutation,
            "frame": "consolidated over every operator and repetition",
            "nadir_factor": NADIR_FACTOR,
            "degenerate_instances": [
                os.path.basename(args.paths[i])
                for i, frame in enumerate(frames)
                if frame is None or np.any(frame.max(axis=0) == frame.min(axis=0))
            ],
        },
        "operators": rows,
        "comparisons": comparisons,
        "pairwise_head_to_head": pairwise_results,
        "per_run_scores": scores,
        "per_instance_hv": {name: [values[(name, rep)] for rep in range(args.repeats)]
                            for name, _ns, _k in operators},
    }
    if args.save_fronts:
        np.savez_compressed(
            os.path.join(out_dir, "fronts.npz"),
            **{f"{name}|{rep}|{index}": np.asarray(front)
               for (name, rep, index), front in fronts.items()
               if front is not None},
        )
    with open(os.path.join(out_dir, "evaluation.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    write_report(os.path.join(out_dir, "report.md"), name, args, rows,
                 comparisons, frames, scores, values, operators,
                 pairwise_results)
    print(f"Guardado en {out_dir}")


if __name__ == "__main__":
    main()
