"""Escritura de artefactos de la comparacion de operadores."""

import json
import os

import numpy as np

from mohh.constants import EVAL_BUDGETS
from mohh.evaluation.results_layout import write_evaluation_layout, write_front
from mohh.llamea.fitness import NADIR_FACTOR, clip_to_reference_box, normalize_front

from mohh.comparison.engine import (
    _ci,
    _fmt,
    _p,
    budget_label,
    protocol_name,
)


def _fronts_payload(fronts):
    return {
        f"{protocol}|{name}|{rep}|{label}|{index}": np.asarray(front)
        for (protocol, name, rep, label, index), front in fronts.items()
        if front is not None and len(front)
    }


def save_fronts(out_dir, fronts):
    """Escribe ``fronts.npz`` de forma atomica (tmp + rename)."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fronts.npz")
    tmp = path + ".tmp.npz"
    try:
        np.savez_compressed(tmp, **_fronts_payload(fronts))
        os.replace(tmp, path)
    except Exception as error:  # noqa: BLE001 - el checkpoint no debe abortar
        print(f"AVISO: no se pudo guardar {path}: {error}", flush=True)


def save_failures(out_dir, failures):
    """Escribe ``failures.json`` de forma atomica."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "failures.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as handle:
            json.dump({"|".join(str(part) for part in key): reason
                       for key, reason in failures.items()}, handle, indent=2)
        os.replace(tmp, path)
    except Exception as error:  # noqa: BLE001
        print(f"AVISO: no se pudo guardar {path}: {error}", flush=True)


def save_json(out_dir, name, payload):
    """Escribe un JSON auxiliar (degenerate.json, hv_problems.json)."""
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


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
            if kind in ("classic", "classic2", "project"):
                text = f"{kind}:{source}"
            else:
                text = source
            handle.write(f"{name}: {text}\n")
    return path


def write_run_metrics_summary(out_dir, n_instances, n_runs, n_failures,
                              total_seconds):
    """`run_metrics/summary.json`, como el original (resumen de la corrida)."""
    summary = {
        "phase": "summary",
        "instances": n_instances,
        "moea_runs": n_runs,
        "failures": n_failures,
        "total_seconds": round(float(total_seconds), 3),
    }
    path = os.path.join(out_dir, "run_metrics", "summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)


def write_ge_layout(out_dir, args, fronts, values, frames, operators, protocols,
                    failures=None, labels=None):
    """Delega en `mohh.evaluation.results_layout` (ver modulo para el detalle).

    Se mantiene el nombre por compatibilidad con tests y llamadores antiguos.
    """
    budgets = list(EVAL_BUDGETS)
    labels = labels or [budget_label(args.mo_generations, args.mo_population_size)]
    return write_evaluation_layout(
        out_dir, args.paths, budgets, operators, protocols, values, frames,
        fronts, args.repeats, failures=failures, labels=labels)


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
            policy_note = ("Un fallo cuenta como HV 0; no se descarta la "
                           "instancia.")
        elif args.fail_policy == "exclude":
            policy_note = ("Un fallo descarta esa celda; la media de cada "
                           "operador usa sus celdas completadas (no pareado).")
        else:
            policy_note = "Un fallo aborta la comparacion."
        lines += [f"- Celdas de fallo contabilizadas: {len(failures)} "
                  f"(politica `{args.fail_policy}`). {policy_note} "
                  "Una corrida que falla deja una celda por presupuesto.", ""]
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
