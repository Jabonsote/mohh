"""Calculo compartido por los scripts de comparacion de operadores.

Marco de normalizacion, HV, puntuaciones por repeticion, contrastes y resumen.
Sin escritura de artefactos: esa parte vive en ``mohh.comparison.writers``.
"""

import time

import numpy as np

from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.llamea.fitness import (
    NADIR_FACTOR,
    clip_to_reference_box,
    comprobate_frame,
    normalize_front,
)
from mohh.shared.statistics import minimum_detectable_difference, paired_summary


def protocol_name(solver, mutation):
    return f"{solver}+mut" if mutation else solver


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


def _log(message):
    """Log de progreso con hora: sin esto las fases largas parecen colgadas."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _safe_frame(pieces, context=""):
    """`_frame_from_points` que no aborta el analisis si el marco es degenerado.

    Un marco degenerado (todos los puntos iguales en algun objetivo, NaN) deja
    los HV de esa instancia a ``None``; el resto de la comparacion sigue.
    """
    try:
        return _frame_from_points(pieces)
    except Exception as error:  # noqa: BLE001 - se registra y se excluye
        _log(f"  AVISO: marco degenerado {context}: "
             f"{type(error).__name__}: {error}")
        return None


def shared_frame(fronts, operators, protocols, n_instances, repeats, labels,
                 out_dir=None):
    """Marco por instancia y presupuesto: frente consolidado de todo.

    Un fallo al construir el marco (p. ej. un frente degenerado que hace
    fallar la NDS) se registra y deja ``None`` para esa instancia: el HV de
    todos sus operadores sera ``None`` (exclusion) en vez de abortar el
    analisis entero.
    """
    frames = {}
    degenerate = {}
    total = len(labels) * n_instances
    done = 0
    started = time.time()
    for label in labels:
        for index in range(n_instances):
            try:
                pieces = [fronts.get((protocol_name(solver, mutation), name, rep,
                                      label, index))
                          for solver, mutation in protocols
                          for name, _ns, _kind, _src in operators
                          for rep in range(repeats)]
                pieces = [p for p in pieces if p is not None and len(p)]
                frame = _frame_from_points(pieces)
                # Un marco con un objetivo sin rango normaliza dividiendo por
                # cero (HV NaN): se excluye entero, como en experiment.py.
                if frame is not None and len(frame):
                    span = (np.asarray(frame.max(axis=0), dtype=float)
                            - np.asarray(frame.min(axis=0), dtype=float))
                    if not np.all(np.isfinite(span)) or np.any(span <= 0):
                        degenerate[f"{label}|{index}"] = (
                            f"marco degenerado (span={span.tolist()})")
                        frame = None
            except Exception as error:  # noqa: BLE001 - se registra y sigue
                frame = None
                degenerate[f"{label}|{index}"] = (
                    f"{type(error).__name__}: {error}")
            if frame is None:
                degenerate.setdefault(
                    f"{label}|{index}",
                    "sin puntos o marco degenerado (vmin == vmax)")
            frames[(label, index)] = frame
            done += 1
            if done % 5 == 0 or done == total:
                elapsed = time.time() - started
                pending = total - done
                _log(f"  marcos consolidados: {done}/{total} "
                     f"({elapsed:.0f}s, "
                     f"{elapsed / done * pending:.0f}s restantes)")
    if degenerate:
        _log(f"  AVISO: {len(degenerate)} marco(s) degenerados/excluidos")
        # Import local: writers importa de engine (evita ciclo al cargar).
        from mohh.comparison.writers import save_json
        save_json(out_dir, "degenerate.json", degenerate)
    return frames


def hv_values(fronts, operators, protocols, frames, repeats, labels,
              failures=None, fail_policy="zero", problems=None):
    """HV de cada (operador, repeticion, instancia) en el marco comun.

    ``frame is None`` significa que la instancia no tiene escala (``vmin ==
    vmax``) y se excluye para todos los operadores, como en
    ``experiment.py``. Un frente vacio es un fallo del operador en esa
    instancia: con ``fail_policy="zero"`` cuenta como HV 0 (peor caso); con
    ``"exclude"`` se elimina el par; con ``"abort"`` se detiene la comparacion.

    ``problems`` (dict opcional) acumula las celdas que no pudieron
    calcularse: un fallo de HV no aborta el analisis, se excluye y se reporta
    en ``hv_problems.json``.
    """
    failures = failures or {}
    problems = problems if problems is not None else {}
    values = {}
    # `frames` puede ser un dict por presupuesto (claves (label, index)) o una
    # lista aplanada al presupuesto de analisis (claves por indice).
    is_dict = isinstance(frames, dict)
    n_instances = (max(k[1] for k in frames) + 1) if is_dict else len(frames)
    for label in (labels or [None]):
      for solver, mutation in protocols:
        protocol = protocol_name(solver, mutation)
        for name, _ns, _kind, _src in operators:
            for rep in range(repeats):
                row = []
                for index in range(n_instances):
                    if is_dict:
                        frame = frames.get((label, index))
                        key = (protocol, name, rep, label, index)
                    else:
                        frame = frames[index]
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
                    try:
                        vmin, vmax = frame.min(axis=0), frame.max(axis=0)
                        if np.any(vmax == vmin):
                            problems[
                                "|".join(str(part) for part in key)
                            ] = "marco degenerado (vmax == vmin)"
                            row.append(None)
                            continue
                        normalized = normalize_front(front, (vmin, vmax))
                        if not np.all(np.isfinite(normalized)):
                            problems[
                                "|".join(str(part) for part in key)
                            ] = "frente normalizado no finito"
                            row.append(None)
                            continue
                        normalized = clip_to_reference_box(normalized)
                        hv = float(compute_hypervolume(
                            np.full(normalized.shape[1], NADIR_FACTOR),
                            normalized
                        ))
                        if not np.isfinite(hv):
                            raise ValueError(f"HV no finito ({hv})")
                        row.append(hv)
                    except Exception as error:  # noqa: BLE001 - se excluye la celda
                        problems["|".join(str(part) for part in key)] = (
                            f"{type(error).__name__}: {error}")
                        row.append(None)
                suffix = (label,) if is_dict else ()
                values[(protocol, name, rep, *suffix)] = row
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
    results = []
    for ours, base in _comparison_pairs(rows):
        frames = []
        for index in range(n_instances):
            pieces = [fronts.get(
                          (protocol_name(solver, mutation), name, rep, index))
                      for solver, mutation in protocols
                      for name in (ours, base)
                      for rep in range(repeats)]
            pieces = [p for p in pieces if p is not None and len(p)]
            frames.append(_safe_frame(pieces, f"par {ours} vs {base} inst={index}"))
        subset = [(ours, None, "llamea", None), (base, None, "classic", None)]
        values = hv_values(fronts, subset, protocols, frames, repeats,
                           [None], failures=failures,
                           fail_policy=fail_policy)
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
            frames.append(_safe_frame(pieces, f"{left} vs {right} inst={index}"))
        values = hv_values(fronts, subset, protocols, frames, repeats,
                            [None], failures=failures,
                            fail_policy=fail_policy)
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


def _comparison_pairs(rows):
    """Pares a contrastar: todo par de operadores evaluados, menos el par
    clasico-clasico (esa comparacion es la del panel, no del operador
    evaluado).

    Antes solo se contrastaban los operadores no clasicos (LLaMEA o GE) contra
    los clasicos, de modo que una comparacion LLaMEA-vs-GE (la del paper) no
    salia: `contrastes.csv` se quedaba vacio.
    """
    evaluados = [r for r in rows if r.get("runs")]
    pairs = []
    for i, a in enumerate(evaluados):
        for b in evaluados[i + 1:]:
            if a["kind"] == "classic" and b["kind"] == "classic":
                continue
            pairs.append((a["operator"], b["operator"]))
    return pairs


def pairwise(scores, operators, rows):
    """Cada operador evaluado contra los demas evaluados (claseicos incluidos).

    El contraste es pareado por repeticion: los dos operadores comparten
    semilla, poblacion inicial y marco, asi que la diferencia es del operador.
    """
    comparisons = []
    for ours, base in _comparison_pairs(rows):
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
