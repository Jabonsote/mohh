"""Analiza y visualiza la diversidad conductual de una corrida LLaMEA.

Entrada normal: `results/<categoria>/<experimento>/diversity.json`, que escribe
`mohh.llamea.run` al terminar. Si la corrida se corto antes, usa
`diversity.jsonl` (traza incremental que escribe el evaluador).

Uso:
    python scripts/analyze_diversity.py results/searches/LLaMEA_single
    python scripts/analyze_diversity.py results/searches/LLaMEA_single \
        --plot results/searches/LLaMEA_single/diversity.png

El mapa conductual usa los mismos ejes que el archivo MAP-Elites de la busqueda:
recombincion (distancia al padre mas cercano) x estructura preservada (orden,
posicion o adyacencia). Cada punto es un operador; el color es su HV. Las
regiones vacias son las que el feedback pide explorar.
"""

import argparse
import csv
import json
import os
import sys

from mohh.llamea.diversity import (
    BehaviorTracker,
    behavior_descriptor,
)


def load_diversity(run_dir):
    """Devuelve `(summary, entries)` desde diversity.json o diversity.jsonl."""
    json_path = os.path.join(run_dir, "diversity.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path) as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("diversity.json is not an object")
            return payload.get("summary", {}), payload.get("entries", [])
        except (OSError, ValueError):
            pass

    lines_path = os.path.join(run_dir, "diversity.jsonl")
    entries = []
    if os.path.isfile(lines_path):
        with open(lines_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    # Una linea truncada por un corte del cluster no debe
                    # impedir analizar las anteriores.
                    break
    return {}, entries


def summary_from_entries(entries):
    """Reconstruye el resumen si solo hay `diversity.jsonl` (corrida cortada).

    Se vuelve a pasar por `BehaviorTracker` con los fingerprints guardados; asi
    la novedad, el clustering y la cobertura son los mismos que en la corrida,
    sin re-ejecutar el LLM ni el MOEA.
    """
    if not entries:
        return {}
    tracker = BehaviorTracker()
    for index, entry in enumerate(entries):
        fingerprint = entry.get("fingerprint")
        if not fingerprint:
            continue
        try:
            tracker.add(
                entry.get("key_sha1") or f"trace-{index}",
                fingerprint,
                float(entry.get("fitness") or 0.0),
                generation=entry.get("generation"),
                description=entry.get("description"),
            )
        except (TypeError, ValueError):
            # Entrada de una version anterior o incompleta: se ignora.
            continue
    return tracker.snapshot()


def print_report(experiment, summary, entries):
    print(f"Experimento: {experiment}")
    print(f"Operadores medidos: {summary.get('n', len(entries))}")
    print(f"Mecanismos distintos (clustering conductual): "
          f"{summary.get('unique_behavior', '?')}")
    entropy = float(summary.get("behavior_entropy") or 0.0)
    effective = float(summary.get("effective_behaviors") or 0.0)
    novelty = float(summary.get("mean_novelty") or 0.0)
    coverage = float(summary.get("archive_coverage") or 0.0)
    print(f"Entropia / mecanismos efectivos: {entropy:.3f} / {effective:.2f}")
    print(f"Novedad media: {novelty:.3f} "
          f"(< umbral = reescritura del mismo mecanismo)")
    print(f"Duplicados conductuales: {int(summary.get('near_duplicates') or 0)}")
    print(f"Cobertura del archivo MAP-Elites: {100.0 * coverage:.1f}%")
    families = summary.get("family_counts") or {}
    if families:
        print("Familias: " + ", ".join(
            f"{name}={count}" for name, count in sorted(families.items())
        ))
    # Validacion del combine conductual: el resumen la trae cuando existe
    # diversity.json; si solo hay JSONL se reconstruye desde las entradas.
    attempts = summary.get("target_family_attempts")
    hits = summary.get("target_family_hits")
    if attempts is None:
        measured = [entry for entry in entries
                    if entry.get("target_family_hit") is not None]
        attempts = len(measured)
        hits = sum(bool(entry.get("target_family_hit")) for entry in measured)
    if attempts:
        rate = 100.0 * float(hits or 0) / float(attempts)
        print(f"Familia objetivo (combine): {int(hits or 0)}/{int(attempts)} "
              f"aciertos ({rate:.1f}%)")
    if entries:
        top = sorted(entries, key=lambda item: -float(item.get("novelty") or 0))[:5]
        print("\nOperadores mas novedosos:")
        for entry in top:
            generation = entry.get("generation")
            family = entry.get("family", "?")
            novelty = float(entry.get("novelty") or 0)
            description = (entry.get("description") or "").splitlines()[0][:90]
            print(f"  gen {generation}: {family:11s} novedad={novelty:.3f} "
                  f"fit={float(entry.get('fitness') or 0):.6f} {description}")


def write_csv(entries, path):
    columns = ["generation", "family", "target_family", "observed_family",
               "target_family_hit", "closest_reference", "closest_distance",
               "novelty", "cell", "coverage", "fitness", "description"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(entries)


def plot(entries, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not entries:
        raise SystemExit("No hay entradas para graficar.")
    recombination, structure, fitness, novelty, generations = [], [], [], [], []
    for entry in entries:
        fingerprint = entry.get("fingerprint")
        if not fingerprint:
            continue
        try:
            x, y = behavior_descriptor(fingerprint)
        except (TypeError, ValueError):
            continue
        recombination.append(x)
        structure.append(y)
        fitness.append(float(entry.get("fitness") or 0))
        novelty.append(float(entry.get("novelty") or 0))
        generations.append(entry.get("generation") or 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    scatter = axes[0].scatter(recombination, structure, c=fitness, cmap="viridis",
                              s=42, edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("recombincion (distancia al padre mas cercano)")
    axes[0].set_ylabel("estructura preservada (orden/posicion/adyacencia)")
    axes[0].set_title("Mapa conductual (color = HV)")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[0], label="HV")

    order = sorted(range(len(novelty)), key=lambda i: (generations[i], i))
    axes[1].plot([novelty[i] for i in order], marker="o", ms=3, lw=1)
    axes[1].axhline(0.08, color="red", ls="--", lw=1, label="umbral de duplicado")
    axes[1].set_xlabel("orden de evaluacion")
    axes[1].set_ylabel("novedad conductual")
    axes[1].set_title("Novedad: 0 = mismo mecanismo")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    families = {}
    for entry in entries:
        family = entry.get("family", "?")
        families[family] = families.get(family, 0) + 1
    axes[2].bar(list(families), list(families.values()), color="#4c72b0")
    axes[2].set_title("Familias de mecanismos")
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160)
    print(f"Grafica: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Directorio de la corrida (o diversity.json).")
    parser.add_argument("--plot", default=None, help="PNG de salida.")
    parser.add_argument("--csv", default=None, help="CSV de entradas.")
    args = parser.parse_args(argv)

    run_dir = args.run
    if os.path.isfile(run_dir):
        run_dir = os.path.dirname(os.path.abspath(run_dir))
    summary, entries = load_diversity(run_dir)
    if not summary:
        summary = summary_from_entries(entries)
    if not entries and not summary:
        raise SystemExit(f"Sin diversity.json/diversity.jsonl en {run_dir}")
    print_report(os.path.basename(run_dir.rstrip("/")), summary, entries)
    if args.csv:
        write_csv(entries, args.csv)
        print(f"CSV: {args.csv}")
    if args.plot:
        plot(entries, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
