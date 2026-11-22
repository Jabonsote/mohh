#!/usr/bin/env python
"""
Resumen y gráficas de los conjuntos de referencia P* de train y test.

Lee los `.pof` de `references_moea` (o de la ruta que se indique) y escribe:

    results/reference_sets/reference_sets_summary.md
    results/reference_sets/reference_sets_sizes.png
    results/reference_sets/reference_sets_examples.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Raiz del proyecto: este script puede vivir en la raiz o copiado en
# `results/analysis_scripts/`, asi que se prueban las dos rutas.
SRC = next(ruta for ruta in (
    os.path.join(HERE, "src"),
    os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "src"),
) if os.path.isdir(ruta))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mohh.core.indicators.reference_sets import read_pof  # noqa: E402


def _leer_split(directorio):
    filas = []
    for nombre in sorted(os.listdir(directorio)):
        if not nombre.endswith("_ref.pof"):
            continue
        instancia = nombre[:-len("_ref.pof")]
        frente = read_pof(os.path.join(directorio, nombre))
        filas.append({
            "instance": instancia,
            "points": int(frente.shape[0]),
            "objectives": int(frente.shape[1]),
        })
    return pd.DataFrame(filas)


def _grafica_sizes(tablas, ruta):
    total = sum(len(tabla) for tabla in tablas.values())
    figura, eje = plt.subplots(figsize=(10, max(8, 0.22 * total)))
    y = np.arange(total)
    pos = 0
    colores = {"train": "#1f77b4", "test": "#d62728"}
    etiquetas = []
    for split, tabla in tablas.items():
        valores = tabla.sort_values("points")["points"].to_numpy()
        eje.barh(y[pos:pos + len(valores)], valores,
                 color=colores[split], alpha=0.8, label=split)
        etiquetas.extend(tabla.sort_values("points")["instance"].tolist())
        pos += len(valores)
    eje.set_yticks(y)
    eje.set_yticklabels(etiquetas, fontsize=5)
    eje.set_xlabel("puntos en P*")
    eje.set_title("Tamaño de los conjuntos de referencia P*")
    eje.grid(axis="x", alpha=0.3)
    eje.legend()
    figura.tight_layout()
    figura.savefig(ruta, dpi=160)
    plt.close(figura)


def _grafica_ejemplos(tablas, directorio, subdirectorio, ruta):
    seleccion = []
    for split, tabla in tablas.items():
        # Los más grandes son los que mejor muestran la forma del frente.
        for _, fila in tabla.sort_values("points", ascending=False).head(3).iterrows():
            seleccion.append((split, fila["instance"]))

    figura, ejes = plt.subplots(2, 3, figsize=(12, 7))
    for eje, (split, instancia) in zip(ejes.ravel(), seleccion):
        ruta_pof = os.path.join(directorio, split, subdirectorio,
                                f"{instancia}_ref.pof")
        frente = read_pof(ruta_pof)
        if frente.shape[1] < 2:
            eje.axis("off")
            continue
        eje.scatter(frente[:, 0], frente[:, 1], s=10, alpha=0.75)
        eje.set_title(f"{split}: {instancia} ({frente.shape[0]} puntos)", fontsize=8)
        eje.set_xlabel("objetivo 1", fontsize=7)
        eje.set_ylabel("objetivo 2", fontsize=7)
        eje.tick_params(labelsize=6)
        eje.grid(alpha=0.25)
    figura.suptitle("Ejemplos de P* (primeros dos objetivos)")
    figura.tight_layout()
    figura.savefig(ruta, dpi=160)
    plt.close(figura)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumen de P*.")
    parser.add_argument("--references", default="references_moea")
    parser.add_argument("--subdir", default="reference_sets",
                        help="subcarpeta de cada split con los *_ref.pof "
                             "(p. ej. reference_sets_pop)")
    parser.add_argument("--output", default=os.path.join("results", "reference_sets"))
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    tablas = {}
    for split in ("train", "test"):
        directorio = os.path.join(args.references, split, args.subdir)
        tablas[split] = _leer_split(directorio)

    # Detalle con ground set si está disponible.
    for split, tabla in tablas.items():
        counts = os.path.join(args.references, "nondominated_counts.csv")
        if not os.path.isfile(counts):
            counts = os.path.join(args.references, split, "nondominated_counts.csv")
        if os.path.isfile(counts):
            extra = pd.read_csv(counts)
            if "split" in extra.columns:
                extra = extra[extra["split"] == split]
            extra = extra[["instance", "ground_set_size", "singleton"]]
            tabla = tabla.merge(extra, on="instance", how="left")
        tablas[split] = tabla
        tabla.to_csv(os.path.join(args.output, f"reference_sets_{split}.csv"),
                     index=False)

    lineas = ["# Conjuntos de referencia P*", ""]
    lineas.append("| split | instancias | puntos P* |")
    lineas.append("|---|---:|---:|")
    for split, tabla in tablas.items():
        puntos = tabla["points"]
        lineas.append(f"| {split} | {len(tabla)} | {int(puntos.sum())} |")
    lineas.extend(["", "## Detalle por instancia", ""])
    for split, tabla in tablas.items():
        lineas.append(f"### {split.capitalize()}")
        lineas.append("")
        lineas.append("| instancia | objetivos | puntos P* | ground set |")
        lineas.append("|---|---:|---:|---:|")
        for _, fila in tabla.sort_values("instance").iterrows():
            ground = "" if pd.isna(fila.get("ground_set_size")) else int(fila["ground_set_size"])
            lineas.append(
                f"| {fila['instance']} | {int(fila['objectives'])} | "
                f"{int(fila['points'])} | {ground} |")
        lineas.append("")

    with open(os.path.join(args.output, "reference_sets_summary.md"), "w") as f:
        f.write("\n".join(lineas) + "\n")

    _grafica_sizes(tablas, os.path.join(args.output, "reference_sets_sizes.png"))
    _grafica_ejemplos(tablas, args.references, args.subdir,
                      os.path.join(args.output, "reference_sets_examples.png"))
    print("Resumen P*:", args.output)


if __name__ == "__main__":
    main()
