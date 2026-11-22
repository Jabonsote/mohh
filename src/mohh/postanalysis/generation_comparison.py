#!/usr/bin/env python
"""
Compara el comportamiento por generación de dos experimentos con la misma
referencia y la misma escala.

Protocolo (el mismo criterio que la evaluación congelada):

* IGD+, DELTA_P y R2 se calculan contra el P* fijo de train. No se recalcula
  la referencia ni se agranda la caja con los frentes medidos.
* HV usa la misma escala congelada para los dos experimentos: mismas
  `vmin/vmax` por instancia y `ref_point = 1.1`, como en `mohh-evaluate`.
* Los valores se leen de `analysis/01_individuals/indicators_*.csv`, que ya
  contienen esos cuatro indicadores por individuo, instancia y generación.

Salida:

    results/generation_comparison/generation_mean.csv
    results/generation_comparison/generation_wide.csv
    results/generation_comparison/generation_by_instance.csv
    results/generation_comparison/generation_comparison.png
    results/generation_comparison/README.md
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICAS = ("HV", "IGD+", "DELTA_P", "R2")


def _leer_experimento(raiz, nombre):
    ruta = os.path.join(raiz, nombre, "analysis", "01_individuals",
                        f"indicators_{nombre}.csv")
    if not os.path.isfile(ruta):
        raise FileNotFoundError(ruta)
    tabla = pd.read_csv(ruta)
    tabla.insert(0, "experiment", nombre)
    return tabla


def _leer_escala(raiz, nombre):
    ruta = os.path.join(raiz, nombre, "analysis", "00_config", "hv_scale.json")
    if not os.path.isfile(ruta):
        return {}
    with open(ruta) as f:
        return json.load(f)


def _limites_distintos(escalas):
    """Instancias en las que las escalas NO comparten `vmin/vmax`.

    Dos escalas pueden tener el mismo `ref_point` y el mismo `source`
    ("consolidado") y, aun asi, limites distintos: en ese caso sus HV no son
    comparables. Mirar solo `ref_point`/`source` da falsos negativos (deja
    pasar escalas incomparables), que es justo el error que hay que evitar al
    mirar generaciones de dos corridas.
    """
    nombres = [n for n, d in escalas.items() if d.get("instances")]
    if len(nombres) < 2:
        return []
    base = escalas[nombres[0]]["instances"]
    problemas = set()
    for otro in nombres[1:]:
        otras = escalas[otro]["instances"]
        problemas |= set(base) ^ set(otras)
        for instancia in set(base) & set(otras):
            lo_a = np.asarray(base[instancia]["vmin"], dtype=float)
            hi_a = np.asarray(base[instancia]["vmax"], dtype=float)
            lo_b = np.asarray(otras[instancia]["vmin"], dtype=float)
            hi_b = np.asarray(otras[instancia]["vmax"], dtype=float)
            if (lo_a.shape != lo_b.shape or hi_a.shape != hi_b.shape
                    or not np.allclose(lo_a, lo_b)
                    or not np.allclose(hi_a, hi_b)):
                problemas.add(instancia)
    return sorted(problemas)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparación por generación.")
    parser.add_argument("--experiments-root",
                        default=os.path.join("results", "experiments_larger"))
    parser.add_argument("--output",
                        default=os.path.join("results", "generation_comparison"))
    parser.add_argument("--experiments", nargs="+",
                        default=["Experiment_001", "Experiment_002"])
    parser.add_argument("--allow-different-scale", action="store_true",
                        help="permite comparar aunque los vmin/vmax no "
                             "coincidan (solo para inspeccion; el HV no es "
                             "comparable).")
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    tablas = [_leer_experimento(args.experiments_root, nombre)
              for nombre in args.experiments]
    largo = pd.concat(tablas, ignore_index=True)

    # Comprobación de escala HV compartida. No basta con `ref_point`/`source`:
    # hay que verificar los `vmin/vmax` reales, porque dos consolidados
    # distintos comparten etiqueta pero no escala.
    escalas = {nombre: _leer_escala(args.experiments_root, nombre)
               for nombre in args.experiments}
    fuentes = {nombre: datos.get("source") for nombre, datos in escalas.items()}
    ref_points = {nombre: datos.get("ref_point") for nombre, datos in escalas.items()}
    if len(set(ref_points.values())) > 1:
        raise RuntimeError(f"ref_point distinto entre experimentos: {ref_points}")
    limite_distinto = _limites_distintos(escalas)
    sin_escala = [nombre for nombre, datos in escalas.items()
                  if not datos.get("instances")]
    difieren = bool(limite_distinto) or bool(sin_escala) \
        or len(set(fuentes.values())) > 1
    if sin_escala:
        mensaje = (f"falta hv_scale.json en {sin_escala}; sin escala no se "
                   f"puede verificar que el HV sea comparable.")
        if not getattr(args, "allow_different_scale", False):
            raise RuntimeError(mensaje)
        print(f"AVISO: {mensaje}")
    if limite_distinto:
        mensaje = (
            f"la escala HV NO es la misma en {len(limite_distinto)} instancia(s) "
            f"({limite_distinto[:8]}{'...' if len(limite_distinto) > 8 else ''}); "
            "los HV por generación NO son comparables. Congela una sola escala "
            "con analyze_gen --hv-scale (o analyze_search --hv-scale) y "
            "reanaliza ambos experimentos con ella.")
        if not getattr(args, "allow_different_scale", False):
            raise RuntimeError(mensaje)
        print(f"AVISO: {mensaje}")

    filas = []
    for (experimento, generacion), grupo in largo.groupby(
            ["experiment", "generation"], sort=True):
        fila = {"experiment": experimento, "generation": int(generacion),
                "n_individuals": int(grupo["individual"].nunique())}
        for metrica in METRICAS:
            valores = grupo[metrica].to_numpy(dtype=float)
            fila[f"{metrica}_mean"] = float(np.nanmean(valores))
            fila[f"{metrica}_std"] = float(np.nanstd(valores))
        filas.append(fila)
    media = pd.DataFrame(filas)
    media.to_csv(os.path.join(args.output, "generation_mean.csv"), index=False)

    ancho = media.pivot(index="generation", columns="experiment",
                        values=[f"{m}_mean" for m in METRICAS])
    ancho.to_csv(os.path.join(args.output, "generation_wide.csv"))

    por_instancia = (largo.groupby(["experiment", "instance", "generation"])[
        list(METRICAS)].mean().reset_index())
    por_instancia.to_csv(
        os.path.join(args.output, "generation_by_instance.csv"), index=False)

    figura, ejes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colores = {"Experiment_001": "#1f77b4", "Experiment_002": "#d62728"}
    for eje, metrica in zip(ejes.ravel(), METRICAS):
        for experimento, grupo in media.groupby("experiment"):
            grupo = grupo.sort_values("generation")
            eje.plot(grupo["generation"], grupo[f"{metrica}_mean"],
                     marker="o", markersize=3, linewidth=1.4,
                     color=colores.get(experimento), label=experimento)
            eje.fill_between(
                grupo["generation"],
                grupo[f"{metrica}_mean"] - grupo[f"{metrica}_std"],
                grupo[f"{metrica}_mean"] + grupo[f"{metrica}_std"],
                color=colores.get(experimento), alpha=0.12)
        eje.set_title(metrica)
        eje.set_xlabel("generación")
        eje.set_ylabel(metrica)
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    figura.suptitle("Comportamiento por generación — train 80 instancias")
    figura.tight_layout()
    figura.savefig(os.path.join(args.output, "generation_comparison.png"),
                   dpi=180)
    plt.close(figura)

    lineas = [
        "# Comparación por generación (train)",
        "",
        "Protocolo congelado, igual para los dos experimentos:",
        "",
        "- **IGD+, DELTA_P y R2**: contra el `P*` fijo de train; sin recalcular "
        "la referencia y sin agrandar la caja con frentes medidos.",
        "- **HV**: misma escala `vmin/vmax` por instancia y `ref_point = 1.1` "
        "en los dos experimentos.",
        "- Cada fila es la media de todos los individuos de esa generación.",
        "",
        f"Experimentos: {', '.join(args.experiments)}.",
        "",
        "## Escala HV",
        "",
        "| experimento | ref_point | origen |",
        "|---|---:|---|",
    ]
    for nombre in args.experiments:
        lineas.append(
            f"| {nombre} | {ref_points.get(nombre)} | {fuentes.get(nombre)} |")
    lineas.append("")
    if limite_distinto:
        lineas.append(
            f"> **NO comparable:** los `vmin/vmax` difieren en "
            f"{len(limite_distinto)} instancia(s). Solo se han escrito los CSV "
            f"porque se pasó `--allow-different-scale`; no interpretes el HV.")
    elif sin_escala:
        lineas.append(
            f"> **Sin verificar:** falta `hv_scale.json` en {sin_escala}; no se "
            f"pudo comprobar que la escala HV sea la misma.")
    elif difieren:
        lineas.append(
            "> **Aviso:** el `source` de la escala HV no coincide. Los valores "
            "de HV sólo son comparables si el contenido de `hv_scale.json` es "
            "el mismo; aquí se comprobó que los `vmin/vmax` sí coinciden.")
    else:
        lineas.append(
            "> La escala HV es la misma en los dos experimentos: se comprobó "
            "`vmin/vmax` instancia por instancia, no solo el `source`.")
    lineas.extend([
        "",
        "Ficheros:",
        "",
        "```text",
        "generation_mean.csv          # largo: experimento/generación/métrica",
        "generation_wide.csv          # ancho: una columna por experimento y métrica",
        "generation_by_instance.csv   # instancia/generación/métrica",
        "generation_comparison.png    # media ± desviación por generación",
        "```",
    ])
    with open(os.path.join(args.output, "README.md"), "w") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"Comparación por generación: {args.output}")


if __name__ == "__main__":
    main()
