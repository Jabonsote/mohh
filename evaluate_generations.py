#!/usr/bin/env python
"""
Evaluación comparable de los frentes consolidados por generación.

Réplica el protocolo de `mohh-evaluate` para las generaciones guardadas:

1. por instancia, toma el frente consolidado de cada `generation_NNNN`;
2. fija UNA escala de HV por instancia con todos los frentes de todas las
   generaciones (NDS consolidado + min/max), y usa `ref_point = [1.1] * m`,
   igual que `mohh-evaluate`;
3. usa los P* fijos de train/test para IGD+, DELTA_P y R2;
4. calcula los indicadores de cada generación con esa escala y esas referencias.

Así los valores de generaciones distintas son comparables entre sí dentro de
cada experimento.

Uso:
    python results/analysis_scripts/evaluate_generations.py \
        results/experiments_larger/Experiment_001
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mohh.analysis.hv_scale import (freeze_hv_scale, instancias_sin_escala,
                                    load_hv_scale, restringir_escala,
                                    save_hv_scale)
from mohh.analysis.indicators_all import (
    INDICATORS,
    compute_indicators,
    indicators_long,
    load_references,
)
from mohh.analysis.layout import find_run_dirs


def _generation_dirs(experimento):
    """Todas las carpetas `generation_NNNN` de la corrida."""
    carpetas = {}
    for run_dir in find_run_dirs(experimento):
        base = os.path.join(run_dir, "generation_results", "generations")
        if not os.path.isdir(base):
            continue
        for nombre in os.listdir(base):
            if nombre.startswith("generation_"):
                try:
                    generacion = int(nombre.split("_")[-1])
                except ValueError:
                    continue
                carpetas.setdefault(generacion, []).append(
                    os.path.join(base, nombre))
    return {g: sorted(v)[0] for g, v in sorted(carpetas.items())}


def _frentes_por_generacion(experimento):
    generaciones = _generation_dirs(experimento)
    if not generaciones:
        raise FileNotFoundError(
            f"no hay generation_results/generations en {experimento}")

    frentes = {}
    for generacion, carpeta in generaciones.items():
        for nombre in sorted(os.listdir(carpeta)):
            if not (nombre.startswith("Instance_Fronts_")
                    and nombre.endswith(".json")):
                continue
            instancia = nombre[len("Instance_Fronts_"):-len(".json")]
            with open(os.path.join(carpeta, nombre)) as f:
                datos = json.load(f)
            puntos = datos.get("consolidated")
            if not puntos:
                continue
            frentes.setdefault(instancia, {})[generacion] = np.asarray(
                puntos, dtype=float)
    return frentes, generaciones


def _figuras(resumen, metricas, directorio, etiqueta):
    os.makedirs(directorio, exist_ok=True)
    for metrica in metricas:
        parte = resumen[resumen["metric"] == metrica]
        if parte.empty:
            continue
        figura, eje = plt.subplots(figsize=(7.5, 4.2))
        eje.plot(parte["generation"], parte["mean"], marker="o",
                 markersize=3, linewidth=1.3, color="#1f77b4")
        eje.set_xlabel("generación")
        eje.set_ylabel(metrica)
        eje.set_title(f"{metrica} por generación — {etiqueta}")
        eje.grid(alpha=0.3)
        if metrica == "HV":
            eje.set_yscale("log")
        figura.tight_layout()
        figura.savefig(os.path.join(directorio, f"convergence_{metrica}.png"),
                       dpi=180)
        plt.close(figura)


def evaluar(experimento, referencias, salida, metricas, escala_hv=None,
            guardar_escala=None):
    frentes, generaciones = _frentes_por_generacion(experimento)
    if not frentes:
        raise ValueError(f"no se encontraron frentes en {experimento}")

    instancias = sorted(frentes)
    referencias = load_references(referencias, instance_names=instancias)

    if escala_hv is not None:
        # Escala congelada externa: la misma para todas las generaciones de
        # este experimento y para cualquier otro que la reutilice.
        faltan = escala_hv.missing(instancias)
        if faltan:
            raise ValueError(
                f"la escala de HV no cubre {len(faltan)} instancia(s): "
                f"{faltan[:8]}{'...' if len(faltan) > 8 else ''}")
        escala = escala_hv
    else:
        # La escala HV se fija con TODAS las generaciones, como en mohh-evaluate.
        por_instancia = {
            instancia: [frentes[instancia][g]
                        for g in generaciones if g in frentes[instancia]]
            for instancia in instancias
        }
        escala = freeze_hv_scale(por_instancia)

    # `mohh-evaluate` invalida las instancias sin escala (vmax == vmin en algun
    # objetivo). Excluirlas es lo que hace que el HV sea comparable con GE.
    escala_completa = escala
    degeneradas = [nombre for nombre in instancias_sin_escala(escala)
                   if nombre in set(instancias)]
    if degeneradas:
        print(f"AVISO: {len(degeneradas)} instancia(s) sin escala "
              f"(vmax == vmin; se excluyen como en mohh-evaluate): "
              f"{degeneradas}")
    instancias_puntuables = [i for i in instancias if i not in degeneradas]
    escala = restringir_escala(escala, instancias_puntuables)
    contextos = escala.contexts()

    filas = []
    for generacion in sorted(generaciones):
        actual = {
            instancia: frentes[instancia][generacion]
            for instancia in instancias_puntuables
            if generacion in frentes[instancia]
        }
        if not actual:
            continue
        medidas = compute_indicators(
            actual, references=referencias, metrics=metricas,
            hv_context=contextos, require_references=True)
        for instancia in medidas.index:
            for metrica in metricas:
                valor = medidas.loc[instancia, metrica]
                if valor is None or (isinstance(valor, float)
                                     and np.isnan(valor)):
                    continue
                filas.append({
                    "generation": generacion,
                    "instance": instancia,
                    "metric": metrica,
                    "value": float(valor),
                })

    largo = pd.DataFrame(filas)
    if largo.empty:
        raise ValueError("no se calculó ningún indicador")
    resumen = (largo.groupby(["metric", "generation"])["value"]
               .agg(["mean", "median", "std", "min", "max", "count"])
               .reset_index())

    os.makedirs(salida, exist_ok=True)
    largo.to_csv(os.path.join(salida, "generation_indicators.csv"),
                 index=False)
    resumen.to_csv(os.path.join(salida, "generation_summary.csv"), index=False)
    save_hv_scale(escala_completa, os.path.join(salida, "hv_scale.json"))
    if guardar_escala:
        save_hv_scale(escala_completa, guardar_escala)

    figuras = os.path.join(salida, "figures")
    _figuras(resumen, metricas, figuras, os.path.basename(experimento))

    print(f"Generaciones : {largo['generation'].nunique()}")
    print(f"Instancias   : {largo['instance'].nunique()}")
    print(f"Filas        : {len(largo)}")
    print(f"Salida       : {salida}")
    return largo, resumen


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Indicadores comparables por generación (estilo mohh-evaluate).")
    parser.add_argument("experiment")
    parser.add_argument("--references", default="references_moea/train/reference_sets")
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", nargs="+", default=list(INDICATORS),
                        choices=list(INDICATORS))
    parser.add_argument("--hv-scale", default=None,
                        help="JSON con la escala de HV congelada; si se omite se "
                             "congela con todas las generaciones de este "
                             "experimento (comparable dentro de él, no entre "
                             "experimentos).")
    parser.add_argument("--save-hv-scale", default=None,
                        help="escribe la escala de HV usada para reutilizarla.")
    args = parser.parse_args(argv)

    salida = args.output or os.path.join(args.experiment,
                                         "analysis_generations")
    escala_hv = load_hv_scale(args.hv_scale) if args.hv_scale else None
    evaluar(args.experiment, args.references, salida, args.metrics,
            escala_hv=escala_hv, guardar_escala=args.save_hv_scale)


if __name__ == "__main__":
    main()
