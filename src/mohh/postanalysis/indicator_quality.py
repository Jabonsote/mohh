#!/usr/bin/env python
"""
Revisión de los indicadores calculados por `analyze_gen.py`.

No recalcula nada: lee `01_individuals/indicators_<experimento>.csv` y deja un
informe con la distribución de cada indicador y su tendencia por generación.
Sirve para detectar indicadores degenerados antes de sacar conclusiones.

Criterios:

* **HV** debe crecer con la generación (mayor es mejor);
* **IGD+**, **DELTA_P** y **R2** deben decrecer (menor es mejor);
* un indicador con desviación ~0 o con más del 90 % de sus valores en el
  borde (0 o 1) se marca como no informativo.

El informe se escribe en `analysis/00_config/indicator_quality.{json,md}`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INDICADORES = {
    "HV": {"direction": "max", "expected": "increasing"},
    "IGD+": {"direction": "min", "expected": "decreasing"},
    "DELTA_P": {"direction": "min", "expected": "decreasing"},
    "R2": {"direction": "min", "expected": "decreasing"},
}


def _tendencia(tabla: pd.DataFrame, metrica: str) -> dict:
    if "generation" not in tabla.columns:
        return {"spearman": None, "expected": INDICADORES[metrica]["expected"]}

    medias = tabla.groupby("generation")[metrica].mean()
    if len(medias) < 3:
        return {"spearman": None, "expected": INDICADORES[metrica]["expected"]}

    correlacion = medias.reset_index().corr(
        method="spearman").loc["generation", metrica]

    return {
        "spearman": float(correlacion),
        "expected": INDICADORES[metrica]["expected"],
    }


def analizar(experiment: str, analysis_dir: str) -> dict:
    nombre = os.path.basename(os.path.normpath(experiment))
    ruta = os.path.join(
        analysis_dir, "01_individuals", f"indicators_{nombre}.csv")

    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"falta {ruta}")

    tabla = pd.read_csv(ruta)
    informe = {
        "experiment": nombre,
        "source": os.path.abspath(ruta),
        "n_rows": int(len(tabla)),
        "metrics": {},
    }

    for metrica, reglas in INDICADORES.items():
        if metrica not in tabla.columns:
            continue

        serie = pd.to_numeric(tabla[metrica], errors="coerce")
        valores = serie.to_numpy(dtype=float)
        finitos = np.isfinite(valores)

        resumen = {
            "direction": reglas["direction"],
            "expected_trend": reglas["expected"],
            "n": int(len(valores)),
            "n_nan": int((~finitos).sum()),
            "n_negative": int((valores[finitos] < 0).sum()),
            "min": float(np.nanmin(valores)) if finitos.any() else None,
            "max": float(np.nanmax(valores)) if finitos.any() else None,
            "mean": float(np.nanmean(valores)) if finitos.any() else None,
            "std": float(np.nanstd(valores)) if finitos.any() else None,
            "fraction_at_1": float(np.mean(np.isclose(
                valores[finitos], 1.0))) if finitos.any() else None,
            "fraction_near_1": float(np.mean(np.isclose(
                valores[finitos], 1.0, atol=0.01))) if finitos.any() else None,
            "fraction_at_0": float(np.mean(np.isclose(
                valores[finitos], 0.0))) if finitos.any() else None,
        }

        resumen["trend"] = _tendencia(tabla, metrica)

        no_informativo = (
            resumen["std"] is None or resumen["std"] <= 1e-12
            or (resumen["fraction_at_1"] or 0) >= 0.90
            or (resumen["fraction_near_1"] or 0) >= 0.50
            or (resumen["fraction_at_0"] or 0) >= 0.90
        )
        resumen["informative"] = not no_informativo

        # Comprobación de dirección: no se falla por ello, sólo se avisa.
        esperado = reglas["expected"]
        spearman = resumen["trend"]["spearman"]
        if spearman is not None:
            if esperado == "increasing":
                resumen["trend_ok"] = spearman > 0
            else:
                resumen["trend_ok"] = spearman < 0
        else:
            resumen["trend_ok"] = None

        informe["metrics"][metrica] = resumen

    informe["warnings"] = [
        f"{metrica}: no informativo (std={datos['std']:.4g}, "
        f"frac_1={datos['fraction_at_1']:.3f}, "
        f"frac_~1={datos['fraction_near_1']:.3f}, "
        f"frac_0={datos['fraction_at_0']:.3f})"
        for metrica, datos in informe["metrics"].items()
        if not datos["informative"]
    ]

    informe["trend_warnings"] = [
        f"{metrica}: tendencia {datos['trend']['spearman']:.3f}, "
        f"se esperaba {datos['expected_trend']}"
        for metrica, datos in informe["metrics"].items()
        if datos["trend_ok"] is False
    ]

    # R2 se mide contra el ideal de P*. Un valor negativo sólo significa que el
    # frente del individuo mejora ese ideal en alguna componente; es legítimo.
    r2 = informe["metrics"].get("R2")
    if r2 and r2["n_negative"] > 0:
        informe["warnings"].append(
            f"R2: {r2['n_negative']} valor(es) negativo(s) "
            f"(mín={r2['min']:.4g}); es posible si el frente mejora el ideal "
            f"de P* y no es un error")

    salida = os.path.join(analysis_dir, "00_config")
    os.makedirs(salida, exist_ok=True)

    with open(os.path.join(salida, "indicator_quality.json"), "w") as handle:
        json.dump(informe, handle, indent=2, ensure_ascii=False)

    with open(os.path.join(salida, "indicator_quality.md"), "w",
              encoding="utf-8") as handle:
        handle.write(f"# Calidad de indicadores — {nombre}\n\n")
        handle.write(f"Fuente: `{ruta}` ({len(tabla)} filas)\n\n")
        handle.write("| indicador | dirección | media | std | mín | máx | "
                     "frac. 1 | frac. ~1 | frac. 0 | tendencia | "
                     "informativo |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
                     ":---:|\n")
        for metrica, datos in informe["metrics"].items():
            spearman = datos["trend"]["spearman"]
            tendencia = "—" if spearman is None else f"{spearman:.3f}"
            handle.write(
                f"| {metrica} | {datos['direction']} | "
                f"{datos['mean']:.4g} | {datos['std']:.4g} | "
                f"{datos['min']:.4g} | {datos['max']:.4g} | "
                f"{datos['fraction_at_1']:.3f} | "
                f"{datos['fraction_near_1']:.3f} | "
                f"{datos['fraction_at_0']:.3f} | {tendencia} | "
                f"{'sí' if datos['informative'] else 'no'} |\n")

        if informe["warnings"]:
            handle.write("\n## Avisos\n\n")
            for aviso in informe["warnings"]:
                handle.write(f"- {aviso}\n")
        if informe["trend_warnings"]:
            handle.write("\n## Tendencias inesperadas\n\n")
            for aviso in informe["trend_warnings"]:
                handle.write(f"- {aviso}\n")

    return informe


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revisa la calidad de los indicadores de un análisis.")
    parser.add_argument("experiment")
    parser.add_argument("--analysis", default=None,
                        help="carpeta analysis (por defecto <experimento>/analysis)")
    return parser


def main(argv=None) -> int:
    args = construir_parser().parse_args(argv)
    experimento = os.path.abspath(os.path.expanduser(args.experiment))
    analisis = os.path.abspath(os.path.expanduser(
        args.analysis or os.path.join(experimento, "analysis")))

    try:
        informe = analizar(experimento, analisis)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"Informe de indicadores: "
          f"{os.path.join(analisis, '00_config', 'indicator_quality.md')}")
    for metrica, datos in informe["metrics"].items():
        estado = "OK" if datos["informative"] else "NO INFORMATIVO"
        print(f"  {metrica:8s} {estado:16s} "
              f"media={datos['mean']:.4g} std={datos['std']:.4g} "
              f"tendencia={datos['trend']['spearman']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
