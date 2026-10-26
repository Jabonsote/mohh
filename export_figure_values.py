#!/usr/bin/env python
"""
Exporta a CSV los datos que dibujan las figuras del análisis.

- `analysis/`: copia centralizada de las tablas que alimentan las figuras de
  indicadores, generaciones, complejidad, operadores y diversidad AST.
- `analysis_search/`: trayectorias normalizadas (media y mejor), boxplots,
  mapas de calor y comparación train/test.
- `results/figure_values/ast_diversity_all.csv`: AST de todos los experimentos.

No modifica las figuras ni los CSV originales.
"""

from __future__ import annotations

import argparse
import os
import shutil

import pandas as pd

import analyze_search as asearch

METRICAS = ("HV", "IGD+", "DELTA_P", "R2")


def _guardar(df, ruta):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    df.to_csv(ruta, index=True)


def _exportar_search(base, metrica):
    tabla = os.path.join(base, "01_tables", "individual_split.csv")
    if not os.path.isfile(tabla):
        return []
    largo = pd.read_csv(tabla)
    destino = os.path.join(base, "figure_values")
    os.makedirs(destino, exist_ok=True)
    rutas = []

    for split in ("train", "test"):
        parte = largo[largo["split"] == split].copy()
        if parte.empty:
            continue

        # Valores individuales de los boxplots.
        ruta = os.path.join(destino, f"boxplot_por_generacion_{split}.csv")
        parte.to_csv(ruta, index=False)
        rutas.append(ruta)

        tablas = asearch.tabla_matriz(parte, METRICAS)
        mejores = asearch.mejor_por_generacion(parte)

        for metrica_actual in METRICAS:
            matriz = tablas.get(metrica_actual)
            if matriz is None or matriz.empty:
                continue

            normalizado, _ = asearch._normaliza_por_instancia(matriz)
            salida = normalizado.T.copy()
            salida["media"] = normalizado.mean(axis=0).to_numpy()
            salida.index.name = "generation"
            ruta = os.path.join(
                destino, f"trayectorias_media_{split}_{metrica_actual}.csv")
            _guardar(salida, ruta)
            rutas.append(ruta)

            parte_mejor = mejores[mejores["metric"] == metrica_actual]
            if not parte_mejor.empty:
                bruto = parte_mejor.pivot(index="instance", columns="generation",
                                          values="mejor")
                norm_mejor, _ = asearch._normaliza_por_instancia(bruto)
                salida_mejor = norm_mejor.T.copy()
                salida_mejor.index.name = "generation"
                ruta = os.path.join(
                    destino, f"trayectorias_mejor_{split}_{metrica_actual}.csv")
                _guardar(salida_mejor, ruta)
                rutas.append(ruta)

            # Mapa de calor: media por instancia y generación.
            ruta = os.path.join(
                destino, f"mapa_calor_{split}_{metrica_actual}.csv")
            _guardar(matriz, ruta)
            rutas.append(ruta)

    # Medias train/test por generación (figuras comparativas).
    for metrica_actual in METRICAS:
        parte = largo[largo["metric"] == metrica_actual]
        if parte.empty:
            continue
        tabla_medias = (parte.groupby(["split", "generation"])["value"]
                        .mean().reset_index())
        ruta = os.path.join(destino, f"medias_train_test_{metrica_actual}.csv")
        tabla_medias.to_csv(ruta, index=False)
        rutas.append(ruta)

    return rutas


def _exportar_analysis(base):
    destino = os.path.join(base, "figure_values")
    os.makedirs(destino, exist_ok=True)
    etiqueta = os.path.basename(os.path.dirname(os.path.abspath(base)))
    copias = {
        "individuals.csv": os.path.join("01_individuals",
                                        f"indicators_{etiqueta}.csv"),
        "convergencia.csv": os.path.join("02_generations",
                                         f"generations_{etiqueta}.csv"),
        "complejidad.csv": os.path.join("04_complexity",
                                        f"complexity_{etiqueta}.csv"),
        "operadores.csv": os.path.join("05_operators", "MOEAD", "operators.csv"),
        "ast_diversity.csv": os.path.join(
            "08_ast_diversity",
            f"ast_diversity_{etiqueta}.csv"),
    }
    rutas = []
    for nombre, relativo in copias.items():
        origen = os.path.join(base, relativo)
        if os.path.isfile(origen):
            ruta = os.path.join(destino, nombre)
            shutil.copy2(origen, ruta)
            rutas.append(ruta)
    return rutas


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exporta los valores de las figuras a CSV.")
    parser.add_argument("--experiments-root",
                        default=os.path.join("results", "experiments_larger"))
    parser.add_argument("--out-root",
                        default=os.path.join("results", "figure_values"))
    args = parser.parse_args(argv)

    ast_filas = []
    total = 0
    for nombre in sorted(os.listdir(args.experiments_root)):
        experimento = os.path.join(args.experiments_root, nombre)
        if not os.path.isdir(experimento) or not nombre.startswith("Experiment_"):
            continue

        search = os.path.join(experimento, "analysis_search")
        analysis = os.path.join(experimento, "analysis")
        if os.path.isdir(search):
            rutas = _exportar_search(search, None)
            total += len(rutas)
            print(f"{nombre}: {len(rutas)} CSV en analysis_search/figure_values")

        if os.path.isdir(analysis):
            rutas = _exportar_analysis(analysis)
            total += len(rutas)
            print(f"{nombre}: {len(rutas)} CSV en analysis/figure_values")

        ast = os.path.join(analysis, "08_ast_diversity",
                           f"ast_diversity_{nombre}.csv")
        if os.path.isfile(ast):
            tabla = pd.read_csv(ast)
            tabla.insert(0, "experiment", nombre)
            ast_filas.append(tabla)

    if ast_filas:
        os.makedirs(args.out_root, exist_ok=True)
        ruta = os.path.join(args.out_root, "ast_diversity_all.csv")
        pd.concat(ast_filas, ignore_index=True).to_csv(ruta, index=False)
        print(f"AST conjunto: {ruta}")
        total += 1

    print(f"Total: {total} CSV")


if __name__ == "__main__":
    main()
