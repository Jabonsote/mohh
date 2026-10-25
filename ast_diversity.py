#!/usr/bin/env python
"""
Paso 4 del pipeline `larger_pipeline`: diversidad de los AST por generación.

Cada individuo de la GE es un operador de variación cuya representación es un
árbol sintáctico (el `phenotype` que guarda `general_info.json`). Este script
mide, generación a generación, cuán distintos son esos árboles:

* **elite**: los primeros `ELITE_SIZE` individuos de la población guardada
  (`generations/generation_NNNN/population.json`, ordenada de mejor a peor por
  el reemplazo elitista);
* **todos**: la población completa de esa generación.

La distancia entre dos AST es la distancia de Jaccard entre sus conjuntos de
subárboles:

    d(A, B) = 1 - |sub(A) ∩ sub(B)| / |sub(A) ∪ sub(B)|

donde `sub(X)` es el conjunto de huellas SHA-1 de todos los subárboles de X.
Dos operadores idénticos dan 0; dos operadores sin ningún subárbol en común dan
1. Para cada generación se reporta la media de las distancias de todos los pares
(la "diversidad" de la población) y la fracción de AST distintos.

La figura tiene dos paneles: diversidad media por pares y proporción de AST
únicos, siempre con dos curvas (elite y todos). Se guarda junto a las tablas en
`<salida>/08_ast_diversity/`, sin tocar las figuras del análisis original.

Uso:

    python larger_pipeline/ast_diversity.py \\
        results/experiments_larger/Experiment_001 \\
        --output results/experiments_larger/Experiment_001/analysis
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mohh.analysis.layout import find_run_dirs  # noqa: E402


class DiversidadError(RuntimeError):
    """No se puede calcular la diversidad de un experimento."""


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


def _etiqueta(nodo: ast.AST) -> str:
    """
    Etiqueta de un nodo AST.

    `Name` y `Constant` se abren para distinguir `order_based` de `reverse` o
    `x` de `y`; el resto basta con el tipo de nodo. Así dos llamadas al mismo
    operador con argumentos distintos no se consideran el mismo subárbol.
    """
    if isinstance(nodo, ast.Name):
        return f"Name:{nodo.id}"
    if isinstance(nodo, ast.Constant):
        return f"Constant:{nodo.value!r}"
    if isinstance(nodo, ast.Attribute):
        return f"Attribute:{nodo.attr}"
    return type(nodo).__name__


def _hijos(nodo: ast.AST) -> list:
    """Hijos en el orden de los campos del AST (estable para un mismo Python)."""
    return list(ast.iter_child_nodes(nodo))


def _huella_subarbol(nodo: ast.AST, memo: dict) -> str:
    """
    Huella SHA-1 de la estructura del subárbol con raíz en `nodo`.

    Es recursiva y memoizada por identidad de nodo; el `phenotype` tiene como
    mucho unos cientos de nodos, así que no hay riesgo de recursión profunda.
    """
    clave = id(nodo)
    if clave in memo:
        return memo[clave]

    partes = [_etiqueta(nodo)]
    partes.extend(_huella_subarbol(hijo, memo) for hijo in _hijos(nodo))
    huella = hashlib.sha1("(".join(partes).encode("utf-8")).hexdigest()[:16]
    memo[clave] = huella
    return huella


def _profundidad(nodo: ast.AST) -> int:
    """Altura del subárbol (una hoja vale 1)."""
    hijos = _hijos(nodo)
    if not hijos:
        return 1
    return 1 + max(_profundidad(hijo) for hijo in hijos)


def _nodos(nodo: ast.AST) -> list:
    """Nodos en preorden."""
    salida = [nodo]
    for hijo in _hijos(nodo):
        salida.extend(_nodos(hijo))
    return salida


def caracterizar_ast(phenotype: str) -> dict:
    """
    Convierte un `phenotype` en las características que usa la diversidad.

    @param phenotype: expresión del operador, tal como la guarda la GE.
    @return: dict con `firma` (AST completo), `subarboles` (conjunto de
        huellas), `nodos`, `profundidad` y `etiquetas` (conteo por etiqueta).
    """
    try:
        arbol = ast.parse(str(phenotype).strip(), mode="eval")
    except SyntaxError as error:
        raise DiversidadError(
            f"phenotype no parseable como expresión Python: {error}") from error

    raiz = arbol.body
    memo = {}
    firma = _huella_subarbol(raiz, memo)
    nodos = _nodos(raiz)

    return {
        "firma": firma,
        "subarboles": set(memo.values()),
        "nodos": len(nodos),
        "profundidad": _profundidad(raiz),
        "etiquetas": Counter(_etiqueta(nodo) for nodo in nodos),
    }


def distancia_jaccard(a: set, b: set) -> float:
    """
    Distancia de Jaccard entre dos conjuntos de subárboles.

    @return: 0 si son idénticos, 1 si no comparten ningún subárbol.
    """
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def diversidad_media(caracteristicas: list) -> float:
    """
    Media de las distancias de Jaccard de todos los pares de una población.

    Una población con menos de dos individuos tiene diversidad 0 (no hay
    ningún par que comparar).
    """
    n = len(caracteristicas)
    if n < 2:
        return 0.0

    total = 0.0
    pares = 0
    for i in range(n):
        sub_i = caracteristicas[i]["subarboles"]
        for j in range(i + 1, n):
            total += distancia_jaccard(sub_i, caracteristicas[j]["subarboles"])
            pares += 1

    return total / pares if pares else 0.0


# ---------------------------------------------------------------------------
# Lectura de la corrida
# ---------------------------------------------------------------------------


def _leer_json(ruta: str):
    with open(ruta) as handle:
        return json.load(handle)


def _elite_size(run_dir: str, override: int | None) -> int:
    """Tamaño de la elite: el del `parameters.txt` o el que pase el usuario."""
    if override is not None:
        return int(override)

    ruta = os.path.join(run_dir, "generation_results", "parameters.txt")
    if os.path.isfile(ruta):
        with open(ruta) as handle:
            for linea in handle:
                if ":" not in linea:
                    continue
                clave, _, valor = linea.partition(":")
                if clave.strip() == "ELITE_SIZE":
                    try:
                        return int(valor.strip())
                    except ValueError:
                        break

    return 15


def _individuo(run_dir: str, nombre: str, cache: dict) -> dict:
    """Características AST de un individuo, con caché por nombre."""
    if nombre in cache:
        return cache[nombre]

    ruta = os.path.join(run_dir, "generation_results", "individuals",
                        nombre, "general_info.json")
    if not os.path.isfile(ruta):
        raise DiversidadError(
            f"falta {ruta}; la población referencia un individuo sin "
            f"general_info.json")

    datos = _leer_json(ruta)
    phenotype = datos.get("phenotype")
    if not phenotype:
        raise DiversidadError(f"{ruta} no tiene 'phenotype'")

    caracteristicas = caracterizar_ast(phenotype)
    caracteristicas["name"] = nombre
    cache[nombre] = caracteristicas
    return caracteristicas


def _generaciones(run_dir: str) -> list:
    """
    Carpetas `generation_NNNN` con su población guardada, ordenadas.

    `population.json` es la población real después del reemplazo y está
    ordenada de mejor a peor. Si faltara, se usa `offspring.json` como
    respaldo y se marca la fuente.
    """
    base = os.path.join(run_dir, "generation_results", "generations")
    if not os.path.isdir(base):
        raise DiversidadError(f"no existe {base}")

    generaciones = []
    for nombre in os.listdir(base):
        ruta = os.path.join(base, nombre)
        if not (os.path.isdir(ruta) and nombre.startswith("generation_")):
            continue

        poblacion = os.path.join(ruta, "population.json")
        fuente = "population"
        if not os.path.isfile(poblacion):
            poblacion = os.path.join(ruta, "offspring.json")
            fuente = "offspring"
        if not os.path.isfile(poblacion):
            continue

        try:
            numero = int(nombre.split("_")[-1])
        except ValueError:
            continue

        generaciones.append((numero, poblacion, fuente))

    return sorted(generaciones)


def analizar_run(run_dir: str, elite_size: int | None = None) -> pd.DataFrame:
    """
    Calcula la diversidad AST de todas las generaciones de una corrida.

    @param run_dir: carpeta `<modelo>_<timestamp>`.
    @param elite_size: tamaño de la elite; si es None se lee del parameters.txt.
    @return: DataFrame con una fila por generación y run.
    """
    elite = _elite_size(run_dir, elite_size)
    cache = {}
    filas = []

    generaciones = _generaciones(run_dir)
    if not generaciones:
        raise DiversidadError(
            f"no hay generaciones con población en {run_dir}")

    for numero, ruta_poblacion, fuente in generaciones:
        nombres = _leer_json(ruta_poblacion)
        if not isinstance(nombres, list):
            raise DiversidadError(
                f"{ruta_poblacion} no es una lista de individuos")

        if not nombres:
            continue

        caracteristicas = [_individuo(run_dir, nombre, cache)
                           for nombre in nombres]
        elite_caracteristicas = caracteristicas[:min(elite, len(caracteristicas))]

        firmas_todas = [c["firma"] for c in caracteristicas]
        firmas_elite = [c["firma"] for c in elite_caracteristicas]

        filas.append({
            "run": os.path.basename(run_dir),
            "generation": int(numero),
            "source": fuente,
            "n_all": len(caracteristicas),
            "n_elite": len(elite_caracteristicas),
            "unique_all": len(set(firmas_todas)),
            "unique_elite": len(set(firmas_elite)),
            "unique_ratio_all": len(set(firmas_todas)) / len(firmas_todas),
            "unique_ratio_elite": (len(set(firmas_elite)) / len(firmas_elite)
                                   if firmas_elite else 0.0),
            "diversity_all": diversidad_media(caracteristicas),
            "diversity_elite": diversidad_media(elite_caracteristicas),
            "nodes_mean_all": float(np.mean([c["nodos"] for c in caracteristicas])),
            "nodes_mean_elite": float(np.mean([c["nodos"] for c in elite_caracteristicas])),
            "depth_mean_all": float(np.mean([c["profundidad"] for c in caracteristicas])),
            "depth_mean_elite": float(np.mean([c["profundidad"] for c in elite_caracteristicas])),
        })

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Figura
# ---------------------------------------------------------------------------


def figura_diversidad(tabla: pd.DataFrame, etiqueta: str, ruta: str) -> str:
    """
    Dibuja los dos paneles de diversidad: distancia media y AST únicos.

    @param tabla: salida de `analizar_run` para una o varias corridas.
    @param etiqueta: nombre del experimento, para el título.
    @param ruta: ruta del PNG.
    @return: la ruta escrita.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = sorted(tabla["run"].unique())
    colores = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    figura, ejes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)

    for indice, run in enumerate(runs):
        datos = tabla[tabla["run"] == run].sort_values("generation")
        color = colores[indice % len(colores)]
        sufijo = f" ({run})" if len(runs) > 1 else ""

        ejes[0].plot(datos["generation"], datos["diversity_all"],
                     marker="o", markersize=3, linewidth=1.4, color=color,
                     label=f"todos{sufijo}")
        ejes[0].plot(datos["generation"], datos["diversity_elite"],
                     marker="s", markersize=3, linewidth=1.4, color=color,
                     linestyle="--", label=f"elite{sufijo}")

        ejes[1].plot(datos["generation"], datos["unique_ratio_all"],
                     marker="o", markersize=3, linewidth=1.4, color=color,
                     label=f"todos{sufijo}")
        ejes[1].plot(datos["generation"], datos["unique_ratio_elite"],
                     marker="s", markersize=3, linewidth=1.4, color=color,
                     linestyle="--", label=f"elite{sufijo}")

    ejes[0].set_ylabel("distancia AST media por pares\n(1 - Jaccard de subárboles)")
    ejes[0].set_ylim(-0.02, 1.02)
    ejes[0].set_title(f"Diversidad AST por generación ({etiqueta})")
    ejes[0].grid(alpha=0.3)

    ejes[1].set_ylabel("AST únicos / individuos")
    ejes[1].set_xlabel("generación")
    ejes[1].set_ylim(-0.02, 1.02)
    ejes[1].grid(alpha=0.3)

    for eje in ejes:
        eje.legend(loc="best", fontsize=8)

    figura.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    figura.savefig(ruta, dpi=200)
    plt.close(figura)

    return ruta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diversidad de los AST (elite y población completa) por "
                    "generación.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("experiment",
                        help="carpeta results/experiments_larger/Experiment_NNN")
    parser.add_argument("--output", default=None,
                        help="carpeta de salida; por defecto "
                             "<experimento>/analysis")
    parser.add_argument("--elite-size", type=int, default=None,
                        help="tamaño de la elite; por defecto ELITE_SIZE del "
                             "parameters.txt")
    parser.add_argument("--no-figure", action="store_true",
                        help="no dibujar el PNG, solo las tablas")
    return parser


def main(argv=None) -> int:
    args = construir_parser().parse_args(argv)

    experimento = os.path.abspath(os.path.expanduser(args.experiment))
    if not os.path.isdir(experimento):
        print(f"error: no existe el experimento: {experimento}", file=sys.stderr)
        return 1

    salida = os.path.abspath(os.path.expanduser(
        args.output or os.path.join(experimento, "analysis")))
    destino = os.path.join(salida, "08_ast_diversity")
    os.makedirs(destino, exist_ok=True)

    etiqueta = os.path.basename(experimento)

    try:
        runs = [ruta for ruta in find_run_dirs(experimento)
                if os.path.isdir(os.path.join(
                    ruta, "generation_results", "generations"))]
        if not runs:
            raise DiversidadError(
                f"no hay corridas con generation_results en {experimento}")

        tablas = [analizar_run(run, args.elite_size) for run in runs]
        tabla = pd.concat(tablas, ignore_index=True)
        if tabla.empty:
            raise DiversidadError(
                f"no se pudo calcular ninguna generación en {experimento}")

        ruta_csv = os.path.join(destino, f"ast_diversity_{etiqueta}.csv")
        tabla.to_csv(ruta_csv, index=False)

        resumen = {
            "experiment": experimento,
            "runs": [os.path.basename(r) for r in runs],
            "elite_size": (args.elite_size if args.elite_size is not None
                           else {os.path.basename(r): _elite_size(r, None)
                                 for r in runs}),
            "generations": int(len(tabla)),
            "diversity_all_mean": float(tabla["diversity_all"].mean()),
            "diversity_elite_mean": float(tabla["diversity_elite"].mean()),
            "unique_ratio_all_mean": float(tabla["unique_ratio_all"].mean()),
            "unique_ratio_elite_mean": float(tabla["unique_ratio_elite"].mean()),
            "csv": ruta_csv,
        }

        ruta_json = os.path.join(destino, f"ast_diversity_{etiqueta}.json")
        with open(ruta_json, "w") as handle:
            json.dump(resumen, handle, indent=2, ensure_ascii=False)

        print("=" * 70)
        print("Diversidad AST por generación")
        print("=" * 70)
        print(f"Experimento   : {experimento}")
        print(f"Corridas      : {len(runs)}")
        print(f"Generaciones  : {len(tabla)}")
        print(f"Diversidad media (todos): {resumen['diversity_all_mean']:.4f}")
        print(f"Diversidad media (elite): {resumen['diversity_elite_mean']:.4f}")
        print(f"AST únicos medios (todos): "
              f"{resumen['unique_ratio_all_mean']:.4f}")
        print(f"AST únicos medios (elite): "
              f"{resumen['unique_ratio_elite_mean']:.4f}")
        print(f"Tabla         : {ruta_csv}")
        print(f"Resumen       : {ruta_json}")

        if not args.no_figure:
            ruta_png = os.path.join(
                destino, f"ast_diversity_{etiqueta}.png")
            figura_diversidad(tabla, etiqueta, ruta_png)
            print(f"Figura        : {ruta_png}")

        return 0

    except (DiversidadError, FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
