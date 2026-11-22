#!/usr/bin/env python
"""
Regenera los conjuntos de referencia P* con N igual a la poblacion real de MOEAD.

El P* que usa el analisis para IGD+, DELTA_P y R2 debe tener la misma
cardinalidad que la poblacion que el MOEAD mantiene durante la busqueda: si el
frente de referencia y la poblacion no tienen el mismo tamano, la media de
distancias que hacen esos indicadores no mide lo mismo para todas las
instancias.

La poblacion real de MOEAD no es `MO_POPULATION_SIZE` a secas: pymoo exige
direcciones de referencia Das-Dennis, cuyo tamano es `C(particiones + m - 1,
m - 1)`, y `mohh.evaluation.algorithms.reference_directions_for_size` elige la
particion mas cercana al objetivo. Con `MO_POPULATION_SIZE = 105` eso da

    m = 2 -> 105    m = 3 -> 105    m = 4 -> 120    m = 5 -> 126

Ese es el N que se pide aqui por instancia.

La extraccion es la oficial del pipeline `set-reference` (RSEIterative,
Falcon-Cardona et al.): mismas 21 corridas, mismas 10000 iteraciones y el
mismo criterio `select="spread"`; cambia `subset_size`.

Si el ground set de una instancia tiene MENOS puntos que N, no se inventa
ninguno: se queda con todo el ground set y la instancia queda marcada como
`truncated` en el reporte. Un frente no dominado pequeno es un resultado, no
un error de medicion.

Salida (los P* originales de `reference_sets/` no se tocan):

    references_moea/<split>/reference_sets_pop/<instance>_ref.pof
    references_moea/<split>/reference_sets_pop/reference_manifest.json
    references_moea/<split>/reference_sets_pop/runs/<instance>/run_NN.pof
    references_moea/reference_sets_pop_report.{csv,md}

Uso:

    python generate_reference_sets_pop.py                  # train + test
    python generate_reference_sets_pop.py --split train
    python generate_reference_sets_pop.py --only KC13-3fl-1un   # piloto
    python generate_reference_sets_pop.py --dry-run         # solo calcula N
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

#: Clone externo con el paquete `setref` y el clon de RSEHH.
SET_REFERENCE_DIR = os.environ.get(
    "SET_REFERENCE_DIR",
    "/home/jabonsote/Desktop/THESIS/MOHH-v1-lab-paralel/SDD/set-reference")
if SET_REFERENCE_DIR not in sys.path:
    sys.path.insert(0, SET_REFERENCE_DIR)

from mohh.evaluation.algorithms import reference_directions_for_size  # noqa: E402
from mohh.evaluation.params import params_dict  # noqa: E402

#: Donde viven los ground sets y donde salen los P* nuevos.
GROUND_DIRNAME = "ground_sets"
OUTPUT_DIRNAME = "reference_sets_pop"


def tamano_poblacion_moead(n_objetivos: int, objetivo: int) -> int:
    """Tamano real de la poblacion de MOEAD para `n_objetivos` objetivos."""
    _, tamano = reference_directions_for_size(n_objetivos, objetivo)
    return int(tamano)


def cargar_ground_sets(directorio: str) -> dict:
    """Lee todos los `*_ground.pof` de un directorio."""
    rutas = sorted(glob.glob(os.path.join(directorio, "*_ground.pof")))
    if not rutas:
        raise FileNotFoundError(
            f"no hay ground sets en {directorio}; primero hay que generarlos")

    ground_sets = {}
    for ruta in rutas:
        nombre = os.path.basename(ruta)[:-len("_ground.pof")]
        frente = np.atleast_2d(np.genfromtxt(ruta))
        ground_sets[nombre] = frente
    return ground_sets


def _hay_dominaidos(frente: np.ndarray) -> bool:
    """True si algun punto del frente esta dominado por otro del mismo."""
    for i in range(len(frente)):
        peor = np.all(frente <= frente[i], axis=1) & np.any(
            frente < frente[i], axis=1)
        if peor.any():
            return True
    return False


def validar(ground_sets: dict, referencias: dict, salida: str,
            peticiones: dict) -> list:
    """
    Comprueba cada P* escrito contra su ground set y contra el N pedido.

    @param peticiones: instancia -> N (poblacion real de MOEAD).
    @return: lista de incidencias (vacia si todo esta bien).
    """
    from setref.pof import read_pof

    incidencias = []
    for instancia, payload in sorted(referencias.items()):
        ruta = os.path.join(salida, f"{instancia}_ref.pof")
        frente = read_pof(ruta)
        ground = np.asarray(ground_sets[instancia], dtype=float)

        if frente.shape[1] != ground.shape[1]:
            incidencias.append(
                f"{instancia}: {frente.shape[1]} objetivos en P* "
                f"!= {ground.shape[1]} en el ground set")

        # Todo punto de P* tiene que ser un punto real del ground set.
        ground_rows = {tuple(fila) for fila in np.round(ground, 6)}
        for fila in np.round(frente, 6):
            if tuple(fila) not in ground_rows:
                incidencias.append(
                    f"{instancia}: el punto {fila.tolist()} no esta en el "
                    f"ground set")
                break

        # P* no debe contener puntos dominados entre si.
        if _hay_dominaidos(frente):
            if _hay_dominaidos(ground):
                incidencias.append(
                    f"{instancia}: el ground set ya contiene puntos "
                    f"dominados y P* los hereda (preexistente)")
            else:
                incidencias.append(
                    f"{instancia}: P* contiene puntos dominados que no "
                    f"estan dominados en el ground set")

        esperado = min(peticiones[instancia], len(ground))
        if len(frente) != esperado:
            incidencias.append(
                f"{instancia}: P* tiene {len(frente)} puntos y se esperaban "
                f"{esperado}")

    return incidencias


def generar_split(split: str, args) -> list:
    """Regenera un split completo.

    @return: (filas del reporte, incidencias de validacion).
    """
    from setref.extract import extract_reference_sets, save_reference_sets

    raiz = os.path.join(args.references, split)
    directorio_ground = os.path.join(raiz, GROUND_DIRNAME)
    directorio_salida = os.path.join(raiz, OUTPUT_DIRNAME)

    ground_sets = cargar_ground_sets(directorio_ground)
    if args.only:
        seleccion = [n for n in args.only if n in ground_sets]
        faltan = sorted(set(args.only) - set(seleccion))
        if faltan:
            raise SystemExit(
                f"{split}: instancias sin ground set: {faltan}")
        ground_sets = {n: ground_sets[n] for n in seleccion}

    # N por instancia: la poblacion real de MOEAD para su numero de objetivos.
    peticiones = {}
    for instancia, ground in ground_sets.items():
        m = int(ground.shape[1])
        peticiones[instancia] = tamano_poblacion_moead(m, args.population_size)

    if args.dry_run:
        for instancia in sorted(peticiones):
            ground = ground_sets[instancia]
            n = peticiones[instancia]
            print(f"  {split:5s} {instancia:18s} m={ground.shape[1]} "
                  f"N={n:4d} |ground|={len(ground):5d} "
                  f"{'TRUNCADO' if len(ground) < n else ''}")
        return [], []

    print(f"[{split}] {len(ground_sets)} instancias -> {directorio_salida}")

    # RSEIterative pide un unico subset_size, asi que se agrupa por N.
    grupos = defaultdict(dict)
    for instancia, n in peticiones.items():
        grupos[n][instancia] = ground_sets[instancia]

    referencias = {}
    for n in sorted(grupos):
        subconjunto = grupos[n]
        print(f"[{split}] N={n}: {len(subconjunto)} instancia(s)", flush=True)
        extraidas = extract_reference_sets(
            subconjunto, subset_size=n, runs=args.runs,
            iterations=args.iterations, select=args.select,
            progress=args.progress)
        referencias.update(extraidas)

    # UNA sola escritura: `save_reference_sets` reescribe el manifiesto entero.
    manifest = save_reference_sets(referencias, directorio_salida,
                                   save_all_runs=args.save_runs)

    incidencias = validar(ground_sets, referencias, directorio_salida,
                          peticiones)
    for incidencia in incidencias:
        print(f"  [aviso] {split}/{incidencia}", flush=True)

    filas = []
    for instancia in sorted(referencias):
        payload = referencias[instancia]
        ground = ground_sets[instancia]
        n = peticiones[instancia]
        obtenido = int(payload["subset_size"])
        filas.append({
            "split": split,
            "instance": instancia,
            "objectives": int(ground.shape[1]),
            "population_size": args.population_size,
            "n_requested": n,
            "n_written": obtenido,
            "ground_size": int(len(ground)),
            "truncated": bool(len(ground) < n),
            "selected_run": int(payload["selected_run"]),
            "selected_seed": int(payload["selected_seed"]),
            "n_runs": int(manifest[instancia]["n_runs"]),
            "select": args.select,
            "iterations": args.iterations,
        })

    return filas, [f"{split}/{i}" for i in incidencias]


def escribir_reporte(filas: list, incidencias: list, ruta_base: str,
                     args) -> None:
    """CSV + Markdown con el tamano de cada P* generado."""
    if not filas:
        return

    campos = list(filas[0])
    with open(ruta_base + ".csv", "w", newline="") as handle:
        escritor = csv.DictWriter(handle, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)

    lineas = [
        "# P* con N = poblacion de MOEAD", "",
        f"Generado con `generate_reference_sets_pop.py` "
        f"(runs={args.runs}, iterations={args.iterations}, "
        f"select={args.select}, MO_POPULATION_SIZE="
        f"{args.population_size}).", "",
        "El tamano pedido es el de la poblacion real de MOEAD para el numero "
        "de objetivos de la instancia (Das-Dennis): 105 para m=2 y m=3, 120 "
        "para m=4 y 126 para m=5.", "",
        "| split | instancia | m | N pedido | N escrito | ground | "
        "truncado | run |", "|---|---|---:|---:|---:|---:|:---:|---:|",
    ]
    for fila in filas:
        lineas.append(
            f"| {fila['split']} | {fila['instance']} | "
            f"{fila['objectives']} | {fila['n_requested']} | "
            f"{fila['n_written']} | {fila['ground_size']} | "
            f"{'si' if fila['truncated'] else 'no'} | "
            f"{fila['selected_run']} |")

    truncados = [f for f in filas if f["truncated"]]
    lineas.extend(["", f"Instancias truncadas (|ground| < N): "
                    f"{len(truncados)}"])
    for fila in truncados:
        lineas.append(f"- {fila['split']}/{fila['instance']}: "
                      f"{fila['ground_size']} < {fila['n_requested']}")

    lineas.extend(["", "## Validacion", "",
                   "Comprobaciones: cabecera `# N m` coherente, todo punto de "
                   "P* presente en el ground set, sin puntos dominados dentro "
                   "de P* y cardinalidad igual a N (o al ground set si este "
                   "es menor).", ""])
    if incidencias:
        for incidencia in incidencias:
            lineas.append(f"- {incidencia}")
    else:
        lineas.append("- sin incidencias")

    with open(ruta_base + ".md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lineas) + "\n")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenera P* con N = poblacion de MOEAD.")
    parser.add_argument("--references", default="references_moea",
                        help="raiz con <split>/ground_sets")
    parser.add_argument("--split", nargs="+", default=["train", "test"],
                        choices=["train", "test"])
    parser.add_argument("--population-size", type=int,
                        default=int(params_dict["MO_POPULATION_SIZE"]),
                        help="MO_POPULATION_SIZE objetivo de MOEAD")
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--select", default="spread",
                        choices=["spread", "first"])
    parser.add_argument("--only", nargs="*", default=None,
                        help="solo estas instancias (piloto)")
    parser.add_argument("--save-runs", action="store_true", default=True)
    parser.add_argument("--no-save-runs", dest="save_runs",
                        action="store_false")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--dry-run", action="store_true",
                        help="solo imprime el N de cada instancia")
    parser.add_argument("--workers", type=int, default=None,
                        help="MOHH_WORKERS para el pool de extraccion")
    return parser


def main(argv=None) -> int:
    args = construir_parser().parse_args(argv)

    if args.workers:
        os.environ["MOHH_WORKERS"] = str(args.workers)

    inicio = time.time()
    filas = []
    incidencias = []
    for split in args.split:
        parte, avisos = generar_split(split, args)
        filas.extend(parte)
        incidencias.extend(avisos)

    if filas:
        # El reporte acumula los splits: regenerar uno solo no borra al otro.
        ruta = os.path.join(args.references, "reference_sets_pop_report")
        previas = {}
        if os.path.isfile(ruta + ".csv"):
            with open(ruta + ".csv", newline="") as handle:
                for fila in csv.DictReader(handle):
                    previas[(fila["split"], fila["instance"])] = fila
        for fila in filas:
            previas[(fila["split"], fila["instance"])] = fila
        combinadas = [previas[clave] for clave in sorted(previas)]
        for fila in combinadas:
            for clave, valor in list(fila.items()):
                if clave in ("objectives", "population_size", "n_requested",
                             "n_written", "ground_size", "selected_run",
                             "selected_seed", "n_runs"):
                    fila[clave] = int(valor)
                elif clave == "truncated":
                    fila[clave] = str(valor).lower() in ("true", "1", "si")
        escribir_reporte(combinadas, incidencias, ruta, args)
        print(f"Reporte: {ruta}.csv / {ruta}.md")

        total = sum(f["n_written"] for f in combinadas)
        truncados = sum(1 for f in combinadas if f["truncated"])
        print(f"{len(combinadas)} P* en el reporte ({len(filas)} regenerados), "
              f"{total} puntos, {truncados} truncados, "
              f"{time.time() - inicio:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
