"""
Complejidad temporal de los operadores, medida sobre las poblaciones reales.

El analisis no inventa permutaciones: usa las poblaciones iniciales que el
MOEA recibio de verdad, guardadas en
`generation_results/initial_solutions/permutation/<instancia>.txt`. Sobre cada
pareja se ejecuta el fenotipo del operador (el mismo `eval` que usa
`HH_Operator._do`) y se cronometra.

Se reportan dos tiempos:

* `per_pair_s`: media del tiempo de una llamada sobre todas las parejas;
* `per_mating_s`: lo que cuesta una generacion completa del MOEA, es decir
  `P` matings (`MO_POPULATION_SIZE`), que es la unidad en la que se comparan
  los operadores en la practica.

Como los operadores fenotipicos usan `np.random`, el tiempo se mide con una
semilla fija (`seed`) para que las ejecuciones sean reproducibles.
"""

import os
import time
import numpy as np
import pandas as pd

from mohh.core.utilities.algorithm.HH_functions import *  # noqa: F401,F403

from mohh.analysis.io_utils import load_initial_permutations

__all__ = [
    "OperatorTimer",
    "time_operator",
    "measure_complexity",
]


class OperatorTimer:
    """
    Cronometra un operador fenotipico sobre poblaciones reales.

    @param phenotype: expresion del operador, tal cual aparece en
        `<modelo>_operators.txt`.
    @param seed: semilla de `np.random` antes de cada llamada, para que el
        operador vea siempre la misma entrada aleatoria y el tiempo sea
        comparable entre operadores.
    """

    def __init__(self, phenotype: str, seed: int = 0) -> None:
        self.phenotype = phenotype
        self.seed = int(seed)

    def call(self, x, y):
        """Ejecuta el operador sobre una pareja y devuelve el hijo."""
        np.random.seed(self.seed)
        namespace = _namespace()
        namespace["x"] = x
        namespace["y"] = y
        return eval(self.phenotype, namespace)

    def time_one(self, x, y, repeats: int = 1) -> float:
        """Tiempo medio (segundos) de `repeats` llamadas sobre (x, y)."""
        if repeats < 1:
            raise ValueError("repeats debe ser >= 1")

        # Una llamada de calentamiento, no se cronometra.
        self.call(x, y)

        muestras = []
        for _ in range(repeats):
            inicio = time.perf_counter()
            self.call(x, y)
            muestras.append(time.perf_counter() - inicio)

        return float(np.median(muestras))


def _namespace() -> dict:
    """Namespace con las funciones de operador y `np`."""
    import mohh.core.utilities.algorithm.HH_functions as helpers
    return {name: getattr(helpers, name) for name in dir(helpers)
            if not name.startswith("_")} | {"np": np}


def time_operator(
    phenotype: str,
    populations: dict,
    repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Cronometra un operador en cada instancia.

    @param phenotype: expresion del operador.
    @param populations: `{instancia: [permutaciones]}` reales.
    @param repeats: llamadas por pareja (se usa la mediana).
    @param seed: semilla fija para `np.random`.
    @return: DataFrame con una fila por instancia y columnas `instance`,
        `n_pairs` y `per_pair_s` (segundos por cruce). El coste por generacion
        se anade en `measure_complexity` con `matings`.
    """
    timer = OperatorTimer(phenotype, seed)
    filas = []

    for instancia, soluciones in sorted(populations.items()):
        soluciones = [np.asarray(s) for s in soluciones]
        n = len(soluciones)
        if n < 2:
            continue

        tiempos = []
        for i in range(n - 1):
            tiempos.append(timer.time_one(soluciones[i], soluciones[i + 1],
                                          repeats=repeats))

        filas.append({
            "instance": instancia,
            "n_pairs": len(tiempos),
            "per_pair_s": float(np.mean(tiempos)),
        })

    return pd.DataFrame(filas)


def measure_complexity(
    operators: dict,
    initial_solutions: dict,
    repeats: int = 5,
    seed: int = 0,
    matings: int = None,
) -> pd.DataFrame:
    """
    Cronometra varios operadores sobre las poblaciones iniciales reales.

    @param operators: `{etiqueta: fenotipo}` (por ejemplo `Best`, `Middle`,
        `Worst`).
    @param initial_solutions: `{instancia: [permutaciones]}`.
    @param repeats: llamadas por pareja.
    @param seed: semilla fija.
    @param matings: numero de cruces por generacion (habitualmente
        `pop_size / 2`). Si se indica se anade la columna `per_generation_s`,
        que es el coste de aplicar el operador a toda una generacion.
    @return: DataFrame largo con una fila por (operador, instancia) y columnas
        `operator`, `instance`, `n_pairs`, `per_pair_s` y, opcionalmente,
        `per_generation_s`.
    """
    filas = []

    for etiqueta, fenotipo in sorted(operators.items()):
        tabla = time_operator(fenotipo, initial_solutions,
                              repeats=repeats, seed=seed)
        if tabla.empty:
            continue
        tabla.insert(0, "operator", etiqueta)
        filas.append(tabla)

    if not filas:
        return pd.DataFrame(columns=["operator", "instance", "n_pairs",
                                     "per_pair_s"])

    resultado = pd.concat(filas, ignore_index=True)

    if matings:
        resultado["per_generation_s"] = resultado["per_pair_s"] * int(matings)

    return resultado


def load_populations(experiment, instances=None) -> dict:
    """
    Carga las poblaciones iniciales reales de un experimento.

    @param experiment: `Experiment` de `io_utils`.
    @param instances: subconjunto de instancias; si es None, todas las que
        tengan archivo de soluciones iniciales.
    @return: `{instancia: [permutaciones]}`.
    """
    poblaciones = {}
    nombres = instances or sorted({
        instancia for ind in experiment.individuals.values()
        for instancia in ind.instances
    })

    for instancia in nombres:
        ruta = experiment.initial_solutions_path(instancia)
        if ruta and os.path.isfile(ruta):
            try:
                poblaciones[instancia] = load_initial_permutations(ruta)
            except ValueError:
                continue

    return poblaciones
