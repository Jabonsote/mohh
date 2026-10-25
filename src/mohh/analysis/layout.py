"""
Resolucion de rutas y organizacion de carpetas del analisis offline.

El repositorio guarda los experimentos en dos formatos distintos:

* `results/Experiment_001/` (generacion + evaluacion en la misma carpeta), y
* `results/<fitness>/Experiment_001/` (un subdirectorio por la fitness con la
  que se genero, por ejemplo `results/R2/Experiment_001`).

Este modulo acepta cualquiera de los dos y devuelve la raiz del experimento
junto con un esquema de carpetas ordenado y numerado, de forma que todos los
artefactos del analisis queden agrupados y en un orden reproducible.
"""

import os

__all__ = [
    "AnalysisLayout",
    "detect_experiment_root",
    "find_run_dirs",
    "INDIVIDUAL_PREFIX",
]

# Prefijo de las carpetas de individuos que escribe `PopulationSaver`.
INDIVIDUAL_PREFIX = "Individual_"


def detect_experiment_root(path: str) -> str:
    """
    Normaliza la ruta de un experimento.

    Acepta:

    * `results/R2/Experiment_001` -> el mismo;
    * `results/R2` -> `results/R2/Experiment_001` si hay exactamente uno;
    * `results/Experiment_001` -> el mismo;
    * `results` -> `results/Experiment_001` si hay exactamente uno.

    @param path: ruta dada por el usuario.
    @return: ruta absoluta de la raiz del experimento.
    """

    path = os.path.abspath(os.path.expanduser(path))

    if not os.path.isdir(path):
        raise FileNotFoundError(f"no existe el directorio: {path}")

    if os.path.basename(path).startswith("Experiment_"):
        return path

    candidatos = sorted(
        os.path.join(path, nombre)
        for nombre in os.listdir(path)
        if nombre.startswith("Experiment_")
        and os.path.isdir(os.path.join(path, nombre))
    )

    if not candidatos:
        raise FileNotFoundError(
            f"{path} no contiene ninguna carpeta Experiment_*; indica la ruta "
            f"de un experimento generado por mohh-generate."
        )

    if len(candidatos) > 1:
        raise ValueError(
            f"{path} contiene {len(candidatos)} experimentos "
            f"({', '.join(os.path.basename(c) for c in candidatos)}); "
            f"indica cual usar."
        )

    return candidatos[0]


def find_run_dirs(experiment_root: str) -> list:
    """
    Carpetas `<modelo>_<timestamp>` de una misma corrida de generacion.

    En el layout guardado por `PopulationSaver` son las subcarpetas de la raiz
    del experimento que contienen `generation_results/`. Las subcarpetas
    `evaluation_results/` se ignoran.

    @param experiment_root: raiz del experimento.
    @return: lista de rutas absolutas, ordenada.
    """

    runs = []

    for nombre in sorted(os.listdir(experiment_root)):
        ruta = os.path.join(experiment_root, nombre)
        if not os.path.isdir(ruta):
            continue
        if os.path.isdir(os.path.join(ruta, "generation_results")):
            runs.append(ruta)

    return runs


class AnalysisLayout:
    """
    Esquema de carpetas numerado de una salida de analisis.

    Las carpetas se crean al vuelo (`ensure`), de modo que un analisis que no
    produce diagramas de diferencias criticas no deja la carpeta vacia.

    La numeracion fija el orden de lectura y hace que las rutas sean estables
    entre ejecuciones.
    """

    #: Carpetas de primer nivel, en orden.
    SECTIONS = (
        "00_config",
        "01_individuals",
        "02_generations",
        "03_sdd",
        "04_complexity",
        "05_operators",
        "06_statistics",
        "07_figures",
    )

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(os.path.expanduser(root))

    def __repr__(self) -> str:
        return f"AnalysisLayout({self.root!r})"

    @property
    def config(self) -> str:
        return os.path.join(self.root, "00_config")

    @property
    def individuals(self) -> str:
        return os.path.join(self.root, "01_individuals")

    @property
    def generations(self) -> str:
        return os.path.join(self.root, "02_generations")

    @property
    def sdd(self) -> str:
        return os.path.join(self.root, "03_sdd")

    @property
    def complexity(self) -> str:
        return os.path.join(self.root, "04_complexity")

    @property
    def operators(self) -> str:
        return os.path.join(self.root, "05_operators")

    @property
    def statistics(self) -> str:
        return os.path.join(self.root, "06_statistics")

    @property
    def figures(self) -> str:
        return os.path.join(self.root, "07_figures")

    def ensure(self, *sections: str) -> None:
        """Crea las secciones indicadas (todas si no se pasa ninguna)."""
        nombres = sections or self.SECTIONS
        for nombre in nombres:
            os.makedirs(os.path.join(self.root, nombre), exist_ok=True)

    def sub(self, seccion: str, *partes: str) -> str:
        """Ruta dentro de una seccion; crea la seccion si hace falta."""
        base = os.path.join(self.root, seccion)
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, *partes)
