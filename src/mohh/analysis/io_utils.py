"""
Lectura de los artefactos que deja una corrida de `mohh-generate`/`mohh-evaluate`.

Todo lo que el analisis necesita del disco entra por aqui: los individuos con
sus frentes por instancia, las generaciones, los operadores publicados en
`<modelo>_operators.txt`, las poblaciones iniciales reales y los CSV de
`mohh-evaluate`. Ninguna funcion de este modulo inventa datos: si un archivo no
existe, lo dice con el nombre del archivo que falta.
"""

import hashlib
import json
import os
import re
import numpy as np
import pandas as pd

from mohh.analysis.layout import INDIVIDUAL_PREFIX, find_run_dirs

__all__ = [
    "Experiment",
    "Individual",
    "load_experiment",
    "load_operators_file",
    "read_parameters_txt",
    "read_pof",
    "write_pof",
    "write_json",
    "read_json",
    "load_initial_permutations",
    "load_initial_seeds",
    "load_evaluation_csvs",
    "file_digest",
]

_INDIVIDUAL_RE = re.compile(rf"^{INDIVIDUAL_PREFIX}(\d+)$")
_GENERATION_RE = re.compile(r"^generation_(\d+)$")
_INSTANCE_JSON_RE = re.compile(r"^instance_(.+)\.json$")
_OPERATORS_RE = re.compile(r"^(?P<label>[^:]+):\s*(?P<operator>.+)$")


# ----------------------------------------------------------------------
# JSON y .pof
# ----------------------------------------------------------------------

def read_json(path: str):
    with open(path) as handle:
        return json.load(handle)


def write_json(path: str, data, indent: int = 4) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=indent, default=_json_default)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"no se puede serializar {type(value).__name__}")


def read_pof(path: str) -> np.ndarray:
    """
    Lee un `.pof` de RSEHH o de `mohh-evaluate`.

    Es el mismo formato que usa `mohh.core.indicators.reference_sets.read_pof`
    (cabecera `# <n> <m>` y una fila por punto), aqui sin la validacion de la
    cabecera porque estos archivos los escribio el propio pipeline.
    """
    front = np.genfromtxt(path, comments="#")
    front = np.atleast_2d(np.asarray(front, dtype=float))
    if front.size == 0:
        raise ValueError(f"{path}: frente vacio")
    return front


def write_pof(path: str, front: np.ndarray) -> None:
    """Escribe un frente en el formato `.pof` usado por el repositorio."""
    front = np.atleast_2d(np.asarray(front, dtype=float))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n, m = front.shape
    with open(path, "w") as handle:
        handle.write(f"# {n} {m}\n")
        for fila in front:
            handle.write(" ".join(f"{valor:1.6e}" for valor in fila) + "\n")


def file_digest(path: str) -> str:
    """SHA-256 (8 hex) de un archivo, para detecting versiones distintas."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for bloque in iter(lambda: handle.read(1 << 16), b""):
            digest.update(bloque)
    return digest.hexdigest()[:16]


# ----------------------------------------------------------------------
# Parametros y operadores publicados
# ----------------------------------------------------------------------

def read_parameters_txt(path: str) -> dict:
    """
    Lee el `parameters.txt` que escribe `PopulationSaver` (clave: valor).

    Los valores son cadenas; el analisis solo los usa para documentar la
    corrida, asi que no se castea.
    """
    if not os.path.isfile(path):
        return {}

    params = {}
    with open(path) as handle:
        for linea in handle:
            if ":" not in linea:
                continue
            clave, _, valor = linea.partition(":")
            params[clave.strip()] = valor.strip()
    return params


def load_operators_file(experiment_root: str) -> dict:
    """
    Operadores publicados por la generacion: `{modelo: {etiqueta: fenotipo}}`.

    `<modelo>_operators.txt` tiene una linea por operador con el formato
    `Etiqueta: expresion`.
    """
    operadores = {}

    for nombre in sorted(os.listdir(experiment_root)):
        if not nombre.endswith("_operators.txt"):
            continue
        modelo = nombre[: -len("_operators.txt")]
        operadores[modelo] = load_operators_file_single(
            os.path.join(experiment_root, nombre))

    return operadores


def load_operators_file_single(path: str) -> dict:
    operadores = {}
    with open(path) as handle:
        for linea in handle:
            linea = linea.strip()
            if not linea:
                continue
            encontrado = _OPERATORS_RE.match(linea)
            if encontrado:
                operadores[encontrado.group("label").strip()] = \
                    encontrado.group("operator").strip()
    return operadores


# ----------------------------------------------------------------------
# Poblaciones iniciales reales
# ----------------------------------------------------------------------

def _parse_bracketed_rows(path: str) -> list:
    filas = []
    with open(path) as handle:
        for linea in handle:
            linea = linea.strip()
            if not (linea.startswith("[") and linea.endswith("]")):
                continue
            cuerpo = linea[1:-1].strip()
            if not cuerpo:
                continue
            filas.append(np.array(
                [float(valor) for valor in cuerpo.split(",")], dtype=float))
    return filas


def load_initial_permutations(path: str) -> list:
    """
    Poblacion inicial real de permutaciones de una instancia.

    `generation_results/initial_solutions/permutation/<instancia>.txt` guarda
    una permutacion por linea con el formato `[0, 3, 1, ...]`. Son las
    permutaciones que el MOEA recibio de verdad, que es lo que usa el analisis
    de complejidad (nunca permutaciones inventadas).
    """
    filas = _parse_bracketed_rows(path)
    if not filas:
        raise ValueError(f"{path}: no hay permutaciones iniciales")
    return [np.rint(fila).astype(int) for fila in filas]


def load_initial_seeds(path: str) -> list:
    """Semillas guardadas en `seeds.txt` (una por linea), si existe."""
    if not os.path.isfile(path):
        return []
    with open(path) as handle:
        return [int(linea) for linea in handle if linea.strip()]


# ----------------------------------------------------------------------
# Individuos y generaciones
# ----------------------------------------------------------------------

class Individual:
    """
    Un individuo de la poblacion de la GE con sus frentes por instancia.

    @param name: `Individual_000123`.
    @param general: contenido de `general_info.json`.
    @param fronts: `{instancia: (n, m) ndarray}` con el frente no dominado.
    @param solution_sets: `{instancia: (n, m) ndarray}` con las permutaciones.
    @param sizes: `{instancia: {number_unique_solutions, ...}}`.
    @param directory: carpeta del individuo.
    """

    __slots__ = ("name", "general", "fronts", "solution_sets", "sizes", "directory")

    def __init__(self, name, general, fronts, solution_sets, sizes, directory):
        self.name = name
        self.general = general
        self.fronts = fronts
        self.solution_sets = solution_sets
        self.sizes = sizes
        self.directory = directory

    @property
    def phenotype(self):
        return self.general.get("phenotype")

    @property
    def born_in_generation(self) -> int:
        return int(self.general.get("born_in_generation", -1))

    @property
    def instances(self) -> list:
        return sorted(self.fronts)

    @property
    def n_objectives(self) -> int:
        for front in self.fronts.values():
            return int(front.shape[1])
        return 0

    @property
    def n_variables(self) -> int:
        for soluciones in self.solution_sets.values():
            if soluciones.size:
                return int(soluciones.shape[1])
        return 0

    def __repr__(self) -> str:
        return f"Individual({self.name!r}, born={self.born_in_generation})"


def _load_individuals(generation_results: str, solution_instances=None,
                      fronts_instances=None, dtype=np.float64) -> dict:
    directorio = os.path.join(generation_results, "individuals")
    individuos = {}

    if not os.path.isdir(directorio):
        return individuos

    for nombre in sorted(os.listdir(directorio)):
        if not _INDIVIDUAL_RE.match(nombre):
            continue
        ruta = os.path.join(directorio, nombre)
        general_path = os.path.join(ruta, "general_info.json")
        if not os.path.isfile(general_path):
            continue

        general = read_json(general_path)
        frentes, conjuntos, tamanos = {}, {}, {}

        for archivo in sorted(os.listdir(ruta)):
            encontrado = _INSTANCE_JSON_RE.match(archivo)
            if not encontrado:
                continue
            instancia = encontrado.group(1)
            datos = read_json(os.path.join(ruta, archivo))

            frente = datos.get("front")
            if frente is not None and (
                    fronts_instances is None
                    or instancia in fronts_instances):
                frentes[instancia] = np.atleast_2d(
                    np.asarray(frente, dtype=dtype))
            soluciones = datos.get("solution_set")
            if soluciones is not None and (
                    solution_instances is None
                    or instancia in solution_instances):
                conjuntos[instancia] = np.atleast_2d(
                    np.asarray(soluciones, dtype=float))
            tamanos[instancia] = {
                clave: datos.get(clave)
                for clave in ("number_unique_solutions",
                              "weak_non_dominated_solutions",
                              "strong_non_dominated_solutions")
            }

        if not frentes and not conjuntos:
            continue

        individuos[general.get("name", nombre)] = Individual(
            nombre, general, frentes, conjuntos, tamanos, ruta)

    return individuos


def _load_generation_index(generation_results: str) -> dict:
    """
    `{generacion: {"population": [...], "offspring": [...]}}` leido de las
    carpetas `generation_NNNN` con `population.json` / `offspring.json`.
    """
    directorio = os.path.join(generation_results, "generations")
    indice = {}

    if not os.path.isdir(directorio):
        return indice

    for nombre in sorted(os.listdir(directorio)):
        encontrado = _GENERATION_RE.match(nombre)
        if not encontrado:
            continue
        generacion = int(encontrado.group(1))
        ruta = os.path.join(directorio, nombre)

        entrada = {"population": [], "offspring": []}
        for etiqueta in ("population", "offspring"):
            archivo = os.path.join(ruta, f"{etiqueta}.json")
            if os.path.isfile(archivo):
                entrada[etiqueta] = read_json(archivo)

        indice[generacion] = entrada

    return indice


def _load_run_metrics(run_dir: str) -> dict:
    """
    Metricas de la corrida: `summary.json`, `progress.json` y la seccion
    `reference_fitness` de cada `generation_NNNN.json`.

    Devuelve
        {"summary": {...}, "progress": {...}, "reference_fitness": [ ... ]}
    donde `reference_fitness` es la lista aplanada de las entradas por
    generacion (cada una con `generation`, `individual`, `metric`, `fitness`,
    `mean_indicator` y `mean_diff`).
    """
    metricas_dir = os.path.join(run_dir, "run_metrics")
    salida = {"summary": {}, "progress": {}, "reference_fitness": []}

    if not os.path.isdir(metricas_dir):
        return salida

    for clave, nombre in (("summary", "summary.json"),
                          ("progress", "progress.json")):
        ruta = os.path.join(metricas_dir, nombre)
        if os.path.isfile(ruta):
            salida[clave] = read_json(ruta)

    for nombre in sorted(os.listdir(metricas_dir)):
        if not _GENERATION_RE.match(nombre.removesuffix(".json")):
            continue
        ruta = os.path.join(metricas_dir, nombre)
        if not os.path.isfile(ruta):
            continue
        datos = read_json(ruta)
        for entrada in datos.get("sections", {}).get("reference_fitness", []):
            salida["reference_fitness"].append(entrada)

    return salida


class Experiment:
    """
    Un experimento de generacion ya cargado en memoria.

    @param root: raiz del experimento (`.../Experiment_001`).
    @param runs: lista de carpetas `<modelo>_<timestamp>`.
    @param operators: `{modelo: {etiqueta: fenotipo}}`.
    @param individuals: `{individuo: Individual}` (union de todas las corridas).
    @param generations: `{generacion: {...}}` (indice de la primera corrida).
    @param run_metrics: metricas por corrida, indexadas por nombre de carpeta.
    @param parameters: `parameters.txt` de cada corrida.
    """

    def __init__(self, root, runs, operators, individuals, generations,
                 run_metrics, parameters) -> None:
        self.root = root
        self.runs = runs
        self.operators = operators
        self.individuals = individuals
        self.generations = generations
        self.run_metrics = run_metrics
        self.parameters = parameters

    @property
    def models(self) -> list:
        return sorted(self.operators)

    @property
    def labels(self) -> list:
        etiquetas = set()
        for porModelo in self.operators.values():
            etiquetas.update(porModelo)
        return sorted(etiquetas)

    @property
    def individuals_by_generation(self) -> dict:
        """`{generacion: [individuos]}` en orden de identificador."""
        agrupado = {}
        for individuo in self.individuals.values():
            agrupado.setdefault(individuo.born_in_generation, []).append(individuo)
        for generacion in agrupado:
            agrupado[generacion].sort(key=lambda ind: ind.name)
        return dict(sorted(agrupado.items()))

    def initial_solutions_path(self, instance_name: str, run_dir: str = None):
        """Ruta de las permutaciones iniciales reales de una instancia."""
        run_dir = run_dir or (self.runs[0] if self.runs else None)
        if run_dir is None:
            return None
        return os.path.join(
            run_dir, "generation_results", "initial_solutions",
            "permutation", f"{instance_name}.txt")


def _iter_individuals(generation_results: str, fronts_instances=None,
                      solution_instances=None, metadata_only=False):
    """Itera individuos leyendo sólo los frentes/permutaciones pedidos."""
    directorio = os.path.join(generation_results, "individuals")
    if not os.path.isdir(directorio):
        return

    for nombre in sorted(os.listdir(directorio)):
        if not _INDIVIDUAL_RE.match(nombre):
            continue
        ruta = os.path.join(directorio, nombre)
        general_path = os.path.join(ruta, "general_info.json")
        if not os.path.isfile(general_path):
            continue

        general = read_json(general_path)
        frentes, conjuntos, tamanos = {}, {}, {}

        if not metadata_only:
            for archivo in sorted(os.listdir(ruta)):
                encontrado = _INSTANCE_JSON_RE.match(archivo)
                if not encontrado:
                    continue
                instancia = encontrado.group(1)
                need_front = (fronts_instances is None
                              or instancia in fronts_instances)
                need_sol = (solution_instances is None
                            or instancia in solution_instances)
                if not need_front and not need_sol:
                    continue

                datos = read_json(os.path.join(ruta, archivo))
                if need_front:
                    frente = datos.get("front")
                    if frente is not None:
                        frentes[instancia] = np.atleast_2d(
                            np.asarray(frente, dtype=float))
                if need_sol:
                    soluciones = datos.get("solution_set")
                    if soluciones is not None:
                        conjuntos[instancia] = np.atleast_2d(
                            np.asarray(soluciones, dtype=float))
                if need_front:
                    tamanos[instancia] = {
                        clave: datos.get(clave)
                        for clave in ("number_unique_solutions",
                                      "weak_non_dominated_solutions",
                                      "strong_non_dominated_solutions")
                    }

        yield Individual(general.get("name", nombre), general, frentes,
                         conjuntos, tamanos, ruta)


def iter_experiment_individuals(path: str, fronts_instances=None,
                                solution_instances=None, metadata_only=False):
    """
    Genera los individuos de un experimento sin mantenerlos todos en memoria.

    Se usa en análisis grandes (`analyze_search`) para no cargar los 1100
    individuos completos. Los flags limitan qué frentes y qué permutaciones se
    conservan por individuo.
    """
    from mohh.analysis.layout import detect_experiment_root

    root = detect_experiment_root(path)
    vistos = set()
    for run_dir in find_run_dirs(root):
        generation_results = os.path.join(run_dir, "generation_results")
        for individuo in _iter_individuals(
                generation_results, fronts_instances=fronts_instances,
                solution_instances=solution_instances,
                metadata_only=metadata_only):
            if individuo.name in vistos:
                continue
            vistos.add(individuo.name)
            yield individuo


def load_experiment(path: str, load_generations=True,
                    load_run_metrics=True, solution_instances=None,
                    fronts_instances=None, dtype=np.float64) -> Experiment:
    """
    Carga un experimento desde disco.

    Los flags permiten cargas ligeras para analisis que no necesitan las
    generaciones ni las metricas de la corrida (por ejemplo `analyze_search`).
    `solution_instances` limita las permutaciones que se conservan en memoria:
    util para experimentos grandes donde solo se reevaluan unas pocas
    instancias de train en test.

    @param path: ruta del experimento (ver `detect_experiment_root`).
    @param load_generations: cargar `generation_results/generations`.
    @param load_run_metrics: cargar `run_metrics`.
    @param solution_instances: nombres de instancia cuyas `solution_set` se
        conservan; None conserva todas.
    @return: `Experiment`.
    """
    from mohh.analysis.layout import detect_experiment_root

    root = detect_experiment_root(path)
    runs = find_run_dirs(root)

    individuals, generations, run_metrics, parameters = {}, {}, {}, {}

    for run_dir in runs:
        nombre = os.path.basename(run_dir)
        generation_results = os.path.join(run_dir, "generation_results")

        individuals.update(_load_individuals(
            generation_results, solution_instances=solution_instances,
            fronts_instances=fronts_instances, dtype=dtype))
        if load_generations:
            generations.update(_load_generation_index(generation_results))
        if load_run_metrics:
            run_metrics[nombre] = _load_run_metrics(run_dir)
        parameters[nombre] = read_parameters_txt(
            os.path.join(generation_results, "parameters.txt"))

    operators = load_operators_file(root)

    return Experiment(root, runs, operators, individuals, generations,
                      run_metrics, parameters)


# ----------------------------------------------------------------------
# CSV de mohh-evaluate
# ----------------------------------------------------------------------

def load_evaluation_csvs(folder: str, instance: str = None) -> pd.DataFrame:
    """
    Lee los CSV de `mohh-evaluate` (`<solver>/<budget>/<instancia>.csv`).

    Se usa para comparar contra las corridas ya terminadas (por ejemplo
    `results/baseline/Experiment_001/evaluation_results`) sin volver a ejecutar
    nada.

    @param folder: carpeta `evaluation_results` (o una subcarpeta de ella).
    @param instance: si se indica, solo se lee el CSV de esa instancia.
    @return: DataFrame largo con columnas `solver`, `budget`, `instance`,
        `config`, `experiment` y `value`.
    """
    filas = []

    for solver in sorted(_subdirs(folder)):
        for budget in sorted(_subdirs(os.path.join(folder, solver))):
            ruta_budget = os.path.join(folder, solver, budget)
            archivos = sorted(f for f in os.listdir(ruta_budget)
                              if f.endswith(".csv"))

            for archivo in archivos:
                nombre = archivo.removesuffix(".csv")
                if instance is not None and nombre != instance:
                    continue

                df = pd.read_csv(os.path.join(ruta_budget, archivo))
                if "Experiment" not in df.columns:
                    continue

                for columna in df.columns:
                    if columna == "Experiment":
                        continue
                    filas.append(pd.DataFrame({
                        "solver": solver,
                        "budget": budget,
                        "instance": nombre,
                        "config": columna,
                        "experiment": df["Experiment"].to_numpy(),
                        "value": df[columna].to_numpy(dtype=float),
                    }))

    if not filas:
        return pd.DataFrame(columns=["solver", "budget", "instance",
                                    "config", "experiment", "value"])

    return pd.concat(filas, ignore_index=True)


def _subdirs(path: str) -> list:
    if not os.path.isdir(path):
        return []
    return [n for n in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, n))]
