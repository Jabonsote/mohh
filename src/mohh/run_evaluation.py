import os

def discover_models(experiment_path: str) -> list:
    """
    Generation models that actually produced operators in this experiment,
    found from the *_operators.txt files mohh-generate writes.
    """

    suffix = "_operators.txt"

    return sorted(f[:-len(suffix)] for f in os.listdir(experiment_path) if f.endswith(suffix))

def _run_dir_for_model(experiment_path: str, model: str) -> str:
    """Localiza la corrida `<model>_<timestamp>` de un experimento."""
    import glob

    candidatos = sorted(
        ruta for ruta in glob.glob(os.path.join(experiment_path, model + "_*"))
        if os.path.isdir(ruta))
    if not candidatos:
        return experiment_path
    return candidatos[-1]


def _elite_size(generation_results: str, population_size: int) -> int:
    """Lee ELITE_SIZE de `parameters.txt`; si no está, usa la población."""
    ruta = os.path.join(generation_results, "parameters.txt")
    if os.path.isfile(ruta):
        with open(ruta) as f:
            for linea in f:
                if linea.startswith("ELITE_SIZE:"):
                    try:
                        return int(linea.split(":", 1)[1].strip())
                    except ValueError:
                        break
    return population_size


def _load_generation_population(generation_results: str, generation: int):
    carpeta = os.path.join(generation_results, "generations",
                           f"generation_{generation:04d}")
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(
            f"no existe la generación {generation} en {carpeta}")
    import json
    for nombre in ("population.json", "offspring.json"):
        ruta = os.path.join(carpeta, nombre)
        if os.path.isfile(ruta):
            with open(ruta) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"la generación {generation} no tiene population.json/offspring.json")


def extract_generation_operators(experiment_path: str, model_names: list,
                                 generation: int):
    """
    Extrae Best/Middle/Worst de una generación concreta.

    Usa el mismo criterio que `run_generation` para la última generación:
    primeros `ELITE_SIZE` individuos del `population.json`, tomando los
    índices 0, (elite-1)//2 y elite-1.
    """
    import json

    operators = []
    for model in model_names:
        run_dir = _run_dir_for_model(experiment_path, model)
        generation_results = os.path.join(run_dir, "generation_results")
        population = _load_generation_population(generation_results, generation)
        if not population:
            raise ValueError(
                f"la generación {generation} de {model} está vacía")

        elite = min(_elite_size(generation_results, len(population)),
                    len(population))
        indices = sorted({0, max(0, (elite - 1) // 2), max(0, elite - 1)})
        individuals_dir = os.path.join(generation_results, "individuals")

        for index in indices:
            nombre = population[index]
            ruta = os.path.join(individuals_dir, nombre, "general_info.json")
            if not os.path.isfile(ruta):
                raise FileNotFoundError(
                    f"falta {ruta} para la generación {generation}")
            with open(ruta) as f:
                datos = json.load(f)
            operators.append(datos["phenotype"])

    return operators


def extract_operators(experiment_path: str, model_names: list,
                      generation: int = None):

    if generation is not None:
        return extract_generation_operators(experiment_path, model_names,
                                            generation)

    operators = []

    for model in model_names:

        file_path = os.path.join(experiment_path, model + "_operators.txt")

        with open(file_path, "r") as f:
            for line in f:
                label, operator = line.split(":", 1)
                operator = operator.strip()

                operators.append(operator)

    return operators
