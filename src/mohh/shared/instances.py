"""Seleccion de instancias compartida por la busqueda y la comparacion."""

import random


def select_instances(paths, amount, seed):
    # `os.listdir` no garantiza orden entre sistemas de ficheros. Ordenar antes
    # de muestrear hace que el mismo `--seed` seleccione las mismas instancias en
    # el cluster y en local (la comparacion pareada depende de ello).
    ordered = sorted(str(path) for path in paths)
    if amount == "all":
        return ordered
    amount = int(amount)
    rng = random.Random(seed)
    return rng.sample(ordered, min(amount, len(ordered)))
