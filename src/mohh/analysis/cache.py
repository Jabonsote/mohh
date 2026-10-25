"""
Cache en disco para que el analisis nunca repita trabajo ya hecho.

La evaluacion de 31 corridas por combinacion es cara. Este modulo guarda el
resultado de cada tarea (un frente por presupuesto) en un `.npz` con un
nombre determinista derivado de la tarea, de forma que:

* relanzar el script reutiliza lo ya calculado;
* borrar un solo `.npz` fuerza a recalcular solo esa tarea;
* el contenido lleva dentro la huella del problema y de los parametros, de modo
  que cambiar el presupuesto o la semilla no se confunda con un resultado
  viejo.

No se genera ningun dato aqui: solo se guarda o se recupera lo que las tareas
ya produjeron.
"""

import hashlib
import json
import os

import numpy as np

__all__ = [
    "task_signature",
    "cache_path",
    "load_cached",
    "save_cached",
    "cached_task",
]


def _digest(payload) -> str:
    texto = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(texto.encode()).hexdigest()[:16]


def task_signature(task: dict) -> str:
    """
    Huella determinista de una tarea de evaluacion.

    Incluye solo lo que cambia el resultado: split, instancia, solver,
    operador, mutacion, presupuesto, generaciones, tamano de poblacion y
    semilla. Los metadatos de solo escritura no entran.

    @param task: dict con la descripcion de la tarea.
    @return: hash hexadecimal de 16 caracteres.
    """
    clave = {
        "split": task.get("split"),
        "instance": task.get("instance"),
        "problem": task.get("problem"),
        "solver": task.get("solver"),
        "cross_type": task.get("cross_type"),
        "cross_operator": task.get("cross_operator"),
        "mutation_operator": task.get("mutation_operator"),
        "budgets": task.get("budgets"),
        "generations": task.get("generations"),
        "population_size": task.get("population_size"),
        "seed": task.get("seed"),
    }
    return _digest(clave)


def cache_path(root: str, task: dict) -> str:
    """
    Ruta del `.npz` de una tarea dentro de `root`.

    La estructura replica la jerarquia logica (split/instancia/solver/
    combinacion) para que sea facil de inspeccionar, y el nombre del archivo
    lleva la huella para que dos tareas distintas nunca se pisen.

    @param root: carpeta raiz de la cache.
    @param task: dict de la tarea.
    @return: ruta absoluta del `.npz`.
    """
    combinacion = "_".join([
        str(task.get("solver", "solver")),
        str(task.get("cross_operator", "cross")),
        str(task.get("cross_type", "type")),
        str(task.get("mutation_operator", "mutation")),
    ])
    partes = [
        root,
        str(task.get("split", "unknown")),
        str(task.get("instance", "unknown")),
        combinacion,
        f"seed{task.get('seed')}",
        f"{task_signature(task)}.npz",
    ]
    return os.path.abspath(os.path.join(*partes))


def save_cached(path: str, fronts: dict, meta: dict = None) -> str:
    """
    Guarda los frentes de una tarea en disco.

    @param path: ruta del `.npz` (se crea la carpeta).
    @param fronts: `{etiqueta_de_presupuesto: (n, m) ndarray}`.
    @param meta: dict de metadatos (tiempos, semilla, firma...).
    @return: la ruta escrita.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    arrays = {}
    for etiqueta, front in fronts.items():
        matriz = np.atleast_2d(np.asarray(front, dtype=float))
        arrays[f"front__{etiqueta}"] = matriz
        arrays[f"size__{etiqueta}"] = np.array(matriz.shape[0])

    if meta:
        arrays["meta__json"] = np.array(json.dumps(meta, default=str))

    np.savez_compressed(path, **arrays)
    return path


def load_cached(path: str):
    """
    Recupera una tarea de la cache.

    @param path: ruta del `.npz`.
    @return: (frentes, meta) o (None, None) si no existe o esta corrupto.
    """
    if not os.path.isfile(path):
        return None, None

    try:
        with np.load(path, allow_pickle=False) as datos:
            fronts = {}
            for clave in datos.files:
                if not clave.startswith("front__"):
                    continue
                etiqueta = clave[len("front__"):]
                fronts[etiqueta] = datos[clave]

            meta = {}
            if "meta__json" in datos.files:
                meta = json.loads(str(datos["meta__json"]))

    except Exception:
        # Un .npz truncado (por ejemplo, una interrupcion) se ignora y la
        # tarea se recalcula en vez de devolver datos incompletos.
        return None, None

    return fronts, meta


def cached_task(root: str, task: dict):
    """
    Devuelve el resultado cacheado de una tarea, o None si hay que calcularla.

    @param root: carpeta raiz de la cache.
    @param task: dict de la tarea.
    @return: (frentes, meta, ruta) o (None, None, ruta) si no hay cache.
    """
    ruta = cache_path(root, task)
    fronts, meta = load_cached(ruta)
    if fronts is None:
        return None, None, ruta
    return fronts, meta, ruta
