"""
Auxiliar de paralelización simple con un pool de procesos.

Ejecuta tareas independientes (corridas MOEA/D o de cualquier otro MOEA)
sobre todos los núcleos disponibles de forma dinámica. Ninguna tarea cambia
su lógica: el proceso principal reparte las tareas, cada worker hace
exactamente lo que haría el proceso principal, y el proceso principal
recoge los resultados y aplica las mismas actualizaciones de estado.
"""

import os

from concurrent.futures import ProcessPoolExecutor, as_completed


def available_cores() -> int:
    """
    Número de workers a usar.

    Orden de prioridad:
    1. MOHH_WORKERS (override explícito, ideal en SLURM).
    2. SLURM_CPUS_PER_TASK / SLURM_CPUS_ON_NODE (lo asignado por sbatch -c).
    3. os.cpu_count() (corrida local fuera de SLURM).

    Esto evita la oversuscripción típica en SLURM: sin esto, os.cpu_count()
    ve los 52 cores físicos del nodo aunque solo te hayan asignado 12 con
    -c 12, y el pool lanzaría 52 procesos sobre 12 cores.
    """
    env_workers = os.environ.get("MOHH_WORKERS", "").strip()
    if env_workers.isdigit() and int(env_workers) >= 1:
        return int(env_workers)

    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var, "").strip()
        # SLURM puede reportar "50(x2)" o listas; quedarnos con el entero líder.
        digits = ""
        for ch in val:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        if digits and int(digits) >= 1:
            return int(digits)

    return max(1, os.cpu_count() or 1)


def run_parallel(fn, tasks: list, on_complete=None) -> list:
    """
    Ejecuta `fn(task)` sobre cada `task` usando un pool de procesos con un
    worker por núcleo. Devuelve los resultados en el mismo orden que `tasks`.

    Si se pasa `on_complete(idx)` (una sentencia que avanza la barra de
    progreso), se invoca en el proceso principal por cada tarea que termina,
    manteniendo la barra viva tal como avanzaba la versión secuencial.

    Si hay una sola tarea (o un solo núcleo disponible) delega en la
    ejecución secuencial normal para no penalizar los casos pequeños.
    """
    if len(tasks) == 0:
        return []

    n_workers = min(available_cores(), len(tasks))

    if n_workers <= 1:
        results = []
        for i, task in enumerate(tasks):
            results.append(fn(task))
            if on_complete is not None:
                on_complete(i)
        return results

    results = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        futures = [pool.submit(fn, task) for task in tasks]
        index_by_future = {fut: i for i, fut in enumerate(futures)}

        for future in as_completed(futures):
            i = index_by_future[future]
            results[i] = future.result()
            if on_complete is not None:
                on_complete(i)

    return results