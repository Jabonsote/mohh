"""
Auxiliar de paralelización simple con un pool de procesos.

Ejecuta tareas independientes (corridas MOEA/D o de cualquier otro MOEA)
sobre todos los núcleos disponibles de forma dinámica. Cada tarea conserva
su lógica y el estado se actualiza igual que en la ejecución secuencial.

Robustez:

* ``on_error="raise"`` (por defecto) conserva el comportamiento histórico:
  una excepción de un worker aborta la tanda. ``on_error="record"`` la
  devuelve como ``{"error": ..., "task": ..., "index": ...}`` para que el
  llamador decida (la evaluación la cuenta como corrida fallida).
* ``timeout`` (segundos, opcional) acota cada tarea. Una tarea que excede el
  límite se marca como fallida y se matan los procesos del pool: sin esto el
  intérprete esperaría para siempre a un MOEA colgado.
"""

import os
import time

from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)


# Centinela: distingue "todavia no hay resultado" de "la tarea devolvio None".
# Con None, una tarea que legitimamente devuelve None se reencolaria sin fin.
MISSING = object()


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


def _as_failure(exc: BaseException, task, index: int, kind: str = "error") -> dict:
    return {
        "error": f"{kind}: {type(exc).__name__}: {exc}" if exc is not None else kind,
        "task": task,
        "index": index,
    }


def run_parallel(fn, tasks: list, on_complete=None,
                 timeout: float | None = None, on_error: str = "raise") -> list:
    """
    Ejecuta ``fn(task)`` sobre cada ``task`` con un proceso por núcleo.

    Devuelve una lista en el mismo orden que ``tasks``. Con
    ``on_error="record"`` una tarea fallida no aborta la tanda: se sustituye
    por ``{"error", "task", "index"}``. Con ``timeout`` las tareas que
    exceden el límite se marcan igual y se matan los procesos pendientes.
    """
    if len(tasks) == 0:
        return []

    if on_error not in ("raise", "record"):
        raise ValueError(f"on_error invalido: {on_error!r}")

    n_workers = min(available_cores(), len(tasks))
    completed = set()

    def notify(index: int):
        if on_complete is not None and index not in completed:
            completed.add(index)
            on_complete(index)

    # Sin paralelismo **y sin timeout**: mismo contrato de errores. Con
    # timeout hace falta un proceso aparte para poder matarlo: una tarea
    # colgada no se puede interrumpir desde dentro del mismo proceso.
    if n_workers <= 1 and timeout is None:
        results = []
        for i, task in enumerate(tasks):
            try:
                results.append(fn(task))
            except Exception as exc:  # noqa: BLE001 - se decide con on_error
                if on_error == "raise":
                    raise
                results.append(_as_failure(exc, task, i))
            notify(i)
        return results

    results = [MISSING] * len(tasks)
    remaining = list(range(len(tasks)))

    # Rondas: si una tarea excede el timeout hay que matar el pool entero (no
    # se puede interrumpir un worker desde fuera). Las tareas que quedaban
    # pendientes NO son culpables: se reencolan en un pool nuevo. Asi un MOEA
    # colgado cuesta una repeticion, no las otras 30.
    while remaining:
        workers = min(n_workers, len(remaining))
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = {pool.submit(fn, tasks[i]): i for i in remaining}
        pending = set(futures)
        timed_out = False
        failed = False

        # El executor arranca las tareas en orden de envio y como maximo
        # `workers` a la vez. Se simula ese orden para medir el timeout desde
        # que la tarea EMPIEZA, no desde que se encola: si no, con 31
        # repeticiones y 12 workers, las 19 en cola "cumplirian" el timeout sin
        # haber corrido y un solo MOEA colgado las mataria a todas.
        not_started = list(remaining)
        started_at = {}
        _now = time.monotonic()
        for i in not_started[:workers]:
            started_at[i] = _now
        del not_started[:workers]

        def _start_next(count):
            moment = time.monotonic()
            for _ in range(count):
                if not not_started:
                    break
                started_at[not_started.pop(0)] = moment

        try:
            while pending:
                done, pending = wait(pending, timeout=1.0,
                                     return_when=FIRST_COMPLETED)
                for future in done:
                    i = futures[future]
                    started_at.pop(i, None)
                    try:
                        results[i] = future.result()
                    except Exception as exc:  # noqa: BLE001 - se decide con on_error
                        if on_error == "raise":
                            failed = True
                            raise
                        results[i] = _as_failure(exc, tasks[i], i)
                    notify(i)
                _start_next(len(done))

                if timeout is None:
                    continue

                now = time.monotonic()
                for future in list(pending):
                    i = futures[future]
                    if i not in started_at or now - started_at[i] <= timeout:
                        continue
                    future.cancel()
                    results[i] = {
                        "error": f"timeout: la tarea excedio {timeout}s",
                        "task": tasks[i],
                        "index": i,
                    }
                    pending.discard(future)
                    started_at.pop(i, None)
                    notify(i)
                    timed_out = True

                if timed_out:
                    break
        finally:
            if timed_out or failed:
                # `_processes` se vacia al hacer shutdown: capturarlo antes.
                processes = list((getattr(pool, "_processes", None) or {}).values())
                for process in processes:
                    try:
                        process.kill()
                    except Exception:  # noqa: BLE001 - mejor esfuerzo
                        pass
                pool.shutdown(wait=False, cancel_futures=True)
            else:
                pool.shutdown(wait=True)

        remaining = [i for i in remaining if results[i] is MISSING]

    for i, value in enumerate(results):
        if value is MISSING:
            results[i] = {
                "error": "sin resultado: la tarea no devolvio nada",
                "task": tasks[i],
                "index": i,
            }
            notify(i)

    return results
