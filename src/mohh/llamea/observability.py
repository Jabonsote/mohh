"""Salida, logging y watchdog de una corrida LLaMEA (extraidos de `run.py`).

`run.py` re-exporta estos nombres para conservar la API publica del modulo.
"""

import logging
import os
import sys
import time


class _Tee:
    """Duplica la salida estandar en consola y en `run.log`."""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, data):
        self.stream.write(data)
        self.handle.write(data)
        self.handle.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return self.stream.isatty()


def _setup_console_log(exp_dir):
    """Escribe stdout y stderr en `<exp_dir>/run.log` (ademas de la consola)."""
    os.makedirs(exp_dir, exist_ok=True)
    handle = open(os.path.join(exp_dir, "run.log"), "a",
                  encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)
    return handle


def _setup_logging(exp_dir):
    """Log de la corrida: `llamea.log` (todo) y `error.log` (solo avisos/errores).

    En el cluster los `.out`/`.err` de SLURM se buffean y un job puede morir sin
    dejar rastro. Estos dos archivos quedan dentro del experimento y `error.log`
    sobrevive incluso a un fallo temprano (servidor caido, modelo ausente), que
    es justo lo que hay que diagnosticar.
    """
    os.makedirs(exp_dir, exist_ok=True)
    logger = logging.getLogger("mohh.llamea")
    logger.setLevel(logging.DEBUG)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    full = logging.FileHandler(os.path.join(exp_dir, "llamea.log"), encoding="utf-8")
    full.setLevel(logging.INFO)
    full.setFormatter(formatter)
    errors = logging.FileHandler(os.path.join(exp_dir, "error.log"), encoding="utf-8")
    errors.setLevel(logging.WARNING)
    errors.setFormatter(formatter)
    logger.addHandler(full)
    logger.addHandler(errors)
    return logger


def _update_latest(results_dir, exp_dir):
    """Actualiza `results/<categoria>/latest` apuntando al run actual.

    El enlace es relativo para que la carpeta de resultados se pueda mover o
    copiar sin romperlo. Si `latest` ya existe y no es un enlace (p. ej. el
    usuario creo una carpeta con ese nombre), no se toca.
    """
    results_dir = os.path.abspath(results_dir)
    exp_dir = os.path.abspath(exp_dir)
    os.makedirs(results_dir, exist_ok=True)
    latest = os.path.join(results_dir, "latest")
    target = os.path.relpath(exp_dir, results_dir)
    try:
        if os.path.lexists(latest):
            if not os.path.islink(latest):
                return None
            if os.readlink(latest) == target:
                return latest
            os.unlink(latest)
        os.symlink(target, latest)
    except OSError:
        return None
    return latest


def _rotate_request_log(path):
    """Preserva un `llm_requests.jsonl` previo antes de empezar una corrida.

    Con el nombre de experimento unico esto casi nunca ocurre, pero si alguien
    reutiliza `EXPERIMENT_NAME` no queremos mezclar peticiones de dos corridas
    (era la causa de "22 peticiones" cuando en realidad solo habia una nueva).
    """
    if not os.path.exists(path):
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(path)))
    rotated = os.path.join(
        os.path.dirname(path), f"llm_requests.{stamp}.jsonl"
    )
    # Si ya existe (dos corridas en el mismo segundo), agregar un sufijo.
    suffix = 1
    while os.path.exists(rotated):
        rotated = os.path.join(
            os.path.dirname(path), f"llm_requests.{stamp}.{suffix}.jsonl"
        )
        suffix += 1
    os.replace(path, rotated)
    return rotated


def _start_stall_watchdog(exp_dir, logger, timeout):
    """Aborta la corrida si pasan `timeout` s sin una generacion nueva.

    Vigila la fecha de `checkpoint.json` (reescrito cada generacion). Si el
    proceso se cuelga (ya paso una vez: una generacion quedo 17 min en estado R;
    ver `areas-de-mejora.md` §14), el checkpoint queda viejo y este hilo lo
    detecta, lo deja en `error.log` y termina el proceso. El mejor operador ya
    esta guardado en `best_so_far.py`, asi que un cuelgue no pierde la corrida.
    """
    if not timeout or timeout <= 0:
        return None
    import threading

    checkpoint = os.path.join(exp_dir, "checkpoint.json")

    def watch():
        while True:
            time.sleep(min(max(timeout / 4.0, 5.0), 120.0))
            if not os.path.exists(checkpoint):
                continue
            stalled = time.time() - os.path.getmtime(checkpoint)
            if stalled > timeout:
                logger.error(
                    "Sin progreso desde hace %.0f s (limite %s s); se aborta. "
                    "El mejor operador esta en best_so_far.py.", stalled, timeout)
                logging.shutdown()
                os._exit(2)

    thread = threading.Thread(target=watch, name="stall-watchdog", daemon=True)
    thread.start()
    return thread
