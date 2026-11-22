"""Escritura atomica y lectura de los logs de LLaMEA (extraidos de `reporting.py`).

`reporting.py` re-exporta estos nombres para conservar la API publica del modulo.
"""

import json
import os

import numpy as np


def _atomic_write_text(path, text):
    """Escribe `path` de forma atomica (tmp + rename) para sobrevivir a un corte."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _atomic_write_json(path, payload):
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _metadata(candidate, key):
    """`candidate.get_metadata(key)` si existe (LLaMEA Solution), o None."""
    getter = getattr(candidate, "get_metadata", None)
    if not callable(getter):
        return None
    try:
        return getter(key)
    except Exception:  # noqa: BLE001 - el log no debe tumbar la corrida
        return None


def _read_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_evolution(log_dir):
    """Lee ``log.jsonl`` y devuelve las evaluaciones en orden de generacion.

    Cada fila agrega ``best_so_far`` y ``rank`` (1 = mejor HV de toda la
    corrida). Solo los individuos con fitness finito y > 0 cuentan como validos
    para el ranking, igual que en el analisis de la busqueda.
    """
    records = _read_jsonl(os.path.join(log_dir, "log.jsonl"))
    rows = []
    for record in records:
        fitness = record.get("fitness")
        try:
            fitness = float(fitness)
        except (TypeError, ValueError):
            fitness = float("nan")
        rows.append(
            {
                "generation": int(record.get("generation", 0) or 0),
                "id": record.get("id"),
                "name": record.get("name", ""),
                "description": record.get("description", ""),
                "fitness": fitness,
                "error": record.get("error"),
                "code": record.get("code", ""),
                "valid": bool(np.isfinite(fitness) and fitness > 0),
            }
        )

    rows.sort(key=lambda r: r["generation"])
    valid = sorted(
        (r["fitness"] for r in rows if r["valid"]),
        reverse=True,
    )
    best_so_far = float("-inf")
    for row in rows:
        if row["valid"]:
            best_so_far = max(best_so_far, row["fitness"])
            row["rank"] = 1 + sum(1 for value in valid if value > row["fitness"])
        else:
            row["rank"] = None
        row["best_so_far"] = best_so_far if best_so_far != float("-inf") else None
    return rows


def tokens_summary(log_dir):
    """Suma de tokens por generacion, si `generation_summary.jsonl` los trae."""
    records = _read_jsonl(os.path.join(log_dir, "generation_summary.jsonl"))
    if not records or not any("tokens_total" in record for record in records):
        return None
    last = next((r for r in reversed(records) if "tokens_total" in r), records[-1])
    summary = {
        "generations": len(records),
        "tokens_total": last.get("tokens_total", 0),
        "tokens_prompt": last.get("tokens_prompt", 0),
        "tokens_completion": last.get("tokens_completion", 0),
        "requests": last.get("requests", 0),
    }
    per_gen = [
        record.get("tokens_total_gen")
        for record in records
        if record.get("tokens_total_gen") is not None
    ]
    if per_gen:
        summary["tokens_total_gen_mean"] = float(np.mean(per_gen))
        summary["tokens_total_gen_max"] = int(np.max(per_gen))
    return summary


def _pair_turns(conversation):
    """Empareja (prompt, respuesta) en orden de aparicion.

    `conversationlog.jsonl` guarda, por turno, la peticion con `role="client"`
    (el prompt completo: rol + tarea + ejemplo + feedback) y la respuesta con
    el nombre del modelo como rol. El orden de los turnos es el de los intentos
    de LLaMEA, que es el orden de las generaciones.
    """
    turns = []
    pending = None
    for entry in conversation:
        content = entry.get("content", "")
        if entry.get("role") == "client":
            if pending is not None:
                turns.append(pending)
            pending = {"prompt": content, "prompt_time": entry.get("time"),
                       "response": "", "response_time": None}
        else:
            if pending is None:
                pending = {"prompt": "", "prompt_time": None,
                           "response": "", "response_time": None}
            pending["response"] = content
            pending["response_time"] = entry.get("time")
            turns.append(pending)
            pending = None
    if pending is not None:
        turns.append(pending)
    return turns


def _request_records(log_dir, request_log=None):
    """Lee `llm_requests.jsonl` y descarta peticiones fallidas.

    Las peticiones fallidas no generan turno en `conversationlog.jsonl`; si se
    incluyeran, desplazarian el emparejamiento. El archivo crudo completo queda
    igualmente en disco para auditoria.
    """
    candidates = []
    if request_log:
        candidates.append(request_log)
    candidates += [
        os.path.join(log_dir, "llm_requests.jsonl"),
        os.path.join(os.path.dirname(log_dir), "llm_requests.jsonl"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return [record for record in _read_jsonl(candidate)
                    if not record.get("error")]
    return []
