"""Logging y reportes de una busqueda LLaMEA, siempre dentro de `results/`.

LLaMEA crea por defecto un directorio ``exp-<fecha>-<nombre>`` en el directorio
de trabajo. Aqui se sustituye por un logger que escribe bajo
``<results-dir>/<experimento>/llamea_logs/`` y que, ademas de ``log.jsonl`` y
``conversationlog.jsonl`` (que ya escribe LLaMEA), agrega una linea por
generacion en ``generation_summary.jsonl``.

Al terminar, :func:`write_report` convierte ``log.jsonl`` en artefactos legibles:
``evolution.csv``, ``report.md`` y ``evolution.png``.
"""

import csv
import hashlib
import json
import os
from datetime import datetime

import numpy as np

from llamea.loggers import ExperimentLogger


def _atomic_write_text(path, text):
    """Escribe `path` de forma atomica (tmp + rename) para sobrevivir a un corte."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _atomic_write_json(path, payload):
    _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


class ResultsExperimentLogger(ExperimentLogger):
    """`ExperimentLogger` de LLaMEA que escribe dentro de `results/`.

    El directorio base se fija en la clase antes de construir `LLaMEA`
    (``ResultsExperimentLogger.results_dir = ...``).
    """

    results_dir = None
    verbose = True
    #: Objeto con `usage_snapshot()` (normalmente `VLLM_LLM`) para registrar
    #: tokens por generacion. Se fija desde `run.py` antes de construir LLaMEA.
    token_source = None
    #: Semillas y configuracion del proceso (se fijan desde `run.py`). Se
    #: escriben en cada registro de `individuals.jsonl` y
    #: `generation_summary.jsonl` para que la trazabilidad sea autosuficiente.
    seed = None
    solver = None
    mo_generations = None
    mo_population_size = None
    mutation = None

    def create_log_dir(self, name=""):
        base = type(self).results_dir or os.getcwd()
        os.makedirs(base, exist_ok=True)
        dirname = os.path.join(base, f"exp-{self.working_date}-{name}")
        os.makedirs(dirname, exist_ok=True)
        for sub in ("configspace", "code"):
            os.makedirs(os.path.join(dirname, sub), exist_ok=True)
        return dirname

    def log_population(self, population):
        super().log_population(population)
        if population:
            self._log_individuals(population)
            self._log_generation_summary(population)

    def counters(self):
        """Resumen acumulado de la corrida (distintos, validos, mejor HV)."""
        return {
            "distinct_operators": len(getattr(self, "_seen_hashes", None) or set()),
            "evaluated_valid": getattr(self, "_evaluated_so_far", 0),
            "best_so_far": getattr(self, "_best_so_far", None),
            "seed": type(self).seed,
            "solver": type(self).solver,
            "mo_generations": type(self).mo_generations,
            "mo_population_size": type(self).mo_population_size,
            "mutation": type(self).mutation,
        }

    def _write_checkpoint(self, best, record):
        """Estado persistente para sobrevivir a un corte del cluster.

        Escribe `checkpoint.json` (progreso) y `best_so_far.py`/`.json` (el mejor
        operador hasta ahora) de forma **atomica**: si el job muere a mitad de
        escritura, el archivo anterior queda intacto. Asi una corrida de 1000
        generaciones nunca pierde el mejor operador encontrado.
        """
        checkpoint = os.path.join(self.dirname, "checkpoint.json")
        payload = {
            "time": record["time"],
            "generation": record["generation"],
            "best_fitness": record["best_fitness"],
            "best_name": record.get("best_name"),
            "counters": self.counters(),
            "tokens_total": record.get("tokens_total"),
            "wall_seconds": record.get("wall_seconds"),
        }
        _atomic_write_json(checkpoint, payload)

        fitness = best.fitness if best is not None else None
        if best is not None and best.code and fitness is not None and np.isfinite(fitness):
            _atomic_write_text(os.path.join(self.dirname, "best_so_far.py"),
                               best.code.rstrip() + "\n")
            _atomic_write_json(os.path.join(self.dirname, "best_so_far.json"), {
                "generation": record["generation"],
                "fitness": float(fitness),
                "name": getattr(best, "name", None),
                "description": getattr(best, "description", None),
                "seed": type(self).seed,
            })

    def _log_individuals(self, population):
        """Una linea por individuo en `individuals.jsonl`.

        Registra todo lo que identifica y hace reproducible al candidato:
        generacion, indice, hash del codigo, descripcion, fitness, error,
        novedad, operador/padres, y **la semilla y configuracion del proceso**.
        Los tokens y la latencia de la peticion que lo genero quedan alineados
        por turno en `prompts/index.jsonl` (aqui no se atribuyen para no inventar
        el reparto cuando hubo reintentos).
        """
        seen = getattr(self, "_seen_hashes", None)
        if seen is None:
            seen = self._seen_hashes = set()
        path = os.path.join(self.dirname, "individuals.jsonl")
        for index, candidate in enumerate(population):
            code = candidate.code or ""
            digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:12]
            seen.add(digest)
            error = getattr(candidate, "error", None)
            record = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "generation": int(getattr(candidate, "generation", 0) or 0),
                "index": index,
                "name": getattr(candidate, "name", None),
                "code_sha1": digest,
                "code_chars": len(code),
                "description": getattr(candidate, "description", None),
                "fitness": (None if candidate.fitness is None
                            else float(candidate.fitness)),
                "error": (str(error)[:300] if error else None),
                "novelty": getattr(candidate, "novelty", None),
                "operator": getattr(candidate, "operator", None),
                "parent_ids": getattr(candidate, "parent_ids", None),
                # Trazabilidad: semilla y configuracion del proceso.
                "seed": type(self).seed,
                "solver": type(self).solver,
                "mo_generations": type(self).mo_generations,
                "mo_population_size": type(self).mo_population_size,
                "mutation": type(self).mutation,
            }
            with open(path, "a") as handle:
                handle.write(json.dumps(record) + "\n")

    def _log_generation_summary(self, population):
        """Agrega una linea por generacion con las metricas agregadas."""
        # Un candidato con error lleva el score de reserva (0), asi que se
        # separa: "validos" son los que tienen fitness finito y sin error.
        valid = [float(p.fitness) for p in population
                 if p.fitness is not None and np.isfinite(p.fitness) and not p.error]
        best = max(
            population,
            key=lambda p: (p.fitness if p.fitness is not None else -np.inf),
        )
        now = datetime.now()
        started = getattr(self, "_start_time", None)
        if started is None:
            started = now
            self._start_time = now
        previous_time = getattr(self, "_last_time", started)
        record = {
            "time": now.isoformat(timespec="seconds"),
            "generation": int(max(getattr(p, "generation", 0) for p in population)),
            "population_size": len(population),
            "evaluated": len(valid),
            "best_fitness": float(np.max(valid)) if valid else None,
            "mean_fitness": float(np.mean(valid)) if valid else None,
            "best_name": best.name,
            "best_description": best.description,
            "errors": sum(1 for p in population if p.error),
            # Tiempos de pared: de la generacion y acumulado de la corrida.
            "wall_seconds_gen": round((now - previous_time).total_seconds(), 3),
            "wall_seconds": round((now - started).total_seconds(), 3),
        }

        # Acumulados de la corrida: operadores distintos vistos, evaluaciones
        # validas y mejor HV historico (para ver la curva de mejora sin post-proceso).
        seen = getattr(self, "_seen_hashes", None) or set()
        evaluated_so_far = getattr(self, "_evaluated_so_far", 0) + len(valid)
        best_so_far = getattr(self, "_best_so_far", None)
        if valid:
            current_best = float(np.max(valid))
            best_so_far = (current_best if best_so_far is None
                           else max(best_so_far, current_best))
        self._evaluated_so_far = evaluated_so_far
        self._best_so_far = best_so_far
        record.update({
            "seed": type(self).seed,
            "solver": type(self).solver,
            "mo_generations": type(self).mo_generations,
            "mo_population_size": type(self).mo_population_size,
            "mutation": type(self).mutation,
            "valid_rate_gen": (len(valid) / len(population)) if population else None,
            "distinct_operators_so_far": len(seen),
            "evaluated_so_far": evaluated_so_far,
            "best_so_far": best_so_far,
        })

        # Tokens acumulados y por generacion (si hay una fuente de usage).
        source = type(self).token_source
        if source is not None and hasattr(source, "usage_snapshot"):
            usage = source.usage_snapshot()
            previous = getattr(self, "_last_usage", None) or {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "requests": 0, "inference_seconds": 0.0,
                "failed_requests": 0,
            }
            tokens_gen = usage["total_tokens"] - previous["total_tokens"]
            seconds_gen = usage.get("inference_seconds", 0.0) - previous.get("inference_seconds", 0.0)
            completion_gen = usage["completion_tokens"] - previous["completion_tokens"]
            record.update({
                "tokens_prompt": usage["prompt_tokens"],
                "tokens_completion": usage["completion_tokens"],
                "tokens_total": usage["total_tokens"],
                "requests": usage["requests"],
                "tokens_prompt_gen": usage["prompt_tokens"] - previous["prompt_tokens"],
                "tokens_completion_gen": completion_gen,
                "tokens_total_gen": tokens_gen,
                # Costo del modelo en esta generacion: latencia de inferencia
                # (cliente) y tokens de salida por segundo.
                "inference_seconds_gen": round(seconds_gen, 3),
                "inference_seconds": round(usage.get("inference_seconds", 0.0), 3),
                "tokens_per_second_gen": (round(completion_gen / seconds_gen, 3)
                                          if seconds_gen > 0 else None),
                "failed_requests": usage.get("failed_requests", 0),
            })
            self._last_usage = usage

        self._last_time = now

        with open(os.path.join(self.dirname, "generation_summary.jsonl"), "a") as file:
            file.write(json.dumps(record) + "\n")

        # Progreso en un archivo de una sola linea (para `tail -f` en el cluster,
        # donde el .out de SLURM se buffea y cuesta ver el avance).
        progress = (
            f"gen {record['generation']}  best={record['best_fitness']}  "
            f"mean={record['mean_fitness']}  valid={record['evaluated']}/"
            f"{record['population_size']}  errors={record['errors']}  "
            f"tokens_gen={record.get('tokens_total_gen')}  "
            f"tok/s={record.get('tokens_per_second_gen')}  "
            f"inference_s_gen={record.get('inference_seconds_gen')}  "
            f"wall_s={record['wall_seconds']}  {record['time']}"
        )
        with open(os.path.join(self.dirname, "progress.txt"), "w") as file:
            file.write(progress + "\n")

        # Checkpoint durable para el cluster (ver `_write_checkpoint`).
        try:
            self._write_checkpoint(best, record)
        except Exception:  # noqa: BLE001 - el checkpoint nunca debe tumbar la corrida
            pass

        if type(self).verbose:
            best_fitness = record["best_fitness"]
            tokens = record.get("tokens_total_gen")
            token_text = f" tokens={tokens}" if tokens is not None else ""
            print(
                f"[gen {record['generation']}] best={best_fitness if best_fitness is None else round(best_fitness, 6)} "
                f"mean={record['mean_fitness'] if record['mean_fitness'] is None else round(record['mean_fitness'], 6)} "
                f"valid={record['evaluated']}/{record['population_size']} "
                f"errors={record['errors']} name={record['best_name']}{token_text}"
                + (f" tok/s={record['tokens_per_second_gen']}"
                   if record.get("tokens_per_second_gen") else "")
                + f" t={record['wall_seconds']}s",
                flush=True,
            )


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
    keys = ("tokens_total", "tokens_prompt", "tokens_completion", "requests")
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


def write_prompt_log(log_dir, usage_history=None, path=None):
    """Guarda los prompts y las respuestas **ordenados por generacion**.

    De `conversationlog.jsonl` (peticion/respuesta en orden) y `log.jsonl`
    (generacion de cada intento) salen:

    - ``prompts/gen-NNNN.md``: un archivo legible por generacion con el prompt
      completo y la respuesta cruda;
    - ``prompts/index.jsonl``: una linea por turno con generacion, candidato,
      tamano del prompt/respuesta, tokens (entrada/salida/total) y **latencia de
      inferencia** (de `usage_history`, en el mismo orden).

    @return la lista de registros del indice.
    """
    directory = path or os.path.join(log_dir, "prompts")
    os.makedirs(directory, exist_ok=True)
    turns = _pair_turns(_read_jsonl(os.path.join(log_dir, "conversationlog.jsonl")))
    rows = load_evolution(log_dir)
    usage_history = list(usage_history or [])
    usage_pointer = 0

    records = []
    by_generation = {}
    for index, turn in enumerate(turns):
        row = rows[index] if index < len(rows) else (rows[-1] if rows else {})
        generation = int(row.get("generation", index) or index)
        usage = {}
        if turn["response"]:
            if usage_pointer < len(usage_history):
                usage = usage_history[usage_pointer] or {}
            usage_pointer += 1
        record = {
            "turn": index + 1,
            "generation": generation,
            "name": row.get("name", ""),
            "prompt_chars": len(turn["prompt"]),
            "response_chars": len(turn["response"]),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "seconds": usage.get("seconds"),
            "prompt_time": turn["prompt_time"],
            "response_time": turn["response_time"],
        }
        records.append(record)
        by_generation.setdefault(generation, []).append((record, turn))

    for generation, entries in sorted(by_generation.items()):
        lines = [f"# Generacion {generation}", ""]
        for record, turn in entries:
            lines += [
                f"## Turno {record['turn']} - `{record['name'] or 'candidato'}`",
                "",
                f"- Prompt: {record['prompt_chars']} caracteres, "
                f"{record['prompt_tokens'] if record['prompt_tokens'] is not None else '-'} tokens",
                f"- Respuesta: {record['response_chars']} caracteres, "
                f"{record['completion_tokens'] if record['completion_tokens'] is not None else '-'} tokens",
                f"- Inferencia: "
                f"{record['seconds'] if record['seconds'] is not None else '-'} s",
                "",
                "### Prompt",
                "",
                "````text",
                turn["prompt"],
                "````",
                "",
                "### Respuesta",
                "",
                "````text",
                turn["response"],
                "````",
                "",
            ]
        with open(os.path.join(directory, f"gen-{generation:04d}.md"), "w") as file:
            file.write("\n".join(lines))

    with open(os.path.join(directory, "index.jsonl"), "w") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    with open(os.path.join(directory, "README.md"), "w") as file:
        file.write(
            "# Prompts y respuestas, ordenados por generacion\n\n"
            "Generado por `mohh.llamea.reporting.write_prompt_log` a partir de\n"
            "`conversationlog.jsonl` (peticiones y respuestas completas) y\n"
            "`log.jsonl` (generacion de cada intento).\n\n"
            "- `gen-NNNN.md`: un archivo por generacion con el prompt completo\n"
            "  (rol + tarea + ejemplo + feedback) y la respuesta cruda del modelo.\n"
            "- `index.jsonl`: una linea por turno, con generacion, candidato,\n"
            "  tamano de prompt/respuesta, tokens de entrada/salida/total y\n"
            "  latencia de inferencia.\n"
        )
    return records


def write_csv(rows, path):
    import csv

    fields = ["generation", "id", "name", "fitness", "rank", "best_so_far",
              "valid", "error", "description"]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value, digits=6):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(rows, path, top_k=10, title="Busqueda LLaMEA", tokens=None,
                   frame_info=None, cost=None):
    valid = [r for r in rows if r["valid"]]
    invalid = [r for r in rows if not r["valid"]]
    best = max(valid, key=lambda r: r["fitness"]) if valid else None

    lines = [f"# {title}", ""]
    lines += [
        "## Resumen",
        "",
        f"- Evaluaciones totales: {len(rows)}",
        f"- Validas (HV > 0): {len(valid)}",
        f"- Invalidas: {len(invalid)}",
        f"- Mejor HV: {_fmt(best['fitness']) if best else '-'}",
        f"- Generaciones registradas: {max((r['generation'] for r in rows), default=0)}",
        "",
    ]

    if frame_info:
        frame = frame_info.get("frame")
        if frame == "ge":
            frame_note = ("frente consolidado del panel y de los candidatos evaluados, "
                          "como la evaluacion GE; comparable con la tabla de la busqueda gramatical")
        elif frame == "stationary":
            frame_note = "solo el panel de referencia (escala abierta, no comparable con la tabla GE)"
        elif frame == "none":
            frame_note = "HV crudo (sin normalizar)"
        else:
            frame_note = "corrida anterior a `frame` (marco fijo del panel)"
        lines += ["## Marco de hipervolumen (HV justo)", "", f"- `frame`: `{frame or '-'}` ({frame_note})"]
        if frame_info.get("note"):
            lines.append(f"- NOTA: {frame_info['note']}")
        degenerate = frame_info.get("degenerate_instances") or []
        if degenerate:
            lines.append(
                f"- Instancias sin escala (excluidas de la media): {', '.join(degenerate)}"
            )
        reference = frame_info.get("reference_hv_final_frame") or {}
        if reference:
            lines.append("- Linea base clasica en el MISMO marco final (comparacion justa):")
            lines.append("")
            lines.append("  | Operador | HV |")
            lines.append("  |---|---:|")
            for name, value in sorted(reference.items(), key=lambda kv: -kv[1]):
                lines.append(f"  | {name} | {_fmt(value)} |")
        if frame_info.get("best_hv_final_frame") is not None:
            lines.append(
                f"- Mejor candidato en el marco final: {_fmt(frame_info['best_hv_final_frame'])}"
            )
        if frame_info.get("test_fitness") is not None:
            lines.append(f"- HV de test (mismo criterio): {_fmt(frame_info['test_fitness'])}")
            test_reference = frame_info.get("test_reference_hv") or {}
            if test_reference:
                baseline = ", ".join(
                    f"{name}={_fmt(value)}" for name, value in
                    sorted(test_reference.items(), key=lambda kv: -kv[1])
                )
                lines.append(f"- Linea base en test, mismo marco: {baseline}")
        lines.append("")

        # Trazabilidad del proceso: semillas, diversidad y parada.
        seeds = frame_info.get("seeds") or {}
        if seeds:
            lines += ["## Semillas y proceso", ""]
            lines.append(f"- Semilla de busqueda (MOEA interno e instancias): "
                         f"`{seeds.get('search')}`")
            if seeds.get("eval_test") is not None:
                lines.append(f"- Semilla de evaluacion en test: `{seeds.get('eval_test')}`")
            llm = seeds.get("llm_sampling") or {}
            if llm:
                lines.append(f"- Muestreo del LLM: temperatura `{llm.get('temperature')}`, "
                             f"max_tokens `{llm.get('max_tokens')}` "
                             "(sin semilla fija: no reproducible bit a bit)")
            if frame_info.get("stop_reason"):
                lines.append(f"- Motivo de parada: `{frame_info['stop_reason']}`")
            if frame_info.get("distinct_operators") is not None:
                lines.append(f"- Operadores distintos: {frame_info['distinct_operators']}")
            if frame_info.get("validity_rate") is not None:
                lines.append(f"- Tasa de validez: {frame_info['validity_rate']:.1%}")
            lines.append("")

    if tokens:
        lines += [
            "## Tokens consumidos",
            "",
            f"- Peticiones al LLM: {tokens.get('requests', 0)}",
            f"- Tokens de prompt (acumulado): {tokens.get('tokens_prompt', 0)}",
            f"- Tokens generados (acumulado): {tokens.get('tokens_completion', 0)}",
            f"- Tokens totales (acumulado): {tokens.get('tokens_total', 0)}",
        ]
        if "tokens_total_gen_mean" in tokens:
            lines.append(
                f"- Tokens totales por generacion (media): "
                f"{tokens['tokens_total_gen_mean']:.1f}"
            )
            lines.append(
                f"- Tokens totales por generacion (maximo): {tokens['tokens_total_gen_max']}"
            )
        lines.append("")

    if cost:
        lines += ["## Costo del modelo (tokens, tiempo de GPU, energia)", ""]
        lines.append(f"- Fuente de las metricas: `{cost.get('source', '-')}`")
        lines.append(f"- Peticiones: {cost.get('requests', '-')}"
                     + (f" (fallidas: {cost['failed_requests']})"
                        if cost.get("failed_requests") else ""))
        lines.append(f"- Tokens de entrada (prompt): {cost.get('prompt_tokens', '-')}")
        lines.append(f"- Tokens de salida (generacion): {cost.get('completion_tokens', '-')}")
        lines.append(f"- Tokens totales: {cost.get('total_tokens', '-')}")
        if cost.get("inference_seconds") is not None:
            lines.append(f"- Tiempo de inferencia (pared, cliente): "
                         f"{cost['inference_seconds']} s")
        if cost.get("tokens_per_second"):
            lines.append(f"- Velocidad de generacion (cliente): "
                         f"{cost['tokens_per_second']} tokens/s")
        if cost.get("wall_seconds"):
            lines.append(f"- Duracion total de la corrida: {cost['wall_seconds']} s")
        if cost.get("gpu_seconds") is not None:
            lines.append(f"- **Tiempo de GPU** (vLLM, prefill + decode): "
                         f"{cost['gpu_seconds']} s ({cost.get('gpu_hours')} GPU-h)")
            if cost.get("gpu_seconds_per_request"):
                lines.append(f"- Tiempo de GPU por peticion: "
                             f"{cost['gpu_seconds_per_request']} s")
            if cost.get("gpu_tokens_per_second"):
                lines.append(f"- Velocidad real de GPU: "
                             f"{cost['gpu_tokens_per_second']} tokens/s")
            if cost.get("gpu_prefill_seconds") is not None:
                lines.append(f"- Desglose: prefill {cost['gpu_prefill_seconds']} s, "
                             f"decode {cost['gpu_decode_seconds']} s")
        if cost.get("ttft_mean_seconds"):
            lines.append(f"- Tiempo al primer token (media): {cost['ttft_mean_seconds']} s")
        if cost.get("e2e_mean_seconds"):
            lines.append(f"- Latencia extremo a extremo (media): {cost['e2e_mean_seconds']} s")
        if cost.get("kv_cache_usage_percent") is not None:
            lines.append(f"- Ocupacion del KV cache: {cost['kv_cache_usage_percent']} %")
        if cost.get("energy_wh") is not None:
            lines.append(f"- Energia estimada ({cost.get('gpu_power_w')} W de GPU): "
                         f"{cost['energy_wh']} Wh")
        if cost.get("gpu_name"):
            lines.append(f"- GPU: {cost['gpu_name']} "
                         f"({cost.get('gpu_memory_used_mb')}/{cost.get('gpu_memory_total_mb')} MB)")
        lines.append("")
        lines.append("> El tiempo de GPU solo lo publica vLLM (`/metrics`); con Ollama o")
        lines.append("> llama.cpp quedan los tokens y la latencia del cliente. Detalle por")
        lines.append("> peticion en `metrics.json` (`per_request`) y por generacion en")
        lines.append("> `generations.csv` (`inference_seconds_gen`, `tokens_per_second_gen`).")
        lines.append("")

    if best:
        lines += [
            "## Mejor operador",
            "",
            f"- Generacion: {best['generation']}",
            f"- Nombre: `{best['name']}`",
            f"- HV: {_fmt(best['fitness'])}",
            f"- Descripcion: {best['description'] or '-'}",
            "",
            "```python",
            best["code"].rstrip(),
            "```",
            "",
        ]

    if valid:
        lines += [
            f"## Top {min(top_k, len(valid))} por HV",
            "",
            "| # | Gen | Nombre | HV | Ranking | Descripcion |",
            "|---|-----|--------|----|---------|-------------|",
        ]
        for position, row in enumerate(
            sorted(valid, key=lambda r: r["fitness"], reverse=True)[:top_k], start=1
        ):
            description = (row["description"] or "").replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {position} | {row['generation']} | `{row['name']}` | "
                f"{_fmt(row['fitness'])} | {row['rank']} | {description[:80]} |"
            )
        lines.append("")

    if invalid:
        from collections import Counter

        errors = Counter((r["error"] or "sin error") for r in invalid)
        lines += ["## Errores mas frecuentes", "", "| Error | Veces |", "|-------|-------|"]
        for error, count in errors.most_common(10):
            text = str(error).replace("|", "/").replace("\n", " ")[:120]
            lines.append(f"| {text} | {count} |")
        lines.append("")

    with open(path, "w") as file:
        file.write("\n".join(lines))


def write_plot(rows, path, title="Busqueda LLaMEA"):
    """Grafica la evolucion del HV (por generacion y mejor acumulado)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib es opcional
        return False

    if not rows:
        return False

    generations = [r["generation"] for r in rows]
    fitness = [r["fitness"] if r["valid"] else np.nan for r in rows]
    best = [r["best_so_far"] if r["valid"] else np.nan for r in rows]

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(generations, fitness, s=18, alpha=0.7, label="HV por generacion")
    axis.plot(generations, best, color="tab:red", linewidth=2, label="Mejor acumulado")
    axis.set_xlabel("Generacion")
    axis.set_ylabel("Hipervolumen (normalizado)")
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def write_generation_csv(log_dir, path=None):
    """Convierte `generation_summary.jsonl` en ``generations.csv``.

    Una fila por generacion con todo lo que el logger haya registrado (HV,
    validos, errores y tokens). Devuelve la ruta escrita, o None si no hay log.
    """
    records = _read_jsonl(os.path.join(log_dir, "generation_summary.jsonl"))
    if not records:
        return None
    path = path or os.path.join(log_dir, "generations.csv")

    # Union de claves en orden de aparicion (las corridas viejas no tienen
    # tokens y las nuevas si; asi el CSV es estable dentro de una corrida).
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def write_seeds(path, seeds):
    """Escribe `seeds.json`: politica y valores de semilla de toda la corrida."""
    with open(path, "w") as handle:
        json.dump(seeds, handle, indent=2)
    return path


def write_report(log_dir, title="Busqueda LLaMEA", top_k=10, frame_info=None,
                 cost=None, usage_history=None):
    """Genera los artefactos legibles de la corrida en `log_dir`.

    ``evolution.csv``, ``report.md``, ``evolution.png``, ``generations.csv`` y
    ``prompts/`` (prompts y respuestas ordenados por generacion).

    @param frame_info: seccion del marco de normalizacion del HV con la linea
        base clasica medida en el mismo marco (`run.py` se lo pasa).
    @param cost: resumen de costo del modelo (`mohh.llamea.metrics.cost_summary`)
        para la seccion de tokens, tiempo de GPU y energia.
    @param usage_history: historial por peticion (`VLLM_LLM.usage_history`) para
        anotar tokens y latencia en `prompts/index.jsonl`.
    @return diccionario de rutas escritas (vacio si no hay log).
    """
    rows = load_evolution(log_dir)
    if not rows:
        return {}

    paths = {
        "csv": os.path.join(log_dir, "evolution.csv"),
        "markdown": os.path.join(log_dir, "report.md"),
    }
    write_csv(rows, paths["csv"])
    write_markdown(rows, paths["markdown"], top_k=top_k, title=title,
                   tokens=tokens_summary(log_dir), frame_info=frame_info,
                   cost=cost)

    generations = write_generation_csv(log_dir)
    if generations:
        paths["generations_csv"] = generations

    prompts = write_prompt_log(log_dir, usage_history=usage_history)
    if prompts:
        paths["prompts"] = os.path.join(log_dir, "prompts")

    plot_path = os.path.join(log_dir, "evolution.png")
    if write_plot(rows, plot_path, title=title):
        paths["plot"] = plot_path
    return paths
