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
import json
import os
from datetime import datetime

import numpy as np

from llamea.loggers import ExperimentLogger


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
            self._log_generation_summary(population)

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
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "generation": int(max(getattr(p, "generation", 0) for p in population)),
            "population_size": len(population),
            "evaluated": len(valid),
            "best_fitness": float(np.max(valid)) if valid else None,
            "mean_fitness": float(np.mean(valid)) if valid else None,
            "best_name": best.name,
            "best_description": best.description,
            "errors": sum(1 for p in population if p.error),
        }

        # Tokens acumulados y por generacion (si hay una fuente de usage).
        source = type(self).token_source
        if source is not None and hasattr(source, "usage_snapshot"):
            usage = source.usage_snapshot()
            previous = getattr(self, "_last_usage", None) or {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "requests": 0,
            }
            record.update({
                "tokens_prompt": usage["prompt_tokens"],
                "tokens_completion": usage["completion_tokens"],
                "tokens_total": usage["total_tokens"],
                "requests": usage["requests"],
                "tokens_prompt_gen": usage["prompt_tokens"] - previous["prompt_tokens"],
                "tokens_completion_gen": usage["completion_tokens"] - previous["completion_tokens"],
                "tokens_total_gen": usage["total_tokens"] - previous["total_tokens"],
            })
            self._last_usage = usage

        with open(os.path.join(self.dirname, "generation_summary.jsonl"), "a") as file:
            file.write(json.dumps(record) + "\n")

        if type(self).verbose:
            best_fitness = record["best_fitness"]
            tokens = record.get("tokens_total_gen")
            token_text = f" tokens={tokens}" if tokens is not None else ""
            print(
                f"[gen {record['generation']}] best={best_fitness if best_fitness is None else round(best_fitness, 6)} "
                f"mean={record['mean_fitness'] if record['mean_fitness'] is None else round(record['mean_fitness'], 6)} "
                f"valid={record['evaluated']}/{record['population_size']} "
                f"errors={record['errors']} name={record['best_name']}{token_text}",
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
                   frame_info=None):
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


def write_report(log_dir, title="Busqueda LLaMEA", top_k=10, frame_info=None):
    """Genera ``evolution.csv``, ``report.md`` y ``evolution.png`` en `log_dir`.

    `frame_info` (opcional) agrega la seccion del marco de normalizacion del HV
    con la linea base clasica medida en el mismo marco (`run.py` se lo pasa).
    Devuelve el diccionario de rutas escritas (vacio si no hay log).
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
                   tokens=tokens_summary(log_dir), frame_info=frame_info)

    generations = write_generation_csv(log_dir)
    if generations:
        paths["generations_csv"] = generations

    plot_path = os.path.join(log_dir, "evolution.png")
    if write_plot(rows, plot_path, title=title):
        paths["plot"] = plot_path
    return paths
