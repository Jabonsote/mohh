"""Estado de una corrida LLaMEA en curso.

Lee `progress.txt` y `llm_requests.jsonl` de una carpeta de resultados y
resume lo que importa para seguir una carrera:

    python scripts/run_state.py results/searches/glm-gptoss2

Muestra las iteraciones validas, los fallos por causa, si el incumbent se
mueve (cuantas veces se ha aceptado un candidato nuevo), el reparto de
estrategias y cuantas respuestas se cortaron por `max_tokens`.
"""
import collections
import json
import os
import re
import sys


def _read_progress(folder):
    rows = []
    path = os.path.join(folder, "progress.txt")
    if not os.path.isfile(path):
        return rows
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 3:
            continue
        row = {"iter": int(parts[1])}
        for item in parts[2:]:
            if "=" in item:
                key, value = item.split("=", 1)
                row[key] = value
        rows.append(row)
    return rows


def _read_incumbent_changes(folder):
    """Veces que cambio el operador incumbent (aceptaciones).

    El prompt alterna entre el incumbent y un prior (cada `--prior-every`
    iteraciones), asi que solo cuentan los cambios de un incumbent a otro.
    """
    path = os.path.join(folder, "llm_requests.jsonl")
    if not os.path.isfile(path):
        return 0
    try:
        from mohh.llamea.priors import load_priors

        priors = {code.strip() for _, code in load_priors()}
    except Exception:  # noqa: BLE001
        priors = set()
    pattern = re.compile(r"REFERENCE CODE\n```python\n(.*?)\n```", re.S)
    changes = 0
    last = None
    for line in open(path, encoding="utf-8"):
        try:
            record = json.loads(line)
        except Exception:  # noqa: BLE001 - linea a medias
            continue
        match = pattern.search(record.get("prompt") or "")
        code = (match.group(1) if match else "").strip()
        if not code or code in priors:
            continue
        if code != last:
            changes += 1
            last = code
    return max(0, changes - 1)


def _read_truncated(folder):
    path = os.path.join(folder, "llm_requests.jsonl")
    if not os.path.isfile(path):
        return 0, 0
    total = truncated = 0
    for line in open(path, encoding="utf-8"):
        try:
            record = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        total += 1
        if (record.get("response") or {}).get("finish_reason") == "length":
            truncated += 1
    return truncated, total


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/searches/latest"
    rows = _read_progress(folder)
    if not rows:
        print(f"sin progress.txt en {folder!r}")
        return 1
    valid = [row for row in rows if row.get("valid") == "1"]
    invalid = [row for row in rows if row.get("valid") != "1"]

    print(f"== {folder}")
    print(f"iteraciones: {len(rows)}   valid={len(valid)} "
          f"({100 * len(valid) / len(rows):.1f}%)   fallos={len(invalid)}")
    causes = collections.Counter(
        (row.get("error") or "?").split(":")[0] for row in invalid)
    print("fallos por causa:", dict(causes))
    if valid:
        ranks = [float(row["best"]) for row in valid]
        print(f"rank del incumbent: min {min(ranks):.4f}  ultimo {ranks[-1]:.4f}")
    print("estrategias:", dict(collections.Counter(
        row.get("strategy", "?") for row in rows)))

    changes = _read_incumbent_changes(folder)
    if changes >= 0:
        print(f"cambios del incumbent (aceptaciones): {changes}")

    truncated, total = _read_truncated(folder)
    if total:
        print(f"peticiones: {total}   truncadas: {truncated} "
              f"({100 * truncated / total:.1f}%)")

    summary = os.path.join(folder, "search_summary.json")
    if os.path.isfile(summary):
        data = json.load(open(summary, encoding="utf-8"))
        print("resumen:", json.dumps(data.get("iterations", {}), ensure_ascii=False))
        print("diversidad:", json.dumps(data.get("code_diversity", {}),
                                       ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
