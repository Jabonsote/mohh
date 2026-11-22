#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume busquedas pequenas y mide diversidad CrystalBLEU en paralelo."""
import argparse
import concurrent.futures
import glob
import json
import os
import re
from collections import Counter, defaultdict

from mohh.analysis.code_diversity import crystalbleu_diversity


def _norm_error(text):
    text = (text or "").split("\n")[0]
    checks = [
        ("Timeout", r"Timeout|exceeded .*seconds"),
        ("syntax", r"SyntaxError|did not compile"),
        ("not permutation", r"not a permutation|not a valid permutation"),
        ("shape/broadcast", r"shape|broadcast|index .* out of bounds|low >= high"),
        ("numpy method", r"ndarray.*(index|pop)|has no attribute"),
        ("KeyError/TypeError", r"KeyError|TypeError|UnboundLocalError|list.remove"),
    ]
    for label, pattern in checks:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return text[:80] or "unknown"


def _read_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _progress(path):
    best = []
    if not os.path.exists(path):
        return best
    for line in open(path, encoding="utf-8"):
        match = re.search(r"best=([0-9.]+)", line)
        hv = re.search(r"hv=([-0-9.eE]+|None)", line)
        if match:
            best.append((float(match.group(1)),
                         None if not hv or hv.group(1) == "None"
                         else float(hv.group(1))))
    return best


def summarize_run(run_dir):
    candidates = _read_jsonl(os.path.join(run_dir, "candidates.jsonl"))
    valid = [row for row in candidates if row.get("valid")]
    codes = [row.get("code", "") for row in valid if row.get("code")]
    errors = Counter(_norm_error(row.get("error")) for row in candidates
                     if not row.get("valid"))
    info = {}
    path = os.path.join(run_dir, "best_so_far.json")
    if os.path.exists(path):
        info = json.load(open(path))
    progress = _progress(os.path.join(run_dir, "progress.txt"))
    initial = progress[0] if progress else (None, None)
    final = progress[-1] if progress else (None, None)
    return {
        "run": os.path.basename(run_dir.rstrip("/")),
        "n": len(candidates),
        "valid": len(valid),
        "valid_rate": len(valid) / len(candidates) if candidates else 0.0,
        "errors": dict(errors),
        "best_generation": info.get("generation"),
        "best_fitness": info.get("fitness"),
        "best_hv": info.get("hv"),
        "initial_fitness": initial[0],
        "initial_hv": initial[1],
        "final_fitness": info.get("fitness", final[0]),
        "final_hv": info.get("hv", final[1]),
        "improved_rank": (info.get("generation") or 0) > 0,
        "improved_hv": (info.get("hv") is not None and initial[1] is not None
                        and info["hv"] > initial[1] + 1e-12),
        "unique_valid_codes": len(set(codes)),
        "descriptions": [row.get("description", "") for row in valid][:10],
    }


def summarize_prompt(path):
    rows = _read_jsonl(path)
    by = defaultdict(list)
    for row in rows:
        by[row["config"]].append(row)
    result = {}
    for name, subset in by.items():
        valid = [row for row in subset if row.get("status") == "ok"]
        codes = [row.get("code", "") for row in valid if row.get("code")]
        errors = Counter(_norm_error(row.get("error")) for row in subset
                         if row.get("status") != "ok")
        result[name] = {
            "n": len(subset), "valid": len(valid),
            "rate": len(valid) / len(subset) if subset else 0.0,
            "errors": dict(errors), "codes": list(set(codes)),
        }
    return result


def add_diversity(items):
    def one(item):
        codes = item.get("codes") or []
        return crystalbleu_diversity(codes) if len(set(codes)) > 1 else 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(one, item) for name, item in items.items()}
        for name, future in futures.items():
            items[name]["diversity"] = future.result()
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--searches",
                        default="results/llamea_small_20261113/searches")
    parser.add_argument("--prompt",
                        default="results/llamea_small_20261113/prompt_ab")
    parser.add_argument("--out",
                        default="results/llamea_small_20261113/analysis.json")
    args = parser.parse_args()

    runs = {}
    for path in sorted(glob.glob(os.path.join(args.searches, "*/"))):
        if os.path.islink(path.rstrip("/")):
            continue
        if os.path.exists(os.path.join(path, "candidates.jsonl")):
            runs[os.path.basename(path.rstrip("/"))] = summarize_run(path)
    prior_path = os.path.join(os.path.dirname(args.out), "initial_priors.json")
    if os.path.exists(prior_path):
        initial = json.load(open(prior_path))
        for name, row in runs.items():
            if name not in initial:
                continue
            prior = initial[name]["prior"]
            row["initial_fitness"] = prior["fitness"]
            row["initial_hv"] = prior["hv"]
            row["improved_rank"] = (row["best_generation"] or 0) > 0
            row["improved_hv"] = (
                row["best_hv"] is not None and prior["hv"] is not None
                and row["best_hv"] > prior["hv"] + 1e-12)
    prompt = summarize_prompt(os.path.join(args.prompt, "results.jsonl"))
    for item in prompt.values():
        item["codes"] = item["codes"]
    add_diversity(prompt)
    # Diversidad de codigos validos por corrida (en paralelo con el resumen).
    run_items = {name: {"codes": [
        row.get("code", "") for row in _read_jsonl(
            os.path.join(args.searches, name, "candidates.jsonl"))
        if row.get("valid") and row.get("code")]} for name in runs}
    add_diversity(run_items)
    for name in runs:
        runs[name]["diversity"] = run_items[name]["diversity"]

    payload = {"runs": runs, "prompt": prompt}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("RUNS")
    for name, row in runs.items():
        print(f"{name:10s} n={row['n']:2d} valid={row['valid']:2d} "
              f"({100*row['valid_rate']:5.1f}%) gen={row['best_generation']} "
              f"fit {row['initial_fitness']}->{row['final_fitness']} "
              f"hv {row['initial_hv']}->{row['final_hv']} "
              f"div={row['diversity']:.3f} "
              f"errors={row['errors']}")
    print("\nPROMPT")
    for name, row in prompt.items():
        print(f"{name:20s} n={row['n']:2d} valid={row['valid']:2d} "
              f"({100*row['rate']:5.1f}%) div={row['diversity']:.3f} "
              f"errors={row['errors']}")


if __name__ == "__main__":
    main()
