"""Tablas de la evaluación terminada (LLaMEA vs OX vs PMX).

Deja los resultados en tablas CSV + un Markdown legible para el paper:

    EVAL_DIR=results/evaluations/llamea_vs_ox_pmx_local \
    python results/analysis_scripts/tablas_evaluacion.py

Salidas (en el directorio de la evaluacion):
    tabla_por_moea.csv       MOEA x presupuesto x mutacion x base: victorias, %
                              efectiva, diferencia, IC95, dz, p y p de Holm
    tabla_por_instancia.csv  MOEA x mutacion x instancia x base: W/P/E
    tabla_no_gana.csv        celdas donde el operador pierde la mayoria
    tablas.md                las tres tablas en Markdown
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "/home/jabonsote/Desktop/MOOHH-R")
from mohh.llamea.statistics import paired_summary, add_holm

D = os.environ.get(
    "EVAL_DIR",
    "/home/jabonsote/Desktop/MOOHH-R/results/evaluations/llamea_vs_ox_pmx_local")
SOLVERS = ("MOEAD", "NSGAII", "SMSEMOA")
BUDGETS = ("10k", "30k", "50k")
MUTS = ("NM", "WM")
MUT_NOMBRE = {"NM": "sin mutacion", "WM": "con mutacion"}
def _panel():
    """(operador evaluado, [bases]) deducidos de evaluation.json o del npz."""
    import json as _json, numpy as _np
    ep = os.path.join(D, "evaluation.json")
    if os.path.exists(ep):
        with open(ep) as fh:
            info = _json.load(fh)
        rows = info.get("operators", [])
        evaluados = [r["operator"] for r in rows if r.get("kind") == "llamea"]
        bases = [r["operator"] for r in rows
                 if r.get("kind") in ("classic", "ge") and r.get("runs")]
        if evaluados and bases:
            return evaluados[0], bases
    # sin evaluation.json: deducir los nombres del fronts.npz
    d = _np.load(os.path.join(D, "fronts.npz"))
    ops = sorted({k.split("|")[1] for k in d.files})
    evaluados = [o for o in ops if o == "llamea"] or [ops[0]]
    bases = [o for o in ops if o != evaluados[0]]
    return evaluados[0], bases


EVAL, BASES = _panel()
OPS = (EVAL,) + tuple(BASES)

# ---------------------------------------------------------------- carga
celdas = defaultdict(dict)
for solver in SOLVERS:
    for budget in BUDGETS:
        d = os.path.join(D, solver, budget)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".csv"):
                continue
            inst = f[:-4]
            with open(os.path.join(d, f)) as fh:
                for row in csv.DictReader(fh):
                    rep = int(row["Experiment"])
                    for col, v in row.items():
                        if col == "Experiment":
                            continue
                        # el nombre del operador puede llevar guiones bajos
                        # (ge_best): se parsea desde la derecha
                        p = col.split("_")
                        mut = "WM" if p[-2] == "WM" else "NM"
                        op = "_".join(p[1:-2])
                        celdas[(solver, budget, mut, inst, rep)][op] = (
                            float(v) if v not in ("", "None") else 0.0)

INSTS = sorted({k[3] for k in celdas})
REPS = sorted({k[4] for k in celdas})


def bloque(solver, budget, mut, inst):
    """[[op_evaluado, base1, base2, ...]] por repeticion, en el mismo orden."""
    out = []
    for rep in REPS:
        c = celdas.get((solver, budget, mut, inst, rep))
        if not c or len(c) < len(OPS):
            continue
        out.append(tuple(c[op] for op in OPS))
    return out


def cuenta(bloques, base_idx):
    w = l = t = t0 = 0
    for b in bloques:
        a, bb = b[0], b[base_idx]
        if a > bb:
            w += 1
        elif a < bb:
            l += 1
        else:
            t += 1
            if a == 0.0 and bb == 0.0:
                t0 += 1
    return w, l, t, t0


# ------------------------------------------------- tabla 1: por MOEA/config
t1 = []
for solver in SOLVERS:
    celdas_moea = []
    for budget in BUDGETS:
        for mut in MUTS:
            bloques = []
            for inst in INSTS:
                bloques += bloque(solver, budget, mut, inst)
            for bidx, base in enumerate(BASES, start=1):
                s = paired_summary([b[0] for b in bloques],
                                   [b[bidx] for b in bloques])
                w, l, t, t0 = cuenta(bloques, bidx)
                s.update({"moea": solver, "presupuesto": budget,
                          "mutacion": MUT_NOMBRE[mut], "base": base,
                          "gana": w, "pierde": l, "empata": t,
                          "empata_fallo": t0, "n": len(bloques),
                          "efectivas": w + l})
                celdas_moea.append(s)
    celdas_moea = add_holm(celdas_moea)
    t1.extend(celdas_moea)

# ------------------------------------------------- tabla 2: por instancia
t2 = []
for solver in SOLVERS:
    for mut in MUTS:
        for inst in INSTS:
            bloques = []
            for budget in BUDGETS:
                bloques += bloque(solver, budget, mut, inst)
            if not bloques:
                continue
            for bidx, base in enumerate(BASES, start=1):
                w, l, t, t0 = cuenta(bloques, bidx)
                t2.append({
                    "moea": solver, "mutacion": MUT_NOMBRE[mut],
                    "instancia": inst, "base": base, "n": len(bloques),
                    "gana": w, "pierde": l, "empata": t,
                    "empata_fallo_ambos": t0,
                    "pct_gana": round(100 * w / len(bloques), 1),
                    "pct_gana_efectiva": round(100 * w / (w + l), 1) if (w + l) else "",
                    "hv_evaluado": round(statistics.mean(b[0] for b in bloques), 6),
                    "hv_base": round(statistics.mean(b[bidx] for b in bloques), 6),
                })

# ------------------------------------------------- tabla 3: donde no gana
t3 = []
for solver in SOLVERS:
    for budget in BUDGETS:
        for mut in MUTS:
            for inst in INSTS:
                bloques = bloque(solver, budget, mut, inst)
                if not bloques:
                    continue
                for bidx, base in enumerate(BASES, start=1):
                    w, l, t, t0 = cuenta(bloques, bidx)
                    if not l > w:
                        continue
                    ml = statistics.mean(b[0] for b in bloques)
                    mp = statistics.mean(b[bidx] for b in bloques)
                    t3.append({
                        "moea": solver, "presupuesto": budget,
                        "mutacion": MUT_NOMBRE[mut], "instancia": inst,
                        "base": base, "gana": w, "pierde": l, "empata": t,
                        "hv_evaluado": round(ml, 6), "hv_base": round(mp, 6),
                        "dif": round(ml - mp, 6),
                        "ratio": round(ml / mp, 4) if mp else "",
                    })

# ------------------------------------------------- escritura
def write_csv(nombre, filas_, campos):
    ruta = os.path.join(D, nombre)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas_)
    print("escrito", ruta)


write_csv("tabla_por_moea.csv", t1, [
    "moea", "presupuesto", "mutacion", "base", "n", "gana", "pierde", "empata",
    "empata_fallo", "efectivas", "operator_mean", "baseline_mean",
    "mean_difference", "sd_difference", "ci95_low", "ci95_high", "cohens_dz",
    "relative", "wilcoxon_p", "wilcoxon_p_holm", "significant_holm"])
write_csv("tabla_por_instancia.csv", t2, [
    "moea", "mutacion", "instancia", "base", "n", "gana", "pierde", "empata",
    "empata_fallo_ambos", "pct_gana", "pct_gana_efectiva", "hv_evaluado", "hv_base"])
write_csv("tabla_no_gana.csv", t3, [
    "moea", "presupuesto", "mutacion", "instancia", "base", "gana", "pierde",
    "empata", "hv_evaluado", "hv_base", "dif", "ratio"])


def md_tabla(filas_, campos, titulo):
    out = [f"### {titulo}", "",
           "| " + " | ".join(campos) + " |",
           "|" + "|".join("---" for _ in campos) + "|"]
    for f in filas_:
        out.append("| " + " | ".join(str(f.get(c, "")) for c in campos) + " |")
    return "\n".join(out)


def fmt(f):
    return (f"{f['gana']}/{f['pierde']}/{f['empata']} "
            f"({100*f['gana']/f['n']:.0f}%)")


md = ["# LLaMEA vs OX vs PMX: tablas de la evaluacion terminada", "",
      f"Directorio: `{D}`", "",
      "Marco compartido por instancia y presupuesto; nadir fijo [1.1]*K; "
      "los empates son siempre fallos de los dos operadores (HV 0).", ""]

md.append("## 1. Por MOEA, presupuesto y mutacion\n")
md.append(md_tabla(
    t1,
    ["moea", "presupuesto", "mutacion", "base", "n", "gana", "pierde", "empata",
     "mean_difference", "ci95_low", "ci95_high", "cohens_dz", "relative",
     "wilcoxon_p_holm"],
    "Contraste pareado por MOEA x presupuesto x mutacion"))

md.append("\n\n## 2. Por instancia (MOEA x mutacion)\n")
md.append(md_tabla(
    t2,
    ["moea", "mutacion", "instancia", "base", "n", "gana", "pierde", "empata",
     "pct_gana", "pct_gana_efectiva", "hv_evaluado", "hv_base"],
    "Victorias por instancia"))

md.append("\n\n## 3. Celdas donde el operador pierde la mayoria contra PMX\n")
md.append(md_tabla(
    t3,
    ["moea", "presupuesto", "mutacion", "instancia", "gana", "pierde",
     "hv_evaluado", "hv_base", "dif", "ratio"],
    "No gana contra PMX"))

ruta_md = os.path.join(D, "tablas.md")
with open(ruta_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md) + "\n")
print("escrito", ruta_md)
print(f"\nresumen: {len(t1)} filas en la tabla por MOEA, {len(t2)} por instancia, "
      f"{len(t3)} celdas donde no gana contra {BASES}")
por_moea = Counter(r["moea"] for r in t3) if False else {}
from collections import Counter
print("celdas donde no gana, por MOEA:", dict(Counter(r["moea"] for r in t3)))
print("operador evaluado:", EVAL, "| bases:", BASES)
