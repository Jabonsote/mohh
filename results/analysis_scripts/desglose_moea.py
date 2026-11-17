import os
"""Desglose científico de la comparación LLaMEA vs OX vs PMX (evaluación terminada).

Usa el mismo modulo de estadistica del repo (`mohh.llamea.statistics`) para que
los numeros sean comparables con el resto del pipeline.

Datos: results/evaluations/llamea_vs_ox_pmx_local (test, 20 instancias,
31 repeticiones, protocolo GE completo: 3 solvers x {con,sin} mutacion,
presupuestos 10k/30k/50k).
"""
import csv, os, json, statistics
from collections import defaultdict

import numpy as np
import sys
sys.path.insert(0, "/home/jabonsote/Desktop/MOOHH-R")
from mohh.llamea.statistics import paired_summary, add_holm, minimum_detectable_difference

D = os.environ.get("EVAL_DIR",
    "/home/jabonsote/Desktop/MOOHH-R/results/evaluations/llamea_vs_ox_pmx_local")
SOLVERS = ("MOEAD", "NSGAII", "SMSEMOA")
BUDGETS = ("10k", "30k", "50k")
MUTS = ("NM", "WM")
OPS = ("llamea", "ox", "pmx")

# ---------------------------------------------------------------- carga
# clave: (solver, budget, mut, instancia, repeticion) -> {op: hv}
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
                        parts = col.split("_")
                        op, mut = parts[1], ("WM" if "WM" in parts else "NM")
                        hv = float(v) if v not in ("", "None") else 0.0
                        celdas[(solver, budget, mut, inst, rep)][(op, mut)] = hv

# LosHV del layout del original: una columna por (solver, op, mut, solver).
# Para cada (solver,budget,inst,rep) hay 2 mutaciones; se separan.
def valores(solver, budget, mut, op):
    out = []
    for inst in sorted({k[3] for k in celdas}):
        for rep in range(1, 32):
            v = celdas.get((solver, budget, mut, inst, rep), {}).get((op, mut))
            if v is not None:
                out.append(v)
    return out

def _fmt_p(p):
    if p is None:
        return "-"
    return "<1e-16" if p < 1e-16 else f"{p:.2e}"

# ------------------------------------------------- T1: media por MOEA/config
print("=" * 118)
print("T1. HV MEDIO POR MOEA, MUTACION Y PRESUPUESTO (20 instancias x 31 repeticiones = 620 pares)")
print("     marco compartido por instancia y presupuesto; nadir fijo [1.1]*K; politica de fallo: HV 0")
print("=" * 118)
print(f"{'MOEA':7s} {'mut':4s} {'presup':7s} {'llamea':>9s} {'sd':>8s} {'pmx':>9s} {'sd':>8s} "
      f"{'ox':>9s} {'sd':>8s} {'ll/pmx':>8s} {'ll/ox':>7s}")
filas_t1 = []
for solver in SOLVERS:
    for budget in BUDGETS:
        for mut in MUTS:
            v = {op: valores(solver, budget, mut, op) for op in OPS}
            m = {op: statistics.mean(v[op]) for op in OPS}
            s = {op: statistics.stdev(v[op]) for op in OPS}
            print(f"{solver:7s} {mut:4s} {budget:7s} {m['llamea']:9.4f} {s['llamea']:8.4f} "
                  f"{m['pmx']:9.4f} {s['pmx']:8.4f} {m['ox']:9.4f} {s['ox']:8.4f} "
                  f"{m['llamea']/m['pmx']:7.2f}x {m['llamea']/m['ox']:6.2f}x")
            filas_t1.append((solver, budget, mut, m, v))
print()
print("Resumen por MOEA (promediando mutacion y presupuesto):")
print(f"{'MOEA':7s} {'llamea':>9s} {'pmx':>9s} {'ox':>9s} {'ll/pmx':>8s} {'ll/ox':>7s}")
for solver in SOLVERS:
    sub = [f for f in filas_t1 if f[0] == solver]
    m = {op: statistics.mean([f[3][op] for f in sub]) for op in OPS}
    print(f"{solver:7s} {m['llamea']:9.4f} {m['pmx']:9.4f} {m['ox']:9.4f} "
          f"{m['llamea']/m['pmx']:7.2f}x {m['llamea']/m['ox']:6.2f}x")

# ------------------------------- T2: contraste pareado por MOEA (Holm intra-MOEA)
print()
print("=" * 118)
print("T2. CONTRASTE PAREADO POR MOEA (620 pares por celda: 20 instancias x 31 repeticiones)")
print("     Wilcoxon de rangos con signo; correccion de Holm dentro de cada MOEA; dz de Cohen; IC95 de la diferencia")
print("=" * 118)
for solver in SOLVERS:
    rows = []
    for budget in BUDGETS:
        for mut in MUTS:
            for base in ("ox", "pmx"):
                v = valores(solver, budget, mut, "llamea")
                b = valores(solver, budget, mut, base)
                s = paired_summary(v, b)
                s.update({"moea": solver, "presupuesto": budget, "mut": mut,
                          "baseline": base,
                          "mdd": minimum_detectable_difference(v, b)})
                rows.append(s)
    rows = add_holm(rows)
    print(f"\n--- {solver} ---")
    print(f"{'presup':7s} {'mut':4s} {'vs':4s} {'n':>4s} {'gana':>5s} {'pierde':>6s} "
          f"{'dif. media':>11s} {'IC95':>22s} {'dz':>7s} {'rel.':>7s} {'p':>9s} {'p_Holm':>9s}")
    for r in rows:
        ci = f"[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]"
        print(f"{r['presupuesto']:7s} {r['mut']:4s} {r['baseline']:4s} {r['pairs']:4d} "
              f"{r['wins']:5d} {r['losses']:6d} {r['mean_difference']:+11.4f} {ci:>22s} "
              f"{r['cohens_dz']:7.2f} {r['relative']:+6.1%} {_fmt_p(r['wilcoxon_p']):>9s} "
              f"{_fmt_p(r['wilcoxon_p_holm']):>9s}")

# ------------------------------- T3: tabla estilo GE (rango medio por instancia)
print()
print("=" * 118)
print("T3. TABLA ESTILO GE: rango medio por instancia (1 = mejor), presupuesto 30k, todos los MOEA")
print("     (como la tabla final de la evolucion gramatical: rango medio de los HV por instancia)")
print("=" * 118)
for solver in SOLVERS:
    for mut in MUTS:
        rangos = defaultdict(list)
        for inst in sorted({k[3] for k in celdas}):
            for rep in range(1, 32):
                hvs = {op: celdas.get((solver, "30k", mut, inst, rep), {}).get((op, mut))
                       for op in OPS}
                hvs = {o: v for o, v in hvs.items() if v is not None}
                if len(hvs) < len(OPS):
                    continue
                orden = sorted(hvs, key=lambda o: -hvs[o])
                for pos, o in enumerate(orden, start=1):
                    rangos[o].append(pos)
        if not rangos:
            continue
        print(f"{solver:7s} {mut:4s} " + "  ".join(
            f"{o}: rango medio {statistics.mean(rangos[o]):.3f} (n={len(rangos[o])})"
            for o in ("llamea", "pmx", "ox")))

# ------------------------------- T4: por instancia (30k, media de todo)
print()
print("=" * 118)
print("T4. POR INSTANCIA: HV medio (los 3 MOEA x 2 mutaciones x 3 presupuestos x 31 rep)")
print("=" * 118)
print(f"{'instancia':22s} {'llamea':>9s} {'pmx':>9s} {'ox':>9s} {'rank':>5s} {'ll/pmx':>8s} {'ll/ox':>8s}")
gana_inst = 0
for inst in sorted({k[3] for k in celdas}):
    acc = {op: [] for op in OPS}
    for (solver, budget, mut, i, rep), d in celdas.items():
        if i != inst:
            continue
        for op in OPS:
            if (op, mut) in d:
                acc[op].append(d[(op, mut)])
    m = {op: statistics.mean(acc[op]) for op in OPS}
    orden = sorted(m, key=lambda o: -m[o])
    rank = orden.index("llamea") + 1
    gana_inst += rank == 1
    r1 = f"{m['llamea']/m['pmx']:7.2f}x" if m['pmx'] else "      -"
    r2 = f"{m['llamea']/m['ox']:7.2f}x" if m['ox'] else "      -"
    print(f"{inst:22s} {m['llamea']:9.4f} {m['pmx']:9.4f} {m['ox']:9.4f} {rank:5d} {r1} {r2}")
print(f"\n  LLaMEA es el mejor operador en {gana_inst}/20 instancias")

# ------------------------------- T5: tabla resumen (presupuesto de analisis 30k)
print()
print("=" * 118)
print("T5. RESUMEN GLOBAL (los 6 protocolos, presupuesto 30k, 31 repeticiones x 20 instancias)")
print("=" * 118)
with open(os.path.join(D, "resumen_operadores.csv")) as fh:
    for row in csv.DictReader(fh):
        print(f"  {row['operador']:7s} ({row['tipo']:7s}) media={float(row['media']):.6f}  "
              f"sd={float(row['desv']):.6f}  mediana={float(row['mediana']):.6f}  "
              f"min={float(row['min']):.6f}  max={float(row['max']):.6f}")
print()
with open(os.path.join(D, "contrastes.csv")) as fh:
    print("  contrastes (del script, marco compartido y por pareja):")
    for row in csv.DictReader(fh):
        print(f"    {row['tipo']:16s} {row['operador']:7s} vs {row['baseline']:4s}  "
              f"gana {row['gana']}/{row['pares']}  dif {float(row['dif_media']):+.4f}  "
              f"rel {float(row['relativo']):+.1%}  dz {float(row['dz']):.2f}  "
              f"p_Holm {float(row['p_holm']):.2e}")
print()
with open(os.path.join(D, "evaluation.json")) as fh:
    d = json.load(fh)
print("  estabilidad del rango (31 repeticiones):", json.dumps(d["rank_stability"], ensure_ascii=False))
