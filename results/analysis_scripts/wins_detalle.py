import os
"""Wins detalladas: por MOEA, mutacion, presupuesto e instancia.

Para cada par (MOEA, presupuesto, mutacion, instancia, repeticion) se comparan
los HV de llamea contra pmx y contra ox (misma semilla, misma poblacion
inicial, mismo frente consolidado -> comparacion pareada).
"""
import csv, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, "/home/jabonsote/Desktop/MOOHH-R")
from mohh.llamea.statistics import paired_summary, add_holm

D = os.environ.get("EVAL_DIR",
    "/home/jabonsote/Desktop/MOOHH-R/results/evaluations/llamea_vs_ox_pmx_local")
SOLVERS = ("MOEAD", "NSGAII", "SMSEMOA")
BUDGETS = ("10k", "30k", "50k")
MUTS = ("NM", "WM")
OPS = ("llamea", "ox", "pmx")

# (solver, budget, mut, inst, rep) -> {op: hv}
celdas = defaultdict(dict)
for solver in SOLVERS:
    for budget in BUDGETS:
        d = os.path.join(D, solver, budget)
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
                        p = col.split("_")
                        op, mut = p[1], ("WM" if "WM" in p else "NM")
                        celdas[(solver, budget, mut, inst, rep)][(op, mut)] = (
                            float(v) if v not in ("", "None") else 0.0)

INSTS = sorted({k[3] for k in celdas})
REPS = range(1, 32)


def bloque(solver, budget, mut, inst):
    """[(llamea, pmx, ox)] por repeticion."""
    out = []
    for rep in REPS:
        c = celdas.get((solver, budget, mut, inst, rep))
        if not c:
            continue
        out.append((c[("llamea", mut)], c[("pmx", mut)], c[("ox", mut)]))
    return out


def cuenta(bloques, base_idx):
    """W/L/E separando los empates por fallo (ambos HV=0) de los empates reales."""
    w = l = t = t0 = 0
    difs = []
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
        difs.append(a - bb)
    return w, l, t, difs, t0


# ---------------------------------------------------------------- T1
print("=" * 120)
print("T1. GANAS AGREGADAS POR MOEA Y MUTACION (todos los presupuestos: 3 x 20 inst x 31 rep = 1860 comparaciones)")
print("=" * 120)
print(f"{'MOEA':7s} {'mut':4s} {'vs':4s} {'gana':>6s} {'pierde':>7s} {'empata':>7s} "
      f"{'empate(0,0)':>12s} {'% gana':>8s} {'dif. media':>11s} {'dz':>6s} {'p_Holm':>10s}")
for solver in SOLVERS:
    filas = []
    for base, bidx in (("pmx", 1), ("ox", 2)):
        bloques = []
        for budget in BUDGETS:
            for inst in INSTS:
                bloques += bloque(solver, budget, mut="NM", inst=inst)
                bloques += bloque(solver, budget, mut="WM", inst=inst)
        w, l, t, difs, t0 = cuenta(bloques, bidx)
        s = paired_summary([b[0] for b in bloques], [b[bidx] for b in bloques])
        s.update({"base": base, "wins": w, "losses": l, "ties": t})
        filas.append(s)
    filas = add_holm(filas)
    for mut in MUTS:
        for r in filas:
            # se recalcula por mutacion
            bloques = []
            for budget in BUDGETS:
                for inst in INSTS:
                    bloques += bloque(solver, budget, mut=mut, inst=inst)
            bidx = 1 if r["base"] == "pmx" else 2
            w, l, t, difs, t0 = cuenta(bloques, bidx)
            s = paired_summary([b[0] for b in bloques], [b[bidx] for b in bloques])
            print(f"{solver:7s} {mut:4s} {r['base']:4s} {w:6d} {l:7d} {t:7d} "
                  f"{t0:12d} {100*w/len(bloques):7.1f}% {s['mean_difference']:+11.4f} "
                  f"{s['cohens_dz']:6.2f} {r['wilcoxon_p_holm']:10.2e}")

# ---------------------------------------------------------------- T2
print()
print("=" * 120)
print("T2. GANAS POR MOEA x PRESUPUESTO (20 inst x 31 rep = 620 comparaciones por celda; NM y WM sumados)")
print("=" * 120)
print(f"{'MOEA':7s} {'presup':7s} {'vs':4s} {'gana':>6s} {'pierde':>7s} {'empata':>7s} "
      f"{'% gana':>8s} {'dif. media':>11s} {'IC95':>22s} {'dz':>6s} {'p_Holm':>10s}")
for solver in SOLVERS:
    filas = []
    for budget in BUDGETS:
        for base, bidx in (("pmx", 1), ("ox", 2)):
            bloques = []
            for inst in INSTS:
                bloques += bloque(solver, budget, "NM", inst)
                bloques += bloque(solver, budget, "WM", inst)
            s = paired_summary([b[0] for b in bloques], [b[bidx] for b in bloques])
            s.update({"budget": budget, "base": base})
            filas.append(s)
    filas = add_holm(filas)
    for r in filas:
        ci = f"[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]"
        print(f"{solver:7s} {r['budget']:7s} {r['base']:4s} {r['wins']:6d} {r['losses']:7d} "
              f"{r['ties']:7d} {100*r['wins']/r['pairs']:7.1f}% {r['mean_difference']:+11.4f} "
              f"{ci:>22s} {r['cohens_dz']:6.2f} {r['wilcoxon_p_holm']:10.2e}")

# ---------------------------------------------------------------- T3
print()
print("=" * 120)
print("T3. POR INSTANCIA: MOEA x mutacion, victorias sobre 93 comparaciones (3 presupuestos x 31 rep)")
print("     W = gana, P = pierde, E = empata   (contra pmx / contra ox)")
print("=" * 120)
for solver in SOLVERS:
    for mut in MUTS:
        print(f"\n--- {solver} / mutacion {mut} ---")
        print(f"{'instancia':22s} {'llamea':>8s} {'pmx':>8s} {'ox':>8s}   "
              f"{'vs pmx W/P/E':>18s} {'%':>6s}   {'vs ox W/P/E':>18s} {'%':>6s}")
        for inst in INSTS:
            bloques = []
            for budget in BUDGETS:
                bloques += bloque(solver, budget, mut, inst)
            ll = [b[0] for b in bloques]
            pm = [b[1] for b in bloques]
            ox = [b[2] for b in bloques]
            wp, lp, tp, _, tp0 = cuenta(bloques, 1)
            wo, lo_, to, _, to0 = cuenta(bloques, 2)
            print(f"{inst:22s} {statistics.mean(ll):8.4f} {statistics.mean(pm):8.4f} "
                  f"{statistics.mean(ox):8.4f}   {f'{wp}/{lp}/{tp}':>18s} "
                  f"{100*wp/len(bloques):5.0f}%   {f'{wo}/{lo_}/{to}':>18s} "
                  f"{100*wo/len(bloques):5.0f}%   (fallos ambos: {tp0}/{to0})")

# ---------------------------------------------------------------- T4
print()
print("=" * 120)
print("T4. CASOS EN LOS QUE NO GANA (contra pmx): celdas (MOEA, presupuesto, mut) donde pierde mas de lo que gana")
print("=" * 120)
print(f"{'MOEA':7s} {'presup':7s} {'mut':4s} {'instancia':22s} {'W':>4s} {'P':>4s} {'E':>4s} "
      f"{'HV llamea':>10s} {'HV pmx':>8s} {'dif':>9s} {'ratio':>7s}")
casos = []
for solver in SOLVERS:
    for budget in BUDGETS:
        for mut in MUTS:
            for inst in INSTS:
                bloques = bloque(solver, budget, mut, inst)
                if not bloques:
                    continue
                w, l, t, difs, _t0 = cuenta(bloques, 1)
                if l > w:
                    ml = statistics.mean([b[0] for b in bloques])
                    mp = statistics.mean([b[1] for b in bloques])
                    casos.append((solver, budget, mut, inst, w, l, t, ml, mp,
                                  ml / mp if mp else float("inf")))
casos.sort(key=lambda c: (c[0], c[1], -c[7]))
for c in casos:
    print(f"{c[0]:7s} {c[1]:7s} {c[2]:4s} {c[3]:22s} {c[4]:4d} {c[5]:4d} {c[6]:4d} "
          f"{c[7]:10.4f} {c[8]:8.4f} {c[7]-c[8]:+9.4f} "
          f"{(f'{c[7]/c[8]:.2f}x' if c[8] else 'inf'):>7s}")
print(f"\n  celdas (MOEA,presup,mut,instancia) donde pierde el mayoría contra pmx: {len(casos)}"
      f" de {len(SOLVERS)*len(BUDGETS)*len(MUTS)*len(INSTS)} posibles")

# ---------------------------------------------------------------- T5
print()
print("=" * 120)
print("T5. RESUMEN POR INSTANCIA (los 3 MOEA x 2 mut x 3 presup x 31 rep = 558 comparaciones por instancia)")
print("=" * 120)
print(f"{'instancia':22s} {'HV llamea':>10s} {'HV pmx':>8s} {'HV ox':>8s} {'rank':>5s} "
      f"{'vs pmx':>16s} {'vs ox':>16s}  {'nota'}")
notas = []
for inst in INSTS:
    bloques = []
    for solver in SOLVERS:
        for budget in BUDGETS:
            for mut in MUTS:
                bloques += bloque(solver, budget, mut, inst)
    ll = [b[0] for b in bloques]
    pm = [b[1] for b in bloques]
    ox = [b[2] for b in bloques]
    wp, lp, tp, _, tp0 = cuenta(bloques, 1)
    wo, lo_, to, _, to0 = cuenta(bloques, 2)
    m = {"llamea": statistics.mean(ll), "pmx": statistics.mean(pm), "ox": statistics.mean(ox)}
    orden = sorted(m, key=lambda o: -m[o])
    rank = orden.index("llamea") + 1
    nota = ""
    if wp < len(bloques) * 0.5:
        nota = f"pierde a pmx en {100*lp/len(bloques):.0f}% de las comparaciones"
    print(f"{inst:22s} {m['llamea']:10.4f} {m['pmx']:8.4f} {m['ox']:8.4f} {rank:5d} "
          f"{f'{wp}/{lp}/{tp}':>16s} {f'{wo}/{lo_}/{to}':>16s}  {nota}")
