# Prompt — Video: el ranking y la evaluación final de 31 repeticiones

> Video animado de 40 s con datos 100 % reales, incrustados. Sin ficheros
> externos. Fuente: `compute_rank` del repo original de GE y
> `results/evaluations/llamea_vs_ge_best/`.

---

## PROMPT

Crea un video animado minimalista, de aproximadamente 40 segundos, que explique **cómo funciona el ranking y la evaluación final de 31 repeticiones** para decidir si un operador de cruce es realmente mejor que otro en mQAP.

Estilo visual: escritorio de programación oscuro, ventanas sencillas, código y tablas legibles, animaciones fluidas y gráficas limpias. Texto mínimo. Sin narración extensa, analogías, personajes ni elementos decorativos.

## Guion de la animación

### Escena 1. Por qué 31 repeticiones (0–5 s)

Ventana con una sola pregunta y su respuesta:

```
One seed can be noise.
Repeat the SAME experiment 31 times.
```

Debajo, la lista de semillas real, desplazándose:

```
base seed 42  →  rep 1 = 42, rep 2 = 43, ... rep 31 = 72
same seed for every operator · same initial population per (instance, rep)
```

Texto final de la escena: `paired comparison`.

### Escena 2. Marco compartido e hipervolumen (5–13 s)

Ventana con los pasos, resaltados uno por uno:

```
for each (instance, budget):
  consolidate ALL fronts of ALL operators and repetitions
  non-dominated sort + comprobate  (peel if <= 2 points)
  vmin, vmax = min/max of that front (per objective)
  normalize: (f - vmin)/(vmax - vmin)
  HV = pymoo.HV(ref_point = [1.1] * K)(normalized_front)
```

A la derecha, un contador que se llena: `20 instances × 3 budgets × 31 reps × 2 protocols = 7440 fronts`. Etiqueta: `one shared frame per (instance, budget)`.

### Escena 3. Dos rankings, una misma idea (13–21 s)

Se abren dos fragmentos de código reales, uno al lado del otro.

Izquierda — ranking **dentro de GE** (`HH_auxiliars.py:6-24`):

```python
sorted_values = sorted(hypervolumes, reverse=True)
ranks = {}
for i, value in enumerate(sorted_values):
    ranks.setdefault(value, []).append(i+1)
average_ranks = {v: sum(r)/len(r) for v, r in ranks.items()}
```

Derecha — ranking de la **evaluación final** (`compare_operators.py:902-926`, `rank_stability`):

```python
for rep in range(n_repeats):
    values = {name: scores[name][rep] for name in names}
    for position, (name, _value) in enumerate(
            sorted(values.items(), key=lambda kv: -kv[1]), start=1):
        ranks[name].append(position)
        if position == 1:
            wins[name] += 1
```

Una sola frase entre ambos: `rank 1 = best · in GE ties share the average position`.

Ejemplo real de empate (fila 10 de `MOEAD/50k/KC11-2fl-1rl.csv`,Experimento 2) aparece brevemente:

```
Best   = 0.6458275283603986   ┐
Middle = 0.6458275283603986   ┘ → both rank 3.5
Worst  = 0.8782180499788266      → rank 1
PMX    = 0.7010592421213068      → rank 2
CX     = 0.5191730211610632      → rank 5
```

### Escena 4. Las 31 repeticiones pareadas (21–28 s)

Tabla minimalista que se llena fila por fila con datos reales de `MOEAD/30k/KC11-2fl-1un.csv`:

```
Rep   Seed   llamea (NM)   ge_best (NM)   Winner
1     42     0.6842453772  0.5549450986   llamea
2     43     ...           ...            llamea
...   ...    ...           ...            ...
31    72     ...           ...            llamea
```

A la derecha, el resultado del ranking de estabilidad:

```
llamea   : 31 / 31 first
ge_best  : 0 / 31 first
minimum detectable difference (95%): 0.006328
```

### Escena 5. El contraste final (28–35 s)

Una tabla limpia con los valores reales de `resumen_operadores.csv`:

```
operator   reps   mean       std
llamea     31     0.475874   0.015508
ge_best    31     0.276445   0.008557
```

Debajo, la fila real de `contrastes.csv`, apareciendo por partes:

```
31 pairs · 31 wins · 0 losses · 0 ties
Δ = +0.199429   95% CI [+0.193101, +0.205757]
relative = +72.1%    dz = 11.56
Wilcoxon p = 9.313e-10    Holm p = 9.313e-10
```

Una sola etiqueta: `signal >> noise`.

### Escena 6. Cierre (35–40 s)

El ciclo completo, con la etapa final resaltada:

```
Codons → Map (BNF) → Phenotype → MOEA/D → HV → rank → 31 paired reps → Wilcoxon + Holm
                                                                        ▲
                                                                    highlighted
```

Texto final:

```
31 paired repetitions turn a rank into evidence.
```

## Datos reales incrustados (verificados en el repo)

- **Ranking GE**: `compute_rank` de `src/mohh/generation/utilities/algorithm/HH_auxiliars.py:6-24` del repo `AdrianMP1/MOHH-Specialized-Operators@main`; usado en `hyperheuristic.py:374` como `ranks = compute_rank(self.population.get_hypervolumes(instance_name))`.
- **Ranking final**: `rank_stability` de `scripts/compare_operators.py:902-931` — rankea operadores dentro de cada repetición por su HV medio sobre instancias y cuenta cuántas veces cada uno queda primero.
- **Semillas**: `base_seed 42`; rep `i` usa semilla `42 + (i-1)` → `42 ... 72` (`seeds.txt` de `llamea_vs_ge_best`).
- **Diseño**: 20 instancias de test, 3 presupuestos (10k/30k/50k), 2 protocolos en esta corrida (MOEAD NM y MOEAD+mut WM), 31 repeticiones → 7440 frentes en `fronts.npz`.
- **HV por repetición**: fila 1 de `MOEAD/30k/KC11-2fl-1un.csv`: `llamea NM 0.6842453772`, `ge_best NM 0.5549450986`, `llamea WM 0.6076597129`, `ge_best WM 0.5337666097`.
- **Resumen**: `resumen_operadores.csv` — llamea `media 0.4758742378`, `desv 0.01550763602`; ge_best `media 0.2764450195`, `desv 0.008556829294`; 31 repeticiones cada uno.
- **Contraste**: `contrastes.csv`, fila `marco_compartido` — `pares=31`, `faltan=0`, `gana=31`, `pierde=0`, `empata=0`, `dif_media=0.1994292184`, `sd_dif=0.01725225432`, `ic95=[0.1931010404, 0.2057573963]`, `relativo=0.7214064436`, `dz=11.55960344`, `p=9.313225746e-10`, `p_holm=9.313225746e-10`.
- **Estabilidad**: `report.md` — llamea 31/31 primero, ge_best 0/31; diferencia mínima detectable 0.006328.
- **Empate real**: `Experiment_002/evaluation_results/MOEAD/50k/KC11-2fl-1rl.csv`, repetición 10 (Best = Middle = 0.6458275283603986 → rank 3.5, verificado con `compute_rank`).

## Instrucciones de implementación

- Genera un video animado real, no una presentación de diapositivas.
- Mantén una sola interfaz de escritorio y transiciones suaves entre ventanas.
- Muestra únicamente el código de ranking, las semillas, el marco compartido, las tablas de HV, los contadores y el contraste estadístico.
- La simulación es determinista: usa exclusivamente los datos incrustados; no requieres ficheros externos ni valores inventados.
- Distingue visualmente el ranking interno de GE (empates promediados) del ranking de estabilidad de la evaluación final (posición por repetición).
- Usa términos técnicos (hipervolumen, rango, repetición pareada, semilla, Wilcoxon, Holm) sin analogías biológicas ni narración.
- No garantices que cada repetición gane: los datos reales muestran 31/31, pero la explicación debe presentarlo como resultado observado, no como promesa.
- No menciones OX en ninguna parte.
- Exporta el resultado como video MP4, resolución 1920 × 1080.

Objetivo final: que cualquier persona con conocimientos básicos de programación entienda, en 40 segundos, que la GE convierte HV en rangos, que la evaluación final repite el experimento 31 veces con semillas idénticas para todos los operadores, y que solo después de promediar y contrastar estadísticamente se puede afirmar que un operador es mejor.

---
