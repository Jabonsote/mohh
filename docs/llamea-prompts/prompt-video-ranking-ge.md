# Prompt — Video: el ranking original de GE (`compute_rank`)

> Prompt autocontenido para generar un video animado. Todo dato está incrustado;
> no requiere ficheros externos. Basado en el código real de
> `AdrianMP1/MOHH-Specialized-Operators@main`.

---

## PROMPT

Crea un video animado minimalista, de aproximadamente 40 segundos, que explique **cómo funciona el ranking original de Evolución Gramatical (GE)** —la función `compute_rank`— para seleccionar y mejorar operadores de cruce en un problema de asignación cuadrática multiobjetivo (mQAP).

Estilo visual: escritorio de programación oscuro, ventanas sencillas, código y expresiones legibles, animaciones fluidas y gráficas limpias. Texto mínimo. Sin narración extensa, analogías, personajes ni elementos decorativos.

## Guion de la animación

### Escena 1. Del frente al número (0–5 s)

Ventana pequeña con el problema y la necesidad de comparar operadores:

```
Problem: Multi-objective Quadratic Assignment Problem (mQAP)
Objective k: f_k(p) = sum_i sum_j W_k[i,j] * D[p_i, p_j]
Representation: facility permutation
Question: which crossover operator is better?
```

Debajo aparece la idea central, en una sola línea:

```
raw hypervolumes are not comparable → convert them to RANKS
```

El texto se desplaza hacia una ventana que representa al motor de GE.

### Escena 2. Cómo se obtiene cada hipervolumen (5–13 s)

Ventana con los cuatro pasos reales, uno por uno, resaltados secuencialmente:

```
1. consolidate ALL fronts of ALL operators in this instance
2. non-dominated sort + comprobate (peel if <= 2 points)
3. vmin, vmax = min/max of that front (per objective)
4. normalize: (f - vmin)/(vmax - vmin)
   HV = pymoo.HV(ref_point = [1.1] * K)(normalized_front)
```

A la derecha, una gráfica de HV por presupuesto con los **valores reales del operador Best** (Experimento 2):

```
10k: 0.312
30k: 0.366
50k: 0.386
```

Etiqueta: `fixed nadir [1.1]^K — never computed from the fronts`. Sin explicación adicional.

### Escena 3. La función `compute_rank` (13–21 s)

Se abre el archivo real `src/mohh/generation/utilities/algorithm/HH_auxiliars.py`. El código aparece escrito en pantalla, línea por línea, con resaltado de sintaxis:

```python
def compute_rank(hypervolumes):
    sorted_values = sorted(hypervolumes, reverse=True)
    ranks = {}
    for i, value in enumerate(sorted_values):
        temp = ranks.get(value, [])
        temp.append(i+1)
        ranks[value] = temp
    average_ranks = {v: sum(r)/len(r) for v, r in ranks.items()}
    return [average_ranks[v] for v in hypervolumes]
```

Animación del algoritmo sobre un ejemplo numérico real: cuatro operadores con HV `[0.366, 0.411, 0.312, 0.411]`.

1. Ordenar descendente: `[0.411, 0.411, 0.366, 0.312]`.
2. Posiciones: `[1, 2, 3, 4]`.
3. Empate en `0.411` → promedio `(1+2)/2 = 1.5`.
4. Resultado en el orden original: `[3, 1.5, 4, 1.5]`.

En pantalla, una sola frase: `rank 1 = best · ties share the average position`.

### Escena 4. Dónde se usa el ranking en GE (21–28 s)

Se abre `src/mohh/generation/algorithm/hyperheuristic.py` en la línea real 374. Aparece el fragmento:

```python
ranks = compute_rank(self.population.get_hypervolumes(instance_name))
for j, individual in enumerate(self.population.individuals):
    individual.fitness_values[instance_name] = ranks[j]
    fitness_values[j][i] = ranks[j]
```

Debajo, una tabla minimalista de una generación real con columnas `individual | HV | rank`, rellenándose fila por fila con los valores de la escena 3.

Luego, una ventana de estado con contadores reales de la corrida:

```
Generation: 14 / 30
Offspring created: 35
Parent pairs attempted: 33
Repeated offspring rejected: 26
```

Y, como pie de pantalla, los totales reales:

```
1086 pairs attempted · 912 repeated offspring rejected (84%)
1050 offspring created · 197 invalid
```

### Escena 5. Ranking por instancia y selección (28–35 s)

Una cuadrícula limpia: filas = individuos, columnas = instancias, celdas = rank. Se rellena con números pequeños y se promedia al final de cada fila:

```
mean rank = (rank_inst_1 + rank_inst_2 + ... + rank_inst_N) / N
```

Se resalta la fila con menor media y se etiqueta `selected`. Al lado, un texto breve:

```
lower mean rank wins
GE internal ranking does not always separate Best from Worst:
  Best  : 0.312 / 0.366 / 0.386
  Worst : 0.386 / 0.411 / 0.411   (difference -0.02 at 10k)
```

A continuación, la deriva real de diversidad de fenotipos: `CrystalBLEU 0.81 → 0.17`, con una línea decreciente limpia. Sin comentario editorial.

### Escena 6. Nuevo ciclo (35–40 s)

El nuevo fenotipo vuelve a evaluarse. Aparece el ciclo completo, con la etapa de ranking resaltada:

```
Codons → Map (BNF) → Phenotype → Execute (MOEA/D) → HV → compute_rank → Select → Vary
                                                      ▲
                                                  highlighted
```

Finaliza con el texto:

```
Grammatical Evolution: automated operator design through
grammar-guided code evolution and rank-based selection.
```

## Datos reales de la corrida (incrustados; no requieres ficheros externos)

- **Ranking**: `compute_rank` de `HH_auxiliars.py:6-24` del repo `AdrianMP1/MOHH-Specialized-Operators@main` (orden descendente, posiciones `i+1`, empates promediados). Se usa en `hyperheuristic.py:374`: `ranks = compute_rank(self.population.get_hypervolumes(instance_name))` y se guarda en `individual.fitness_values[instance_name]` (líneas 377-379).
- **Gramática BNF** (idéntica a la ejecutada en la corrida, 8 producciones):

```
<expr>        ::= <blocks> | <binary_func>(<expr>,<expr>)
               | <unary_func>(<expr>) | <var>
<blocks>      ::= <orderbased>
<binary_func> ::= map_list | alternate_elements
<unary_func>  ::= move_right | reverse
<orderbased>  ::= order_based(<fill>,<preserve>,<var>,<expr>)
<fill>        ::= fill_first_occurring
<preserve>    ::= preserve_elements | preserve_segments
<var>         ::= x | y
```

- **Genoma**: primeros 20 codones de un individuo real: `152, 197, 177, 178, 4, 191, 240, 222, 125, 181, 194, 80, 176, 53, 135, 101, 192, 213, 32, 90`. Datos reales del individuo: **120 nodos, 80 codones usados de 100, profundidad del árbol 42**.
- **Fenotipo** (inicio real, truncado):

```
order_based(fill_first_occurring,preserve_elements,y,
  order_based(fill_first_occurring,preserve_elements,x,
    order_based(fill_first_occurring,preserve_elements,y, ...)))
```

- **Pareto**: tres puntos reales del frente del operador Best (el frente completo tiene 13 puntos):
  `(2.154e6, 1.532e7)`, `(2.165e6, 1.516e7)`, `(2.273e6, 1.443e7)`. Completar la curva interpolando monotónicamente entre ellos, sin puntos fuera del rango.
- **Hipervolúmenes**: Best `0.312 / 0.366 / 0.386` (10k/30k/50k); Worst `0.386 / 0.411 / 0.411`.
- **Contadores**: generación real `14/30`; 35 descendientes creados, 33 pares intentados, 26 descendientes repetidos rechazados. Totales: 1086 pares, 912 repetidos, 1050 creados, 197 inválidos.
- **Diversidad**: CrystalBLEU `0.81 → 0.17`.

## Instrucciones de implementación

- Genera un video animado real, no una presentación de diapositivas.
- Mantén una sola interfaz de escritorio y transiciones suaves entre ventanas.
- Muestra únicamente la gramática, los codones, el fenotipo, el HV, el código de `compute_rank`, la tabla de rangos, los contadores, el Pareto y la diversidad.
- La simulación es determinista: usa exclusivamente los datos incrustados; no requieres ficheros externos ni valores inventados.
- Distingue visualmente la fase de mapeo (gramática) de la de ejecución y evaluación (MOEA/D), y resalta la etapa de ranking dentro del ciclo.
- Usa los términos técnicos del método (genoma, codones, fenotipo, hipervolumen, rango, descendientes) sin analogías biológicas ni narración.
- No garantices que cada generación mejore: muestra los valores reales, incluido el caso en que Worst supera a Best (diferencia -0.02).
- No menciones OX en ninguna parte.
- Exporta el resultado como video MP4, resolución 1920 × 1080.

Objetivo final: que cualquier persona con conocimientos básicos de programación entienda, en 40 segundos, que la GE convierte hipervolúmenes en rangos (empates promediados), que esos rangos son la señal de selección dentro de la población, y que ese ranking es el que dirige la evolución del siguiente operador de cruce.

---
