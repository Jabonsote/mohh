# Punto de referencia, cálculo de HV y comparabilidad

Fuente: `AdrianMP1/MOHH-Specialized-Operators@main`,
`src/mohh/evaluation/experiment.py` y `src/mohh/core/utilities/algorithm/MO.py`.

## 1. Punto de referencia (nadir)

`experiment.py:397-399`:

```python
# Compute Nadir point
nadir_point = np.array([1.1, 1.1]) # Fixed for normalized PFAs
#nadir_point = compute_nadir_point(set(best_front))
```

**Es fijo**, no se calcula a partir de los frentes (`compute_nadir_point`
está comentado). El valor `[1.1, 1.1]` corresponde a los 2 objetivos de las
instancias KC. Nuestro arnés usa `np.full(K, NADIR_FACTOR)` con
`NADIR_FACTOR = 1.1` (`fitness.py:693`), idéntico para K=2.

`MO.py:58-66`:

```python
def compute_hypervolume(ref_point, front):
    indicator = HV(ref_point=ref_point)   # pymoo.indicators.hv.HV
    return indicator(front)
```

Es pymoo puro, sin normalización ni transformación adicional.

## 2. Marco de normalización (consolidated)

Por `(instancia, presupuesto)`:

1. `experiment.py:335` — se unen **todos los puntos de todos los frentes**
   de todas las configuraciones y experimentos:
   `consolidated[label] = consolidated[label].union([tuple(p) for p in pareto_front.tolist()])`.
2. `experiment.py:386` — frente no dominado de esa unión:
   `non_dominated_sorting_vectorized(consolidated[label])[0]`.
3. `experiment.py:368-374` — `comprobate`: si el frente tiene ≤2 puntos y
   quedan puntos en el consolidado, se pelan los puntos del frente y se
   recalcula, hasta que el frente tenga >2 puntos o no quede nada.
4. `experiment.py:391-395` — los límites salen de ese frente:
   `min_val, max_val = np.min(best_front, axis=0), np.max(best_front, axis=0)`.

Nuestro equivalente es `_frame_from_points` (`compare_operators.py:479-492`):
`union → np.unique → NDS → comprobate_frame`, y `shared_frame`
(`:570-622`) construye ESE marco con todos los operadores, protocolos y
repeticiones de la comparación. `comprobate_frame` (`fitness.py:785-806`)
replica el bucle de pelado. ✔

## 3. Normalización y HV

`normalize_values` (`experiment.py:171-173`):

```python
return (data - vmin) / (vmax - vmin)
```

Por objetivo, con los `vmin/vmax` del marco. Después
`experiment.py:420-422`:

```python
front = normalize_values(front, boundaries[evals][0], boundaries[evals][1])
hv = compute_hypervolume(nadir_point, front)
```

Nuestro `hv_values` (`compare_operators.py:683-694`) hace exactamente eso:
`normalize_front` (`fitness.py:696-709`) es la misma fórmula y luego
`compute_hypervolume(np.full(K, 1.1), normalized)`. ✔

**Paso extra:** `clip_to_reference_box` (`fitness.py:712-723`) recorta a
`[0, 1.1]`. En el marco GE es **no-op**: los puntos normalized caen en `[0,1]`
porque el marco se construye con esos mismos puntos. Para puntos fuera de la
caja, pymoo ya los ignora (contribuyen 0) y el recorte los deja en 1.1 (también
0). Por tanto el HV coincide con el original.

**Frente mediano de los `.pof`:** `experiment.py:429-435` elige la repetición
cuyo HV está en la posición mediana (`np.argsort(column_data)[n//2]`) y escribe
ese frente. Nuestro `_median_front` hace lo mismo. ✔

**Marcos degenerados:** si `vmax == vmin`, el original produce NaN e
invalida la instancia. Nuestro arnés la excluye y lo reporta en
`degenerate.json` / `hv_problems.json`. ✔

## 4. Comparabilidad (el punto delicado)

El HV **solo es comparable dentro de un mismo experimento**, porque la
normalización depende del marco, y el marco depende de qué operadores entren
en la comparación:

- En la GE original, el marco de una instancia incluye **todas** sus
  configuraciones (Best/Middle/Worst × 3 solvers × 2 mutaciones + CX/PMX).
  Por eso los HV de Best, CX y PMX son comparables entre sí: comparten marco.
- En `llamea_vs_ge_best`, el marco incluye solo `llamea` y `ge_best` (2
  operadores × 6 protocolos). Es el equivalente a un experimento GE de dos
  configuraciones. **Sus HV no son comparables en valor absoluto con los HV
  del Experiment_001/002 de la GE**, aunque el protocolo de cálculo sea
  idéntico.
- Para que dos operadores sean comparables deben evaluarse en la **misma**
  corrida con el mismo `GE_PROTOCOL` (3 solvers, 2 mutaciones, presupuestos
  10k/30k/50k) y el arnés construye un único marco por `(instancia,
  presupuesto)` con todos ellos. Es lo que hace `shared_frame`.
- El arnés además escribe un marco **por pareja** (`pairwise_design`) para el
  cabez a cabeza, equivalente a un experimento GE de dos configuraciones;
  es el análisis que se debe citar para comparar dos operadores.

**Regla práctica:** comparar siempre dentro del mismo directorio de salida
(mismo `NAME`), nunca cruzar HV entre corridas distintas ni contra los CSV de
la GE original. Para contrastar con la GE, hay que reevaluar sus operadores
en la misma corrida (o usar el marco de la pareja).

## 5. Lo que garantiza que el cálculo sea el del original

| Elemento | Original | Arnés |
|---|---|---|
| Nadir | `[1.1, 1.1]` fijo | `[1.1]*K` fijo |
| Unión | todos los puntos, `set` de tuplas | todos los puntos, `np.unique` |
| No dominados | `non_dominated_sorting_vectorized` (con `sign`) | `_nondominated` (mismo `sign`) |
| Pelado | `comprobate` si ≤2 puntos | `comprobate_frame` |
| Límites | min/max por objetivo del mejor frente | igual |
| Normalización | `(f-vmin)/(vmax-vmin)` | `normalize_front` |
| HV | pymoo `HV(ref_point=nadir)` | igual |
| Frente `.pof` | repetición mediana por HV | `_median_front` |
| Marco degenerado | NaN → instancia inválida | exclusión + reporte |
