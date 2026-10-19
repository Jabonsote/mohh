# Áreas de mejora del prompt y su justificación

Este documento identifica qué se puede mejorar en los prompts de
`src/mohh/llamea/prompts.py`, **por qué** (evidencia publicada y evidencia
local propia) y **cómo verificarlo**. No se deriva de la gramática de MOHH-R:
el prompt describe el problema y el contrato, no el árbol de derivación.

Referencias: claves de [`referencias.md`](referencias.md). Evidencia local:
[`pruebas-locales.md`](pruebas-locales.md).

---

## 1. Restricciones negativas por fallos observados (la mejora de mayor impacto)

**Problema.** La validación de contrato sólo comprueba que la salida sea una
permutación. Eso deja pasar operadores que *parecen* válidos pero dañan la
búsqueda:

| Fallo observado | Efecto real |
|---|---|
| `x[idx] = ...`, `np.random.shuffle(x)`, `x.sort()` | El operador **muta los padres in-place**. En pymoo `x` es una vista del array de la población: corrompe individuos ya evaluados. |
| `return x` / `return y` | No hay recombinación; el cruce degenera en copia. |
| `return np.random.permutation(n)` | Ignora a los padres; es una mutación aleatoria, no un cruce. |

En las corridas locales, varios operadores aceptados terminaron en
`return np.random.permutation(n)` ([`pruebas-locales.md`](pruebas-locales.md),
punto 5).

**Evidencia.** FunSearch y LLaMEA coinciden en que el prompt debe fijar el
contrato de forma explícita porque el modelo no infiere las precondiciones del
harnés [funsearch, llamea]. Las restricciones negativas concretas provienen de
nuestros registros de fallos, no de la literatura.

**Cambio aplicado.** Bloque "Hard requirements" ampliado: prohibición explícita
de mutar `x`/`y` in-place, de devolver un padre sin cambios y de ignorar a los
padres.

**Cómo verificarlo.** Contar en `log.jsonl` los errores de tipo
`crossover failed on an instance` y comparar la tasa de operadores degenerados
(`return np.random.permutation`) antes/después.

---

## 2. Feedback interpretable: el ranking como señal de búsqueda

**Problema.** El feedback anterior decía sólo "Mean hypervolume: X". Un modelo
chico no sabe si X es bueno, malo o mejor que la generación anterior, así que no
puede decidir qué conservar.

**Evidencia.** LLaMEA y ReEvo basan la refinación en el *feedback* de la
evaluación y en la comparación con lo ya visto; la "mutación elitista" de ReEvo
usa explícitamente el mejor individuo como referencia [llamea, reevo]. Los
trabajos de auto-refinamiento con LLM muestran que la señal debe ser
**comparable** para que el modelo pueda corregir [opro, evoprompt].

**Cambio aplicado.** Tres piezas:
1. El `feedback` del fitness ahora empieza con
   `Ranking: this crossover is #R of N evaluated so far ... Best so far: V`.
2. El `task_prompt` explica que la señal es ese ranking normalizado y pide
   "alcanzar el rank 1".
3. Los operadores `refine`/`redesign` indican usar el ranking para decidir qué
   cambio conservar.

**Cómo verificarlo.** Medir la pendiente del mejor HV acumulado y la tasa de
operadores que superan al mejor previo, con y sin ranking en el feedback.

---

## 3. Autochequeo antes de responder

**Problema.** La mayoría de los fallos de los modelos chicos son salidas que no
son permutación (duplicados o valores faltantes). El modelo no verifica su
propio resultado.

**Evidencia.** El razonamiento paso a paso mejora tareas de razonamiento y
reduce errores de ejecución [cot]; EoH pide "primero piensa, luego implementa"
[eoh]. Un chequeo barato (`assert`/construcción por diseño) convierte un error
silencioso en un error detectable.

**Cambio aplicado.** Se pide un autochequeo ("verify on paper: every value
0..n-1 exactly once") y se sugiere un `assert` barato en el código.

**Cómo verificarlo.** Tasa de `ValueError: ... valid permutation` por generación.

---

## 4. Corregir la inconsistencia del rango de `n`

**Problema.** El prompt decía "for ANY n >= 2" y a la vez "the benchmark uses
n = 11..100". Un modelo literal puede intentar casos límite (`n = 2`) que el
núcleo de MOHH-R no soporta (`preserve_segments` entra en bucle infinito), o
ignorar el rango real y hardcodear tamaños.

**Evidencia.** Los prompts con especificaciones contradictorias aumentan la
varianza de salida [prompt-report]. En nuestras pruebas, la validación se tuvo
que restringir a `n = 11, 24, 50` por ese motivo
([`pruebas-locales.md`](pruebas-locales.md)).

**Cambio aplicado.** Se mantiene "cualquier n >= 2" como contrato matemático
pero se aclara que el benchmark usa `n = 11..100` y que debe ser robusto hasta
unos cientos. No se promete soporte para n = 2 en el núcleo.

**Cómo verificarlo.** Que ningún operador generado dependa de un `n` fijo
(buscar constantes tipo `n == 24` en `log.jsonl`).

---

## 5. Enriquecer la "Design idea" (thought + code)

**Problema.** La instrucción anterior pedía "explica el mecanismo", lo que
produce descripciones genéricas ("preserve order") que no ayudan a la
refinación.

**Evidencia.** EoH coevoluciona *thoughts* y código y reporta mejores
resultados cuando el *thought* describe el mecanismo y su razón [eoh]. ReEvo
usa reflexiones para justificar cambios [reevo].

**Cambio aplicado.** La "# Design idea:" debe nombrar (a) la familia o familias
que combina, (b) qué estructura de cada padre preserva y (c) su complejidad
esperada en `n`.

**Cómo verificarlo.** Correlación entre la especificidad de la descripción
(p. ej., longitud o menciones de familias) y el HV del candidato.

---

## 6. Explicar la semántica del segundo descendiente

**Problema.** El contrato permite devolver `(child1, child2)`, pero no explica
que si se devuelve un solo hijo el MOEA usa `child[::-1]` como segundo
descendiente. El modelo no puede aprovechar esa decisión.

**Evidencia.** `LLMOperator` replica la semántica de `HH_Operator`
(`child[::-1]`); los cruces de QAP suelen producir dos hijos complementarios
[crossover-review, misevicius]. La decisión de controlar ambos hijos es una
palanca de diversidad.

**Cambio aplicado.** Se documenta que el retorno puede ser `(c1, c2)` y que
ambos deben ser permutaciones **nuevas**.

**Cómo verificarlo.** Comparar diversidad del frente final (tamaño y dispersión)
entre corridas que devuelven uno vs. dos hijos.

---

## 7. Anclar menos al ejemplo (sesgo de copia)

**Problema.** Con modelos ≤ 1.5B, casi todos los operadores válidos son copias
del OX del ejemplo ([`pruebas-locales.md`](pruebas-locales.md)). El ejemplo fija
el contrato pero también ancla la creatividad.

**Evidencia.** El *best-shot prompting* de FunSearch usa un esqueleto y ejemplos
del mejor programa, no una solución completa; LLaMEA advierte que un ejemplo
demasiado fuerte sesga la población [funsearch, llamea]. El sesgo de copia es
mayor cuanto menor es el modelo [prompt-report].

**Cambio aplicado (parcial).** Se refuerza "do NOT simply copy it" y se pide
nombrar la familia; el ejemplo sigue siendo uno solo para no romper el contrato.
Ya están implementadas las tres variantes para medirlo: `--skeleton-example`
(esqueleto mínimo) y `--no-example` (sin ejemplo; cuidado: `example_prompt=None`
haría que LLaMEA inyecte su propio ejemplo de *random search*, por eso se pasa
cadena vacía).

**Cómo verificarlo.** Ablación de tres prompts (ejemplo completo / esqueleto /
sin ejemplo) con el mismo modelo, semilla y presupuesto; comparar HV y
diversidad de operadores distintos.

**Resultado con 0.5B.** El esqueleto dio 0/30 válidos y "sin ejemplo" escribió
OX de memoria (1 distinto, HV = OX). La conclusión provisional es que, por
debajo de ~3B, el ejemplo completo es imprescindible como plantilla sintáctica
y el anclaje dominante es el conocimiento previo del modelo, no el texto del
ejemplo. Falta repetir la ablación con 7B+.

---

## 8. Alineación con la métrica real (HV normalizado)

**Problema.** El prompt decía "hypervolume" sin aclarar la normalización. El HV
crudo no es comparable entre instancias ni con la búsqueda GE.

**Evidencia.** El proyecto original normaliza cada frente con los límites del
frente consolidado y usa nadir fijo `[1.1]*K`
(`src/mohh/evaluation/experiment.py`); sin eso, el número que ve el LLM no
significa lo mismo entre generaciones [mohh].

**Cambio aplicado.** El fitness usa la misma normalización y el prompt dice
explícitamente que el score es el HV medio normalizado "like the MOHH-R
pipeline". El ranking se calcula sobre ese valor.

**Cómo verificarlo.** Que el HV de un operador de referencia (p. ej. OX) sea
estable entre corridas y del orden del que reporta la búsqueda GE.

---

## Plan de ablación (para justificar cada cambio con datos)

Todas las corridas van a `results/prompts/<variante>/` con el mismo modelo,
semilla, instancias y presupuesto:

| Variante | Qué cambia | Hipótesis |
|---|---|---|
| `baseline` | prompt anterior (sin restricciones negativas ni ranking) | referencia |
| `improved` | prompt actual | menos errores, mejor HV |
| `no_example` | sin ejemplo | menos copias, más diversidad (a costa de validez) |
| `skeleton` | ejemplo reducido a firma | equilibrio validez/diversidad |
| `refine_only` | sin operador `redesign` | más explotación |
| `temp06` | temperatura 0.6 | más diversidad (a costa de validez) |

Métricas por variante: operadores válidos, operadores distintos, mejor HV,
pendiente del mejor acumulado y errores por tipo. `results/*/…/report.md` ya
incluye la tabla; `scripts/analyze_llamea_runs.py` agrega el conteo de válidos y
códigos distintos.

### Primer resultado (0.5B, 30 generaciones)

| Variante | Válidos | Distintos | Mejor HV |
|---|---:|---:|---:|
| `example` | 30/30 | 1 | 0.426177 |
| `refineonly` | 30/30 | 2 | 0.426177 |
| `noexample` (cadena vacía) | 29/30 | 1 | 0.426177 |
| `llameadefault` (ejemplo de LLaMEA) | 24/30 | 6 | 0.102935 |
| `skeleton` | 0/30 | 0 | 0.000000 |
| `temp06` | 2/30 | 1 | 0.108861 |

Con 0.5B la hipótesis del esqueleto **no se cumple**: el modelo necesita el
ejemplo completo como plantilla sintáctica. Y quitar el ejemplo tampoco ayuda:
el modelo escribe OX de memoria (1 operador distinto, HV = OX de la línea
base), así que el anclaje también está en la lista de familias del
`task_prompt`. La tabla no invalida la mejora #7 para modelos grandes, pero sí
muestra que la ablación debe repetirse con 7B+ antes de adoptarla por defecto.
Detalle y lectura en
[`pruebas-locales.md`](pruebas-locales.md#ablaciones-de-prompt-30-generaciones).

---

## Resumen de cambios ya implementados

1. Prohibición de mutar los padres in-place y de devolver copias/permutaciones
   aleatorias.
2. Feedback con ranking (#R de N) y mejor valor hasta ahora.
3. Instrucción de autochequeo de permutación.
4. Rango de `n` coherente con el benchmark.
5. "# Design idea" con familia, estructura preservada y complejidad.
6. Semántica explícita de los dos descendientes.
7. HV normalizado como el proyecto original (y mencionado en el prompt).
8. Operadores `refine`/`redesign` guiados por el ranking.
