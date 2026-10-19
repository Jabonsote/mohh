# Diseño de los prompts (basado en evidencia)

Este documento justifica cada parte del prompt que usa la integración LLaMEA de
MOHH-R para generar operadores de cruce sobre mQAP. La idea rectora: en vez de
traducir la gramática a lenguaje natural, se diseña el prompt a partir de las
prácticas que la literatura reporta como efectivas para la **generación
automática de heurísticas/algoritmos con LLMs** y del **conocimiento de
dominio sobre operadores de permutación**.

---

## 1. Qué se quiere generar y cómo se evalúa

El LLM debe producir una función

```python
def crossover(x, y):
    # x, y : permutaciones de range(n)
    # return: una permutación, o una tupla (child1, child2)
```

que se inyecta como operador de cruce del MOEA interno (MOEA/D, NSGA-II o
SMS-EMOA) que resuelve el mQAP. La calidad se mide con el **hipervolumen medio**
del frente de Pareto obtenido sobre las mismas instancias `train` que usa la
búsqueda por gramática de MOHH-R. Esto es exactamente el *evaluate* de FunSearch:
una función de evaluación determinista que puntúa programas candidatos
([Romera-Paredes et al., 2024](referencias.md#funsearch)).

Consecuencia de diseño: el prompt debe describir **el objetivo (hipervolumen)**,
**el contrato exacto** y **las restricciones de validez**, porque el LLM no ve
el evaluador.

---

## 2. Principios y su respaldo

### 2.1 Rol explícito de experto (`ROLE_PROMPT`)

*Evidencia.* LLaMEA antepone una descripción de rol ("*The starting prompt
provides the role description first*") antes de la tarea
([van Stein & Bäck, 2024](referencias.md#llamea)). ReEvo usa un *system prompt*
equivalente ("*You are an expert in the domain of optimization heuristics*")
([Ye et al., 2024](referencias.md#reevo)). El *role prompting* aparece además
como técnica estándar en la taxonomía de
[Schulhoff et al., 2024](referencias.md#prompt-report).

*Aplicación.* `ROLE_PROMPT` define a un experto en computación evolutiva y
optimización combinatoria, especializado en operadores de recombinación para
problemas de permutaciones multiobjetivo, que escribe Python correcto y original.

### 2.2 Tarea precisa: problema + contrato + objetivo + restricciones (`task_prompt`)

*Evidencia.* LLaMEA resume su receta como "*a precise task prompt (including a
sample algorithm, code interface requirements, and mutation instruction)*"
([van Stein & Bäck, 2024](referencias.md#llamea)). FunSearch parte de una
especificación del problema como código (función `evaluate`) más una
implementación inicial y, opcionalmente, un *skeleton* del que solo se evoluciona
la lógica crítica ([Romera-Paredes et al., 2024](referencias.md#funsearch)).

*Aplicación.* El `task_prompt` incluye:
- la definición formal del mQAP (permutación, matrices de flujo `W_k`, distancia
  `D`, `f_k(p) = Σ_i Σ_j W_k[i,j]·D[p[i],p[j]]`) siguiendo
  [Knowles & Corne, 2003](referencias.md#knowles);
- dónde se usará el operador y con qué MOEA interno (parametrizado);
- el **contrato exacto** de `crossover` y las **restricciones duras**
  (permutación válida, solo `numpy`, sin estado global, costo razonable);
- el objetivo: maximizar el hipervolumen y **generalizar** a tamaños/instancias
  no vistas.

> Nota sobre generalización: el propio paper de MOHH-R observa que el operador
> mejor rankeado en entrenamiento **no siempre generaliza** mejor a instancias
> mayores ([Morales-Paredes et al., 2025](referencias.md#mohh)). Por eso el
> prompt enfatiza robustez y no ajuste a una instancia.

### 2.3 Conocimiento de dominio (familias de operadores)

*Evidencia.* ReEvo inyecta explícitamente *domain knowledge* en sus prompts y
reflexiones ([Ye et al., 2024](referencias.md#reevo)). La literatura de
operadores de permutación muestra que distintas familias preservan estructura
distinta y que **ninguna domina** en QAP
([Knowles & Corne, 2003](referencias.md#knowles);
[revisión de operadores de cruce, 2015](referencias.md#crossover-review);
[Misevičius, 2018](referencias.md#misevicius)).

*Aplicación.* El prompt enumera cuatro familias y **qué preservan**:
- orden/precedencia (OX, OBX),
- posición (PBX, PMX),
- adyacencia/enlace (ERX) — relevante en mQAP porque `f_k` es una suma de
  productos por pares,
- ciclos (CX).

y señala la observación empírica de que combinaciones adaptativas generalizan
mejor. Nótese que esto es *conocimiento de dominio*, no una reescritura de la
gramática de MOHH-R.

### 2.4 Pensar primero, codificar después (`output_format_prompt`)

*Evidencia.* EoH coevoluciona **"thoughts" (ideas en lenguaje natural) y código**
y demuestra con ablaciones que usar ambos supera a evolucionar solo código
([Liu et al., 2024](referencias.md#eoh)); en todos sus prompts pide "primero
describir y luego implementar". El *chain-of-thought* mejora el razonamiento
antes de responder ([Wei et al., 2022](referencias.md#cot)).

*Aplicación.* Se pide un campo `# Design idea:` (2-3 frases) **antes** de
`# Description:` (una línea) y del bloque de código. Se mantiene
`# Description:` de una línea porque es lo que parsea LLaMEA
(`extract_algorithm_description`).

El bloque de código se especifica de forma explícita (**arrancar con
`import numpy as np` y definir una función `crossover(x, y)` a nivel de módulo,
no una clase ni un método**). Los modelos pequeños envuelven el operador en una
clase o lo dejan como método; decirlo literalmente reduce rechazos sin cambiar
el espacio de búsqueda, y lo confirma el contrato de evaluación
(`compile_operator`). Esto se verificó en corridas locales
([`pruebas-locales.md`](pruebas-locales.md)).

### 2.5 Ejemplo de referencia acotado (`example_prompt`)

*Evidencia.* FunSearch usa un *skeleton* con estructura previa
([Romera-Paredes et al., 2024](referencias.md#funsearch)). LLaMEA advierte que
un ejemplo puede **sesgar** la búsqueda hacia soluciones parecidas y, por eso,
elige un ejemplo simple ([van Stein & Bäck, 2024](referencias.md#llamea)).

*Aplicación.* Se da **un solo** ejemplo (OX simple) solo para fijar el contrato y
el estilo, con la instrucción explícita de **no copiarlo** y proponer algo
distinto. El conocimiento de familias (2.3) da la "materia prima" para ir más
allá sin mostrar más código que sesgue.

### 2.6 Refinamiento y rediseño entre generaciones (`operators`)

*Evidencia.* LLaMEA construye el prompt de refinamiento con la tarea, el resumen
de la población, el individuo seleccionado (código, descripción, feedback y
error) y la instrucción "*Either refine or redesign to improve the algorithm*"
([van Stein & Bäck, 2024](referencias.md#llamea)). ReEvo aporta reflexiones
(comparar peor/mejor código) y una mutación elitista
([Ye et al., 2024](referencias.md#reevo)). EoH organiza sus estrategias en
exploración (E1/E2) y modificación (M1/M2/M3)
([Liu et al., 2024](referencias.md#eoh)). EvoPrompt y OPRO son evidencia
temprana de que LLM+EA (o LLM-como-optimizador) funciona
([Guo et al., 2024](referencias.md#evoprompt);
[Yang et al., 2024](referencias.md#opro)).

*Aplicación.* Se pasan dos operadores de un solo padre (compatibles con
`n_parents=1`): `refine` (explotación) y `redesign` (exploración). El feedback de
hipervolumen por instancia y los errores ya los aporta la función de evaluación
(`OperatorEvaluator`).

---

## 3. Estructura final y mapeo a LLaMEA

| Slot | Se propaga | Uso |
|---|---|---|
| `role_prompt` | pregunta inicial | rol de experto |
| `task_prompt` | pregunta inicial y de refinamiento | problema + contrato + dominio |
| `example_prompt` | pregunta inicial | ejemplo mínimo (OX) |
| `output_format_prompt` | pregunta inicial y de refinamiento | idea + descripción + código |
| `operators` | refinamiento | `refine` / `redesign` |

LLaMEA concatena `role_prompt + task_prompt + example_prompt + output_format_prompt`
en la inicialización y, en cada refinamiento, la tarea + resumen de población +
individuo + operador + formato. Ver `prompt-final.md` para el texto literal.

---

## 4. Decisiones y descartes

- **No basado en la gramática.** El prompt no menciona la BNF ni los primitivos
  de MOHH-R; describe familias de operadores como conocimiento de dominio. (Los
  primitivos siguen disponibles en el *namespace* de ejecución por
  compatibilidad, pero el prompt no depende de ellos: `primitive_namespace()`.)
- **Sin nombres de funciones de MOHH-R en el prompt.** Se busca un espacio de
  búsqueda abierto (código Python libre), como en EoH/FunSearch/ReEvo.
- **Un ejemplo, no varios.** Menos sesgo de *in-context*
  ([van Stein & Bäck, 2024](referencias.md#llamea)).
- **Objetivo único (hipervolumen).** Coincide con la métrica de MOHH-R; no se
  introduce una métrica nueva.
- **Instrucción de validez explícita.** Los modelos pequeños fallan a menudo
  con la permutación; el contrato lo dice literalmente para reducir fallos.

---

## 5. Riesgos conocidos y mitigaciones

| Riesgo | Mitigación en el diseño |
|---|---|
| El ejemplo sesga hacia OX | Un solo ejemplo + instrucción explícita de novedad + familias en el texto |
| Código inválido (no permutación) | Contrato con "cualquier violación puntúa 0" + feedback por error |
| Sobreajuste a una instancia | Instrucción de generalidad + énfasis en distintos tamaños `n` |
| `random` de Python no determinista | Se exige `np.random.*` y se siembra `np.random` por corrida |
| Modelos ≤1.5B generan poco código válido | Recomendar 7B+ (ver `modelos.md`); el feedback reintenta |
| `numpy`/`typing` sin importar | El *namespace* de ejecución aporta `np`/`numpy` y `List/Tuple/Dict/...` |
| Operador que no termina (`while True`) | `_call_with_timeout` (SIGALRM) acota validación e instancia |
| Respuesta sin `def crossover` | `extract_algorithm_code` lanza `NoCodeException` (evita un `IndexError` interno de LLaMEA) |
| Operador que solo copia a un padre (`return x`, `np.array(x, copy=False)`) | Chequeo de novedad (`recombination_novelty`): 0 automático + anunciado en el prompt; ver §7.2.1 de `cambios.md` |
| Permutación aleatoria que ignora a los padres | Sigue aceptada por contrato (la novedad es alta); se penaliza con el ranking/HV y en el texto del prompt |

Las últimas filas salieron de corridas reales con un modelo pequeño; ver
[`pruebas-locales.md`](pruebas-locales.md).

Las mejoras posteriores (restricciones negativas, ranking en el feedback,
autochequeo, coherencia del rango de `n`, `Design idea` enriquecida y semántica
de los dos descendientes) están documentadas con su justificación en
[`mejoras-prompt.md`](mejoras-prompt.md).

---

## 6. Cómo editar o regenerar los prompts

1. Edita `src/mohh/llamea/prompts.py` (constantes y `_TASK_TEMPLATE`).
2. Regenera el texto de `prompt-final.md` con:

   ```bash
   python -m mohh.llamea.export_prompts docs/llamea-prompts/prompt-final.md
   ```

   (script disponible en `src/mohh/llamea/export_prompts.py`).
3. Verifica con `pytest tests/test_llamea.py -q`.
