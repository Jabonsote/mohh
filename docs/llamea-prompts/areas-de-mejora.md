# Áreas de mejora del bucle de búsqueda (más allá del prompt)

Este documento complementa [`mejoras-prompt.md`](mejoras-prompt.md). Ahí se
justifican los cambios **de prompt**; aquí se listan las mejoras del **bucle
completo** (modelo, evaluación, señal de recompensa, paralelismo, seguridad y
observabilidad), con la evidencia local que las motiva y una propuesta concreta.

La prioridad se fijó con dos criterios: (a) ¿cambia la conclusión científica?
y (b) ¿puede invalidar una corrida larga sin que nos demos cuenta? Primero lo
que sesga o invalida resultados; después lo que solo cuesta tiempo o dinero.

---

## P0 — Afectan la validez de las conclusiones

### 1. Capacidad del modelo: el 0.5B no busca, copia

**Problema.** Con `Qwen2.5-0.5B-Instruct` el bucle funciona pero no explora.
En 100 generaciones con MOEA/D y SMS-EMOA produjo **un solo operador distinto**,
idéntico al OX del ejemplo. El "mejor HV" coincide exactamente con la línea base
de OX (0.426177 y 0.316706), así que el resultado del LLM **es** el resultado
del operador clásico.

**Evidencia.**
[`pruebas-locales.md`](pruebas-locales.md): 1 distinto en 100 generaciones
(con ejemplo); 1 distinto también sin ejemplo (escribe OX de memoria, HV = OX);
0/30 válidos con esqueleto y 2/30 válidos a temperatura 0.6. Con el Coder 7B
(mismo prompt, misma configuración, 30 generaciones) la tasa de validez va de
**0/30 a 22/30** entre réplicas y, con el marco GE, tres de tres se anclaron en
el OX del ejemplo (mejor HV 0.396–0.425, por debajo de PBX). A 30 generaciones la
varianza de muestreo domina: ninguna conclusión se sostiene con una corrida. Con
**100 generaciones** (`..._b100_ge`) el mismo modelo sale del anclaje: 51/100
válidos, 21 operadores distintos y un OBX con reparación de adyacencia que gana a
todos los clásicos en train (0.511 frente a PBX 0.423) y en test (0.215 frente a
0.209). El presupuesto es, por sí solo, la palanca más grande.

**Propuesta.**
- Usar 7B+ servido con vLLM en el cluster (ya previsto en
  [`slurm/llamea-generate.slrm`](../../slurm/llamea-generate.slrm)).
- Registrar siempre la **tasa de validez** y el **número de operadores
  distintos**; un experimento con 1 distinto no es un experimento de búsqueda.
- Reportar el HV del mejor operador contra la línea base clásica en la misma
  tabla, nunca solo el número absoluto.

**Costo/beneficio.** Alto beneficio, costo de infraestructura ya cubierto. Es la
mejora con más impacto sobre la calidad de lo encontrado.

### 1b. El hipervolumen debe usar el marco de la evaluación GE (implementado)

**Problema.** El HV solo es comparable si todos los frentes se normalizan en el
**mismo marco por instancia**. La evaluación gramatical
(`src/mohh/evaluation/experiment.py`) usa como marco el frente no dominado
consolidado de *todas* las configuraciones evaluadas (todos los operadores ×
repeticiones) y luego normaliza con `(f - vmin) / (vmax - vmin)` hasta el nadir
fijo `[1.1] * K`. La primera versión del bucle usaba solo un panel fijo
(OX/PBX/PMX/CX/ERX) como marco.

**Evidencia.** Medido con las funciones originales sobre las 8 instancias de
train y el mejor operador guardado de
`results/searches/qwen25coder7b_moead_b30_feedback`:

| Operador | Marco = solo panel | Marco GE (envolvente de todo lo evaluado) |
|---|---:|---:|
| OX / PBX / PMX / CX / ERX | 0.426 / 0.505 / 0.390 / 0.149 / 0.254 | 0.174 / 0.224 / 0.225 / 0.092 / 0.103 |
| mejor candidato | **4.433042** | **0.583228** |

Con el marco fijo el HV del candidato no tiene cota (4.43 frente a 0.43–0.50 del
panel) y no se puede poner en la misma tabla que la búsqueda GE. Peor: en una
instancia donde el panel colapsa (`vmin == vmax`, `KC59-2fl-1rl`) el HV de test
salió **2.75e14**, un artefacto de normalización, no un resultado.

**Propuesta (implementada).**
- Marco por instancia = frente no dominado consolidado del panel de referencia
  **más todos los candidatos ya evaluados** (opción `frame="ge"`, por defecto).
- Si un candidato amplía la envolvente, los HV de los candidatos anteriores se
  recalculan desde los frentes cacheados (solo hipervolumen, sin repetir MOEA).
- Instancias sin escala (`vmax == vmin`): se excluyen de la media, como el
  original las invalida con NaN, y se reportan en el feedback y en
  `llamea_best_info.json`.
- `--stationary-frame` conserva el comportamiento del panel fijo para comparar.

**Verificación.** La implementación reproduce **exactamente** los números
calculados con las funciones originales (`normalize_values` +
`non_dominated_sorting_vectorized` + `compute_hypervolume`), y con
`--stationary-frame` reproduce bit a bit la corrida anterior
(candidato 4.433042, OX 0.426177). Detalle en `cambios.md` §7.1.

**Pendiente.** El experimento 3+3 ya está hecho. **No** se observa que el marco GE
aplaste la búsqueda: con el marco fijo del panel (más gradiente) dos de tres
réplicas también se quedaron al nivel del OX, y una tercera sí superó a PBX
(0.546 frente a 0.505 en la caja del panel). Es decir, la diferencia entre
corridas la explica la varianza del muestreo (la tasa de validez del 7B va de
0/30 a 22/30 entre réplicas), no el marco del feedback. Detalle y tablas en
[`pruebas-locales.md`](pruebas-locales.md).

La otra decisión de diseño que queda abierta: el marco GE es el de la gramática
original y sirve para la tabla de una corrida, pero el marco fijo del panel es un
**metro fijo** y por eso es el único que permite comparar candidatos de corridas
distintas entre sí y contra los clásicos.

**Costo/beneficio.** Bajo; es requisito para publicar la comparación contra la
tabla GE. Pendiente relacionado: `--eval-repeats` (§6), porque GE promedia
varias repeticiones por instancia y aquí cada instancia se evalúa una vez.

### 1c. La tabla de una corrida no es una comparación (implementado)

**Problema.** El reporte de una corrida compara al candidato con la línea base en
el marco de *esa* corrida, con **una sola semilla por instancia** y eligiendo al
"mejor" operador con información de todos los candidatos. Nada de eso es una
comparación entre operadores:

1. **Marco por corrida**: el HV de un candidato depende de los demás candidatos
   de la corrida (el marco GE incluye a todos). Dos corridas nunca son
   comparables, y la tabla de una corrida favorece al candidato que la definió.
2. **Una semilla**: la varianza entre semillas del MOEA es del mismo orden que la
   diferencia que se quiere detectar (el propio `experiment.py` usa **31**
   repeticiones por configuración).
3. **Selección post-hoc**: `llamea_best_operator_final_frame.py` es el argmax del
   marco final, elegido después de ver todos los candidatos. Es un sesgo
   optimista; el operador que el bucle entrega sin mirar atrás es
   `llamea_best_operator.py`.
4. **Sin prueba estadística**: "0.511 frente a 0.423" no dice si la diferencia
   sobrevive al ruido, ni cuántas veces gana de 11.
5. **Presupuesto interno distinto**: el bucle usa 30/30 y la búsqueda GE 300/105.

**Arreglo (implementado).** `scripts/compare_operators.py` hace la comparación
que sí es válida: mismos conjuntos, **mismas semillas para todos los operadores**
(pareado), **N repeticiones independientes** (11 por defecto), **un solo marco por
instancia** (el consolidado de todas las configuraciones, como `experiment.py`;
las instancias sin escala se excluyen y se reportan), y una sección
**cabeza a cabeza** donde cada pareja se renormaliza con el marco de esas dos
configuraciones (un "experimento GE de dos configuraciones"), para que un tercer
operador muy bueno no aplaste a los demás a HV ≈ 0. Reporta media, desviación,
mediana, victorias/derrotas, diferencia media, tamaño de efecto (`dz`) y
Wilcoxon pareado.

**Resultado (11 repeticiones, 8 instancias de test).** El operador de la corrida
de 100 generaciones **no** supera a PBX: −1.8% (p = 0.90; gana 6 de 11). Sí supera
a OX, PMX, CX y ERX. El operador de la corrida de 30 generaciones con feedback
diagnóstico (`feedback_b30`) sí gana **11 de 11** contra los cinco clásicos
(p = 0.0010, `dz` = 2.6–3.3; +122% frente a PBX). Es decir: la tabla dentro de la
corrida sobrevaloraba al candidato de 100 generaciones y subestimaba al de 30.
Detalle en [`comparacion-justa.md`](comparacion-justa.md).

**Pendiente.** El fitness *durante* la búsqueda sigue usando una semilla por
instancia (`--eval-repeats`): promediar R repeticiones reduciría el ruido del
ranking a costa de R× el MOEA interno. Y la selección final sigue siendo
post-hoc; lo correcto es seleccionar con train y validar en un conjunto aparte
(punto 3).

### 1d. `--mutation` era un no-op (implementado)

**Problema.** El núcleo GE tiene `with_mutation` **comentado**
(`multiobjective.py:35`) y el arnés LLaMEA cargaba siempre
`solver.load_operator("mutation", "NullMutation")`: la bandera `--mutation` no
hacía nada. Se descubrió al correr una ablación con y sin la bandera: los frentes
salían **idénticos** bit a bit, así que la ablación no medía nada.

**Arreglo.** Con `--mutation` el arnés inyecta `Swap_Mutation` (la mutación de
permutación que usa `experiment.py`, con `MOEAD_MUTATION_PROBABILITY = 0.05`) por
atributo del solver, igual que ya inyecta el cruce: el núcleo no se toca. Test
`test_mutation_flag_is_not_a_noop` (los frentes con y sin mutación ahora difieren).

**Por qué importa para la comparación.** Refuerza la lectura de §1c: como el MOEA
del proyecto corre sin mutación (`MO_MUTATION_BOOL = False`), la única fuente de
exploración es el cruce, y un cruce agresivo (el del operador de 30 generaciones
con feedback) hace de mutación. Comparar con el panel de clásicos conservadores
sin decirlo escondía esa diferencia.

### 2. `exec` de código generado sin aislamiento

**Problema.** `compile_operator` ejecuta el código del LLM con `exec` en el
proceso de búsqueda, sin límite de memoria ni de CPU por evaluación. Un
operador con un bucle infinito o que reserve memoria tumba la corrida; uno
malicioso (o simplemente curioso) puede leer/escribir archivos. Hoy solo hay
timeout por evaluación y captura de excepciones.

**Propuesta.**
- Ejecutar cada evaluación en un subproceso con `resource.setrlimit`
  (CPU, memoria, sin red) y matarlo al expirar.
- Rechazar imports peligrosos (`os`, `sys`, `subprocess`, `open`) en el
  validador estático; permitir solo `numpy` y builtins.
- En el cluster, correr la búsqueda en un contenedor sin credenciales.

**Costo/beneficio.** Medio. No cambia los resultados, pero evita perder corridas
largas y es requisito para ejecutar código de un modelo abierto sin supervisión.

### 3. Ranking global cuando hay varios workers

**Problema.** `HVArchive` vive en el proceso. Con `--max-workers > 1`, cada
worker tiene su propio archivo y el ranking que recibe el LLM ("#3 de 40") es
local al worker, no el global. El feedback deja de ser comparable entre
generaciones.

**Propuesta.**
- Mantener `max_workers=1` en las corridas de reporte (hoy es el default).
- Si se necesita paralelismo, usar un `multiprocessing.Manager` con una lista
  compartida o un servidor de ranking (p. ej. un socket/Redis local) y calcular
  el rank en el proceso principal.

**Costo/beneficio.** Bajo si se documenta y se deja en 1; medio si se quiere
paralelizar de verdad. Ya está anotado en el docstring de `fitness.py`; falta
el candado explícito (`--max-workers > 1` + `--no-shared-rank`).

### 4. Fuga de información por el conjunto de test

**Problema.** `--eval-test` evalúa los mejores operadores sobre test al final,
pero el mejor se elige mirando train. Con 1000 generaciones y muchos operadores
el máximo de train es un estimador optimista; si además se repiten corridas y
se elige la mejor mirando test, test deja de ser test.

**Propuesta.**
- Selección final **solo** con train; test se evalúa una vez y se reporta.
- Si se comparan variantes de prompt/modelo, decidir con train (o con un
  validation split) y usar test solo para la tabla final.
- Documentar en `run.json` que test no participó en la selección.

**Costo/beneficio.** Bajo y evita el error metodológico más caro de corregir
después.

---

## P1 — Afectan la eficiencia de la búsqueda

### 5. Recompensa binaria: válido / inválido (y válido pero degenerado)

**Problema.** La señal es `HV` o `0`. Un operador que falla en la instancia 7
recibe exactamente lo mismo que un `return x`, y el LLM no recibe el diagnóstico
estructurado (solo el texto del error en el feedback). Con modelos chicos, la
mayoría del presupuesto se va en respuestas inválidas (24/30 válidas sin
ejemplo, 0/30 con esqueleto).

**Problema relacionado (evidencia nueva de la corrida de 1000).** La validación
"es una permutación" es demasiado débil: en
`results/searches/qwen05b_moead_noexample_b1000`, 22 de los operadores válidos
distintos resultaron ser, tras el análisis, **`return np.array(x, copy=False)`**
con comentarios copiados de la sección de conocimiento de dominio. Todos
obtuvieron el mismo HV exacto (0.102935), porque si el cruce devuelve siempre el
primer padre el MOEA no recombina nada y el frente final es idéntico. El bucle
gastó ~900 evaluaciones en el mismo operador degenerado sin detectarlo, y el
ranking nunca lo penalizó (todos empatan → todos son "#1 of N").

**Propuesta.**
- **Implementado**: diagnóstico barato antes de la evaluación cara
  (`recombination_novelty`): aplica el candidato a 8 pares aleatorios y exige
  que el hijo **difiera de ambos padres** (distancia de Hamming > 0) y que no
  sea una vista (`np.shares_memory(child, x)`); si no, score 0 con feedback
  explícito `degenerate: returns a parent`. En la corrida de 1000 esto habría
  descartado ~90 % de los candidatos degenerados antes de gastar el MOEA.
- **Implementado**: desempate del ranking por novedad de recombinación, para
  que "válido pero copión" no quede primero cuando todo empata en HV.
- **Implementado**: feedback de validación con diagnóstico
  (`validate_operator`): dice el tamaño que falló, qué se encontró en lugar de
  una permutación (huecos con `-1`, duplicados, faltantes, longitud), si el
  operador **mutó los padres in-place** o devolvió una **vista**, y — cuando el
  código lanza una excepción — la línea exacta del operador que falló
  (el código se compila con nombre propio y se registra en `linecache`).
  Evidencia: con `Qwen2.5-Coder-7B-Instruct` y el mensaje genérico anterior,
  27/30 generaciones recibieron el mismo texto y **0/30 operadores válidos**
  (todas el mismo OX con off-by-one); con el mensaje nuevo el bucle ve el
  `AssertionError` y la línea del `assert`, y mejora la tasa de validez
  (ver [`pruebas-locales.md`](pruebas-locales.md)).
- **Pendiente**: códigos de error legibles por máquina en el log
  (`reason=not_a_permutation`), índice de la primera instancia que falla
  (`first_failing_instance=j`) y el traceback recortado dentro del JSONL.
- **Pendiente**: recompensa intermedia por validación (p. ej. 0.01 si compila y
  devuelve una permutación válida en `n=5`, aunque falle en una instancia real).
- **Pendiente**: reintento automático con el error como contexto antes de
  gastar una generación (1 reintento barato).

**Costo/beneficio.** Medio; puede subir mucho la tasa de validez útil con
modelos chicos, que es el cuello de botella actual. En la corrida de 1000 el
diagnóstico barato habría descartado ~90 % de los candidatos degenerados antes
de gastar el MOEA interno.

### 6. Evaluación con una sola semilla por instancia

**Problema.** Cada operador se evalúa con un `seed` fijo por instancia. La
diferencia entre dos operadores puede ser ruido del MOEA interno (30/30), sobre
todo con HV normalizado y frentes chicos. El ranking que ve el LLM puede
"premiar" suerte.

**Propuesta.**
- `--eval-repeats k` (3–5 semillas) y fitness = media; el ranking usa la media.
- Reportar desviación entre semillas; si es comparable a la diferencia entre
  operadores, la comparación no es concluyente.
- Costo lineal en `k`: con el caché por código, repetir solo afecta a los
  operadores nuevos.

**Costo/beneficio.** Medio-alto; mejora la señal y la credibilidad de la tabla.

### 7. Las líneas base no están dentro del bucle

**Problema.** El ranking es "#1 de N operadores generados", no "#1 de N
incluyendo los clásicos". Un operador puede ser el mejor del LLM y seguir por
debajo de OX/PBX/CX. El LLM no recibe esa referencia.

**Propuesta.**
- Sembrar el `HVArchive` con el HV de OX/PBX/PMX/CX/ERX (ya calculados por
  `scripts/llamea_examples.py`) antes de la primera generación.
- Feedback: `Ranking: #4 of 46 (best classic: PBX 0.5047; you: 0.4262)`.
- Así el LLM sabe si está cerca o lejos, no solo su posición relativa.

**Costo/beneficio.** Bajo (los números ya existen) y alto: convierte el ranking
en una señal absoluta y comparable entre corridas.

### 8. Desajuste de presupuesto: MOEA interno 30/30 vs GE 300/105

**Problema.** El operador se selecciona con un MOEA interno de 30 generaciones
y 30 individuos, mientras que la búsqueda GE original usa 300/105. Un crossover
que brilla a corto plazo puede no hacerlo a largo; el ranking puede estar
midiendo otra cosa que el experimento final.

**Propuesta.**
- Calibrar: correr los operadores clásicos con 30/30 y con 300/105 y ver si el
  orden se conserva. Si no, subir el presupuesto interno (aunque sea con caché
  y menos generaciones de búsqueda).
- Registrar `mo_generations`/`mo_population_size` en `run.json` (ya se hace) y
  no comparar HV entre corridas con presupuestos distintos.

**Costo/beneficio.** Medio; primero hay que medir si el orden cambia (es un
experimento barato con los 5 clásicos).

### 9. Sin criterio de parada por estancamiento

**Problema.** El presupuesto es fijo (`--budget`). En 1000 generaciones con un
modelo que copia el ejemplo se gastan ~1.8k tokens por generación sin mejorar
nada. No hay parada por "sin mejora en k generaciones" ni por presupuesto de
tokens.

**Propuesta.**
- `--patience k` (parar si el mejor HV no mejora en k generaciones) y
  `--max-tokens-total` (presupuesto de costo).
- Registrar en `run.json` el motivo de parada.

**Costo/beneficio.** Bajo; ahorra GPU/tiempo en corridas largas.

---

## P2 — Observabilidad y reproducibilidad

### 10. Caché por código (implementado)

**Problema.** En corridas largas el LLM repite el mismo operador muchas veces
(OX 99 veces de 100). Reevaluar es tirar el MOEA interno.

**Estado.** Implementado en `OperatorEvaluator`: clave = código normalizado;
se reutiliza `score` y `per_instance`, se sigue registrando en `HVArchive` (el
ranking cuenta la repetición) y el feedback indica `(reused cached evaluation)`.
Con 0.5B el ahorro es casi todo el tiempo de evaluación; con 7B+ el ahorro es
menor pero real en las generaciones de refinamiento.

**Cuidado.** El caché es por proceso; no se comparte entre workers (ver punto
3). Si se cambia el presupuesto interno o las instancias, hay que invalidarlo
(hoy se crea un evaluador nuevo por corrida, así que no hay riesgo).

### 11. Tokens y costo (implementado)

**Problema.** No había forma de saber cuánto costaba una corrida ni de comparar
variantes de prompt por costo.

**Estado.** `VLLM_LLM` acumula `usage` por petición; `generation_summary.jsonl`
tiene acumulados y deltas por generación; `report.md` incluye una sección de
tokens y `tokens.json` guarda el histórico. El prompt domina el costo
(~1.5–1.8k tokens por generación con el ejemplo completo), así que acortar el
ejemplo reduce costo y anclaje a la vez.

**Pendiente.** Añadir el costo en dinero (precio por millón de tokens) cuando
se use un proveedor de pago; con vLLM local solo importa el tiempo de GPU.

### 12. Reportes de corridas largas

**Problema.** `evolution.png` y `evolution.csv` con 1000 puntos son difíciles
de leer y el CSV pesa. Además el log por operador (`log.jsonl`) guarda el código
completo de cada intento.

**Propuesta.**
- Graficar con media móvil y marcar solo los cambios de mejor operador.
- Guardar el código completo solo para operadores distintos (deduplicar por
  hash); para los repetidos, una referencia al hash.
- Añadir al reporte "generaciones hasta el mejor" y "generaciones sin mejora".

**Costo/beneficio.** Bajo; mejora la lectura de las corridas de 1000.

### 13. Reproducibilidad entre cluster y local

**Problema.** El resultado depende de la versión del modelo, del servidor
(vLLM vs llama.cpp), de `temperature`, `max_tokens`, del orden de las
peticiones y de la semilla del MOEA. Hoy `run.json` guarda casi todo, pero no
la revisión del modelo ni la versión del servidor.

**Propuesta.**
- Registrar `model_revision` (o hash del snapshot), versión de vLLM y
  `max_tokens` (ya está).
- Fijar `--seed` explícito en las corridas de comparación (hoy hay default).
- Un `--dry-run` que imprima la configuración efectiva y valide el servidor
  antes de gastar GPU.

**Costo/beneficio.** Bajo; imprescindible para publicar.

### 14. Código de módulo sin límite de tiempo (implementado)

**Problema.** El `exec` del código del LLM no tenía timeout. El modelo suele
anexar un *self-check* que llama a `crossover` al importar, y si ese código no
termina — aquí un `while` que buscaba una adyacencia que no existía
(`while abs(child1[j] - child1[(j + 1) % n]) != 1: j = (j + 1) % n`) — el
proceso entero se quedaba girando al 100 % de CPU **sin límite**, sin log y sin
poder recuperarse.

**Evidencia.** `results/searches/qwen25coder7b_moead_b30_st_r3_cuelgue`: la
corrida quedó 17 min en la generación 9 con el proceso en estado `R` (100 % de un
núcleo), la GPU ociosa (`ollama ps` vacío) y `log.jsonl` sin crecer. Los caminos
ya protegidos (`validate_operator`, `recombination_novelty`, cada instancia del
MOEA) tienen límite, así que el cuelgue estaba solo en el `exec` sin acotar.

**Arreglo (implementado).** `compile_operator(code, timeout=COMPILE_TIMEOUT)` usa
`_call_with_timeout` alrededor del `exec` y devuelve al LLM un mensaje accionable
("remove any example/self-check that calls crossover at import time"). El
`_TimeoutError` dentro de una llamada al operador también se reporta con la línea
exacta, que es lo que el modelo necesita para corregir.

**Limitación.** El límite usa `SIGALRM`, que solo funciona en el hilo principal:
con `max_workers > 1` (joblib) el guardia no aplica, así que sigue pendiente el
sandbox por subproceso (punto 7). Tampoco protege código C/numpy que no devuelva
el control a Python.

---

## Hoja de ruta sugerida

| Prioridad | Acción | Esfuerzo | Impacto |
|---|---|---|---|
| 1 | Correr con 7B+ en vLLM y reportar validez/distintos | Bajo | Alto |
| 2 | Feedback contra la línea base clásica (hecho; no se sembró el archivo) | Bajo | Alto |
| 2b | Marco de HV de la evaluación GE (hecho) | Bajo | Alto |
| 2c | Comparación pareada con 11 repeticiones y prueba (hecho) | Bajo | Alto |
| 2d | `--mutation` dejó de ser un no-op (hecho) | Bajo | Alto |
| 3 | Selección final solo con train (+ validation) | Bajo | Alto |
| 4 | `--patience` y presupuesto de tokens | Bajo | Medio |
| 5 | Chequeo de degeneración (hecho) + feedback con diagnóstico (hecho) + 1 reintento | Medio | Alto |
| 6 | `--eval-repeats` (3 semillas) | Medio | Medio |
| 7 | Sandbox de `exec` (subproceso + límites) | Medio | Medio |
| 7b | Timeout del `exec` de código de módulo (hecho) | Bajo | Alto |
| 8 | Calibrar el presupuesto interno 30/30 vs 300/105 | Medio | Medio |
| 9 | Ranking compartido si se paraleliza | Alto | Bajo hoy |
| 10 | Reportes con media móvil y deduplicación de código | Bajo | Bajo |
| 11 | Réplicas marco GE vs marco fijo (hecho: 3+3) | Bajo | Medio |

Las acciones 1–4 son las que cambiarían la conclusión de las corridas actuales;
el resto mejora robustez, costo y presentación.
