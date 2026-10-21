# Comparación justa entre operadores

Este documento explica por qué la tabla de una corrida de LLaMEA **no** es una
comparación entre operadores, cómo se corrigió (`scripts/compare_operators.py`) y
qué dicen los resultados.

## 1. Los cinco fallos de comparación

| # | Fallo | Consecuencia |
|---|---|---|
| 1 | **Marco por corrida.** El HV se normaliza con el frente consolidado de los operadores de referencia *y de todos los candidatos de esa corrida*. | El HV de un candidato depende de los demás candidatos; dos corridas nunca son comparables y la tabla de una corrida favorece al candidato que la definió. |
| 2 | **Una sola semilla por instancia.** El fitness del bucle corre el MOEA una vez por instancia. | La desviación entre semillas es del mismo orden que la diferencia que se quiere detectar. El propio `experiment.py` usa **31** repeticiones por configuración. |
| 3 | **Selección post-hoc.** `llamea_best_operator_final_frame.py` es el argmax del marco final, elegido después de ver todos los candidatos. | Sesgo optimista: el operador reportado no es el que el bucle habría entregado sin mirar atrás (`llamea_best_operator.py`). |
| 4 | **Sin prueba estadística.** Se reportaba "0.511 frente a 0.423". | No dice si la diferencia sobrevive al ruido, ni en cuántas repeticiones gana. |
| 5 | **Contrato de descendientes distinto.** El arnés (`LLMOperator`) usa `child[::-1]` como segundo hijo cuando el operador devuelve uno solo. | Un operador que devuelve dos hijos distintos no es comparable con los clásicos de un solo hijo. |

## 2. El diseño que sí es válido

`scripts/compare_operators.py` corre **N repeticiones independientes** (11 por
defecto) y, para cada par (instancia, repetición), usa la **misma semilla en
todos los operadores**. Así todos parten de la misma población inicial y se
comparan **diferencias pareadas**, no medias independientes (varianza mucho
menor, prueba de Wilcoxon).

El HV se mide en **un solo marco por instancia**: el frente no dominado
consolidado de *todas* las configuraciones (todos los operadores × todas las
repeticiones), normalizado `(f - vmin) / (vmax - vmin)` con nadir fijo
`[1.1] * K`, igual que `mohh/evaluation/experiment.py`. Las instancias sin escala
(`vmin == vmax`) se excluyen y se reportan.

Además se reporta una sección **cabeza a cabeza**: cada pareja se renormaliza con
el marco de *esas dos* configuraciones, que es lo que sería un experimento GE de
dos configuraciones. Hace falta porque un tercer operador muy bueno (aquí
`feedback_b30`) domina la envolvente del marco compartido y aplasta a los demás a
HV ≈ 0, lo que exagera las diferencias relativas.

Los clásicos pueden correrse también en su versión de **dos hijos**
(`classic2:ox` → `(ox(x, y), ox(y, x))`) para cerrar el fallo 5.

La estadística reporta IC95 de la diferencia media y ajusta con Holm la familia
de comparaciones de cada tabla. Un fallo del operador en una instancia cuenta
como HV 0 (`--fail-policy zero`, default) en vez de eliminar el par; `exclude` y
`abort` están disponibles. Para la tabla final, `--ge-protocol` activa el
protocolo completo de `experiment.py` (MOEA/D + NSGA-II + SMS-EMOA, con y sin
`Swap_Mutation`, 300/105 y 11 repeticiones), todo bajo el mismo marco
consolidado.

Salida en `results/evaluations/<nombre>/` (`report.md`, `evaluation.json`,
`fronts.npz`). Reproducir:

```bash
python3 scripts/compare_operators.py \
  --operator b100_ge=results/examples/llamea_found_operators/qwen25coder7b_moead_b100_ge.py \
  --operator feedback_b30=results/examples/llamea_found_operators/qwen25coder7b_moead_b30_feedback.py \
  --operator ox=classic:ox --operator pbx=classic:pbx --operator pmx=classic:pmx \
  --operator cx=classic:cx --operator erx=classic:erx \
  --operator ox2=classic2:ox --operator pbx2=classic2:pbx --operator pmx2=classic2:pmx \
  --operator cx2=classic2:cx --operator erx2=classic2:erx \
  --split test --instances 8 --repeats 11 --save-fronts \
  --name b100_ge_vs_clasicos_11x

# Tabla final con el protocolo GE completo (caro):
python3 scripts/compare_operators.py ... --ge-protocol
```

## 3. Resultados (11 repeticiones, 8 instancias de test, MOEA/D 30/30)

Marco compartido de las 13 configuraciones (media de las 11 repeticiones):

| Operador | Tipo | Media | Desv. | Mediana |
|---|---|---:|---:|---:|
| feedback_b30 | descubierto | 0.252647 | 0.058872 | 0.245308 |
| pbx2 | clásico (2 hijos) | 0.108100 | 0.021042 | 0.116264 |
| pbx | clásico | 0.108003 | 0.020402 | 0.108692 |
| b100_ge | descubierto | 0.107449 | 0.014000 | 0.104508 |
| ox2 | clásico (2 hijos) | 0.100356 | 0.017500 | 0.105994 |
| pmx2 | clásico (2 hijos) | 0.094900 | 0.017942 | 0.092768 |
| ox | clásico | 0.094046 | 0.015195 | 0.095726 |
| pmx | clásico | 0.084704 | 0.018943 | 0.080251 |
| erx | clásico | 0.071390 | 0.009181 | 0.068746 |
| erx2 | clásico (2 hijos) | 0.067519 | 0.010033 | 0.068758 |
| cx | clásico | 0.054396 | 0.020411 | 0.051100 |
| cx2 | clásico (2 hijos) | 0.053396 | 0.012155 | 0.054542 |

Cabeza a cabeza (marco propio de cada pareja, 11 pares):

| Operador | vs | Media op | Media clásico | Gana | Pierde | Relativo | dz | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| b100_ge | ox | 0.219505 | 0.188036 | 9 | 2 | +16.7% | 0.51 | 0.083 |
| b100_ge | pbx | 0.187420 | 0.160673 | 6 | 5 | +16.6% | 0.31 | 0.520 |
| b100_ge | ox2 | 0.228942 | 0.210063 | 8 | 3 | +9.0% | 0.25 | 0.465 |
| b100_ge | pbx2 | 0.195382 | 0.168659 | 8 | 3 | +15.8% | 0.35 | 0.240 |
| b100_ge | pmx | 0.188411 | 0.125618 | 11 | 0 | +50.0% | 1.00 | 0.001 |
| b100_ge | pmx2 | 0.239375 | 0.147767 | 9 | 2 | +62.0% | 1.13 | 0.005 |
| b100_ge | cx | 0.219675 | 0.076799 | 11 | 0 | +186.0% | 2.17 | 0.001 |
| b100_ge | cx2 | 0.215180 | 0.072460 | 11 | 0 | +197.0% | 2.31 | 0.001 |
| b100_ge | erx | 0.226307 | 0.108262 | 11 | 0 | +109.0% | 1.87 | 0.001 |
| b100_ge | erx2 | 0.219768 | 0.098787 | 11 | 0 | +122.5% | 2.11 | 0.001 |
| feedback_b30 | ox | 0.265370 | 0.103403 | 11 | 0 | +156.6% | 2.77 | 0.001 |
| feedback_b30 | pbx | 0.253883 | 0.104741 | 11 | 0 | +142.4% | 2.63 | 0.001 |
| feedback_b30 | ox2 | 0.242884 | 0.087118 | 11 | 0 | +178.8% | 2.32 | 0.001 |
| feedback_b30 | pbx2 | 0.226008 | 0.086654 | 11 | 0 | +160.8% | 2.36 | 0.001 |
| feedback_b30 | pmx | 0.246067 | 0.077482 | 11 | 0 | +217.6% | 3.35 | 0.001 |
| feedback_b30 | cx | 0.243790 | 0.044849 | 11 | 0 | +443.6% | 3.28 | 0.001 |
| feedback_b30 | erx | 0.249153 | 0.066060 | 11 | 0 | +277.2% | 3.45 | 0.001 |

Con 11 pares, el p mínimo alcanzable en Wilcoxon es 0.0010; con 10 comparaciones
por operador conviene leerlo con Bonferroni (0.05 / 10 = 0.005).

### Qué cambia respecto a la conclusión de la corrida

El reporte de la corrida de 100 generaciones decía que su operador superaba a
PBX (0.511 frente a 0.423 en el marco de esa corrida, que incluye al propio
candidato). La comparación pareada dice otra cosa:

- `b100_ge` **no** supera a PBX: 6 de 11 victorias, p = 0.52 (cabeza a cabeza) y
  p = 0.97 en el marco compartido. Está **al nivel de PBX**.
- `b100_ge` supera a OX solo marginalmente (9/11, p = 0.083) y a OX de dos hijos
  no significativamente (8/11, p = 0.465).
- `b100_ge` sí supera con claridad a PMX, CX y ERX (11/11 o 9/11, p ≤ 0.005).
- El operador de la corrida de **30** generaciones con feedback diagnóstico
  (`qwen25coder7b_moead_b30_feedback.py`) gana **11 de 11** a los diez clásicos,
  incluidos los de dos hijos, con `dz` de 2.2 a 4.0. Su ventaja no viene de
  devolver dos hijos.

Es decir: la tabla dentro de la corrida sobrevaloró al candidato de 100
generaciones (y el marco compartido subestima al de 30 al aplastar a los demás a
0, que es justo el motivo de la sección cabeza a cabeza).

### Por qué el operador de 30 generaciones es tan bueno

`qwen25coder7b_moead_b30_feedback.py` combina OBX + PBX + un intercambio de
pares adyacentes + un intercambio de ciclos: es un operador **agresivo** (la
novedad medida respecto a los padres es 0.46–0.60, frente a 0.2–0.4 de los
clásicos). Como el MOEA del proyecto corre con `MO_MUTATION_BOOL = False`, la
exploración depende solo del cruce: un cruce agresivo hace de mutación. Los
clásicos del panel son todos conservadores (orden, posición, mapeo, ciclos,
adyacencia).

Comprobación con `--mutation` (el MOEA muta además del cruce): ver sección 4.

## 4. Robustez (ablation de mutación)

`results/evaluations/b100_ge_vs_clasicos_11x_mut/` corre la misma comparación con
`--mutation` (MOEA/D mutando con `Swap_Mutation` además del cruce), 11
repeticiones y las mismas semillas.

| Operador | vs | Gana | Pierde | Media op | Media clásico | Relativo | dz | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| b100_ge | ox | 11 | 0 | 0.259206 | 0.192475 | +34.7% | 1.98 | 0.001 |
| b100_ge | ox2 | 11 | 0 | 0.258314 | 0.192308 | +34.3% | 2.29 | 0.001 |
| b100_ge | pbx | 9 | 2 | 0.206069 | 0.183251 | +12.5% | 0.60 | 0.042 |
| b100_ge | pbx2 | 7 | 4 | 0.172239 | 0.167112 | +3.1% | 0.11 | 0.465 |
| feedback_b30 | ox | 11 | 0 | 0.248315 | 0.093724 | +164.9% | 3.86 | 0.001 |
| feedback_b30 | ox2 | 11 | 0 | 0.226131 | 0.079075 | +186.0% | 3.45 | 0.001 |
| feedback_b30 | pbx | 11 | 0 | 0.225684 | 0.118073 | +91.1% | 2.25 | 0.001 |
| feedback_b30 | pbx2 | 11 | 0 | 0.229172 | 0.111565 | +105.4% | 2.20 | 0.001 |

La conclusión es la misma: `b100_ge` queda **al nivel de PBX** (con la mutación,
9/11 y p = 0.042 frente a PBX, pero 7/11 y p = 0.465 frente a PBX de dos hijos) y
`feedback_b30` gana 11/11 a todo. Lo que cambia es el margen: la mutación ayuda
más a los clásicos (el margen de `feedback_b30` sobre PBX baja de +142% a +91%) y
`b100_ge` se separa más de OX (de +16.7% a +34.7%).

> Esta ablación no se pudo hacer antes porque `--mutation` era un no-op: el núcleo
> GE tiene `with_mutation` comentado y el arnés cargaba siempre `NullMutation`, de
> modo que con y sin la bandera los frentes salían idénticos. Ya está corregido
> (`Swap_Mutation` inyectada por atributo; ver `cambios.md` §7.7).

## 5. Limitaciones que quedan

1. **El fitness durante la búsqueda sigue con una semilla por instancia.** La
   evaluación final (`--eval-repeats`, 11 por defecto) y `compare_operators.py`
   ya usan repeticiones pareadas, pero el bucle sigue rankeando candidatos con una
   semilla: promediar R repeticiones *dentro de la búsqueda* reduciría el ruido
   del ranking a costa de R× el MOEA interno.
2. **Selección post-hoc.** En esta corrida el argmax del marco final y el mejor
   en línea son el mismo operador *en comportamiento* (los dos archivos solo
   difieren en comentarios y un `assert`; el HV es idéntico a 6 decimales), pero
   eso hay que comprobarlo, no suponerlo. Lo correcto es seleccionar con train y
   validar en un conjunto aparte.
3. **Presupuesto interno.** La comparación usa 30/30 (el del bucle) y la búsqueda
   GE usa 300/105; con más presupuesto las diferencias entre operadores
   conservadores suelen encogerse. Falta repetir la comparación con 300/105.
4. **Instancias de test reutilizadas.** Los 8 conjuntos de test se han usado para
   reportar varias corridas; mientras no se *seleccione* con ellos no hay fuga,
   pero lo limpio es reservar un test virgen para la conclusión final.
5. **Un solo solver.** Todo esto es MOEA/D; la conclusión podría no transferir a
   NSGA-II o SMS-EMOA.

## 6. Marco GE original y reproducción (`scripts/compare_ge_frame.py`)

Para comparar LLaMEA contra los operadores de GE hace falta re-ejecutar **todos**
los operadores con el arnés de la evaluación original (`experiment.py`), no con
el arnés de LLaMEA (`LLMOperator` + marco estacionario). `compare_ge_frame.py`
hace exactamente eso:

- mismo MOEA interno (`MOEAD`, `NSGAII`, `SMSEMOA`) con y sin `Swap_Mutation`;
- `PMX_Cross`/`CX_Cross` de GE (clases exactas) si se piden como
  `classic:pmx`/`classic:cx`;
- un solo marco por instancia: unión de los frentes de **todas** las
  configuraciones y repeticiones, `comprobate` (peeling cuando el frente tiene
  ≤2 puntos), normalización `(f - vmin)/(vmax - vmin)` y nadir fijo `[1.1] * K`,
  sin clipping (idéntico a `experiment.py`);
- comparación pareada por repetición (Wilcoxon + Holm).

**Artefactos** (en `results/evaluations_ge_frame/<nombre>/`), con el mismo
formato que la evaluación original:

| Ruta | Contenido |
|---|---|
| `initial_solutions/{real,permutation}/<instancia>/experiment_NNN.txt` | poblaciones iniciales (los individuos) |
| `initial_solutions/{real,permutation}/<instancia>/seeds.txt` | semillas del MOEA por experimento |
| `operators/*.py`, `operators/*.phenotype.txt`, `operators.json` | los operadores evaluados (LLaMEA y GE) |
| `fronts/<instancia>/best_front_consolidated{,_norm}.pof` | marco GE de la instancia |
| `fronts.npz` | frentes crudos por (protocolo, operador, repetición, instancia) |
| `evaluation.json`, `report.md` | HV por operador/instancia, comparaciones y procedencia |

**Reproducir una evaluación GE ya guardada** (mismos individuos y mismas
semillas), p. ej. `Experiment_001`:

```bash
python3 scripts/compare_ge_frame.py \
  --operator ge1=ge:/ruta/Experiment_001/MOEAD_operators.txt \
  --operator pmx=classic:pmx --operator cx=classic:cx \
  --dataset mqap_original --split test --instances all --repeats 11 \
  --init-from /ruta/Experiment_001/evaluation_results/initial_solutions \
  --reuse-seeds --solver MOEAD,NSGAII,SMSEMOA --mutation both \
  --mo-generations 500 --mo-population-size 105 --budget 50k \
  --name reproduce_ge_experiment_001
```

`--init-from` carga los individuos y `--reuse-seeds` sus semillas
(`seeds.txt` por instancia); el marco y el HV se recalculan con el mismo
procedimiento que `experiment.py`. Con `--instance-names` se puede reproducir
una sola instancia para comprobar contra su CSV original.
