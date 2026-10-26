"""
Analisis del comportamiento de la busqueda: train y test, por generacion.

Este modulo responde a una pregunta distinta de la que responde
`evaluate_31.py`: no compara algoritmos, sino que sigue la BUSQUEDA a lo largo
de las generaciones y mira como se comportan los individuos que produce.

QUE HACE, PASO A PASO

1. Carga una corrida ya terminada de `mohh-generate` (`results/.../Experiment_NNN`).
   Cada individuo trae su frente en TRAIN y sus permutaciones (`solution_sets`).
2. Mide cada individuo en TRAIN con los P* de train.
3. Reevalua las MISMAS permutaciones sobre las instancias de TEST y las mide con
   los P* de test. No se ejecuta ningun algoritmo: solo se vuelven a evaluar las
   soluciones que la busqueda ya encontro. Asi se mide generalizacion de verdad,
   con el mismo objeto y no con una poblacion distinta.
4. Escribe tablas por generacion, por instancia y por generacion x instancia.
5. Dibuja:
   - boxplot de los individuos de cada generacion (dispersion de la poblacion),
   - boxplot train frente a test en escala normalizada,
   - una linea por instancia a lo largo de las generaciones (comportamiento
     entre instancias por generacion),
   - mapa de calor instancia x generacion.
6. Resume la convergencia: dispersion, separacion entre generaciones y pendiente
   en la segunda mitad de la corrida.

POR QUE SE PUEDE EVALUAR EN TEST

En datasets emparejados, train y test son la misma familia con distinta
semilla: el generador KC nombra `KC{n}-{k}fl-{semilla}{tipo}`, y coincide
`(n, k, distribucion)`. Una permutacion optimizada en `KC10-2fl-1rl` es valida
para `KC10-2fl-2rl`; lo que cambian son las matrices.

En datasets held-out (por ejemplo el dataset larger) puede no existir esa
coincidencia exacta. `emparejar_train_test` usa entonces una instancia de train
con el mismo numero de instalaciones `n` (una permutacion de tamano `n` se
puede reevaluar en cualquier instancia con `n` instalaciones) y omite los test
que no tengan ningun train de ese tamano. `summary.json` deja constancia en
`test_skipped` y `pairing`.

COMO LEERLO, Y COMO NO LEERLO

- El HV se mide con una escala congelada por instancia y split, igual que
  `mohh-evaluate`: el frente no dominado consolidado de TODOS los individuos da
  `vmin`/`vmax` y el punto de referencia es 1.1. Sin congelarla, los HVs de
  generaciones distintas no serian comparables y las trayectorias serian ruido.
- Entre train y test NO se comparan valores absolutos: son instancias distintas
  con matriz de distancias distinta. Lo comparable es la TRAYECTORIA: si una
  generacion mejora en train, ¿mejora en test? Si divergen, hay sobreajuste a
  train.
- DELTA_P puede saturar contra su techo de 1.0. Cuando eso ocurre el aviso
  aparece en consola y en `summary.json`, y el indicador no debe usarse para
  decidir nada.

@author jhonnatan
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import re

import numpy as np
import pandas as pd

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_SRC = os.path.join(_RAIZ, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mohh.analysis.indicators_all import (INDICATORS, compute_indicators,
                                          load_references)
from mohh.analysis.io_utils import (iter_experiment_individuals,
                                     load_experiment, write_json)
from mohh.analysis.hv_scale import HvScale, save_hv_scale
from mohh.core.indicators.reference_sets import (
    NADIR_MARGIN, ReferenceSet, pooled_reference_sets)
from analyze_gen import diagnostico_saturacion


class AnalisisSearchError(RuntimeError):
    """Falta algo para poder seguir la busqueda."""


# ---------------------------------------------------------------------------
# Emparejar cada instancia de test con su equivalente de train
# ---------------------------------------------------------------------------


def firma(instancia):
    """
    Devuelve la parte de un nombre de instancia que define su dimension.

    `KC10-2fl-1rl` -> `('KC10', '2fl', 'rl')`. Dos instancias con la misma firma
    tienen el mismo numero de variables, el mismo numero de objetivos y la misma
    distribucion de flujos, luego una permutacion de una es valida en la otra
    aunque las matrices cambien.

    El sufijo de distribucion importa y no se puede ignorar: `rl` son flujos
    aleatorios y `un` flujos uniformes, y son problemas con dificultad distinta.
    Emparejar un `2un` con un `1rl` mediria la solucion contra el problema
    equivocado.

    @param instancia: nombre de la instancia.
    @return: tupla con el tamano, los objetivos y la distribucion.
    """
    partes = instancia.split("-")
    if len(partes) < 3 or not re.fullmatch(r"\d+fl", partes[1]) \
            or not re.fullmatch(r"\d+\w*", partes[2]):
        raise AnalisisSearchError(
            f"nombre de instancia no reconocido: {instancia!r}; se esperaba "
            f"el formato KC{{n}}-{{k}}fl-{{semilla}}{{tipo}}")
    semilla_y_tipo = partes[2]
    return partes[0], partes[1], semilla_y_tipo[1:]


def emparejar_train_test(instancias_train, instancias_test,
                          estricto=True, verbose=False):
    """
    Empareja cada instancia de test con un train compatible.

    Por defecto exige la misma firma (n, k y distribucion), que es el caso de
    los datasets emparejados. Si `estricto=False`, cuando no hay firma exacta
    se intenta emparejar por el mismo numero de instalaciones `n`: una
    permutacion de tamano `n` se puede reevaluar en cualquier instancia con
    `n` instalaciones, aunque `k` o la distribucion cambien. Las instancias
    de test sin train del mismo `n` se omiten en lugar de abortar el
    analisis, para que el mismo codigo sirva para datasets held-out.

    @param instancias_train: nombres de train.
    @param instancias_test: nombres de test.
    @param estricto: si es True, un test sin equivalente exacto lanza
        `AnalisisSearchError` (comportamiento historico).
    @param verbose: imprime los emparejamientos aproximados y las omisiones.
    @return: `{instancia_test: instancia_train}`.
    """
    train_por_firma = {}
    train_por_n = {}
    for nombre in instancias_train:
        train_por_firma.setdefault(firma(nombre), []).append(nombre)
        train_por_n.setdefault(nombre.split("-")[0], []).append(nombre)

    emparejamiento = {}
    sin_pareja = []
    aproximados = []
    for nombre in instancias_test:
        candidatos = train_por_firma.get(firma(nombre))
        if candidatos:
            if len(candidatos) > 1:
                raise AnalisisSearchError(
                    f"varias instancias de train encajan con {nombre}: "
                    f"{sorted(candidatos)}; el emparejamiento seria ambiguo")
            emparejamiento[nombre] = candidatos[0]
            continue

        candidatos_n = sorted(train_por_n.get(nombre.split("-")[0], []))
        if candidatos_n:
            elegido = candidatos_n[0]
            emparejamiento[nombre] = elegido
            aproximados.append((nombre, elegido))
            continue

        sin_pareja.append(nombre)

    if aproximados and verbose:
        print("  AVISO: emparejamiento aproximado por numero de instalaciones:")
        for test, train in aproximados:
            print(f"    {test} -> {train}")

    if sin_pareja:
        mensaje = (
            f"{len(sin_pareja)} instancia(s) de test no tienen train con el "
            f"mismo numero de instalaciones y se omiten: {sin_pareja}")
        if estricto:
            raise AnalisisSearchError(
                f"{len(sin_pareja)} instancia(s) de test no tienen equivalente "
                f"en train por dimension (n, k y distribucion): {sin_pareja}")
        print(f"  AVISO: {mensaje}", flush=True)

    if not emparejamiento:
        raise AnalisisSearchError(
            "ninguna instancia de test pudo emparejarse con train")

    return emparejamiento



# ---------------------------------------------------------------------------
# Carga de problemas y reevaluacion
# ---------------------------------------------------------------------------


def cargar_problemas(rutas, solution_type="permutation", population_size=105):
    """
    Carga las instancias como problemas de pymoo.

    @param rutas: `{nombre: ruta del .txt}`.
    @param solution_type: 'permutation' o 'real'.
    @param population_size: solo se usa para construir el objeto `Instance`.
    @return: `{nombre: QuadraticAssignment}`.
    """
    from mohh.evaluation.params import Params

    # `Instance.load_problem` guarda las soluciones iniciales al cargar, asi que
    # necesita un destino valido o falla antes de devolver el problema.
    Params().update_params({
        "FILE_PATH_INITIAL_SOLUTIONS": os.environ.get(
            "MOHH_TMP_SOLUTIONS", "/tmp/mohh_analysis_search"),
        "POPULATION_SIZE": population_size,
    })

    from mohh.evaluation.problem.instance import Instance

    problemas = {}
    for nombre, ruta in sorted(rutas.items()):
        instancia = Instance(population_size=population_size,
                             solution_type=solution_type)
        instancia.load_problem("qap", ruta, 1)
        problemas[nombre] = instancia.problem
    return problemas


def frentes_de_soluciones(soluciones, problemas, bloque=64):
    """
    Reevalua permutaciones y devuelve el frente no dominado de cada solucion.

    Los valores de objetivo se recalculan porque train y test son problemas
    distintos: un frente de train no dice nada sobre la misma permutacion en
    test.

    @param soluciones: `{instancia: (n, n_variables) ndarray}`.
    @param problemas: `{instancia: problema}`.
    @param bloque: cuantas permutaciones se evaluan de una vez.
    @return: `{instancia: (n_frente, k) ndarray}`.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    frentes = {}
    for instancia, matriz in soluciones.items():
        problema = problemas.get(instancia)
        if problema is None or matriz is None:
            continue

        matriz = np.atleast_2d(np.asarray(matriz))
        if matriz.size == 0:
            continue

        # `QuadraticAssignment` usa la permutacion como indice de las matrices,
        # asi que tiene que llegar como entero y no como float.
        if matriz.dtype.kind == "f" and np.all(matriz == np.round(matriz)):
            matriz = matriz.astype(int)

        filas = []
        for inicio in range(0, matriz.shape[0], bloque):
            filas.append(np.asarray(
                problema.evaluate(matriz[inicio:inicio + bloque]), dtype=float))
        valores = np.vstack(filas)

        frente = valores[NonDominatedSorting().do(
            valores, only_non_dominated_front=True)]

        # Se descartan filas repetidas: el frente debe ser un conjunto de
        # soluciones distintas, no la misma repetida.
        frentes[instancia] = np.unique(frente, axis=0)
    return frentes


def frentes_de_split(experimento, nombres, problemas, emparejamiento, verbose):
    """
    Une los frentes de todos los individuos para un split.

    Para train se toman los frentes ya guardados en la corrida. Para test se
    reevalua la permutacion de cada train en su equivalente de test.

    @param experimento: `Experiment`.
    @param nombres: lista de individuos.
    @param problemas: problemas del split.
    @param emparejamiento: `{test: train}`, o None para train.
    @param verbose: imprimir progreso.
    @return: `{individuo: {instancia: ndarray}}`.
    """
    por_individuo = {}

    for i, nombre in enumerate(nombres):
        individuo = experimento.individuals[nombre]

        if emparejamiento is None:
            por_individuo[nombre] = {
                instancia: np.atleast_2d(np.asarray(frente, dtype=float))
                for instancia, frente in individuo.fronts.items()
                if np.asarray(frente).size
            }
        else:
            # Se reevalua la solucion de train sobre TODAS las instancias de
            # test compatibles con ella.
            por_instancia_test = {}
            for instancia_test, instancia_train in emparejamiento.items():
                soluciones = individuo.solution_sets.get(instancia_train)
                if soluciones is None:
                    continue
                frente = frentes_de_soluciones(
                    {instancia_test: soluciones}, problemas).get(instancia_test)
                if frente is not None and frente.size:
                    por_instancia_test[instancia_test] = frente
            por_individuo[nombre] = por_instancia_test

        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(nombres)} individuos", flush=True)

    return por_individuo


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------


def _frentes_entrenamiento(individuo):
    """Frentes de train de un individuo, sin copias innecesarias."""
    return {instancia: np.atleast_2d(np.asarray(frente, dtype=float))
            for instancia, frente in individuo.fronts.items()
            if np.asarray(frente).size}


def _consolidar(actual, nuevos):
    """Frente no dominado de `actual` + `nuevos` (minimizacion)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    bloques = ([actual] if actual is not None and actual.size else []) + list(nuevos)
    if not bloques:
        return actual
    puntos = np.vstack(bloques)
    indices = np.asarray(
        NonDominatedSorting().do(puntos, only_non_dominated_front=True))
    return puntos[indices]


def _recolectar_escalas_train(ruta_experimento, instancias, verbose=True,
                              chunk=25):
    """
    Primera pasada de train con memoria acotada.

    En vez de guardar los ~1100 frentes de cada instancia, se consolidan por
    lotes de individuos. Devuelve los frentes no dominados consolidados por
    instancia y los minimos/maximos de todos los puntos (para la caja pooled).
    """
    consolidado = {}
    pooled_min = {}
    pooled_max = {}
    buffer = {}
    total = 0

    def volcar():
        for instancia, frentes in list(buffer.items()):
            consolidado[instancia] = _consolidar(
                consolidado.get(instancia), frentes)
        buffer.clear()

    for i, individuo in enumerate(iter_experiment_individuals(
            ruta_experimento, fronts_instances=instancias,
            solution_instances=set())):
        total += 1
        for instancia, frente in _frentes_entrenamiento(individuo).items():
            if frente.size:
                bajo = frente.min(axis=0)
                alto = frente.max(axis=0)
                if instancia in pooled_min:
                    pooled_min[instancia] = np.minimum(
                        pooled_min[instancia], bajo)
                    pooled_max[instancia] = np.maximum(
                        pooled_max[instancia], alto)
                else:
                    pooled_min[instancia] = bajo
                    pooled_max[instancia] = alto
            buffer.setdefault(instancia, []).append(frente)
        if (i + 1) % chunk == 0:
            volcar()
            if verbose and (i + 1) % 200 == 0:
                print(f"    {i + 1} individuos (train, escala)", flush=True)
    volcar()
    return consolidado, pooled_min, pooled_max, total


def _referencias_pooled(referencias, pooled_min, pooled_max):
    """Escala de P* ampliada con los extremos de los frentes medidos."""
    salida = {}
    for nombre, referencia in referencias.items():
        bajo = np.minimum(referencia.front.min(axis=0),
                          pooled_min.get(nombre, referencia.front.min(axis=0)))
        alto = np.maximum(referencia.front.max(axis=0),
                          pooled_max.get(nombre, referencia.front.max(axis=0)))
        span = alto - bajo
        span = np.where(span > 0, span, 1.0)
        nadir = alto + NADIR_MARGIN * span
        salida[nombre] = ReferenceSet(
            name=nombre, front=referencia.front, ideal=bajo,
            nadir=nadir, span=span)
    return salida


def _medir_train_streaming(ruta_experimento, instancias, nombres, born,
                           contextos, referencias, metricas, verbose=True):
    """
    Segunda pasada de train: mide cada individuo y descarta sus frentes.
    """
    filas = []
    for i, individuo in enumerate(iter_experiment_individuals(
            ruta_experimento, fronts_instances=instancias,
            solution_instances=set())):
        frentes = {instancia: frente
                   for instancia, frente in _frentes_entrenamiento(individuo).items()
                   if instancia in contextos}
        if frentes:
            medidas = compute_indicators(
                frentes, references=referencias, metrics=metricas,
                hv_context=contextos, require_references=False)
            generacion = born.get(individuo.name,
                                  individuo.born_in_generation)
            for instancia in medidas.index:
                for metrica in metricas:
                    valor = medidas.loc[instancia, metrica]
                    if valor is None or (isinstance(valor, float)
                                         and np.isnan(valor)):
                        continue
                    filas.append({
                        "individual": individuo.name,
                        "generation": generacion,
                        "instance": instancia,
                        "metric": metrica,
                        "value": float(valor),
                    })
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(nombres)} individuos (train, medida)",
                  flush=True)

    if not filas:
        raise AnalisisSearchError(
            "ningun individuo pudo medirse en train; revisa las referencias")
    return pd.DataFrame(filas)


def escala_hv_por_instancia(frentes_por_individuo, ref_point=1.1):
    """
    Escala de HV congelada por instancia, sobre todos los individuos.

    Es la misma construccion que `mohh-evaluate`: frente no dominado consolidado
    del conjunto a medir, `vmin`/`vmax` de sus extremos y punto de referencia
    1.1 en la escala normalizada. Congelarla es lo que hace comparables los HVs
    de generaciones distintas; reajustarla por generacion produciria
    trayectorias sin significado.

    @param frentes_por_individuo: `{individuo: {instancia: ndarray}}`.
    @param ref_point: punto de referencia normalizado.
    @return: `HvScale`.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    por_instancia = {}
    for frentes in frentes_por_individuo.values():
        for instancia, frente in frentes.items():
            por_instancia.setdefault(instancia, []).append(
                np.atleast_2d(np.asarray(frente, dtype=float)))

    bounds = {}
    for instancia, frentes in por_instancia.items():
        apilado = np.vstack(frentes)
        consolidado = apilado[
            NonDominatedSorting().do(apilado, only_non_dominated_front=True)]
        bounds[instancia] = (consolidado.min(axis=0),
                             consolidado.max(axis=0))

    return HvScale(bounds=bounds, ref_point=ref_point, source="consolidado")


def medir_split(experimento, nombres, frentes_por_individuo, escala,
                referencias, metricas):
    """
    Mide los individuos con las escalas de HV y de referencia ya congeladas.

    Las dos escalas se derivan del MISMO conjunto de frentes (todos los
    individuos del split), asi que son identicas para todos ellos y los valores
    son comparables entre individuos y entre generaciones.

    @param experimento: `Experiment`.
    @param nombres: lista de individuos.
    @param frentes_por_individuo: salida de `frentes_de_split`.
    @param escala: `HvScale` de este split.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param metricas: metricas a calcular.
    @return: (tabla larga, referencias efectivamente usadas).
    """
    contextos = escala.contexts()

    # Escala de las metricas relativas a P*: la caja de la union de P* con
    # todos los frentes que se miden en este split. Se calcula una vez y se
    # reutiliza para todos los individuos, que es lo que hace comparables sus
    # valores. Sin esto, un P* de un solo punto (KC10-2fl-2un, KC20-2fl-2un)
    # deja `span` en 1 y los indicadores salen en la escala cruda de los
    # objetivos: IGD+ del orden de 1e5 y DELTA_P siempre igual a 1.
    frentes_consolidados = {}
    for por_instancia in frentes_por_individuo.values():
        for instancia, frente in por_instancia.items():
            if instancia in contextos:
                frentes_consolidados.setdefault(instancia, []).append(frente)

    referencias_usadas = pooled_reference_sets(referencias,
                                                frentes_consolidados)

    filas = []
    for nombre in nombres:
        frentes = {instancia: frente
                   for instancia, frente in frentes_por_individuo[nombre].items()
                   if instancia in contextos}
        if not frentes:
            continue

        # `require_references=False` porque una instancia puede quedarse sin
        # frente medido y entonces sus indicadores de referencia salen como
        # NaN en vez de abortar el resto del split. HV no depende de P*.
        medidas = compute_indicators(frentes, references=referencias_usadas,
                                     metrics=metricas, hv_context=contextos,
                                     require_references=False)

        generacion = experimento.individuals[nombre].born_in_generation
        for instancia in medidas.index:
            for metrica in metricas:
                valor = medidas.loc[instancia, metrica]
                if valor is None or (isinstance(valor, float)
                                     and np.isnan(valor)):
                    continue
                filas.append({
                    "individual": nombre,
                    "generation": generacion,
                    "instance": instancia,
                    "metric": metrica,
                    "value": float(valor),
                })

    if not filas:
        raise AnalisisSearchError(
            "ningun individuo pudo medirse; revisa que las referencias "
            "correspondan a las instancias del split")

    return pd.DataFrame(filas), referencias_usadas


# ---------------------------------------------------------------------------
# Convergencia
# ---------------------------------------------------------------------------


def resumen_convergencia(largo):
    """
    Como evoluciona la busqueda: dispersion y separacion entre generaciones.

    Cuatro medidas por metrica:
    - `rango_relativo`: amplitud entre la mejor y la peor generacion, en
      porcentaje sobre el valor final. Se acerca a 0 cuando todas las
      generaciones rinden parecido.
    - `dispersion_promedio`: desviacion tipica media de los individuos dentro
      de cada generacion. Si baja, la poblacion se homogeneiza.
    - `paso_entre_generaciones`: separacion media entre la media de una
      generacion y la de la siguiente. Si cae, las generaciones se parecen
      cada vez mas.
    - `pendiente_final`: ajuste lineal sobre la segunda mitad. Su magnitud
      indica si la busqueda seguia moviendse o ya se habia estabilizado.

    @param largo: tabla larga de indicadores.
    @return: DataFrame con una fila por metrica.
    """
    filas = []
    for metrica, grupo in largo.groupby("metric"):
        medias = grupo.groupby("generation")["value"].mean().sort_index()
        dispersiones = grupo.groupby("generation")["value"].std()

        pasos = medias.diff().abs().dropna()

        mitad = len(medias) // 2
        x = np.arange(mitad, len(medias), dtype=float)
        y = medias.to_numpy()[mitad:]
        pendiente = float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else float("nan")

        ultimo = float(medias.iloc[-1])
        amplitud = float(medias.max() - medias.min())

        filas.append({
            "metric": metrica,
            "rango_relativo": amplitud / abs(ultimo) if ultimo else float("nan"),
            "dispersion_promedio": float(dispersiones.mean()),
            "paso_entre_generaciones": float(pasos.mean()) if len(pasos) else float("nan"),
            "pendiente_final": pendiente,
            "generaciones": len(medias),
        })

    return pd.DataFrame(filas).sort_values("metric").reset_index(drop=True)


def tabla_por_generacion(largo):
    """
    Resume los indicadores por generacion: media, mediana y dispersion.

    @param largo: tabla larga.
    @return: DataFrame con una fila por (metrica, generacion).
    """
    return (largo.groupby(["metric", "generation"])["value"]
            .agg(["mean", "median", "std", "min", "max", "count"])
            .reset_index())


def tabla_por_instancia(largo):
    """
    Resume los indicadores por instancia y generacion: una trayectoria por
    instancia, que es la vista de comportamiento entre instancias.

    @param largo: tabla larga.
    @return: DataFrame con una fila por (metrica, instancia, generacion).
    """
    return (largo.groupby(["metric", "instance", "generation"])["value"]
            .agg(["mean", "median", "std", "count"])
            .reset_index())


def tabla_matriz(largo, metricas):
    """
    Pasa la vista por instancia a matriz instancia x generacion, que es lo que
    necesita el mapa de calor.

    @param largo: tabla larga.
    @param metricas: metricas a incluir.
    @return: dict metrica -> DataFrame indexada por instancia.
    """
    tablas = {}
    for metrica in metricas:
        grupo = largo[largo["metric"] == metrica]
        if grupo.empty:
            continue
        tablas[metrica] = (grupo.groupby(["instance", "generation"])["value"]
                           .mean().unstack("generation"))
    return tablas


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def guardar(figura, ruta):
    """
    Guarda una figura y la cierra.

    @param figura: figura de matplotlib.
    @param ruta: fichero de salida.
    """
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def figura_boxplot_por_generacion(largo, metricas, directorio, split):
    """
    Boxplot de los individuos de cada generacion.

    Si la busqueda converge, la caja se estrecha con las generaciones y el
    Bigote deja de crecer. Una caja que no se estrecha indica que la poblacion
    sigue explorando, no que la busqueda falle.

    @param largo: tabla larga.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    @param split: 'train' o 'test'.
    """
    for metrica in metricas:
        grupo = largo[largo["metric"] == metrica]
        if grupo.empty:
            continue

        generaciones = sorted(grupo["generation"].unique())
        datos = [grupo.loc[grupo["generation"] == g, "value"].to_numpy()
                 for g in generaciones]

        figura, eje = plt.subplots(figsize=(max(7, len(generaciones) * 0.45), 4.5))
        caja = eje.boxplot(datos, tick_labels=[str(g) for g in generaciones],
                           patch_artist=True,
                           showfliers=False, widths=0.7)
        for elemento in caja["medians"]:
            elemento.set_color("black")
            elemento.set_linewidth(1.2)
        caja["boxes"][0].set_facecolor("#9ecae1")

        eje.set_title(f"{metrica}")
        eje.set_xlabel("generacion de nacimiento")
        eje.set_ylabel(metrica)
        eje.grid(axis="y", alpha=0.3)

        guardar(figura, os.path.join(
            directorio, f"cajas/generacion_{split}_{metrica}.png"))


def figura_train_test(largo, metricas, directorio):
    """
    Boxplot train frente a test, en escala normalizada por instancia.

    Train y test son instancias distintas, asi que sus valores absolutos no se
    comparan. Se normaliza cada valor dentro de su instancia y de su metrica
    (0 = minimo de esa instancia, 1 = maximo) para comparar la FORMA de las
    distribuciones: si la caja de test esta mas abajo, el entrenamiento no
    esta transfiriendo.

    @param largo: tabla larga con las filas de ambos splits.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    """
    for metrica in metricas:
        grupo = largo[largo["metric"] == metrica]
        if grupo.empty or grupo["split"].nunique() < 2:
            continue

        normalizado = grupo.copy()
        limites = grupo.groupby("instance")["value"].transform(
            lambda s: s.max() - s.min())
        normalizado["norm"] = (grupo["value"] -
                               grupo.groupby("instance")["value"].transform("min")) \
            / limites.replace(0, np.nan)

        datos, etiquetas, colores = [], [], []
        paleta = {"train": "#3182bd", "test": "#e6550d"}
        for split in ("train", "test"):
            valores = normalizado.loc[normalizado["split"] == split,
                                      "norm"].dropna().to_numpy()
            if valores.size:
                datos.append(valores)
                etiquetas.append(split)
                colores.append(paleta[split])

        if not datos:
            continue

        figura, eje = plt.subplots(figsize=(5.5, 4.5))
        caja = eje.boxplot(datos, tick_labels=etiquetas, patch_artist=True,
                           widths=0.55, showfliers=False)
        for parche, color in zip(caja["boxes"], colores):
            parche.set_facecolor(color)

        eje.set_title(f"{metrica} train-test")
        eje.set_ylabel("valor normalizado (0-1)")
        eje.grid(axis="y", alpha=0.3)

        guardar(figura, os.path.join(directorio,
                                     f"cajas/train_test_{metrica}.png"))


def mejor_por_generacion(largo):
    """
    El mejor individuo de cada generacion: el minimo, porque todos los
    indicadores se minimizan.

    Es la trayectoria de la mejor solucion encontrada hasta cada generacion. La
    media la arrastran todos los individuos de esa generacion, incluidos los que
    ya no aportan nada, asi que baja mas despacio y no deja distinguir si la
    busqueda sigue avanzando o si se estanca. El minimo si lo dice.

    @param largo: tabla larga.
    @return: DataFrame con una fila por (split, metrica, instancia,
        generacion), o por (metrica, instancia, generacion) si la tabla no trae
        la columna `split`.
    """
    claves = ["metric", "instance", "generation"]
    if "split" in largo.columns:
        # Imprescindible cuando se pasan los dos splits juntos: sin `split` en
        # el groupby, la generacion 3 de train y la de test acabarian en la
        # misma fila y el boxplot compararia particiones que no se corresponden.
        claves.insert(0, "split")

    return (largo.groupby(claves)["value"]
            .min().reset_index().rename(columns={"value": "mejor"}))


def figura_boxplot_generacion_train_test(largo, metricas, directorio):
    """
    Train y test en la misma caja, generacion a generacion, con el mejor
    individuo de cada una.

    Las dos particiones son instancias distintas, asi que sus valores no se
    comparan en crudo. Se normaliza cada valor dentro de su instancia (0 = su
    minimo, 1 = su maximo) y se compara la FORMA: si la caja de test baja menos o
    se aplana mas que la de train, el entrenamiento no esta transfiriendo. Es
    el unico sitio donde se ven las dos particiones en el mismo eje de
    generaciones, que es lo que hace falta para leer la brecha.

    @param largo: tabla larga con las filas de ambos splits.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    """
    mejores = mejor_por_generacion(largo)
    paleta = {"train": "#3182bd", "test": "#e6550d"}

    for metrica in metricas:
        parte = mejores[mejores["metric"] == metrica]
        if parte.empty or parte["split"].nunique() < 2:
            continue

        normalizado = parte.copy()
        limites = parte.groupby("instance")["mejor"].transform(
            lambda s: s.max() - s.min())
        normalizado["norm"] = (parte["mejor"] -
                               parte.groupby("instance")["mejor"]
                               .transform("min")) / limites.replace(0, np.nan)

        # Una instancia cuyo mejor valor no cambia entre generaciones tiene
        # rango 0 y al normalizar sale NaN, asi que se queda fuera de la caja.
        # Para HV esto es frecuente (en las instancias `un` el mejor HV es 0 en
        # todas las generaciones, porque ningun individuo aporta volumen) y si
        # no se dice, el boxplot parece completo cuando se ha comido la mitad
        # de los datos.
        perdidos = int(normalizado["norm"].isna().sum())
        if perdidos:
            print(f"  nota: {metrica} train-test, {perdidos} de {len(parte)} "
                  f"valores fuera de la caja por instancia sin rango "
                  f"(constante entre generaciones)", flush=True)

        generaciones = sorted(parte["generation"].unique())
        posiciones = np.arange(len(generaciones), dtype=float)
        ancho = 0.38
        series = {}

        for offset, split in ((-ancho / 2 - 0.01, "train"),
                              (ancho / 2 + 0.01, "test")):
            datos = []
            for g in generaciones:
                valores = normalizado.loc[
                    (normalizado["split"] == split) &
                    (normalizado["generation"] == g), "norm"].dropna()
                datos.append(valores.to_numpy())
            if any(d.size for d in datos):
                series[split] = (posiciones + offset, datos)

        if len(series) < 2:
            continue

        figura, eje = plt.subplots(
            figsize=(max(7, len(generaciones) * 0.5), 4.5))
        for split, (posiciones_split, datos) in series.items():
            caja = eje.boxplot(datos, positions=posiciones_split, widths=ancho,
                               patch_artist=True, showfliers=False,
                               manage_ticks=False)
            for parche in caja["boxes"]:
                parche.set_facecolor(paleta[split])
            for mediana in caja["medians"]:
                mediana.set_color("black")
                mediana.set_linewidth(1.1)

        eje.set_xticks(posiciones)
        eje.set_xticklabels([str(g) for g in generaciones])
        eje.set_title(f"{metrica} mejor train-test")
        eje.set_xlabel("generacion")
        eje.set_ylabel("normalizado")

        mangos = [Line2D([], [], color=paleta[s], linewidth=8, label=s)
                  for s in ("train", "test") if s in series]
        eje.legend(handles=mangos, fontsize=8)
        eje.grid(axis="y", alpha=0.3)

        guardar(figura, os.path.join(
            directorio, f"cajas/generacion_train_test_{metrica}.png"))


def _normaliza_por_instancia(matriz):
    """
    Normaliza cada fila a [0, 1] dentro de su propia instancia.

    Se normaliza por instancia porque los valores absolutos de una instancia
    aplastan visualmente a las demas. Una instancia constante (rango 0) no tiene
    nada que normalizar: se deja en 0, que es su valor, en vez de en NaN, que
    la hacia desaparecer de la figura sin avisar. Es el caso de las instancias
    `un` con HV, donde el mejor valor es 0 en todas las generaciones.

    @param matriz: DataFrame instancia x generacion.
    @return: (matriz normalizada, numero de instancias constantes).
    """
    rango = matriz.max(axis=1) - matriz.min(axis=1)
    constante = rango <= 0
    normalizado = matriz.sub(matriz.min(axis=1), axis=0).div(
        rango.replace(0, np.nan), axis=0)
    normalizado = normalizado.mask(constante, 0.0)
    return normalizado, int(constante.sum())


def matriz_generaciones(largo, metrica):
    """
    Matriz instancia x generacion con la media de los individuos.

    Es la matriz que necesita el test de Friedman: cada fila es un bloque
    (instancia) y cada columna un metodo (generacion), de modo que todas las
    generaciones tienen valor en todas las instancias y los rangos son
    comparables entre instancias de escalas distintas.

    @param largo: tabla larga.
    @param metrica: indicador.
    @return: DataFrame instancia x generacion.
    """
    grupo = largo[largo["metric"] == metrica]
    if grupo.empty:
        return pd.DataFrame()
    return grupo.pivot_table(index="instance", columns="generation",
                             values="value", aggfunc="mean")


def matriz_instancias(largo, metrica):
    """
    Matriz individuo x instancia con el valor de cada uno.

    La inversa de la anterior: aqui los bloques son los individuos (1100, que
    es de donde sale la potencia del test) y los metodos son las instancias.
    Es lo que responde que instancias son estadisticamente mas o menos
    distintas entre si.

    @param largo: tabla larga.
    @param metrica: indicador.
    @return: DataFrame individuo x instancia.
    """
    grupo = largo[largo["metric"] == metrica]
    if grupo.empty:
        return pd.DataFrame()
    return grupo.pivot_table(index="individual", columns="instance",
                             values="value", aggfunc="mean")


def figura_diferencias_criticas_juntas(largo, metricas, directorio):
    """
    Los mismos diagramas CD con train y test juntos.

    Son los que interesan para el objetivo real del experimento: si el operador
    mejora en train y no en test, el CD conjunto lo enseña. Si train y test
    estuvieran separados no se podria ver que instancias de train y de test se
    comportan igual, que es justamente lo que
    se quiere ver de la generalizacion.

    @param largo: tabla larga con los dos splits.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    """
    from mohh.analysis.stats import critical_difference_diagram

    for metrica in metricas:
        matriz = matriz_generaciones(largo, metrica)
        if not matriz.empty and matriz.shape[0] >= 2 and matriz.shape[1] >= 2:
            ruta = os.path.join(directorio, "generaciones",
                                f"junto_{metrica}.png")
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            critical_difference_diagram(matriz, metric=metrica, title=metrica,
                                        path=ruta)

        matriz = matriz_instancias(largo, metrica)
        if not matriz.empty and matriz.shape[1] >= 2:
            ruta = os.path.join(directorio, "instancias", f"junto_{metrica}.png")
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            critical_difference_diagram(matriz, metric=metrica, title=metrica,
                                        path=ruta)


def figura_diferencias_criticas(largo, metricas, directorio, split):
    """
    Diagramas de diferencias criticas de Nemenyi.

    Dos familias, porque el diagrama necesita bloques y metodos y aqui las dos
    particiones dan para las dos:

    - `generaciones/<split>_<metrica>`: metodos = generaciones, bloques =
      instancias. Dice en que generacion la mejora del promedio pasa a ser
      estadisticamente significativa.
    - `instancias/<split>_<metrica>`: metodos = instancias, bloques =
      individuos. Dice que instancias son estadisticamente mas o menos distintas
      entre si, que es la pregunta de las particiones con escalas distintas.

    Un CD por instancia suelta NO se puede dibujar: con una sola instancia hay
    un unico bloque y el CD de Nemenyi sale tan grande que ninguna diferencia
    resulta significativa. Se avisa en vez de escribir 22 diagramas vacios.

    @param largo: tabla larga del split.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    @param split: 'train' o 'test'.
    """
    from mohh.analysis.stats import critical_difference_diagram

    for metrica in metricas:
        matriz = matriz_generaciones(largo, metrica)
        if not matriz.empty and matriz.shape[0] >= 2 and matriz.shape[1] >= 2:
            ruta = os.path.join(directorio, "generaciones",
                                f"{split}_{metrica}.png")
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            critical_difference_diagram(matriz, metric=metrica, title=metrica,
                                        path=ruta)
        elif not matriz.empty:
            print(f"  nota: {metrica} sin CD de generaciones en {split}: "
                  f"hacen falta 2 instancias y 2 generaciones y hay "
                  f"{matriz.shape[0]} y {matriz.shape[1]}", flush=True)

        matriz = matriz_instancias(largo, metrica)
        if not matriz.empty and matriz.shape[1] >= 2:
            ruta = os.path.join(directorio, "instancias",
                                f"{split}_{metrica}.png")
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            critical_difference_diagram(matriz, metric=metrica, title=metrica,
                                        path=ruta)


def figura_trayectorias(largo, tablas, metricas, directorio, split):
    """
    Trayectorias por instancia. Dos versiones de la misma vista:

    - `trayectorias_<split>_<metrica>`: la media de cada instancia mas la linea
      de la media entre instancias. Es la vista con la que ya se leia el
      analisis, y no se toca.
    - `trayectorias_mejor_<split>_<metrica>`: el mejor individuo de cada
      generacion, sin la linea de media. Es la que no se aplana cuando el resto
      de la poblacion empeora, asi que separa "la busqueda encontro algo mejor"
      de "la media se movio".

    @param largo: tabla larga del split.
    @param tablas: dict metrica -> matriz instancia x generacion (medias).
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    @param split: 'train' o 'test'.
    """
    mejores = mejor_por_generacion(largo)

    for metrica in metricas:
        matriz = tablas.get(metrica)
        if matriz is None or matriz.empty:
            continue

        normalizado, constantes = _normaliza_por_instancia(matriz)
        if constantes:
            print(f"  nota: {metrica} trayectorias {split}, "
                  f"{constantes} instancia(s) constantes dibujadas en 0",
                  flush=True)

        figura, eje = plt.subplots(figsize=(8, 4.5))
        instancias = list(normalizado.index)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0, vmax=max(1, len(instancias) - 1))
        for posicion, instancia in enumerate(instancias):
            serie = normalizado.loc[instancia]
            eje.plot(serie.index, serie.to_numpy(), marker="o", markersize=3,
                     linewidth=1.2, alpha=0.8,
                     color=cmap(norm(posicion)))

        medias = normalizado.mean(axis=0)
        eje.plot(medias.index, medias.to_numpy(), color="black", linewidth=2.4,
                 label="media")

        eje.set_title(f"{metrica} media")
        eje.set_xlabel("generacion")
        eje.set_ylabel("normalizado")
        eje.grid(alpha=0.3)
        # 80 instancias no caben en una leyenda: colorbar ordenada.
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = figura.colorbar(sm, ax=eje, pad=0.015, fraction=0.035)
        cbar.set_label("instancia (orden alfabético)", fontsize=7)
        eje.legend(handles=[Line2D([], [], color="black", linewidth=2.4,
                                   label="media")], loc="best", fontsize=8)

        guardar(figura, os.path.join(directorio,
                                     f"trayectorias/{split}_{metrica}.png"))

        # Version sin media: el mejor individuo de cada generacion.
        parte = mejores[mejores["metric"] == metrica]
        if parte.empty:
            continue

        bruto = parte.pivot(index="instance", columns="generation",
                            values="mejor")
        norm_mejor, constantes_mejor = _normaliza_por_instancia(bruto)
        if constantes_mejor:
            print(f"  nota: {metrica} trayectorias mejor {split}, "
                  f"{constantes_mejor} instancia(s) constantes dibujadas en 0",
                  flush=True)

        figura, eje = plt.subplots(figsize=(8, 4.5))
        instancias = list(norm_mejor.index)
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(vmin=0, vmax=max(1, len(instancias) - 1))
        for posicion, instancia in enumerate(instancias):
            serie = norm_mejor.loc[instancia]
            eje.plot(serie.index, serie.to_numpy(), marker="o", markersize=3,
                     linewidth=1.2, alpha=0.8,
                     color=cmap(norm(posicion)))

        eje.set_title(f"{metrica} mejor")
        eje.set_xlabel("generacion")
        eje.set_ylabel("normalizado")
        eje.grid(alpha=0.3)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = figura.colorbar(sm, ax=eje, pad=0.015, fraction=0.035)
        cbar.set_label("instancia (orden alfabético)", fontsize=7)

        guardar(figura, os.path.join(
            directorio, f"trayectorias/mejor_{split}_{metrica}.png"))


def figura_trayectorias_por_instancia(largo, metricas, directorio, split):
    """
    Una subgrafica por instancia, con su media y su mejor individuo.

    Las trayectorias agregadas esconden que hay instancias que no avanzan: al
    normalizar por instancia todas arrancan en 0 y solo se ve el promedio. Con
    un panel por instancia se ve cuales se estacan.

    @param largo: tabla larga del split.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    @param split: 'train' o 'test'.
    """
    mejores = mejor_por_generacion(largo)

    for metrica in metricas:
        grupo = largo[largo["metric"] == metrica]
        if grupo.empty:
            continue

        instancias = sorted(grupo["instance"].unique())
        if not instancias:
            continue

        columnas = min(4, len(instancias))
        filas = int(np.ceil(len(instancias) / columnas))
        figura, ejes = plt.subplots(filas, columnas,
                                    figsize=(3.2 * columnas, 2.6 * filas),
                                    squeeze=False, sharex=True)

        for indice, instancia in enumerate(instancias):
            eje = ejes[indice // columnas][indice % columnas]
            parte = grupo[grupo["instance"] == instancia]
            medias = parte.groupby("generation")["value"].mean().sort_index()

            mejor = mejores[(mejores["metric"] == metrica) &
                            (mejores["instance"] == instancia)]
            mejor = mejor.set_index("generation")["mejor"].sort_index()

            eje.plot(medias.index, medias.to_numpy(), color="#3182bd",
                     linewidth=1.4, label="media")
            if not mejor.empty:
                eje.plot(mejor.index, mejor.to_numpy(), color="#e6550d",
                         linewidth=1.4, label="mejor")

            eje.set_title(instancia, fontsize=8)
            eje.grid(alpha=0.3)
            if indice == 0:
                eje.legend(fontsize=7)

        for indice in range(len(instancias), filas * columnas):
            ejes[indice // columnas][indice % columnas].axis("off")

        figura.suptitle(f"{metrica} instancias", fontsize=11)
        figura.supxlabel("generacion")
        figura.supylabel("valor")
        figura.tight_layout()

        guardar(figura, os.path.join(
            directorio, f"trayectorias/instancia_{split}_{metrica}.png"))


def figura_mapa_calor(tablas, metricas, directorio, split):
    """
    Mapa de calor instancia x generacion.

    De una vistazo muestra donde falla la busqueda: una fila que se apaga antes
    que las demas es una instancia donde el MOEA no esta estabilizando.

    @param tablas: dict metrica -> matriz instancia x generacion.
    @param metricas: metricas a dibujar.
    @param directorio: carpeta de figuras.
    @param split: 'train' o 'test'.
    """
    for metrica in metricas:
        matriz = tablas.get(metrica)
        if matriz is None or matriz.empty:
            continue

        datos = matriz.to_numpy(dtype=float)
        figura, eje = plt.subplots(
            figsize=(max(7, matriz.shape[1] * 0.3), max(3, matriz.shape[0] * 0.4)))
        imagen = eje.imshow(datos, aspect="auto", cmap="viridis")
        figura.colorbar(imagen, ax=eje, label=f"{metrica} (media)")

        eje.set_xticks(range(matriz.shape[1]))
        eje.set_xticklabels([str(c) for c in matriz.columns], fontsize=7,
                             rotation=90)
        eje.set_yticks(range(matriz.shape[0]))
        eje.set_yticklabels(matriz.index, fontsize=7)
        eje.set_title(f"{metrica} calor")
        eje.set_xlabel("generacion")

        guardar(figura, os.path.join(directorio,
                                     f"calor/{split}_{metrica}.png"))


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------


def validar_referencias(referencias, minimo=1, verbose=True):
    """
    Comprueba que los P* cargados no estan vacios y avisa si alguno es un punto.

    Un P* con un solo punto NO invalida los indicadores: desde la escala
    poolingada (`pooled_reference_sets`) la caja de normalizacion sale de la
    union de P* con los frentes medidos, de modo que IGD+, R2 y DELTA_P siguen
    estar bien definidos y en escala ~[0, 1]. Antes esta funcion descartaba esas
    instancias y se quedaban solo con HV; ahora la regla es la misma para todas
    y no se descarta ninguna.

    Lo que si es un error real es un P* vacio, porque entonces no hay ningun
    punto de referencia contra el que medir.

    @param referencias: dict nombre -> `ReferenceSet`.
    @param minimo: numero minimo de puntos; 1 = acepta P* de un punto.
    @param verbose: avisar por pantalla.
    @return: (referencias validas, lista de nombres descartados).
    """
    validas = {}
    descartadas = []
    unitarios = []

    for nombre, referencia in referencias.items():
        if referencia.size < minimo:
            descartadas.append(nombre)
        elif referencia.size == 1:
            unitarios.append(nombre)
            validas[nombre] = referencia
        else:
            validas[nombre] = referencia

    if unitarios and verbose:
        print(f"  nota: {len(unitarios)} P* con un unico punto "
              f"({', '.join(sorted(unitarios))}). Un P* de un punto significa "
              f"que el ground set del que sale tiene un punto; ese caso solo se "
              f"acepta aqui si esta confirmado. Si no lo esta, la referencia "
              f"puede estar incompleta (ver extract.py). La escala de "
              f"normalizacion se amplia a los valores medidos en vez de salir "
              f"solo de P* (asi IGD+/R2/DELTA_P no salen en escala bruta y "
              f"los cuatro indicadores se calculan tambien ahi).",
              flush=True)

    if descartadas and verbose:
        print(f"  AVISO: {len(descartadas)} P* vacio(s) en este split "
              f"({', '.join(sorted(descartadas))}). Sin puntos de referencia no "
              f"se puede calcular IGD+/R2/DELTA_P.", flush=True)

    return validas, sorted(descartadas)


def preparar_split(nombres_instancias, directorio_datos, directorio_referencias,
                   etiqueta, verbose=True):
    """
    Carga referencias y problemas de un split.

    @param nombres_instancias: nombres de las instancias del split.
    @param directorio_datos: carpeta con los `.txt` de las instancias.
    @param directorio_referencias: carpeta con los `P*`.
    @param etiqueta: 'train' o 'test', para mensajes.
    @return: (referencias, problemas).
    """
    faltantes = [nombre for nombre in nombres_instancias
                 if not os.path.isfile(os.path.join(directorio_datos,
                                                    f"{nombre}.txt"))]
    if faltantes:
        raise AnalisisSearchError(
            f"faltan {len(faltantes)} fichero(s) de instancia de {etiqueta} en "
            f"{directorio_datos}: {faltantes[:5]}")

    referencias = load_references(directorio_referencias,
                                   instance_names=list(nombres_instancias))
    referencias, descartadas = validar_referencias(referencias, verbose=verbose)
    problemas = cargar_problemas({
        nombre: os.path.join(directorio_datos, f"{nombre}.txt")
        for nombre in nombres_instancias})

    return referencias, problemas, descartadas


def ejecutar(ruta_experimento, salida, metricas=tuple(INDICATORS),
             datos_train="datasets/mqap/train", datos_test="datasets/mqap/test",
             referencias_train="set-reference/train/reference_sets",
             referencias_test="set-reference/test/reference_sets",
             figuras=True, verbose=True):
    """
    Analiza una corrida completa en train y test.

    @param ruta_experimento: carpeta `Experiment_NNN` de una corrida.
    @param salida: carpeta donde escribir tablas y figuras.
    @param metricas: metricas a calcular.
    @param datos_train: carpeta con los `.txt` de train.
    @param datos_test: carpeta con los `.txt` de test.
    @param referencias_train: carpeta con los P* de train.
    @param referencias_test: carpeta con los P* de test.
    @param figuras: dibujar las figuras.
    @param verbose: imprimir progreso.
    @return: diccionario con el resumen.
    """
    if not os.path.isdir(ruta_experimento):
        raise AnalisisSearchError(
            f"no existe el experimento: {ruta_experimento}")

    os.makedirs(salida, exist_ok=True)
    os.makedirs(os.path.join(salida, "00_config"), exist_ok=True)
    figuras_dir = os.path.join(salida, "07_figures")
    tablas_dir = os.path.join(salida, "01_tables")
    os.makedirs(tablas_dir, exist_ok=True)

    inicio = time.time()

    # Los P* se llaman `<instancia>_ref.pof`; el nombre de la instancia es la
    # parte anterior a ese sufijo.
    test_names = sorted(
        os.path.splitext(f)[0].removesuffix("_ref")
        for f in os.listdir(referencias_test)
        if f.endswith("_ref.pof"))
    train_names = sorted(
        os.path.splitext(f)[0] for f in os.listdir(datos_train)
        if f.endswith(".txt"))

    # El emparejamiento se calcula antes de cargar el experimento para poder
    # conservar en memoria sólo las permutaciones que se van a reevaluar.
    emparejamiento = emparejar_train_test(
        train_names, test_names, estricto=False, verbose=verbose)
    train_necesarios = set(emparejamiento.values())

    # Nombres y generacion de nacimiento sin cargar frentes ni permutaciones.
    nombres = []
    born = {}
    for individuo in iter_experiment_individuals(
            ruta_experimento, metadata_only=True):
        nombres.append(individuo.name)
        born[individuo.name] = individuo.born_in_generation
    nombres = sorted(nombres)
    if not nombres:
        raise AnalisisSearchError(f"el experimento no tiene individuos: "
                                  f"{ruta_experimento}")

    instancias_train = list(train_names)
    faltan = sorted(train_necesarios - set(instancias_train))
    if faltan:
        raise AnalisisSearchError(
            f"el experimento no contiene las instancias de train emparejadas: "
            f"{faltan}")

    if verbose:
        print(f"  {len(nombres)} individuos, {len(instancias_train)} instancias "
              f"de train, {len(emparejamiento)} de test")

    largos = {}
    escalas = {}
    escalas_referencia = {}
    descartados_por_split = {}

    # --- train: dos pasadas, sin retener los 1100 individuos completos -----
    if verbose:
        print("  [train] cargando referencias", flush=True)
    referencias_train_set = load_references(
        referencias_train, instance_names=list(instancias_train))
    referencias_train_set, descartadas = validar_referencias(
        referencias_train_set, verbose=verbose)
    descartados_por_split["train"] = descartadas

    if verbose:
        print("  [train] primera pasada (escala)", flush=True)
    consolidado, pooled_min, pooled_max, total = _recolectar_escalas_train(
        ruta_experimento, instancias_train, verbose=verbose)
    if total != len(nombres):
        raise AnalisisSearchError(
            f"se esperaban {len(nombres)} individuos y se leyeron {total}")

    bounds = {instancia: (frente.min(axis=0), frente.max(axis=0))
              for instancia, frente in consolidado.items()}
    escala_train = HvScale(bounds, source="consolidado")
    escalas["train"] = escala_train
    referencias_usadas_train = _referencias_pooled(
        referencias_train_set, pooled_min, pooled_max)
    escalas_referencia["train"] = referencias_usadas_train
    del consolidado, pooled_min, pooled_max

    if verbose:
        print("  [train] segunda pasada (medida)", flush=True)
    largo_train = _medir_train_streaming(
        ruta_experimento, instancias_train, nombres, born,
        escala_train.contexts(), referencias_usadas_train, metricas,
        verbose=verbose)
    largo_train["split"] = "train"
    largos["train"] = largo_train

    # --- test: solo se cargan las permutaciones de train que se reevaluan --
    if verbose:
        print("  [test] cargando permutaciones de train", flush=True)
    experimento = load_experiment(
        ruta_experimento, load_generations=False, load_run_metrics=False,
        fronts_instances=set(), solution_instances=train_necesarios)

    if verbose:
        print("  [test] cargando referencias y problemas", flush=True)
    instancias_test = sorted(emparejamiento)
    referencias_test_set, problemas_test, descartadas = preparar_split(
        instancias_test, datos_test, referencias_test, "test", verbose)
    descartados_por_split["test"] = descartadas

    if verbose:
        print("  [test] reevaluando individuos", flush=True)
    frentes_test = frentes_de_split(experimento, nombres, problemas_test,
                                    emparejamiento, verbose)
    escala_test = escala_hv_por_instancia(frentes_test)
    escalas["test"] = escala_test
    largo_test, referencias_usadas_test = medir_split(
        experimento, nombres, frentes_test, escala_test, referencias_test_set,
        metricas)
    escalas_referencia["test"] = referencias_usadas_test
    largo_test["split"] = "test"
    largos["test"] = largo_test

    largo = pd.concat(largos.values(), ignore_index=True)
    largo.to_csv(os.path.join(tablas_dir, "individual_split.csv"), index=False)

    for split, escala in escalas.items():
        save_hv_scale(escala, os.path.join(salida, "00_config",
                                           f"hv_scale_{split}.json"))

    # La escala de las metricas relativas a P* queda a mano en el resumen: es
    # lo que hace comparables los valores entre individuos, asi que sin ella
    # los CSV no se pueden reproducir ni auditar.
    resumen_referencias = {}
    for nombre_split, por_instancia in escalas_referencia.items():
        detalle = {}
        for nombre_instancia, ref in por_instancia.items():
            detalle[nombre_instancia] = {
                "n_puntos": int(ref.size),
                "ideal": [float(v) for v in ref.ideal],
                "nadir": [float(v) for v in ref.nadir],
                "span": [float(v) for v in ref.span],
            }
        resumen_referencias[nombre_split] = dict(sorted(detalle.items()))

    resumen = {"experiment": ruta_experimento,
               "individuals": len(nombres),
               "train_instances": list(instancias_train),
               "test_instances": sorted(emparejamiento),
               "test_skipped": sorted(set(test_names) - set(emparejamiento)),
               "pairing": "auto-por-tamano",
               "metrics": list(metricas),
               "convergence": {},
               "saturation": {},
               "referencias_descartadas": descartados_por_split,
               "escala_referencias": resumen_referencias}

    for split, parte in largos.items():
        tabla_por_generacion(parte).to_csv(
            os.path.join(tablas_dir, f"por_generacion_{split}.csv"), index=False)
        tabla_por_instancia(parte).to_csv(
            os.path.join(tablas_dir, f"por_instancia_{split}.csv"), index=False)

        convergencia = resumen_convergencia(parte)
        convergencia["split"] = split
        resumen["convergence"][split] = convergencia.to_dict("records")

        # El diagnostico trabaja con una columna por metrica, asi que la tabla
        # larga se abre a ancha antes de pasarsela.
        ancha = parte.pivot_table(index=["individual", "generation",
                                         "instance"],
                                  columns="metric", values="value").reset_index()
        saturacion = diagnostico_saturacion(ancha, metricas)
        resumen["saturation"][split] = saturacion
        if verbose and saturacion:
            print(f"  AVISO DELTA_P ({split}): {saturacion}")

        if figuras:
            figura_boxplot_por_generacion(parte, metricas, figuras_dir, split)
            tablas = tabla_matriz(parte, metricas)
            figura_trayectorias(parte, tablas, metricas, figuras_dir, split)
            figura_trayectorias_por_instancia(parte, metricas, figuras_dir,
                                              split)
            figura_mapa_calor(tablas, metricas, figuras_dir, split)
            figura_diferencias_criticas(parte, metricas, figuras_dir, split)

    if figuras:
        figura_train_test(largo, metricas, figuras_dir)
        figura_boxplot_generacion_train_test(largo, metricas, figuras_dir)
        figura_diferencias_criticas_juntas(largo, metricas, figuras_dir)
        # El CD por instancia suelta no existe con esta construccion, asi que
        # se dice una sola vez por experimento y no una por split.
        print("  nota: sin CD por instancia suelta (Friedman necesita mas de un "
              "bloque); las diferencias por instancia van en "
              "diferencias/instancias/", flush=True)

    convergencia_total = pd.concat(
        [pd.DataFrame(v) for v in resumen["convergence"].values()],
        ignore_index=True)
    convergencia_total.to_csv(os.path.join(tablas_dir, "convergencia.csv"),
                              index=False)

    resumen["duration_seconds"] = round(time.time() - inicio, 1)
    write_json(os.path.join(salida, "summary.json"), resumen)

    if verbose:
        print(f"  listo en {resumen['duration_seconds']}s -> {salida}")

    return resumen


def parse_args(argv=None):
    """
    Interpreta los argumentos de linea de comandos.

    @param argv: lista de argumentos, o None para usar `sys.argv`.
    @return: `argparse.Namespace`.
    """
    analizador = argparse.ArgumentParser(
        description=("Analiza el comportamiento de la busqueda en train y test "
                     "por generacion, reevaluando los individuos ya generados."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    analizador.add_argument("experiment", help="carpeta Experiment_NNN de la corrida")
    analizador.add_argument("--output", required=True, help="carpeta de salida")
    analizador.add_argument("--metrics", default=",".join(INDICATORS),
                            help="metricas separadas por comas")
    analizador.add_argument("--train-data", default="datasets/mqap/train")
    analizador.add_argument("--test-data", default="datasets/mqap/test")
    analizador.add_argument("--train-references",
                            default="references_moea/train/reference_sets")
    analizador.add_argument("--test-references",
                            default="references_moea/test/reference_sets")
    analizador.add_argument("--no-figuras", action="store_true",
                            help="solo escribir tablas, sin dibujar")
    analizador.add_argument("--quiet", action="store_true",
                            help="no imprimir progreso")
    return analizador.parse_args(argv)


def main(argv=None):
    """
    Punto de entrada.

    @param argv: argumentos de linea de comandos.
    @return: codigo de salida.
    """
    argumentos = parse_args(argv)

    metricas = [m.strip() for m in argumentos.metrics.split(",") if m.strip()]
    desconocidas = [m for m in metricas if m not in INDICATORS]
    if desconocidas:
        print(f"ERROR metricas desconocidas: {desconocidas}; "
              f"validas: {list(INDICATORS)}", file=sys.stderr)
        return 2

    try:
        ejecutar(ruta_experimento=argumentos.experiment,
                 salida=argumentos.output,
                 metricas=metricas,
                 datos_train=argumentos.train_data,
                 datos_test=argumentos.test_data,
                 referencias_train=argumentos.train_references,
                 referencias_test=argumentos.test_references,
                 figuras=not argumentos.no_figuras,
                 verbose=not argumentos.quiet)
    except AnalisisSearchError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())