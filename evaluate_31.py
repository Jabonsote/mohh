#!/usr/bin/env python
"""
Evaluacion comparativa de 31 replicas con poblaciones iniciales compartidas.

Este script NO es una copia de `mohh-evaluate`: existe porque el pipeline por
defecto usa semillas basadas en `datetime.now().microsecond` y poblaciones
iniciales generadas al azar dentro de cada proceso, de modo que dos
configuraciones distintas no parten de la misma poblacion y sus resultados no son
comparables replica a replica. Aqui:

* las poblaciones iniciales se generan UNA vez por (instancia, replica) con una
  semilla determinista y se reutilizan en todas las configuraciones, solvers y
  splits, de modo que la comparacion replica a replica es exacta;
* las semillas del MOEA tambien son deterministas y dependen solo de
  (semilla base, instancia, replica), no del reloj;
* cada tarea se cachea en `.npz`, asi que la ejecucion es reanudable;
* los cuatro indicadores (HV, IGD+, DELTA_P, R2) se calculan offline con
  `mohh.analysis`, con la misma convencion de normalizacion y las mismas
  referencias P* que usa el resto del repositorio.

No se fabrica ningun dato: las poblaciones iniciales son permutaciones de las
instancias reales y los frente salen de ejecutar los MOEAs sobre ellas.

Uso:

    # planificacion completa (no ejecuta nada)
    python evaluate_31.py --list

    # inspeccion en seco de cuantas tareas hay
    python evaluate_31.py --dry-run

    # evaluacion real, reanudable
    python evaluate_31.py --split test --budgets 10k 30k

Las referencias P* de cada split deben existir. Para `train` estan en
`set-reference/train/reference_sets`; para `test` en
`set-reference/test/reference_sets`.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

_RAIZ = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_RAIZ, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mohh.analysis.cache import load_cached, save_cached
from mohh.analysis.indicators_all import (
    INDICATORS,
    METRIC_LABELS,
    IndicatorsError,
    compute_indicators,
    fit_hv_contexts,
    load_references,
)
from mohh.analysis.layout import AnalysisLayout
from mohh.core.indicators.reference_sets import (ReferenceSetError,
                                                pooled_reference_sets)

#: Indicadores que necesitan un P* valido. Si el P* de una instancia esta
#: degenerado, estos no se pueden calcular para ella.
_METRICAS_REFERENCIA = ("IGD+", "DELTA_P", "R2")

#: Numero minimo de puntos para dar un P* por utilizable. Con la escala
#: poolingada (`pooled_reference_sets`) un P* de un solo punto ya es valido: la
#: caja de normalizacion sale de la union de P* con los frentes medidos, de modo
#: que IGD+/R2/DELTA_P quedan en ~[0, 1] y son comparables con el resto. Solo un
#: P* VACIO es un error, porque entonces no hay ningun punto de referencia.
_MIN_PUNTOS_REFERENCIA = 1


def referencia_ausente(nombre, referencias):
    """
    Indica si a una instancia le falta el P* entero.

    Un P* ausente es un problema de preparacion de la corrida y se reporta como
    error: sin el no se puede calcular IGD+, R2 ni DELTA_P, y medir en silencio
    dejaria huecos en las medias sin avisar.

    @param nombre: nombre de la instancia.
    @param referencias: dict nombre -> `ReferenceSet`.
    @return: True si no hay P* para esa instancia.
    """
    return referencias is None or nombre not in referencias


def referencia_invalida(nombre, referencias, minimo=_MIN_PUNTOS_REFERENCIA):
    """
    Indica si el P* de una instancia existe pero esta vacio.

    Que exista pero tenga menos de `minimo` puntos es distinto de que falte. Con
    la escala poolingada, un P* de un unico punto NO es un problema: las dos
    instancias `KC10-2fl-2un` y `KC20-2fl-2un` lo tienen (su frente verdadero
    es un punto, verificado con busqueda independiente) y antes quedaban fuera
    de tres de los cuatro indicadores. Ahora se miden igual que las demas.

    @param nombre: nombre de la instancia.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param minimo: numero minimo de puntos aceptable.
    @return: True si el P* existe pero es demasiado pequeno.
    """
    if referencia_ausente(nombre, referencias):
        return False
    return referencias[nombre].size < minimo


def advertencias_referencias(referencias, minimo=_MIN_PUNTOS_REFERENCIA):
    """
    Lista las instancias cuyo P* esta vacio, y anota las de un solo punto.

    @param referencias: dict nombre -> `ReferenceSet`.
    @param minimo: numero minimo de puntos aceptable.
    @return: lista de nombres con P* inutilizable.
    """
    return sorted(nombre for nombre in referencias
                  if referencias[nombre].size < minimo)


def pstar_de_un_punto(referencias):
    """
    Instancias cuyo P* tiene un unico punto.

    No es un error: se informative porque son las que antes quedaban sin
    IGD+/R2/DELTA_P, y conviene poder confirmar que ahora si se miden.

    @param referencias: dict nombre -> `ReferenceSet`.
    @return: lista de nombres, ordenada.
    """
    return sorted(nombre for nombre in referencias
                  if referencias[nombre].size == 1)

#: Mapeo de la etiqueta del experimento al presupuesto en evaluaciones.
PRESUPUESTOS_POR_ETIQUETA = {
    "10k": 10500,
    "30k": 31500,
    "50k": 52500,
}

#: Mutaciones usadas por el pipeline de evaluacion del repositorio.
MUTACIONES = ("NullMutation", "Swap_Mutation")

#: Descripcion de cada indicador, para `--list` y `--help`.
DESCRIPCION_METRICAS = {
    "HV": "hypervolumen (mayor es mejor, requiere estimar ref_point)",
    "IGD+": "distancia media invertida Generation+ (menor es mejor)",
    "DELTA_P": "distancia media al frente P* (menor es mejor)",
    "R2": "indicador R2 de la union de referencias (menor es mejor)",
}

#: Cruces estandar contra los que se compara el operador generado.
CRUCES_ESTANDAR = (("PMX_Cross", "PMX"), ("CX_Cross", "CX"))


class EvaluacionError(RuntimeError):
    """Fallo de la evaluacion, con un mensaje accionable."""


# ---------------------------------------------------------------------------
# Semillas deterministas
# ---------------------------------------------------------------------------


def semilla_para(semilla_base: int, *partes) -> int:
    """
    Semilla estable de 32 bits derivada de la semilla base y unas partes.

    Se usa un hash en vez de una suma para que cambiar el numero de replicas no
    desplace las semillas de las anteriores: la replica 3 de una corrida de 31
    tiene la misma semilla que la replica 3 de una corrida de 11.

    @param semilla_base: semilla global del experimento.
    @param partes: identificadores (instancia, replica, ...).
    @return: entero en [0, 2**31).
    """
    texto = "|".join(str(p) for p in (semilla_base,) + partes)
    digesto = hashlib.sha1(texto.encode("utf-8")).hexdigest()
    return int(digesto[:8], 16) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Poblaciones iniciales compartidas
# ---------------------------------------------------------------------------


def ruta_poblacion(cache_dir: str, instancia: str, replica: int,
                   n_variables: int) -> str:
    """Ruta de la poblacion inicial cacheada de (instancia, replica)."""
    return os.path.join(cache_dir, "initial_populations", instancia,
                        f"replica_{replica:03d}_n{n_variables}.txt")


def generar_poblacion(n_variables: int, population_size: int, semilla: int):
    """
    Permutacion inicial determinista.

    Se usa `MOSampling`, el mismo generador que employa el pipeline
    (`Instance.create_initial_solution`), de modo que las permutaciones tienen
    exactamente la misma distribucion y el mismo formato que las que se
    usarian sin este script.

    @param n_variables: numero de instalaciones a permutar.
    @param population_size: numero de soluciones.
    @param semilla: semilla de `np.random`.
    @return: array (population_size, n_variables) de enteros.
    """
    from mohh.core.problem.instance import MOSampling

    np.random.seed(semilla)
    return np.asarray(MOSampling(kind="Natural").generate(n_variables,
                                                           population_size),
                      dtype=int)


def cargar_poblacion(ruta: str):
    """Lee una poblacion inicial del formato propio (una permutacion por linea)."""
    soluciones = []
    with open(ruta, "r") as archivo:
        for linea in archivo:
            linea = linea.strip()
            if linea:
                soluciones.append([int(v) for v in linea.split()])
    return np.asarray(soluciones, dtype=int)


def guardar_poblacion(ruta: str, soluciones) -> None:
    """Escribe una poblacion inicial en el formato propio."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    temporal = ruta + ".tmp"
    with open(temporal, "w") as archivo:
        for solucion in soluciones:
            archivo.write(" ".join(str(v) for v in solucion) + "\n")
    os.replace(temporal, ruta)


def poblacion_compartida(cache_dir: str, instancia: str, replica: int,
                         n_variables: int, population_size: int,
                         semilla_base: int):
    """
    Devuelve la poblacion inicial de (instancia, replica), generandola una vez.

    La misma poblacion se usa para todos los operadores, todos los solvers y
    todos los splits. Es la pieza que hace comparables las 31 replicas.

    @param cache_dir: carpeta de cache.
    @param instancia: nombre de la instancia.
    @param replica: indice de replica (1-based).
    @param n_variables: numero de instalaciones.
    @param population_size: tamano de la poblacion.
    @param semilla_base: semilla global.
    @return: lista de permutaciones.
    """
    ruta = ruta_poblacion(cache_dir, instancia, replica, n_variables)

    if os.path.isfile(ruta):
        try:
            poblacion = cargar_poblacion(ruta)
            if (poblacion.shape == (population_size, n_variables)
                    and all(sorted(fila) == list(range(n_variables))
                            for fila in poblacion)):
                return poblacion
        except (OSError, ValueError):
            # Cache corrupta o incompleta: se regenera en vez de fallar.
            pass

    semilla = semilla_para(semilla_base, "poblacion", instancia, replica)
    poblacion = generar_poblacion(n_variables, population_size, semilla)
    guardar_poblacion(ruta, poblacion)
    return poblacion


# ---------------------------------------------------------------------------
# Tareas
# ---------------------------------------------------------------------------


def firma_tarea(tarea: dict) -> str:
    """
    Firma estable de una tarea, para cachear y reanudar.

    Se calcula aqui y no con `mohh.analysis.cache.task_signature` porque el
    nombre del archivo lo lleva esta firma, y el operador fenotipico puede
    ocupar cientos de caracteres (una ruta construida con el nombre del
    operador se passaria de largo). Ademas la firma incluye la huella de la
    poblacion inicial, que es lo que hace comparables las replicas.

    @param tarea: dict con las claves que determinan el resultado.
    @return: hash corto.
    """
    relevante = ("instance", "solver", "config", "mutation", "experiment",
                 "n_variables", "population_size", "base_seed", "seed",
                 "budgets", "population_signature")
    partes = [f"{clave}={tarea.get(clave)}" for clave in relevante]
    return hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()[:20]


def cache_ruta(cache_dir: str, tarea: dict) -> str:
    """Ruta del `.npz` de una tarea."""
    return os.path.join(cache_dir, "tasks",
                        f"{firma_tarea(tarea)}.npz")


def _ejecutar_tarea(tarea: dict) -> dict:
    """
    Ejecuta una tarea en el proceso actual y devuelve sus frentes por
    presupuesto. Se usa tanto en el proceso principal como en el pool.
    """
    from mohh.evaluation.experiment import build_solver
    from mohh.evaluation.params import Params

    Params().update_params(tarea["params"])

    semilla = tarea["seed"]
    random.seed(semilla)
    np.random.seed(semilla)

    instancia = tarea["instance_object"]
    instancia.set_initial_solutions(experiment=tarea["experiment"],
                                    kind="permutation")

    solver = build_solver(tarea["solver"], tarea["cross_type"],
                          tarea["cross_operator"], tarea["mutation"])

    inicio = time.perf_counter()
    solver.load_instance(instancia)
    solver.start_model(semilla)
    instantaneos = solver.solve_instance()
    duracion = time.perf_counter() - inicio

    frentes = {etiqueta: np.asarray(instantaneos[etiqueta], dtype=float)
               for etiqueta in tarea["budgets"]
               if etiqueta in instantaneos
               and instantaneos[etiqueta] is not None
               and np.asarray(instantaneos[etiqueta]).size > 0}

    return {"frentes": frentes, "solve_s": duracion, "seed": semilla}


def construir_tareas(instancias, configuraciones, solvers, replicas,
                     presupuestos, params_dict, poblaciones):
    """
    Construye la rejilla completa de tareas.

    @param instancias: `{nombre: Instance}`.
    @param configuraciones: lista de dicts con `cross_type`, `cross_operator`,
        `mutation`, `config`.
    @param solvers: nombres de solver (MOEAD, NSGAII, SMSEMOA).
    @param replicas: numero de replicas (31 por defecto).
    @param presupuestos: etiquetas de presupuesto (10k, 30k, 50k).
    @param params_dict: parametros para el singleton `Params`.
    @param poblaciones: `{instancia: {replica: array de permutaciones}}`.
    @return: lista de tareas.
    """
    tareas = []

    parametros = {k: v for k, v in params_dict.items() if k != "BNF_GRAMMAR"}
    semilla_base = params_dict["BASE_SEED"]

    for nombre_instancia, instancia in instancias.items():
        for replica in range(1, replicas + 1):
            poblacion = poblaciones[nombre_instancia][replica]
            firma_poblacion = hashlib.sha1(
                "\n".join(" ".join(str(v) for v in fila) for fila in poblacion)
                .encode("utf-8")).hexdigest()[:16]

            # La semilla del MOEA depende de (base, instancia, replica), nunca
            # del reloj ni del solver: las tres réplicas se ejecutan con la misma
            # semilla, que es justamente lo que hace comparables a las
            # configuraciones.
            semilla = semilla_para(semilla_base, "moea", nombre_instancia,
                                   replica)

            for configuracion in configuraciones:
                for solver in solvers:
                    tarea = {
                        "params": parametros,
                        "instance": nombre_instancia,
                        "instance_object": instancia,
                        "experiment": replica,
                        "replicate": replica,
                        "solver": solver,
                        "cross_type": configuracion["cross_type"],
                        "cross_operator": configuracion["cross_operator"],
                        "mutation": configuracion["mutation"],
                        "config": configuracion["config"],
                        "budgets": list(presupuestos),
                        "seed": semilla,
                        "base_seed": semilla_base,
                        "population_size": instancia.population_size,
                        "n_variables": instancia.n_variables,
                        "population_signature": firma_poblacion,
                    }
                    tareas.append(tarea)

    return tareas


def ejecutar_tareas(tareas, cache_dir, usar_cache=True, forzar=False,
                    verbose=True, log_dir=None):
    """
    Ejecuta las tareas, saltando las que ya estan en cache.

    @param tareas: lista de `construir_tareas`.
    @param cache_dir: carpeta de cache.
    @param usar_cache: si False, recalcula aunque exista el `.npz`.
    @param forzar: si True, ignora la cache.
    @param verbose: imprime el progreso.
    @param log_dir: directorio para volcar lineas TASK_OK por
        solver/config/mutation; si es None no se escribe nada.
    @return: lista de resultados en el mismo orden que `tareas`.
    """
    from mohh.parallel import run_parallel

    usar_cache = usar_cache and not forzar
    resultados = [None] * len(tareas)
    pendientes = []
    posiciones = []

    for indice, tarea in enumerate(tareas):
        ruta = cache_ruta(cache_dir, tarea)
        if usar_cache and os.path.isfile(ruta):
            frentes, meta = load_cached(ruta)
            if frentes is not None:
                resultados[indice] = {"frentes": frentes,
                                      "solve_s": meta.get("solve_s"),
                                      "seed": meta.get("seed"),
                                      "cache": True}
                continue
            # Cache ilegible o corrupta: se recalcula.
        pendientes.append(tarea)
        posiciones.append(indice)

    if verbose:
        print(f"Tareas: {len(tareas)} ({len(tareas) - len(pendientes)} desde "
              f"cache, {len(pendientes)} a ejecutar)")

    if not pendientes:
        return resultados

    os.makedirs(os.path.join(cache_dir, "tasks"), exist_ok=True)

    hecho = {"n": 0}
    total = len(pendientes)

    def al_terminar(_indice):
        hecho["n"] += 1
        if verbose:
            pct = 100.0 * hecho["n"] / total
            print(f"\r  progreso: {hecho['n']:6d}/{total} tareas ({pct:5.1f}%)",
                  end="", flush=True)
            if hecho["n"] == total:
                print()

    for posicion, resultado in zip(posiciones,
                                   run_parallel(_ejecutar_tarea, pendientes,
                                                on_complete=al_terminar)):
        save_cached(cache_ruta(cache_dir, tareas[posicion]),
                    resultado["frentes"],
                    meta={"solve_s": resultado.get("solve_s"),
                          "seed": resultado.get("seed")})
        resultados[posicion] = resultado
        t = tareas[posicion]
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            ruta = os.path.join(
                log_dir, f"{t['solver']}_{t['config']}_{t['mutation']}.log")
            with open(ruta, "a") as fh:
                fh.write(
                    f"TASK_OK solver={t['solver']} config={t['config']} "
                    f"mutation={t['mutation']} instance={t['instance']}\n")

    return resultados


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------


def medir(frentes_por_instancia, referencias, metricas, contextos=None):
    """
    Calcula los indicadores de un conjunto de frentes.

    @param frentes_por_instancia: `{instancia: (n, m) ndarray}`.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param metricas: subconjunto de `INDICATORS`.
    @param contextos: escala de HV por instancia.
    @return: dict metrica -> float.
    """
    tabla = compute_indicators(frentes_por_instancia, references=referencias,
                               metrics=metricas, hv_context=contextos)
    return {metrica: float(tabla[metrica].mean()) for metrica in metricas}


def indicadores_completos(frentes_por_instancia, referencias, metricas,
                         contextos=None):
    """Igual que `medir`, pero devuelve el detalle por instancia."""
    return compute_indicators(frentes_por_instancia, references=referencias,
                              metrics=metricas, hv_context=contextos)


def construir_filas(resultados, tareas, presupuestos, referencias, metricas,
                    contextos_por_presupuesto):
    """
    Convierte los resultados de las tareas en una tabla larga.

    @param resultados: salida de `ejecutar_tareas`.
    @param tareas: la misma lista de tareas, para leer la metadata.
    @param presupuestos: etiquetas de presupuesto.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param metricas: indicadores a calcular.
    @param contextos_por_presupuesto: `{presupuesto: HypervolumeContexts}`.
    @return: DataFrame con una fila por (instancia, config, solver, replica,
        presupuesto).
    """
    # Contextos de HV por presupuesto: ajustados con TODOS los frentes de ese
    # presupuesto, para que los HVs de distintas réplicas/configuraciones sean
    # comparables (igual que el pipeline, que normaliza con el consolidado).
    filas = []
    faltantes = []

    for indice, tarea in enumerate(tareas):
        resultado = resultados[indice]
        if resultado is None:
            continue

        for etiqueta in presupuestos:
            frente = resultado["frentes"].get(etiqueta)
            if frente is None or np.asarray(frente).size == 0:
                continue

            filas.append({
                "instance": tarea["instance"],
                "config": tarea["config"],
                "solver": tarea["solver"],
                "mutation": tarea["mutation"],
                "replicate": tarea["replicate"],
                "budget": etiqueta,
                "front_size": int(np.asarray(frente).shape[0]),
                "solve_s": resultado.get("solve_s"),
                "seed": resultado.get("seed"),
                "_front": np.asarray(frente, dtype=float),
            })

    if not filas:
        raise EvaluacionError(
            "ninguna tarea devolvio frentes; revisa el presupuesto maximo "
            "(MO_GENERATIONS) y que las tareas se hayan ejecutado.")

    # Escala comun por presupuesto: se ajusta con TODOS los frentes de todas las
    # configuraciones, solvers y replicas de ese presupuesto (el consolidado),
    # para que los HVs sean comparables. Si se normalizara cada frente por si
    # mismo, dos algoritmos con distinta escala de valores darian HVs sin
    # significancia.
    #
    # La MISMA idea se aplica a las metricas relativas a P*: la caja sale de la
    # union de P* con el consolidado de cada presupuesto. Asi un P* de un solo
    # punto no deforma la escala, y todas las instancias se miden con los cuatro
    # indicadores por igual.
    consolidados_por_presupuesto = {}
    for etiqueta in presupuestos:
        por_instancia = {}
        for fila in filas:
            if fila["budget"] != etiqueta:
                continue
            por_instancia.setdefault(fila["instance"], []).append(
                fila["_front"])
        consolidados_por_presupuesto[etiqueta] = {
            nombre: np.vstack(frentes)
            for nombre, frentes in por_instancia.items()}

    if "HV" in metricas and contextos_por_presupuesto is None:
        contextos_por_presupuesto = {
            etiqueta: fit_hv_contexts({"consolidado": consolidado})
            for etiqueta, consolidado in consolidados_por_presupuesto.items()}

    # Solo se poolingan las instancias que realmente se miden en ese
    # presupuesto. Pooling tambien las demas dejaria en la escala guardada
    # entradas sin poolear, con el `span` de convenio 1.0 de un P* de un punto,
    # que es justo el valor degenerado que este cambio elimina.
    referencias_por_presupuesto = {}
    for etiqueta, consolidado in consolidados_por_presupuesto.items():
        subset = {nombre: referencias[nombre] for nombre in consolidado
                  if nombre in referencias}
        referencias_por_presupuesto[etiqueta] = pooled_reference_sets(
            subset, consolidado)

    registro = []
    for fila in filas:
        # Guarda defensiva, hoy inalcanzable: `read_pof` y `_build` rechazan un P*
        # vacio, asi que `size` es siempre >= 1 y con la escala poolingada un P*
        # de un solo punto se mide igual que cualquier otro. Se deja el check
        # para que, si alguna vez pudiera existir un P* sin puntos, saliera un
        # NaN explicito en vez de contaminar las medias en silencio.
        metricas_fila = metricas
        if referencia_invalida(fila["instance"], referencias):
            metricas_fila = [m for m in metricas
                             if m not in _METRICAS_REFERENCIA]
            if not metricas_fila:
                continue

        contexto = (contextos_por_presupuesto.get(fila["budget"])
                    if "HV" in metricas_fila else None)
        referencias_fila = referencias_por_presupuesto.get(fila["budget"],
                                                            referencias)
        try:
            valores = indicadores_completos(
                {fila["instance"]: fila["_front"]}, referencias_fila,
                metricas_fila, contexto)
        except IndicatorsError as error:
            raise EvaluacionError(
                f"indicadores de {fila['instance']} "
                f"({fila['config']}, {fila['solver']}, replica "
                f"{fila['replicate']}, {fila['budget']}): {error}") from error

        base = {k: v for k, v in fila.items() if k != "_front"}
        for metrica in metricas:
            valor = valores.loc[fila["instance"], metrica]
            base[metrica] = (np.nan if metrica not in metricas_fila
                             or not np.isfinite(valor) else float(valor))
        registro.append(base)

    tabla = pd.DataFrame(registro)

    # La escala se viaja en `.attrs` para poder guardarla sin cambiar la
    # firma de esta funcion (la usan varios sitios y sus tests).
    detalle = {}
    if presupuestos:
        for nombre_instancia, ref in referencias_por_presupuesto[
                next(iter(presupuestos))].items():
            detalle[nombre_instancia] = {
                "n_puntos": int(ref.size),
                "ideal": [float(v) for v in ref.ideal],
                "nadir": [float(v) for v in ref.nadir],
                "span": [float(v) for v in ref.span],
            }
    tabla.attrs["escala_referencias"] = dict(sorted(detalle.items()))

    return tabla


# ---------------------------------------------------------------------------
# Estadisticas
# ---------------------------------------------------------------------------


def escribir_estadisticas(por_presupuesto, layout, etiqueta, metricas,
                          con_figuras=True, max_columnas=25):
    """
    Friedman + Nemenyi y boxplots por presupuesto.

    Los bloques son las INSTANCIAS y los algoritmos son las configuraciones
    (operador + mutacion + solver), que es el diseno de Demšar. El valor de
    cada celda es la media de las 31 replicas.

    @param por_presupuesto: `{presupuesto: DataFrame largo}`.
    @param layout: `AnalysisLayout`.
    @param etiqueta: sufijo de los archivos.
    @param metricas: indicadores.
    @param con_figuras: dibujar diagramas.
    @param max_columnas: tope de columnas de los diagramas.
    @return: dict presupuesto -> resumen.
    """
    from mohh.analysis import stats as stats_mod

    resumen = {}

    for etiqueta_presupuesto, tabla in sorted(por_presupuesto.items()):
        por_config = {}

        for metrica in metricas:
            clave = (tabla.groupby(["config", "instance"])[metrica]
                     .mean().reset_index())
            valores = {}
            for nombre in sorted(clave["config"].unique()):
                fila = clave[clave["config"] == nombre]
                valores[nombre] = fila[metrica].to_numpy(dtype=float)

            if len(valores) < 2:
                continue

            frame = pd.DataFrame(valores)
            frame.index.name = "instance"

            base = {"n_configs": int(frame.shape[1]),
                    "n_instances": int(frame.shape[0])}

            if frame.shape[0] >= 3 and frame.shape[1] >= 3:
                statistic, pvalue = stats_mod.friedman_test(frame)
                base.update({"friedman_statistic": float(statistic),
                             "friedman_p": float(pvalue),
                             "cd_nemenyi": stats_mod.nemenyi_cd(
                                 frame.shape[0], frame.shape[1])})
            else:
                base["nota"] = ("Friedman necesita >= 3 instancias y >= 3 "
                                "configuraciones; se omite.")

            frame.to_csv(layout.sub(
                layout.statistics,
                f"values_{etiqueta}_{etiqueta_presupuesto}_{metrica}.csv"))

            if con_figuras and frame.shape[1] <= max_columnas:
                # Un fallo al dibujar no puede tumbar la evaluacion: los CSV y
                # las tablas ya estan escritos y son el resultado de verdad.
                try:
                    stats_mod.critical_difference_diagram(
                        frame, metric=metrica,
                        path=layout.sub(layout.figures,
                                        f"cd_{etiqueta}_{etiqueta_presupuesto}_"
                                        f"{metrica}.png"),
                        title=f"{METRIC_LABELS.get(metrica, metrica)} "
                              f"({etiqueta_presupuesto})")
                    stats_mod.boxplot_configs(
                        frame, metric=metrica,
                        path=layout.sub(layout.figures,
                                        f"boxplot_{etiqueta}_"
                                        f"{etiqueta_presupuesto}_{metrica}.png"))
                except Exception as error:  # noqa: BLE001 - degradar, no fallar
                    print(f"  aviso: no se pudo dibujar {metrica} en "
                          f"{etiqueta_presupuesto} ({error})")

            por_config[metrica] = base

        if por_config:
            resumen[etiqueta_presupuesto] = por_config

    return resumen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def construir_parser():
    """Parser de argumentos de `evaluate_31.py`."""
    parser = argparse.ArgumentParser(
        description="Evaluacion comparativa de 31 replicas con poblaciones "
                    "iniciales compartidas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "experiment", nargs="?", default=None,
        help="ruta del experimento (results/.../Experiment_NNN). Si se da, se "
             "leen sus operadores y toda la salida queda en "
             "'<experimento>/evaluation'")
    parser.add_argument(
        "--operators", default=None,
        help="fichero `<model>_operators.txt` de la generacion; si se omite y "
             "no hay experimento, solo se comparan los cruces estandar")
    parser.add_argument("--models", nargs="+", default=["MOEAD"],
                        help="modelos de los operadores, en el mismo orden que "
                             "el fichero de operadores")
    parser.add_argument("--solvers", nargs="+", default=None,
                        help="solvers; por defecto los tres de "
                             "mohh.evaluation.params")
    parser.add_argument("--replicates", type=int, default=31,
                        help="numero de replicas por configuracion")
    parser.add_argument("--population-size", type=int, default=105,
                        help="tamano de poblacion del MOEA")
    parser.add_argument("--generations", type=int, default=500,
                        help="generaciones del MOEA (el presupuesto maximo se "
                             "deriva de aqui)")
    parser.add_argument("--budgets", nargs="+", default=None,
                        help="presupuestos a registrar; por defecto los tres "
                             "del paper (10k, 30k, 50k) mas los añadidos "
                             "con --custom-budget")
    parser.add_argument("--custom-budget", action="append", default=[],
                        metavar="LABEL=EVALS",
                        help="presupuesto adicional, repetible "
                             "(p.ej. --custom-budget smoke=210)")
    parser.add_argument("--metrics", nargs="+", default=list(INDICATORS),
                        choices=list(INDICATORS),
                        help="indicadores a calcular")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                        help="split de instancias a evaluar")
    parser.add_argument("--splits", nargs="+", default=None,
                        choices=["train", "test"],
                        help="splits a evaluar en la MISMA corrida, cada uno con "
                             "sus propias referencias P*. Por defecto, con un "
                             "experimento se usan train y test; sin "
                             "experimento, solo --split")
    parser.add_argument("--instances", nargs="+", default=None,
                        help="instancias concretas; por defecto, todas las "
                             "del split")
    parser.add_argument("--base-seed", type=int, default=20260930,
                        help="semilla global; de ella salen todas las demas")
    parser.add_argument("--cache", default=None,
                        help="carpeta de cache (poblaciones y tareas); por "
                             "defecto '<salida>/cache'")
    parser.add_argument("--output", default=None,
                        help="carpeta de resultados; por defecto "
                             "'<experimento>/evaluation' si se da un "
                             "experimento, o 'results/evaluate_31'")
    parser.add_argument("--references", default=None,
                        help="carpeta de P*; por defecto la del split elegido")
    parser.add_argument("--dataset", default="mqap",
                        help="dataset de las instancias")
    parser.add_argument("--problem", default="QAP", help="tipo de problema")
    parser.add_argument("--solution-type", default="Natural",
                        help="tipo de solucion")
    parser.add_argument("--limit-instances", type=int, default=None,
                        help="usar solo las primeras N instancias (depuracion)")
    parser.add_argument("--log-dir", default=None,
                        help="directorio para volcar lineas TASK_OK por "
                             "solver/config/mutation")
    parser.add_argument("--dry-run", action="store_true",
                        help="planifica y comprueba referencias sin ejecutar")
    parser.add_argument("--list", action="store_true",
                        help="lista las opciones disponibles y termina")
    parser.add_argument("--force", action="store_true",
                        help="ignorar la cache de tareas")
    parser.add_argument("--no-figures", action="store_true",
                        help="no dibujar diagramas")

    return parser


def cargar_operadores(ruta: str, modelos: list) -> list:
    """
    Lee los fenotipos de `<model>_operators.txt`.

    @param ruta: fichero de operadores.
    @param modelos: nombres de los modelos, en orden.
    @return: lista de fenotipos, de tres en tres (Best, Middle, Worst).
    """
    from mohh.analysis.io_utils import load_operators_file_single

    operadores = load_operators_file_single(ruta)

    # El fichero trae una entrada por corrida `<modelo>`; se concatenan los
    # fenotipos de todos los modelos en el orden en que se aparecerian.
    fenotipos = []
    for modelo in modelos:
        if modelo not in operadores:
            raise EvaluacionError(
                f"'{ruta}' no tiene operadores para el modelo '{modelo}'; "
                f"modelos disponibles: {sorted(operadores)}")
        entrada = operadores[modelo]
        if isinstance(entrada, dict):
            for etiqueta in ("Best", "Middle", "Worst"):
                if etiqueta not in entrada:
                    raise EvaluacionError(
                        f"'{ruta}' ({modelo}) no tiene la etiqueta "
                        f"'{etiqueta}'; hay {sorted(entrada)}")
                fenotipos.append(entrada[etiqueta])
        else:
            fenotipos.extend(entrada)

    if len(fenotipos) % 3 != 0:
        raise EvaluacionError(
            f"'{ruta}' tiene {len(fenotipos)} fenotipos, que no es multiplo de "
            f"3; se esperan Best/Middle/Worst por modelo.")

    return fenotipos


def configuraciones_desde(fenotipos, modelos, mutaciones=MUTACIONES):
    """
    Construye la lista de configuraciones a comparar.

    Por cada modelo se anaden sus tres operadores (Best/Middle/Worst) con las
    dos mutaciones, y despues los dos cruces estandar con una sola mutacion
    (para no multiplicar inutilmente el numero de tareas).

    @param fenotipos: lista de fenotipos de tres en tres.
    @param modelos: nombres de los modelos.
    @param mutaciones: mutaciones a combinar.
    @return: lista de dicts de configuracion.
    """
    configuraciones = []

    for indice in range(0, len(fenotipos), 3):
        modelo = modelos[indice // 3]
        for etiqueta, fenotipo in zip(("Best", "Middle", "Worst"),
                                      fenotipos[indice:indice + 3]):
            for mutacion in mutaciones:
                sufijo = "NM" if mutacion == "NullMutation" else "WM"
                configuraciones.append({
                    "cross_type": "operator_template",
                    "cross_operator": fenotipo,
                    "mutation": mutacion,
                    "config": f"{modelo}_{etiqueta}_{sufijo}",
                })

    for operador, etiqueta in CRUCES_ESTANDAR:
        configuraciones.append({
            "cross_type": "crossover",
            "cross_operator": operador,
            "mutation": "NullMutation",
            "config": f"Standard_{etiqueta}_NM",
        })

    return configuraciones


# ---------------------------------------------------------------------------
# Resumen final: el operador generado frente a los estados base
# ---------------------------------------------------------------------------


def _es_base(config: str) -> bool:
    """True si la configuracion es un cruce estandar, no un operador generado."""
    return config.startswith("Standard_")


def comparar_contra_bases(por_presupuesto, metricas, alpha=0.05):
    """
    Responde si cada operador generado bate a los estados base.

    El contraste es pareado por INSTANCIA: para cada instancia se compara la
    media de las 31 replicas de la configuracion con la media de las 31 replicas
    de la referencia. Como las replicas comparten poblacion inicial y semilla
    (ver `poblacion_compartida` y `construir_tareas`), la comparacion replica a
    replica es exacta, y promediar antes de testear es lo correcto porque las
    replicas de una misma instancia no son independientes.

    La referencia de cada familia es el MEJOR estado base de esa instancia, no
    uno fijo: exigirle a un operador que gane al mejor de PMX y CX es la
    comparacion fuerte, y es la que interesa para un operador generado.

    @param por_presupuesto: `{presupuesto: DataFrame largo}`.
    @param metricas: indicadores a comparar.
    @param alpha: nivel de significacion para marcar una victoria.
    @return: lista de dicts, uno por (presupuesto, metrica, configuracion).
    """
    from scipy import stats as scipy_stats

    filas = []

    for etiqueta, tabla in sorted(por_presupuesto.items()):
        bases = sorted(c for c in tabla["config"].unique() if _es_base(c))
        generados = sorted(c for c in tabla["config"].unique()
                           if not _es_base(c))

        if not bases or not generados:
            continue

        for metrica in metricas:
            if metrica not in tabla.columns:
                continue

            # instancia x configuracion, con la media de las replicas.
            celdas = (tabla.groupby(["instance", "config"])[metrica]
                      .mean().unstack("config"))

            if celdas.shape[0] < 3 or celdas.shape[1] < 2:
                continue

            # Referencia por instancia: el mejor de los estados base. HV se
            # maximiza, el resto se minimiza.
            mejor_base = (celdas[bases].idxmax(axis=1) if metrica == "HV"
                          else celdas[bases].idxmin(axis=1))
            base_por_instancia = pd.Series(
                [celdas.loc[instancia, mejor_base[instancia]]
                 for instancia in celdas.index], index=celdas.index)

            # `signo` hace que "mejor" sea siempre una diferencia negativa.
            signo = -1.0 if metrica == "HV" else 1.0
            base_alineada = base_por_instancia * signo

            pruebas = []
            for config in generados:
                if config not in celdas.columns:
                    continue
                diferencias = (celdas[config] * signo) - base_alineada
                if np.allclose(diferencias.to_numpy(dtype=float), 0.0):
                    continue
                try:
                    estadistico, p_bruto = scipy_stats.wilcoxon(
                        diferencias.to_numpy(dtype=float))
                except ValueError:
                    continue
                pruebas.append({
                    "config": config,
                    "n_instances": int(diferencias.notna().sum()),
                    "statistic": float(estadistico),
                    "p_raw": float(p_bruto),
                    "mean_diff": float(diferencias.mean()),
                })

            if not pruebas:
                continue

            # Correccion de Holm sobre la familia de configuraciones.
            tabla_p = pd.DataFrame(pruebas).sort_values("p_raw",
                                                         kind="mergesort")
            tabla_p["p_holm"] = stats_holm(
                tabla_p["p_raw"].to_numpy(dtype=float))

            mejor_config = (celdas.mean(axis=0).idxmax() if metrica == "HV"
                            else celdas.mean(axis=0).idxmin())

            for _, prueba in tabla_p.iterrows():
                filas.append({
                    "budget": etiqueta,
                    "metric": metrica,
                    "config": prueba["config"],
                    "kind": "individual",
                    "mean": float(celdas[prueba["config"]].mean()),
                    "mean_diff": prueba["mean_diff"],
                    "n_instances": prueba["n_instances"],
                    "wilcoxon_stat": prueba["statistic"],
                    "p_raw": prueba["p_raw"],
                    "p_holm": float(prueba["p_holm"]),
                    "beats_best_baseline": bool(prueba["mean_diff"] < 0
                                                and prueba["p_holm"] < alpha),
                    "is_best_config": bool(prueba["config"] == mejor_config),
                    "best_baselines": "|".join(bases),
                })

    return filas


def stats_holm(p_valores):
    """
    Correccion de Holm sobre p-valores ya ordenados de menor a mayor.

    Holm es paso a decremento, asi que se aplica sobre el orden ascendente y el
    resultado se devuelve en ese mismo orden (no reordenado).

    @param p_valores: secuencia de p-valores.
    @return: p-valores ajustados, en el orden de entrada.
    """
    p = np.asarray(p_valores, dtype=float)
    n = p.size
    if n == 0:
        return p

    ajustado = np.empty(n, dtype=float)
    acumulado = 0.0

    for indice in range(n):
        candidato = min(1.0, (n - indice) * p[indice])
        acumulado = max(acumulado, candidato)
        ajustado[indice] = acumulado

    return ajustado


def escribir_summary_final(filas, layout, etiqueta):
    """
    Escribe el veredicto final en CSV y en texto legible.

    @param filas: salida de `comparar_contra_bases`.
    @param layout: `AnalysisLayout` donde escribir.
    @param etiqueta: sufijo de los archivos.
    @return: ruta del CSV escrito, o None si no hay filas.
    """
    if not filas:
        return None

    tabla = pd.DataFrame(filas).sort_values(
        ["budget", "metric", "p_holm", "config"], kind="mergesort")

    ruta = layout.sub(layout.statistics, f"summary_final_{etiqueta}.csv")
    tabla.to_csv(ruta, index=False)

    # Version legible: una linea por veredicto.
    ruta_txt = layout.sub(layout.statistics, f"summary_final_{etiqueta}.txt")
    with open(ruta_txt, "w") as archivo:
        archivo.write("¿El operador generado bate a los estados base?\n")
        archivo.write("=" * 72 + "\n\n")
        archivo.write(
            "Referencia: el MEJOR estado base (PMX/CX) de cada instancia, "
            "comparado\npareado por instancia sobre la media de las replicas.\n"
            "Negativo en `mean_diff` = el operador es mejor. `p_holm` < 0.05 = "
            "significativo.\n\n")

        for (presupuesto, metrica), grupo in tabla.groupby(
                ["budget", "metric"], sort=True):
            archivo.write(f"[{presupuesto} / {metrica}]\n")
            for _, fila in grupo.iterrows():
                if fila["beats_best_baseline"]:
                    veredicto = "GANA"
                elif fila["mean_diff"] < 0:
                    veredicto = "gana pero NO significativo"
                else:
                    veredicto = "pierde"
                estrella = " *" if fila["is_best_config"] else ""
                archivo.write(
                    f"  {fila['config']:<28} {veredicto:<28} "
                    f"diff={fila['mean_diff']:+.6g} "
                    f"p_holm={fila['p_holm']:.3e}{estrella}\n")
            archivo.write("\n")

    return ruta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    """
    Punto de entrada.

    @param argv: argumentos; si es None se leen de la linea de comandos.
    @return: codigo de salida (0 correcto, 1 error).
    """
    args = construir_parser().parse_args(argv)

    if args.list:
        print("Presupuestos disponibles:")
        for etiqueta, valor in sorted(PRESUPUESTOS_POR_ETIQUETA.items()):
            print(f"  {etiqueta}: {valor} evaluaciones")
        print("\nIndicadores disponibles:")
        for metrica, descripcion in DESCRIPCION_METRICAS.items():
            print(f"  {metrica:<8} {descripcion} "
                  f"(archivo: {METRIC_LABELS[metrica]})")
        print("\nEjemplo:")
        print("  python evaluate_31.py --dry-run")
        print("  python evaluate_31.py results/R2/Experiment_001")
        return 0

    inicio = time.perf_counter()

    try:
        # --- presupuestos de esta corrida ---------------------------------
        presupuestos = dict(PRESUPUESTOS_POR_ETIQUETA)

        for especificacion in args.custom_budget:
            if "=" not in especificacion:
                raise EvaluacionError(
                    f"--custom-budget '{especificacion}' no tiene el formato "
                    f"LABEL=EVALS")
            etiqueta, _, valor = especificacion.partition("=")
            try:
                presupuestos[etiqueta] = int(valor)
            except ValueError:
                raise EvaluacionError(
                    f"--custom-budget '{especificacion}': '{valor}' no es un "
                    f"entero") from None

        # Un --custom-budget que no se selecciona no hace nada, y eso es una
        # trampa: se pide un presupuesto de depuracion y se acaban corriendo los
        # de verdad. Si el usuario no pasa --budgets, se selectionan los de
        # siempre MAS los nuevos, que es lo que casi siempre se quiere.
        if args.budgets is None:
            args.budgets = list(PRESUPUESTOS_POR_ETIQUETA)
            args.budgets.extend(
                esp.partition("=")[0] for esp in args.custom_budget)
        else:
            args.budgets = list(args.budgets)

        desconocidos = [b for b in args.budgets if b not in presupuestos]
        if desconocidos:
            raise EvaluacionError(
                f"presupuesto(s) desconocido(s): {desconocidos}. Disponibles: "
                f"{sorted(presupuestos)}. Usa --custom-budget LABEL=EVALS para "
                f"anadir uno.")

        # --- experimento (opcional) ----------------------------------------
        raiz_experimento = None
        operadores_por_modelo = {}

        if args.experiment:
            from mohh.analysis.io_utils import load_operators_file
            from mohh.analysis.layout import detect_experiment_root

            raiz_experimento = detect_experiment_root(args.experiment)
            operadores_por_modelo = load_operators_file(raiz_experimento)
            if not operadores_por_modelo:
                raise EvaluacionError(
                    f"'{raiz_experimento}' no tiene ningun fichero "
                    f"'<modelo>_operators.txt'; no hay operadores que evaluar.")

        # --- splits --------------------------------------------------------
        if args.splits:
            splits = list(dict.fromkeys(args.splits))
        elif raiz_experimento:
            # Un experimento se evalua en train y test: es lo que pide comparar
            # el comportamiento en las dos particiones con las mismas replicas.
            splits = ["train", "test"]
        else:
            splits = [args.split]

        if len(splits) > 1 and args.references:
            raise EvaluacionError(
                "--references no se puede usar con varios splits: cada uno usa "
                "las suyas en 'set-reference/<split>/reference_sets'. Deja el "
                "flag para un solo split.")

        # --- salida y cache -------------------------------------------------
        if args.output:
            salida = args.output
        elif raiz_experimento:
            salida = os.path.join(raiz_experimento, "evaluation")
        else:
            salida = "results/evaluate_31"

        cache = args.cache or os.path.join(salida, "cache")

        # --- operadores ------------------------------------------------------
        modelos = sorted(operadores_por_modelo) or list(args.models)

        if operadores_por_modelo:
            fenotipos = cargar_operadores_desde(operadores_por_modelo, modelos)
        elif args.operators:
            fenotipos = cargar_operadores(args.operators, modelos)
        else:
            fenotipos = []
            print("aviso: sin experimento ni --operators, solo se comparan "
                  "los cruces estandar")

        print(f"Experimento   : {raiz_experimento or '(ninguno)'}")
        print(f"Splits        : {', '.join(splits)}")
        print(f"Salida        : {salida}")
        print(f"Cache         : {cache}")

        resultados = {}

        for split in splits:
            print(f"\n{'=' * 72}\nSPLIT {split.upper()}\n{'=' * 72}")
            resultados[split] = evaluar_split(args, split, presupuestos,
                                              fenotipos, modelos,
                                              cache, salida, len(splits) > 1)

        # --- resumen cruzando splits ----------------------------------------
        if len(resultados) > 1:
            print(f"\n{'=' * 72}\nRESUMEN FINAL (train + test)\n{'=' * 72}")
            escribir_resumen_cruzado(resultados, args, salida)

        print(f"\nCompletado en {time.perf_counter() - inicio:.1f}s")
        print(f"Resultados en {salida}")
        return 0

    except (EvaluacionError, IndicatorsError, ReferenceSetError,
            FileNotFoundError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def cargar_operadores_desde(operadores_por_modelo, modelos):
    """
    Aplana `{modelo: {etiqueta: fenotipo}}` en la lista de tres en tres.

    @param operadores_por_modelo: lo que devuelve `load_operators_file`.
    @param modelos: modelos a usar, en orden.
    @return: lista de fenotipos.
    """
    if not operadores_por_modelo:
        return []

    fenotipos = []

    for modelo in modelos:
        if modelo not in operadores_por_modelo:
            raise EvaluacionError(
                f"el experimento no tiene operadores para el modelo "
                f"'{modelo}'; disponibles: {sorted(operadores_por_modelo)}")

        entrada = operadores_por_modelo[modelo]

        if isinstance(entrada, dict):
            for etiqueta in ("Best", "Middle", "Worst"):
                if etiqueta not in entrada:
                    raise EvaluacionError(
                        f"el experimento ({modelo}) no tiene la etiqueta "
                        f"'{etiqueta}'; hay {sorted(entrada)}")
                fenotipos.append(entrada[etiqueta])
        else:
            fenotipos.extend(entrada)

    if len(fenotipos) % 3 != 0:
        raise EvaluacionError(
            f"hay {len(fenotipos)} fenotipos, que no es multiplo de 3; se "
            f"esperan Best/Middle/Worst por modelo.")

    return fenotipos


def _prefijo_split(split: str, anidar: bool) -> str:
    """
    Prefijo de los nombres de archivo de un split.

    Con un solo split el prefijo es vacio, de modo que los nombres coinciden con
    los que producia la version anterior del script (`runs_10k.csv`,
    `plan.csv`, ...). Con varios splits cada archivo lleva el nombre del split
    delante, porque todos acabarían en la misma carpeta raiz.

    @param split: nombre del split.
    @param anidar: si los splits se anidan en subcarpetas propias.
    @return: '' o '<split>_'.
    """
    return f"{split}_" if anidar else ""


def evaluar_split(args, split, presupuestos, fenotipos, modelos, cache,
                  salida, anidar):
    """
    Evalua un split completo y deja los resultados en disco.

    @param args: `argparse.Namespace`.
    @param split: 'train' o 'test'.
    @param presupuestos: `{etiqueta: evaluaciones}`.
    @param fenotipos: lista de fenotipos de tres en tres.
    @param modelos: nombres de los modelos.
    @param cache: carpeta de cache.
    @param salida: carpeta raiz de resultados.
    @param anidar: si True, la salida de este split va a `<salida>/<split>`.
    @return: dict con `split`, `layout`, `por_presupuesto` y `resumen`.
    """
    from mohh.evaluation.params import Params, set_params
    from mohh.core.utilities.instance_utils import instance_paths
    from mohh.evaluation.problem.instance import Instance

    params_dict = {
        "SOLVERS": args.solvers or ["MOEAD", "NSGAII", "SMSEMOA"],
        "N_EXPERIMENTS": args.replicates,
        "EVAL_BUDGETS": {etiqueta: presupuestos[etiqueta]
                         for etiqueta in args.budgets},
        "MO_POPULATION_SIZE": args.population_size,
        "MO_GENERATIONS": args.generations,
        "DATASET": args.dataset,
        "DATASET_TRAIN": "train" if split == "train" else None,
        "DATASET_TEST": "test" if split == "test" else None,
        "PROBLEM_NAME": args.problem,
        "SOLUTION_TYPE": args.solution_type,
        "SAVE": False,
        "LOG_METRICS": False,
        "RANDOM_SEED": args.base_seed,
        "BASE_SEED": args.base_seed,
    }

    set_params(params_dict)

    params = Params()
    solvers = list(params["SOLVERS"])

    # --- referencias de ESTE split --------------------------------------
    if args.references:
        directorio_ref = args.references
    else:
        candidato = os.path.join("references_moea", split, "reference_sets")
        if os.path.isdir(candidato):
            directorio_ref = candidato
        else:
            directorio_ref = os.path.join(
                "set-reference", split, "reference_sets")

    necesita = any(m in ("IGD+", "DELTA_P", "R2") for m in args.metrics)

    referencias = {}
    if necesita:
        if not os.path.isdir(directorio_ref):
            raise EvaluacionError(
                f"faltan las referencias P* de '{split}' en "
                f"'{directorio_ref}'. Los indicadores que las necesitan no se "
                f"pueden calcular.")
        try:
            referencias = load_references(directorio_ref)
        except ReferenceSetError as error:
            raise EvaluacionError(
                f"las referencias P* de '{split}' no son utilizables "
                f"({directorio_ref}): {error}") from error

# Un P* VACIO no da error al cargar, pero no se puede usar. Un P* de un solo
        # punto ya NO es un caso especial: con la escala poolingada la caja sale
        # de la union de P* con los frentes medidos, asi que los cuatro
        # indicadores quedan definidos y comparables. Antes esas instancias se
        # median solo con HV y quedaban fuera de las medias.
        degeneradas = advertencias_referencias(referencias)
        if degeneradas:
            print(f"AVISO: {len(degeneradas)} P* vacio(s) en '{split}' "
                  f"({', '.join(degeneradas)}). Sin puntos de referencia no se "
                  f"puede calcular {'/'.join(_METRICAS_REFERENCIA)}.",
                  file=sys.stderr, flush=True)

        unitarios = pstar_de_un_punto(referencias)
        if unitarios:
            print(f"nota: {len(unitarios)} P* con un unico punto en '{split}' "
                  f"({', '.join(unitarios)}); su frente real es de un punto, "
                  f"asi que la escala de normalizacion se amplia a los valores "
                  f"medidos y los cuatro indicadores se calculan tambien "
                  f"para ellas.",
                  file=sys.stderr, flush=True)

    configuraciones = configuraciones_desde(fenotipos, modelos)

    # --- instancias de ESTE split ---------------------------------------
    rutas = instance_paths(train=(split == "train"))

    if args.instances:
        rutas = [r for r in rutas
                 if os.path.basename(r).removesuffix(".txt")
                 in set(args.instances)]

    if args.limit_instances is not None:
        rutas = rutas[:args.limit_instances]

    if not rutas:
        raise EvaluacionError(f"no hay instancias para el split '{split}'")

    print(f"Split         : {split}")
    print(f"Instancias    : {len(rutas)}")
    print(f"Configuraciones: {len(configuraciones)}")
    print(f"Solvers       : {solvers}")
    print(f"Replicas      : {args.replicates}")
    print(f"Presupuestos  : {args.budgets}")
    print(f"Referencias   : {len(referencias)} P* de {directorio_ref}")

    layout = AnalysisLayout(os.path.join(salida, split) if anidar else salida)
    layout.ensure()

    prefijo = _prefijo_split(split, anidar)

    # `Instance.load_problem` escribe las poblaciones que crea en
    # `FILE_PATH_INITIAL_SOLUTIONS`, asi que hay que dirigirlas al cache propio
    # ANTES de cargar nada; y cada split usa un subcache para que train y test
    # no se pisen.
    params["RESULTS_PATH"] = layout.root
    params["FILE_PATH_INITIAL_SOLUTIONS"] = os.path.join(
        cache, split, "initial_solutions_repo")
    os.makedirs(params["FILE_PATH_INITIAL_SOLUTIONS"], exist_ok=True)

    cache_split = os.path.join(cache, split)

    instancias = {}
    for ruta in rutas:
        instancia = Instance(args.population_size, args.solution_type)
        instancia.load_problem(args.problem, ruta, args.replicates, "")
        instancias[instancia.instance_name] = instancia

    # --- poblaciones iniciales compartidas -------------------------------
    print("\nPoblaciones iniciales compartidas...")
    poblaciones = {}
    for nombre, instancia in instancias.items():
        populations = {
            replica: poblacion_compartida(
                cache_split, nombre, replica, instancia.n_variables,
                args.population_size, args.base_seed)
            for replica in range(1, args.replicates + 1)}

        instancia.initial_solutions_permutations_experiments = {
            f"Experiment_{replica:03d}": poblacion
            for replica, poblacion in populations.items()}

        poblaciones[nombre] = populations
        print(f"  {nombre}: {args.replicates} poblaciones x "
              f"{args.population_size} permutaciones")

    # --- planificacion ----------------------------------------------------
    tareas = construir_tareas(instancias, configuraciones, solvers,
                              args.replicates, args.budgets,
                              params.get_params(), poblaciones)

    print(f"\nTareas totales: {len(tareas)} "
          f"({len(tareas) // max(1, len(instancias))} por instancia)")

    if args.dry_run:
        plan = pd.DataFrame([{
            "split": split,
            "instance": t["instance"], "config": t["config"],
            "solver": t["solver"], "mutation": t["mutation"],
            "replicate": t["replicate"],
            "budgets": ",".join(t["budgets"]),
            "task_signature": firma_tarea(t),
        } for t in tareas])
        ruta_plan = layout.sub(layout.config, f"{prefijo}plan.csv")
        plan.to_csv(ruta_plan, index=False)
        print(f"\n--dry-run: no se ejecuta nada.")
        print(f"Plan escrito en {ruta_plan}")
        return {"split": split, "layout": layout, "por_presupuesto": {},
                "instancias": sorted(instancias), "dry_run": True}

    # --- ejecucion ---------------------------------------------------------
    print()
    resultados = ejecutar_tareas(tareas, cache_split, usar_cache=True,
                                 forzar=args.force, log_dir=args.log_dir)

    pendientes = sum(1 for r in resultados if r is None)
    if pendientes:
        raise EvaluacionError(
            f"{pendientes} tarea(s) sin resultado; no se puede analizar")

    # --- indicadores -------------------------------------------------------
    print("\nCalculando indicadores...")
    por_presupuesto = {}

    for etiqueta in args.budgets:
        tabla = construir_filas(resultados, tareas, [etiqueta], referencias,
                                args.metrics, None)
        tabla.insert(0, "split", split)
        por_presupuesto[etiqueta] = tabla
        tabla.to_csv(layout.sub(layout.individuals,
                                f"{prefijo}runs_{etiqueta}.csv"), index=False)
        print(f"  {etiqueta}: {len(tabla)} filas")

    # --- agregados ---------------------------------------------------------
    print("\nAgregando por configuracion...")
    agregados = []
    for etiqueta, tabla in por_presupuesto.items():
        grupo = (tabla.groupby(["config", "solver", "budget"])
                 .agg(n_replicates=("replicate", "count"),
                      front_size_mean=("front_size", "mean"),
                      solve_s_mean=("solve_s", "mean"),
                      **{metrica: (metrica, "mean")
                         for metrica in args.metrics})
                 .reset_index())
        grupo.insert(0, "split", split)
        for metrica in args.metrics:
            grupo[f"{metrica}_std"] = (
                tabla.groupby(["config", "solver", "budget"])[metrica]
                .std().reset_index(drop=True))
            grupo[f"{metrica}_median"] = (
                tabla.groupby(["config", "solver", "budget"])[metrica]
                .median().reset_index(drop=True))
        agregados.append(grupo)

    pd.concat(agregados, ignore_index=True).to_csv(
        layout.sub(layout.individuals, f"{prefijo}summary.csv"), index=False)

    # --- SDD por configuracion --------------------------------------------
    print("\nSDD por configuracion...")
    for etiqueta, tabla in por_presupuesto.items():
        for metrica in args.metrics:
            matriz = (tabla.groupby(["instance", "config"])[metrica]
                      .mean().unstack("config"))
            if matriz.shape[0] < 2 or matriz.shape[1] < 2:
                continue
            from mohh.core.utilities.algorithm.sdd import SDD
            ranking = SDD(metrica, "mean").ranking(matriz)
            ranking.insert(0, "budget", etiqueta)
            ranking.insert(0, "split", split)
            ranking.to_csv(layout.sub(layout.sdd,
                                      f"{prefijo}sdd_{etiqueta}_{metrica}.csv"),
                           index=False)

    # --- estadisticas ------------------------------------------------------
    print("\nEstadisticas...")
    etiqueta_est = split if anidar else "evaluate_31"
    resumen_est = escribir_estadisticas(
        por_presupuesto, layout, etiqueta_est, args.metrics,
        con_figuras=not args.no_figures)

    # --- ¿el operador bate a los estados base? ----------------------------
    print("\n¿Bate a los estados base?")
    filas = comparar_contra_bases(por_presupuesto, args.metrics)
    escribir_summary_final(filas, layout, split)
    if filas:
        marco = pd.DataFrame(filas)
        for metrica in args.metrics:
            sub = marco[marco["metric"] == metrica]
            if sub.empty:
                continue
            ganadoras = sub[sub["beats_best_baseline"]]["config"].nunique()
            print(f"  {metrica}: {ganadoras}/{sub['config'].nunique()} "
                  f"configuraciones ganan al mejor estado base")
    else:
        print("  sin configuraciones generadas que comparar")

    # --- configuracion guardada -------------------------------------------
    from mohh.analysis.io_utils import write_json

    # La escala de las metricas relativas a P* se guarda junto a la
    # configuracion. Es la que hace comparables los valores entre replicas y
    # configuraciones, y como ahora depende de la union de P* con el
    # consolidado de cada presupuesto, sin estos numeros los CSV no se pueden
    # reproducir ni comparar con otra corrida. Mismo motivo por el que
    # `analyze_search` guarda su escala.
    escala_referencias = {
        etiqueta: tabla.attrs.get("escala_referencias", {})
        for etiqueta, tabla in por_presupuesto.items()}

    write_json(layout.sub(layout.config, f"{prefijo}evaluate_31.json"), {
        "split": split,
        "replicates": args.replicates,
        "budgets": args.budgets,
        "metrics": list(args.metrics),
        "solvers": solvers,
        "configs": [c["config"] for c in configuraciones],
        "instances": sorted(instancias),
        "references_dir": os.path.abspath(directorio_ref),
        "base_seed": args.base_seed,
        "population_size": args.population_size,
        "generations": args.generations,
        "cache": os.path.abspath(cache_split),
        "shared_populations": True,
        "seeds": "SHA-1 de (base_seed, instancia, replica); sin reloj",
        "escala_referencias": escala_referencias,
        "nota_escala": (
            "IGD+/R2/DELTA_P se normalizan con la caja de la union de P* con "
            "el consolidado de frentes de cada presupuesto. Los valores NO son "
            "comparables entre presupuestos distintos (cada uno tiene su caja) "
            "ni con otra corrida cuyo consolidado sea diferente."),
    })

    write_json(layout.sub(layout.config, f"{prefijo}statistics.json"),
               resumen_est)

    return {"split": split, "layout": layout, "por_presupuesto": por_presupuesto,
            "instancias": sorted(instancias), "resumen": resumen_est,
            "dry_run": False}


def escribir_resumen_cruzado(resultados, args, salida):
    """
    Tabla que enfrenta train y test, split a split.

    Responde a la pregunta que un unico split no puede: si el operador generado
    mejora en train Y en test (generaliza) o solo en uno (sobreajusta). La
    comparacion usa la media de las replicas de cada configuracion.

    @param resultados: `{split: salida de evaluar_split}`.
    @param args: `argparse.Namespace`.
    @param salida: carpeta raiz.
    """
    from mohh.analysis.io_utils import write_json

    layout = AnalysisLayout(salida)
    # En la raiz solo se escribe configuracion y la tabla de generalizacion.
    # Las demas secciones (incluida 07_figures) viven dentro de cada split, asi
    # que crearlas aqui dejaba un 07_figures vacio que parecia un fallo de
    # generacion de figuras.
    layout.ensure(layout.config, layout.statistics)

    filas = []

    for split, datos in sorted(resultados.items()):
        if datos["dry_run"]:
            continue
        for etiqueta, tabla in datos["por_presupuesto"].items():
            for metrica in args.metrics:
                if metrica not in tabla.columns:
                    continue
                grupo = (tabla.groupby("config")[metrica]
                         .agg(["mean", "std", "count"]).reset_index())
                for _, fila in grupo.iterrows():
                    filas.append({
                        "split": split,
                        "budget": etiqueta,
                        "metric": metrica,
                        "config": fila["config"],
                        "kind": "baseline" if _es_base(fila["config"])
                                else "individual",
                        "mean": float(fila["mean"]),
                        "std": float(fila["std"]) if pd.notna(fila["std"])
                               else float("nan"),
                        "n": int(fila["count"]),
                    })

    if not filas:
        print("sin datos que comparar entre splits")
        return

    tabla = pd.DataFrame(filas)

    # Tabla ancha: config x (split, metrica) en todos los presupuestos, que es
    # la comparacion que se lee de un vistazo.
    ancho = tabla.pivot_table(index="config", columns=["split", "metric"],
                              values="mean")

    salida_tabla = layout.sub(layout.statistics, "train_vs_test.csv")
    tabla.to_csv(salida_tabla, index=False)
    ancho.to_csv(layout.sub(layout.statistics, "train_vs_test_wide.csv"))

    # Marca de generalizacion: mejora en ambos splits respecto al mejor estado
    # base de cada split.
    marcas = []
    for (etiqueta, metrica), grupo in tabla.groupby(["budget", "metric"],
                                                    sort=True):
        bases = grupo[grupo["kind"] == "baseline"]
        generados = grupo[grupo["kind"] == "individual"]
        if bases.empty or generados.empty:
            continue
        for config in generados["config"].unique():
            fila = generados[generados["config"] == config]
            if fila.empty:
                continue
            gane = {}
            for split in sorted(grupo["split"].unique()):
                sub_g = fila[fila["split"] == split]
                sub_b = bases[bases["split"] == split]
                if sub_g.empty or sub_b.empty:
                    continue
                valor_g = float(sub_g["mean"].iloc[0])
                mejor_b = (float(sub_b["mean"].max()) if metrica == "HV"
                           else float(sub_b["mean"].min()))
                gane[split] = (valor_g > mejor_b if metrica == "HV"
                               else valor_g < mejor_b)
            marcas.append({
                "budget": etiqueta, "metric": metrica, "config": config,
                **{f"beats_baseline_{s}": bool(v) for s, v in gane.items()},
                "generalizes": bool(all(gane.values())) if len(gane) > 1
                               else bool(next(iter(gane.values()))),
            })

    if marcas:
        pd.DataFrame(marcas).to_csv(
            layout.sub(layout.statistics, "generalization.csv"), index=False)

    write_json(layout.sub(layout.config, "splits.json"), {
        "splits": sorted(resultados),
        "instances": {s: d["instancias"] for s, d in sorted(resultados.items())},
        "replicates": args.replicates,
        "budgets": list(args.budgets),
        "metrics": list(args.metrics),
    })

    print(f"Tabla train/test: {salida_tabla}")
    print(f"Generalizacion : "
          f"{layout.sub(layout.statistics, 'generalization.csv')}")


if __name__ == "__main__":
    sys.exit(main())
