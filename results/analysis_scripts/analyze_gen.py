#!/usr/bin/env python
"""
Analisis offline de una corrida de `mohh-generate`.

Lee los individuos y las generaciones que el pipeline ya guardo, y produce, sin
volver a ejecutar el AG y sin inventar datos:

* los cuatro indicadores multiobjetivo (HV, IGD+, DELTA_P, R2) de cada
  individual's, por instancia y su media, contra las referencias P*;
* la SDD de cada generacion y su ranking, con la misma definicion que usa el
  pipeline en vivo (`mohh.core.utilities.algorithm.sdd.SDD`);
* tiempos de los operadores fenotipicos sobre las poblaciones iniciales reales;
* los operadores exportados como modulos de Python ejecutables;
* estadisticas (Friedman + Nemenyi, boxplots, Wilcoxon contra una referencia) y
  el progreso por generacion.

Uso minimo:

    python analyze_gen.py results/R2/Experiment_001

Ejemplos:

    # metricas contra las referencias de train
    python analyze_gen.py results/R2/Experiment_001

    # explicita donde estan las referencias
    python analyze_gen.py results/R2/Experiment_001 \\
        --references set-reference/train/reference_sets

    # solo las piezas que interesen, sin estadisticas ni diagramas
    python analyze_gen.py results/R2/Experiment_001 \\
        --sections indicators sdd --no-figures

El calculo de los indicadores y de la SDD se apoya en `mohh.analysis`, que ya
reproduce exactamente lo que el pipeline midio durante la generacion: para la
corrida real de este repositorio, el R2 y la SDD reconstruidos aqui coinciden
con los valores guardados en `run_metrics/generation_*.json` con un error maximo
del orden de 1e-14.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Permite ejecutar el script desde la raiz del repositorio sin instalar el
# paquete (`python analyze_gen.py`), que es como se usa en la practica.
_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_SRC = os.path.join(_RAIZ, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mohh.analysis import complexity as complexity_mod
from mohh.analysis import phenotype as phenotype_mod
from mohh.analysis import stats as stats_mod
from mohh.analysis.cache import load_cached, save_cached
from mohh.analysis.hv_scale import (
    HvScale,
    HvScaleError,
    load_hv_scale,
    load_hv_scale_from_evaluation,
    save_hv_scale,
)
from mohh.analysis.indicators_all import (
    INDICATORS,
    METRIC_LABELS,
    IndicatorsError,
    compute_indicators,
    fit_hv_contexts,
    load_references,
)
from mohh.analysis.io_utils import load_experiment, write_json
from mohh.analysis.layout import AnalysisLayout, detect_experiment_root
from mohh.analysis.sdd_all import sdd_for_metric, write_sdd
#: Secciones del analisis, en el orden de `AnalysisLayout.SECTIONS`.
SECCIONES = ("indicators", "sdd", "complexity", "operators", "statistics",
             "generations")

#: Individuos por generation: cada generacion es un bloque de la SDD.
GENERACION_POR_DEFECTO = 0


class AnalisisError(RuntimeError):
    """Fallo del analisis, con un mensaje accionable."""


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------


def diagnostico_saturacion(largo, metricas):
    """
    Detecta indicadores que han quedado pegados a un extremo de su rango.

    DELTA_P vale 1.0 cuando ningun punto del frente entra en la caja de
    referencia (`deltap.py:142`), y su nadir esta solo un margen por encima del
    mejor punto de P*, asi que con P* muy buenos los individuos generated quedan
    fuera y el indicador colapsa contra su techo. Cuando eso pasa deja de
    distinguir: dos soluciones muy distintas reciben el mismo valor.

    La comprobacion tambien sirve para IGD+ y R2, que no se saturan, pero así
    queda registrado si en algún momento empiezan a hacerlo.

    @param largo: tabla larga de indicadores.
    @param metricas: metricas pedidas.
    @return: dict metrica -> diagnostico.
    """
    from mohh.analysis.indicators_all import INDICATORS as _IND

    ceilings = {"DELTA_P": 1.0, "IGD+": None, "R2": None, "HV": None}

    salida = {}
    for metrica in metricas:
        if metrica not in largo.columns or metrica not in ceilings:
            continue

        valores = largo[metrica].to_numpy(dtype=float)
        valores = valores[np.isfinite(valores)]
        if valores.size == 0:
            continue

        techo = ceilings[metrica]
        if techo is None:
            continue

        en_techo = float((np.isclose(valores, techo, rtol=0, atol=1e-9)).mean())
        casi = float((valores >= techo - 0.01).mean())
        # Un indicador inutil es el que ademas apenas varia.
        std = float(np.std(valores))

        salida[metrica] = {
            "n": int(valores.size),
            "valor_maximo": techo,
            "fraccion_en_el_techo": round(en_techo, 4),
            "fraccion_a_1_punto_del_techo": round(casi, 4),
            "desviacion_tipica": round(std, 6),
            "saturado": bool(casi >= 0.20),
        }

    return salida


def _inencias_de(experimentos):
    """
    Instancias vistas por uno o varios experimentos.

    `Experiment` no expone la lista de instancias; sale de los frentes de sus
    individuos.

    @param experimentos: lista de `Experiment`.
    @return: conjunto de nombres de instancia.
    """
    return {instancia
            for experimento in experimentos
            for nombre in sorted(experimento.individuals)
            for instancia in experimento.individuals[nombre].fronts}


def _freeze_hv_streaming(experimentos, chunk=25):
    """
    Congela la escala de HV consolidando por lotes de individuos.

    Evita el pico de memoria de `freeze_hv_scale` cuando hay cientos de
    individuos por instancia: mantiene sólo el frente consolidado y un buffer
    pequeño, no la lista completa de frentes.
    """
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    consolidado = {}
    buffer = {}

    def volcar():
        for instancia, frentes in list(buffer.items()):
            bloques = ([consolidado[instancia]]
                       if instancia in consolidado else []) + frentes
            puntos = np.vstack(bloques)
            indices = np.asarray(
                NonDominatedSorting().do(
                    puntos, only_non_dominated_front=True))
            consolidado[instancia] = puntos[indices]
        buffer.clear()

    for experimento in experimentos:
        nombres = sorted(experimento.individuals)
        for i, nombre in enumerate(nombres):
            for instancia, frente in experimento.individuals[nombre].fronts.items():
                buffer.setdefault(instancia, []).append(
                    np.atleast_2d(np.asarray(frente, dtype=float)))
            if (i + 1) % chunk == 0:
                volcar()
    volcar()

    if not consolidado:
        raise HvScaleError("no hay frentes con los que congelar la escala")

    bounds = {instancia: (frente.min(axis=0), frente.max(axis=0))
              for instancia, frente in consolidado.items()}
    return HvScale(
        bounds,
        source="consolidado:" + ",".join(
            os.path.basename(e.root) for e in experimentos))


def escala_hv_compartida(experimentos, metricas, budget=None, escala=None):
    """
    Resuelve la escala de HV con la que se van a medir los indicadores.

    El HV solo es comparable si todos los experimentos que se comparan comparten
    `vmin/vmax` y el mismo `ref_point`. Por eso el orden de preferencia es:

    1. la escala congelada que se pasa, o la de `--hv-scale`;
    2. la que `mohh-evaluate` ya escribio en
       `evaluation_results/run_metrics/evaluation_<instancia>.json`
       (`sections.best_fronts`), que es la unica escala historica existente;
    3. el consolidado de los experimentos dados, con la misma regla que
       `mohh-evaluate` (min/max del frente no dominado consolidado);
    4. sin escala congelada: la de cada experimento por separado, que NO es
       comparable entre ellos.

    @param experimentos: lista de `Experiment`.
    @param metricas: metricas pedidas; si no incluyen HV, no hace falta escala.
    @param budget: presupuesto al leer la escala de `evaluation_results`.
    @param escala: `HvScale` ya construida, si la hay.
    @return: (contexto, escala, comparable, motivo) donde `contexto` es un
        `HypervolumeContexts` o None, `escala` un `HvScale` o None, `comparable`
        dice si los HVs se pueden comparar entre experimentos, y `motivo` explica
        el porque cuando no.
    """
    if "HV" not in metricas:
        return None, None, True, ""

    if escala is not None:
        faltan = escala.missing(_inencias_de(experimentos))
        if faltan:
            raise AnalisisError(
                f"la escala de HV no cubre {len(faltan)} instancia(s): "
                f"{faltan}")
        return escala.contexts(), escala, True, ""

    # 2) la escala que dejo mohh-evaluate.
    try:
        desde_evaluacion = load_hv_scale_from_evaluation(
            experimentos[0].root, budget)
        faltan = desde_evaluacion.missing(_inencias_de(experimentos))
        if not faltan:
            return (desde_evaluacion.contexts(), desde_evaluacion, True, "")
    except HvScaleError:
        pass

    # 3) consolidar lo que hay, con la regla de mohh-evaluate.
    try:
        congelada = _freeze_hv_streaming(experimentos)
        return congelada.contexts(), congelada, True, ""
    except HvScaleError as error:
        motivo = str(error)

    # 4) cada experimento por su cuenta: no comparable.
    if len(experimentos) == 1:
        frente = {nombre: experimentos[0].individuals[nombre].fronts
                  for nombre in sorted(experimentos[0].individuals)}
        contexto = fit_hv_contexts(frente)
        return (contexto, None, False,
                "el HV se ha normalizado con los individuos de este propio "
                "experimento, asi que es comparable entre individuos pero NO "
                f"entre experimentos. Motivo del intento de congelarla: {motivo}")

    contexto = fit_hv_contexts({
        nombre: experimentos[0].individuals[nombre].fronts
        for nombre in sorted(experimentos[0].individuals)})
    return (contexto, None, False,
            "el HV se ha normalizado con los individuos del primer "
            "experimento, asi que NO es comparable entre experimentos")


def calcular_indicadores(experimento, referencias, metricas=INDICATORS,
                         contexto_hv=None):
    """
    Calcula los indicadores de todos los individuos del experimento.

    Los indicadores de minimizacion (IGD+, DELTA_P, R2) no dependen de la escala:
    la fija el propio conjunto de referencia, igual que en el pipeline, y por eso
    son comparables entre experimentos. El HV si depende, asi que se mide con la
    escala que le pase `escala_hv_compartida`; si no se le pasa ninguna, se ajusta
    con los individuos del propio experimento y sus HVs no salen de aqui.

    @param experimento: `Experiment` de `io_utils`.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param metricas: subconjunto de `INDICATORS`.
    @param contexto_hv: `HypervolumeContexts` con la escala compartida.
    @return: (largo, ancho) donde el largo tiene una fila por
        (individuo, instancia) y el ancho una columna por indicador mas
        `mean_indicator`.
    """
    nombres = sorted(experimento.individuals)

    if not nombres:
        raise AnalisisError(
            f"{experimento.root} no tiene ningun individuo; nada que analizar.")

    contexto = contexto_hv
    if "HV" in metricas and contexto is None:
        frente = {nombre: experimento.individuals[nombre].fronts
                  for nombre in nombres}
        contexto = fit_hv_contexts(frente)

    filas = []
    for nombre in nombres:
        individual = experimento.individuals[nombre]
        medidas = compute_indicators(
            individual.fronts, references=referencias,
            metrics=metricas, hv_context=contexto)

        for instancia in medidas.index:
            fila = {"individual": nombre,
                    "instance": instancia,
                    "generation": individual.born_in_generation,
                    "number_nodes": individual.general.get("number_nodes"),
                    "depth": individual.general.get("depth"),
                    "codons_usage": individual.general.get("codons_usage")}
            for metrica in metricas:
                fila[metrica] = float(medidas.loc[instancia, metrica])
            filas.append(fila)

    largo = pd.DataFrame(filas)

    # Media por individual's: es la magnitud que consume la SDD.
    medias = (largo.groupby("individual", sort=True)[list(metricas)]
              .mean()
              .rename(columns={m: f"{m}_mean" for m in metricas}))
    metadatos = largo.groupby("individual", sort=True)[
        ["generation", "number_nodes", "depth", "codons_usage"]].first()
    # El indice pasa a columna: el resto del analisis agrupa por `individual`
    # y necesita poder referirse a el como columna.
    ancho = metadatos.join(medias).reset_index()

    for metrica in metricas:
        clave = f"{metrica}_mean"
        ancho[clave] = pd.to_numeric(ancho[clave], errors="coerce")

    return largo, ancho


# ---------------------------------------------------------------------------
# SDD
# ---------------------------------------------------------------------------


def poblaciones_por_evaluacion(experimento):
    """
    Individuos evaluados en cada evaluacion, tal como los registro el pipeline.

    La SDD solo es comparable dentro del mismo conjunto de columnas, porque
    `b_i` es el mejor valor conocido de cada instancia Y ESO DEPENDE DEL
    CONJUNTO. La agrupacion correcta por tanto no se deduce de
    `born_in_generation` (la poblacion es de estado estacionario: cada
    evaluacion mide 50 supervivientes mas 35 descendientes, y los supervivientes
    nacieron en generaciones distintas), sino de la seccion `reference_fitness`
    de cada `run_metrics/generation_NNNN.json`, que es el registro fiel de que
    individuos se midieron en cada evaluacion.

    @param experimento: `Experiment` de `io_utils`.
    @return: `{(corrida, evaluacion): [nombres de individuos]}`, con las listas
        en el orden en que se evaluaron.
    """
    por_evaluacion = {}

    for corrida, metricas in sorted(experimento.run_metrics.items()):
        entradas = metricas.get("reference_fitness") or []
        for entrada in entradas:
            clave = (corrida, int(entrada["generation"]))
            por_evaluacion.setdefault(clave, []).append(entrada["individual"])

    return por_evaluacion


def calcular_sdd_por_evaluacion(experimento, metricas_sdd, referencias,
                                contextos):
    """
    Calcula la SDD de cada evaluacion con su poblacion real.

    @param experimento: `Experiment` de `io_utils`.
    @param metricas_sdd: metricas para las que se calcula la SDD.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param contextos: escala de HV por instancia ajustada con todos los
        individuos, o None si no se calcula HV.
    @return: dict metrica -> DataFrame largo con el ranking de cada evaluacion.
    """
    from mohh.core.utilities.algorithm.sdd import SDD

    resultados = {}
    poblaciones = poblaciones_por_evaluacion(experimento)

    if not poblaciones:
        return resultados

    for metrica in metricas_sdd:
        filas = []

        for (corrida, evaluacion), individuos in sorted(poblaciones.items()):
            # Un individual's ausente (una corrida parcial) haria la SDD
            # silenciosamente distinta de la online, asi que se avisa.
            faltan = [nombre for nombre in individuos
                      if nombre not in experimento.individuals]
            if faltan:
                raise AnalisisError(
                    f"la evaluacion {evaluacion} de '{corrida}' referencia "
                    f"{len(faltan)} individuo(s) sin datos de frente "
                    f"(p.ej. {faltan[:3]}); no se puede reconstruir su SDD.")

            if len(individuos) < 2:
                continue

            frentes = {nombre: experimento.individuals[nombre].fronts
                       for nombre in individuos}

            try:
                resultado = sdd_for_metric(
                    frentes, metrica, candidates=individuos,
                    references=referencias,
                    hv_context=contextos if metrica == "HV" else None)
            except (IndicatorsError, ValueError) as error:
                raise AnalisisError(
                    f"no se pudo calcular la SDD de '{metrica}' en la "
                    f"evaluacion {evaluacion}: {error}") from error

            ranking = SDD(metrica, "mean").ranking(resultado["matrix"])
            rangos = {operador: posicion + 1 for posicion, operador
                      in enumerate(ranking["Operador"].tolist())}

            for _, fila in ranking.iterrows():
                operador = fila["Operador"]
                filas.append({
                    "run": corrida,
                    "evaluation": evaluacion,
                    "metric": metrica,
                    "individual": operador,
                    "sdd": float(fila["SDD (Fitness)"]),
                    "mean_diff": float(fila["Mean_Diff (x_bar %)"]),
                    "rank": rangos[operador],
                })

        if filas:
            resultados[metrica] = pd.DataFrame(filas).sort_values(
                ["run", "evaluation", "rank"], kind="mergesort")

    return resultados


# ---------------------------------------------------------------------------
# Estadisticas
# ---------------------------------------------------------------------------


def _bloques(largo, metrica, unidades):
    """
    Construye la matriz `instancias x algoritmos` que usa el post-hoc.

    @param largo: tabla con una fila por (individuo, instancia) y la columna
        del indicador.
    @param metrica: nombre del indicador.
    @param unidades: 'generation' o 'individual'.
    @return: DataFrame con instancias en el indice.
    """
    if unidades == "generation":
        # Cada celda es la media de los individuos de esa generacion en esa
        # instancia: asi el bloque es la instancia y el algoritmo la
        # generacion, que es el diseno que Friedman presupone.
        pivote = largo.pivot_table(index="instance", columns="generation",
                                   values=metrica, aggfunc="mean")
        pivote.columns = [f"gen_{int(c):03d}" for c in pivote.columns]
        pivote = pivote.sort_index(axis=1)
        pivote.index.name = "instance"
        return pivote

    pivote = largo.pivot_table(index="instance", columns="individual",
                               values=metrica, aggfunc="mean")
    pivote = pivote[sorted(pivote.columns)]
    pivote.index.name = "instance"
    return pivote


def suelo_p_wilcoxon(n):
    """
    Menor p-valor bilateral que puede dar un Wilcoxon de rangos con `n` muestras.

    Con n pares, el signo de cada diferencia aporta un bit, y el caso extremo
    (todas las diferencias del mismo signo) tiene probabilidad `2/2^n`. Ese es el
    suelo del test: ningun conjunto de datos, por perfecto que sea, puede bajar
    de el. Con las 6 instancias de train da 0.03125, asi que tras una correccion
    de Holm sobre ~30 comparaciones nada alcanza 0.05 y el test no tiene potencia
    para decidir nada.

    @param n: numero de muestras pareadas.
    @return: p-valor minimo, o None si n es insuficiente.
    """
    if n < 1:
        return None
    return 2.0 / (2.0 ** n)


def advertencias_post_hoc(frame, unidades):
    """
    Dice por que un post-hoc sobre una sola corrida no es un test de algoritmos.

    Con `--stats-units generation` o `individual`, las columnas de la matriz no
    son metodos que compitan entre si: son instantes consecutivos de una misma
    corrida (o individuos sueltos de una misma poblacion). Friedman y Wilcoxon
    presuponen bloques independientes, y generaciones consecutivas de un GA estan
    fuertemente autocorrelacionadas. Ademas, con pocas instancias el Wilcoxon no
    tiene potencia.

    @param frame: matriz `bloques x columnas` del post-hoc.
    @param unidades: 'generation' o 'individual'.
    @return: lista de motivos; vacia si no hay nada que advertir.
    """
    motivos = []

    nombres = {
        "generation": "generaciones consecutivas de una sola corrida",
        "individual": "individuos sueltos de una sola poblacion",
    }
    motivos.append(
        f"las columnas son {nombres.get(unidades, unidades)}, no metodos que "
        f"compitan entre si; {'las generaciones consecutivas estan autocorrelacionadas, lo que rompe la independencia de los bloques' if unidades == 'generation' else 'los individuos de una poblacion no son replicas independientes'}")

    suelo = suelo_p_wilcoxon(frame.shape[0])
    if suelo is not None:
        motivos.append(
            f"con {frame.shape[0]} instancias, el menor p-valor bilateral "
            f"alcanzable es 2/2^{frame.shape[0]} = {suelo:.5f}, de modo que "
            f"tras corregir {frame.shape[1]} comparaciones con Holm "
            f"nada puede quedar por debajo de 0.05")

    return motivos


def escribir_estadisticas(largo, metricas, layout, etiqueta, con_figuras=True,
                          unidades="generation", max_algoritmos=25,
                          permitir=False):
    """
    Friedman + Nemenyi, boxplots y Wilcoxon sobre los datos del experimento.

    La unidad de comparacion NO es el individual's: una corrida de generacion
    puede tener cientos de individuos y un post-hoc de Nemenyi sobre ellos no
    tiene sentido estadistico (los individuos no son metodos, son muestras de
    una misma poblacion). La unidad por defecto es la GENERACION, y los bloques
    son las INSTANCIAS, que es el diseno que Friedman presupone. El Wilcoxon
    contra la mejor generacion responde a la pregunta util de esta tesis: si la
    poblacion mejora a medida que avanza la generacion.

    @param largo: tabla larga devuelta por `calcular_indicadores` (una fila por
        (individuo, instancia)).
    @param metricas: metricas a comparar.
    @param layout: `AnalysisLayout` de destino.
    @param etiqueta: sufijo de los archivos.
    @param con_figuras: si False, no se dibujan diagramas.
    @param unidades: 'generation' (una columna por generacion) o 'individual'
        (una columna por individual's).
    @param max_algoritmos: numero maximo de columnas antes de omitir los
        diagramas.
    @param permitir: si True, ejecuta el post-hoc aunque las columnas no sean
        metodos comparables. Por defecto False: se escribe en su lugar
        `POST_HOC_SKIPED.txt` con el motivo, porque con una sola corrida el
        post-hoc no es un test de algoritmos y, con pocas instancias, el
        Wilcoxon no tiene potencia para concluir nada.
    @return: dict metrica -> dict con el resumen del analisis.
    """
    resumen = {}

    if unidades not in ("generation", "individual"):
        raise AnalisisError(
            f"--stats-units debe ser 'generation' o 'individual', "
            f"no '{unidades}'")

    for metrica in metricas:
        if metrica not in largo.columns:
            continue

        frame = _bloques(largo, metrica, unidades)

        if frame.shape[1] < 2:
            continue

        if np.isnan(frame.to_numpy()).any():
            raise AnalisisError(
                f"'{metrica}' tiene valores ausentes; no se puede comparar. "
                f"Revisa las referencias P* cargadas.")

        base = {"n_candidates": int(frame.shape[1]),
                "n_blocks": int(frame.shape[0]),
                "units": unidades}

        motivos = advertencias_post_hoc(frame, unidades)

        if not permitir and motivos:
            base["post_hoc"] = "omitido"
            base["motivos"] = motivos
            base["suelo_p_wilcoxon"] = suelo_p_wilcoxon(frame.shape[0])

            # El descriptivo se conserva: la tabla instancia x generacion es
            # util para ver la evolucion, lo que no se puede es inferir de ella.
            frame.to_csv(layout.sub(layout.statistics,
                                    f"values_{etiqueta}_{metrica}.csv"))

            resumen[metrica] = base
            continue

        if frame.shape[0] >= 3 and frame.shape[1] >= 3:
            statistic, pvalue = stats_mod.friedman_test(frame)
            base.update({"friedman_statistic": float(statistic),
                         "friedman_p": float(pvalue),
                         "cd_nemenyi": stats_mod.nemenyi_cd(frame.shape[0],
                                                            frame.shape[1])})
        else:
            base["nota"] = ("Friedman necesita >= 3 bloques y >= 3 algoritmos; "
                            "se omite.")

        resumen[metrica] = base

        frame.to_csv(layout.sub(layout.statistics,
                                f"values_{etiqueta}_{metrica}.csv"))

        # Wilcoxon contra la columna con mejor media: mayor para HV, menor para
        # las metricas de minimizacion.
        referencia = (frame.mean(axis=0).idxmax() if metrica == "HV"
                      else frame.mean(axis=0).idxmin())

        if frame.shape[0] >= 6:
            wilcoxon = stats_mod.wilcoxon_vs_reference(frame,
                                                       reference=referencia)
            wilcoxon.to_csv(layout.sub(
                layout.statistics,
                f"wilcoxon_{etiqueta}_{metrica}.csv"), index=False)

        if not con_figuras:
            continue

        if frame.shape[1] > max_algoritmos:
            print(f"  {metrica}: {frame.shape[1]} columnas; se omite el "
                  f"diagrama (>{max_algoritmos}). Sube --stats-max-columns "
                  f"si lo quieres.")
            continue

        try:
            stats_mod.critical_difference_diagram(
                frame, metric=metrica,
                path=layout.sub(layout.figures, f"cd_{etiqueta}_{metrica}.png"),
                title=f"{METRIC_LABELS.get(metrica, metrica)} ({etiqueta})")
            stats_mod.boxplot_configs(
                frame, metric=metrica,
                path=layout.sub(layout.figures,
                                f"boxplot_{etiqueta}_{metrica}.png"))
        except Exception as error:  # noqa: BLE001 - degradar, no fallar
            print(f"  aviso: no se pudo dibujar {metrica} ({error})")

    return resumen


# ---------------------------------------------------------------------------
# Progreso por generacion
# ---------------------------------------------------------------------------


def escribir_generaciones(ancho, metricas, layout, etiqueta):
    """
    Resume cada generacion: media de la SDD y de los indicadores.

    @param ancho: tabla larga de individuals.
    @param metricas: metricas incluidas.
    @param layout: `AnalysisLayout` de destino.
    @param etiqueta: sufijo de los archivos.
    @return: DataFrame con una fila por generacion.
    """
    columnas_medias = [f"{m}_mean" for m in metricas
                       if f"{m}_mean" in ancho.columns]

    resumen = (ancho.groupby("generation", sort=True)
               .agg(n_individuals=("individual", "count"),
                    **{columna: (columna, "mean")
                       for columna in columnas_medias})
               .reset_index())

    resumen.to_csv(layout.sub(layout.generations,
                              f"generations_{etiqueta}.csv"), index=False)
    return resumen


def grafico_convergencia(resumen, metricas, layout, etiqueta):
    """
    Dibuja la evolucion de cada indicador a lo largo de las generaciones.

    Es la figura que responde a "va convergiendo?": una curva que baja (IGD+,
    DELTA_P, R2) o que sube (HV) y se estabiliza indica que la poblacion mejora y
    luego satura; una curva que oscila indica que la generacion no aporta.

    @param resumen: tabla de `escribir_generaciones`.
    @param metricas: metricas incluidas.
    @param layout: `AnalysisLayout` de destino.
    @param etiqueta: sufijo de los archivos.
    @return: lista de rutas escritas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rutas = []

    for metrica in metricas:
        columna = f"{metrica}_mean"
        if columna not in resumen.columns:
            continue

        serie = resumen.sort_values("generation")
        valores = serie[columna].to_numpy(dtype=float)

        if np.isnan(valores).all() or len(valores) < 2:
            continue

        figura, eje = plt.subplots(figsize=(7.0, 4.0))

        eje.plot(serie["generation"], valores, marker="o", markersize=3,
                 linewidth=1.2, color="#1f77b4")
        # Media movil: separa la tendencia del ruido de una sola generacion.
        if len(valores) >= 5:
            ventana = max(3, len(valores) // 10)
            suavizado = pd.Series(valores).rolling(
                ventana, min_periods=1, center=True).mean()
            eje.plot(serie["generation"], suavizado, linewidth=2.0,
                     color="#d62728", label=f"media movil ({ventana})")

        if metrica == "HV":
            eje.set_ylabel("HV (mayor es mejor)")
            eje.set_yscale("log")
        else:
            eje.set_ylabel(f"{METRIC_LABELS.get(metrica, metrica)} "
                           f"(menor es mejor)")

        eje.set_xlabel("generacion de nacimiento")
        eje.set_title(f"Convergencia {METRIC_LABELS.get(metrica, metrica)} "
                      f"({etiqueta})")
        eje.grid(alpha=0.3)
        if len(valores) >= 5:
            eje.legend(loc="best", fontsize=8)
        figura.tight_layout()

        ruta = layout.sub(layout.figures,
                          f"convergence_{etiqueta}_{metrica}.png")
        figura.savefig(ruta, dpi=200)
        plt.close(figura)
        rutas.append(ruta)

    return rutas


# ---------------------------------------------------------------------------
# Complejidad y operadores
# ---------------------------------------------------------------------------

def escribir_complejidad(experimento, layout, etiqueta, repeticiones, semilla,
                         cruces):
    """
    Cronometra los operadores fenotipicos sobre las poblaciones reales.

    @param experimento: `Experiment` de `io_utils`.
    @param layout: `AnalysisLayout` de destino.
    @param etiqueta: sufijo de los archivos.
    @param repeticiones: llamadas por pareja.
    @param semilla: semilla de `np.random`.
    @param cruces: cruces por generacion, para el coste por generacion.
    @return: DataFrame largo, o None si no hay poblaciones iniciales.
    """
    poblaciones = complexity_mod.load_populations(experimento)

    if not poblaciones:
        return None

    filas = []
    for modelo, operadores in sorted(experimento.operators.items()):
        tabla = complexity_mod.measure_complexity(
            operadores, poblaciones, repeats=repeticiones, seed=semilla,
            matings=cruces)
        if tabla.empty:
            continue
        tabla.insert(0, "model", modelo)
        filas.append(tabla)

    if not filas:
        return None

    resultado = pd.concat(filas, ignore_index=True)
    resultado.to_csv(layout.sub(layout.complexity,
                                f"complexity_{etiqueta}.csv"), index=False)
    return resultado


def _boxplot_horizontal(eje, datos, etiquetas):
    """
    Boxplot horizontal compatible con varias versiones de matplotlib.

    `orientation="horizontal"` sustituyo a `vert=False` en matplotlib 3.11, y
    `vert` se retira en la 3.13. Se prueba la forma moderna y se cae a la
    antigua si el matplotlib instalado es anterior.

    @param eje: `matplotlib.axes.Axes`.
    @param datos: lista de listas de valores.
    @param etiquetas: lista de nombres.
    @return: lo que devuelve `Axes.boxplot`.
    """
    try:
        return eje.boxplot(datos, orientation="horizontal",
                           tick_labels=etiquetas, patch_artist=True)
    except TypeError:
        return eje.boxplot(datos, vert=False, tick_labels=etiquetas,
                           patch_artist=True)


def grafico_complejidad(complejidad, layout, etiqueta):
    """
    Boxplot de los segundos por pareja y por generacion de cada operador.

    La pregunta que responde es si el coste del operador generado es comparable
    al de los cruces estandar: un operador que multiplica por diez el tiempo de
    una pareja puede mejorar la calidad y aun asi ser inviable en la practica.

    @param complejidad: tabla larga de `escribir_complejidad`.
    @param layout: `AnalysisLayout` de destino.
    @param etiqueta: sufijo de los archivos.
    @return: lista de rutas escritas.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if complejidad is None or complejidad.empty:
        return []

    rutas = []
    columnas_tiempo = [c for c in ("per_pair_s", "per_generation_s")
                       if c in complejidad.columns]

    for columna in columnas_tiempo:
        datos = complejidad.dropna(subset=[columna])
        if datos.empty:
            continue

        # Una caja por (modelo, operador): cada punto es una instancia.
        grupos = (datos.groupby(["model", "operator"], sort=True)[columna]
                  .apply(list))

        etiquetas = [f"{modelo}/{operador}"
                     for modelo, operador in grupos.index]

        figura, eje = plt.subplots(
            figsize=(max(7.0, 0.9 * len(grupos) + 3.0), 4.5))

        # `orientation` sustituye a `vert` en matplotlib 3.11; se usa la forma
        # moderna cuando existe para no depender de una opcion retirada.
        caja = _boxplot_horizontal(eje, grupos.tolist(), etiquetas)

        for caja_individual in caja["boxes"]:
            caja_individual.set_facecolor("#aec7e8")
            caja_individual.set_edgecolor("#1f77b4")

        # Puntos por encima de la caja: una instancia = una muestra.
        pos = np.concatenate([np.full(len(v), i + 1)
                              for i, v in enumerate(grupos.tolist())])
        eje.scatter(pos + np.random.RandomState(0).uniform(
            -0.08, 0.08, size=pos.size),
            np.concatenate([np.asarray(v) for v in grupos.tolist()]),
            s=14, color="#d62728", alpha=0.6, zorder=3)

        eje.set_xscale("log")
        eje.set_xlabel("segundos" + (" por pareja" if columna == "per_pair_s"
                                      else " por generacion"))
        eje.set_title(f"Complejidad de los operadores ({etiqueta})")
        eje.grid(axis="x", alpha=0.3)
        figura.tight_layout()

        nombre = "per_pair" if columna == "per_pair_s" else "per_generation"
        ruta = layout.sub(layout.figures,
                          f"complexity_{nombre}_{etiqueta}.png")
        figura.savefig(ruta, dpi=200)
        plt.close(figura)
        rutas.append(ruta)

    return rutas


def escribir_operadores(experimento, layout, etiqueta):
    """
    Exporta cada operador fenotipico como modulo de Python.

    @param experimento: `Experiment` de `io_utils`.
    @param layout: `AnalysisLayout` de destino.
    @param etiqueta: sufijo de los archivos.
    @return: lista de rutas escritas.
    """
    rutas = []

    for modelo, operadores in sorted(experimento.operators.items()):
        base = layout.sub(layout.operators, modelo)
        os.makedirs(base, exist_ok=True)

        indice = []
        for nombre, fenotipo in sorted(operadores.items()):
            archivo = os.path.join(
                base, f"{phenotype_mod.sanitize_filename(nombre)}.py")
            try:
                phenotype_mod.write_python_module(
                    fenotipo, archivo,
                    name=f"operar_{phenotype_mod.sanitize_filename(nombre)}")
            except (OSError, ValueError, SyntaxError) as error:
                raise AnalisisError(
                    f"no se pudo exportar el operador '{nombre}' de "
                    f"'{modelo}': {error}") from error
            rutas.append(archivo)
            indice.append({"label": nombre, "phenotype": fenotipo,
                           "functions": phenotype_mod.used_functions(fenotipo),
                           "arity": phenotype_mod.arity_of(fenotipo),
                           "module": os.path.basename(archivo)})

        pd.DataFrame(indice).to_csv(os.path.join(base, "operators.csv"),
                                    index=False)

    return rutas


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _digest_referencia(referencia) -> str:
    """
    Huella corta del contenido de un `ReferenceSet`.

    `ReferenceSet` no expone un `digest`, asi que se calcula sobre el frente en
    escala original: si el `.pof` cambia, cambia la huella y la cache no se
    reutiliza por error.
    """
    import hashlib

    frente = np.ascontiguousarray(np.asarray(referencia.front, dtype=float))
    return hashlib.sha1(frente.tobytes()).hexdigest()[:12]


def _digest_individuos(experimento) -> str:
    """
    Huella corta de la poblacion de individuos de un experimento.

    Sin esto, dos experimentos distintos que compartieran carpeta de cache
    devolverian los indicadores del primero: la clave solomiraba las metricas y
    las referencias, no QUIEN se estaba midiendo. La huella cubre el nombre de
    cada individuo, su generacion de nacimiento y las instancias que tiene, que
    es lo que determina el valor de cada celda.

    @param experimento: `Experiment` de `io_utils`.
    @return: hash corto.
    """
    import hashlib

    resumen = hashlib.sha1()

    for nombre in sorted(experimento.individuals):
        individuo = experimento.individuals[nombre]
        resumen.update(
            f"{nombre}|{individuo.born_in_generation}|"
            f"{','.join(individuo.instances)}\n".encode("utf-8"))

    return resumen.hexdigest()[:16]


def clave_cache(metricas, referencias, experimento=None, escala_hv=None):
    """
    Firma estable de un conjunto de metricas + referencias + experimento.

    El experimento forma parte de la clave a proposito: los indicadores de
    `results/CP/Experiment_001` y los de `results/R2/Experiment_001` se miden
    contra las mismas P*, pero sobre poblaciones distintas, y confundirlos
    daria por conclusion que dos experimentos coinciden.

    Ademas, el HV solo es comparable cuando comparte la escala `vmin/vmax`. Por
    eso la clave tambien incluye una huella de esa escala cuando se usa.

    @param metricas: metricas pedidas.
    @param referencias: dict nombre -> `ReferenceSet`.
    @param experimento: `Experiment` de `io_utils`, o None.
    @param escala_hv: `HvScale` o None.
    @return: hash corto.
    """
    import hashlib

    partes = [",".join(sorted(metricas))]

    if experimento is not None:
        partes.append(f"exp={os.path.abspath(experimento.root)}")
        partes.append(f"ind={_digest_individuos(experimento)}")

    for nombre in sorted(referencias):
        partes.append(f"{nombre}:{_digest_referencia(referencias[nombre])}")

    if escala_hv is not None:
        partes.append(f"hv:{escala_hv.source}:{escala_hv.ref_point}")
        for nombre in sorted(escala_hv.bounds):
            lo, hi = escala_hv.bounds[nombre]
            partes.append(f"hvb:{nombre}:{','.join(f'{x:.8g}' for x in lo.tolist())}|"
                          f"{','.join(f'{x:.8g}' for x in hi.tolist())}")

    return hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()[:16]


def cargar_cache(cache_dir, metricas, referencias, experimento=None,
                 escala_hv=None):
    """
    Devuelve la tabla larga cacheada, o None si no hay o no encaja.

    @param cache_dir: carpeta de cache.
    @param metricas: metricas pedidas.
    @param referencias: referencias con las que se calculo.
    @param experimento: `Experiment` de `io_utils`, o None.
    @return: DataFrame largo o None.
    """
    if not cache_dir:
        return None

    ruta = os.path.join(
        cache_dir,
        f"indicators_{clave_cache(metricas, referencias, experimento, escala_hv)}.csv")
    if not os.path.isfile(ruta):
        return None

    try:
        return pd.read_csv(ruta)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def construir_parser():
    """Parser de argumentos de `analyze_gen.py`."""
    parser = argparse.ArgumentParser(
        description="Analisis offline de una corrida de mohh-generate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("experiment",
                        help="ruta del experimento (results/.../Experiment_NNN)")
    parser.add_argument(
        "--references", default="references_moea/train/reference_sets",
        help="carpeta con las referencias P* (.pof)")
    parser.add_argument(
        "--output", default=None,
        help="carpeta de salida; por defecto '<experimento>/analysis'")
    parser.add_argument(
        "--metrics", nargs="+", default=list(INDICATORS), choices=list(INDICATORS),
        help="indicadores a calcular")
    parser.add_argument(
        "--sdd-metrics", nargs="+", default=list(INDICATORS),
        choices=list(INDICATORS),
        help="metricas para las que ademas se escribe el ranking SDD por "
             "generacion; por defecto, las cuatro")
    parser.add_argument(
        "--sections", nargs="+", default=list(SECCIONES), choices=list(SECCIONES),
        help="partes del analisis a ejecutar")
    parser.add_argument(
        "--no-figures", action="store_true",
        help="no dibujar diagramas de diferencias criticas ni boxplots")
    parser.add_argument(
        "--stats-units", default="generation",
        choices=["generation", "individual"],
        help="unidad de comparacion del post-hoc: generacion (recomendado) o "
             "individual")
    parser.add_argument(
        "--allow-unpaired-stats", action="store_true",
        help="ejecutar Friedman/Nemenyi/Wilcoxon aunque las columnas no sean "
             "metodos comparables (generaciones o individuos de una sola "
             "corrida). Los p-valores que salen no respaldan conclusiones; "
             "por defecto el post-hoc se omite y se explica por que")
    parser.add_argument(
        "--stats-max-columns", type=int, default=25,
        help="columnas maximas antes de omitir los diagramas por ilegibilidad")
    parser.add_argument(
        "--complexity-repeats", type=int, default=5,
        help="llamadas por pareja al cronometrar un operador")
    parser.add_argument("--seed", type=int, default=0,
                        help="semilla de np.random para la complejidad")
    parser.add_argument(
        "--matings", type=int, default=None,
        help="cruces por generacion para el coste por generacion")
    parser.add_argument(
        "--cache", default=None,
        help="carpeta de cache de indicadores (reutiliza el calculo si los "
             "datos no cambian)")
    parser.add_argument(
        "--force", action="store_true",
        help="ignorar la cache y recalcular siempre")
    parser.add_argument(
        "--hv-scale", default=None,
        help="ruta de una escala de HV congelada (JSON) a reutilizar; si no se "
             "indica, se intenta reconstruirla desde evaluation_results si "
             "existe")
    parser.add_argument(
        "--hv-budget", default=None,
        choices=["10k", "30k", "50k"],
        help="presupuesto a usar al leer la escala de HV de evaluation_results")

    return parser


def main(argv=None):
    """
    Punto de entrada.

    @param argv: argumentos; si es None se leen de la linea de comandos.
    @return: codigo de salida (0 correcto, 1 error).
    """
    args = construir_parser().parse_args(argv)

    inicio = time.perf_counter()

    try:
        raiz = detect_experiment_root(args.experiment)
        # `analyze_gen` no reevalua permutaciones: solo necesita frentes,
        # metadatos y el indice de generaciones. Evitar cargar los
        # `solution_set` de las 80 instancias reduce mucho la memoria en
        # experimentos grandes (1100 individuos).
        experimento = load_experiment(
            raiz, load_generations=False, solution_instances=set(),
            dtype=np.float32)
        # El contenido de `generations` no se usa: solo se necesita saber
        # cuantas evaluaciones hubo. Se evita cargar ~600 MB de JSON.
        n_generaciones = 0
        for corrida in experimento.runs:
            carpeta = os.path.join(corrida, "generation_results", "generations")
            if os.path.isdir(carpeta):
                n_generaciones += sum(
                    1 for nombre in os.listdir(carpeta)
                    if nombre.startswith("generation_"))
        experimento.generations = {i: {} for i in range(n_generaciones)}
        layout = AnalysisLayout(args.output or os.path.join(raiz, "analysis"))
        layout.ensure()

        etiqueta = os.path.basename(raiz)

        write_json(layout.sub(layout.config, "experiment.json"), {
            "experiment_root": raiz,
            "runs": experimento.runs,
            "models": experimento.models,
            "labels": experimento.labels,
            "n_individuals": len(experimento.individuals),
            "n_generations": len(experimento.generations),
            "instances": sorted({i for ind in experimento.individuals.values()
                                 for i in ind.instances}),
            "metrics": list(args.metrics),
            "sdd_metrics": list(args.sdd_metrics),
            "sections": list(args.sections),
            "references_dir": os.path.abspath(args.references),
        })

        print(f"Experimento   : {raiz}")
        print(f"Individuos    : {len(experimento.individuals)}")
        # `generations` son los ficheros `run_metrics/generation_*.json`, uno por
        # evaluacion. Las generaciones de nacimiento (las que agrupan a los
        # individuos) pueden ser una menos, porque la ultima evaluacion consume
        # individuos nacidos antes.
        nacimientos = len({individual.born_in_generation
                           for individual in experimento.individuals.values()})
        print(f"Evaluaciones  : {len(experimento.generations)} "
              f"(generaciones de nacimiento: {nacimientos})")
        print(f"Salida        : {layout.root}")

        metricas = tuple(args.metrics)

        # Las metricas que necesitan P* fallan pronto y con un mensaje claro si
        # las referencias no estan disponibles.
        necesita_referencias = any(m in ("IGD+", "DELTA_P", "R2")
                                   for m in metricas)
        referencias = {}

        if necesita_referencias:
            if not os.path.isdir(args.references):
                raise AnalisisError(
                    f"faltan las referencias P* en '{args.references}'. Los "
                    f"indicadores {sorted(m for m in metricas if m in ('IGD+', 'DELTA_P', 'R2'))} "
                    f"no se pueden calcular sin ellas. Genera las referencias "
                    f"o quita esos indicadores con --metrics.")
            referencias = load_references(args.references)
            print(f"Referencias   : {len(referencias)} conjuntos P* "
                  f"de {args.references}")

        # Escala de HV. Se resuelve antes de calcular nada porque de ella depende
        # la comparabilidad del HV entre experimentos.
        escala_hv = None
        if "HV" in metricas:
            if args.hv_scale:
                try:
                    escala_hv = load_hv_scale(args.hv_scale)
                except HvScaleError as error:
                    raise AnalisisError(str(error)) from error
                print(f"Escala HV     : congelada de {args.hv_scale} "
                      f"({len(escala_hv)} instancias, "
                      f"ref_point={escala_hv.ref_point})")
            else:
                escala_hv = None

        contexto_hv, escala_usada, hv_comparable, motivo_hv = (
            escala_hv_compartida([experimento], metricas, args.hv_budget,
                                 escala_hv))

        if "HV" in metricas:
            if escala_usada is not None:
                print(f"Escala HV     : {len(escala_usada)} instancias, "
                      f"ref_point={escala_usada.ref_point}, "
                      f"origen={escala_usada.source}")
                # La escala se escribe junto a los resultados: sin ella el HV no
                # se puede auditar ni rehacer, y antes no quedaba registrada en
                # ningun sitio.
                ruta_escala = save_hv_scale(
                    escala_usada,
                    layout.sub(layout.config, "hv_scale.json"))
                print(f"Escala HV     : escrita en {ruta_escala}")
            if not hv_comparable:
                print(f"AVISO HV      : {motivo_hv}")

        largo = ancho = None
        saturacion = {}

        if "indicators" in args.sections or "generations" in args.sections:
            largo = None if args.force else cargar_cache(
                args.cache, metricas, referencias, experimento, escala_usada)

            if largo is not None:
                print("Indicadores   : desde cache")
                metadatos = largo.groupby("individual", sort=True)[
                    ["generation", "number_nodes", "depth",
                     "codons_usage"]].first()
                medias = (largo.groupby("individual", sort=True)[list(metricas)]
                          .mean()
                          .rename(columns={m: f"{m}_mean"
                                           for m in metricas}))
                ancho = metadatos.join(medias).reset_index()
            else:
                largo, ancho = calcular_indicadores(experimento, referencias,
                                                    metricas, contexto_hv)
                print(f"Indicadores   : {len(largo)} filas "
                      f"({len(ancho)} individuos x "
                      f"{largo['instance'].nunique()} instancias)")

                if args.cache:
                    os.makedirs(args.cache, exist_ok=True)
                    largo.to_csv(os.path.join(
                        args.cache,
                        f"indicators_{clave_cache(metricas, referencias, experimento, escala_usada)}.csv"),
                        index=False)

            # Los CSV se escriben siempre, tambien desde cache: la carpeta de
            # resultados debe estar completa aunque el calculo se haya reutilizado.
            largo.to_csv(layout.sub(layout.individuals,
                                    f"indicators_{etiqueta}.csv"), index=False)
            ancho.to_csv(layout.sub(layout.individuals,
                                    f"per_individual_{etiqueta}.csv"))

            saturacion = diagnostico_saturacion(largo, metricas)
            for metrica, datos in sorted(saturacion.items()):
                if datos["saturado"]:
                    print(f"AVISO {metrica:<7}: saturado contra el techo "
                          f"({datos['valor_maximo']}). "
                          f"{datos['fraccion_en_el_techo'] * 100:.0f}% de las "
                          f"celdas valen exactamente el techo y "
                          f"{datos['fraccion_a_1_punto_del_techo'] * 100:.0f}% "
                          f"estan a menos de 0.01; desviacion tipica "
                          f"{datos['desviacion_tipica']:.4f}. No distingue "
                          f"individuos: no lo uses para decidir ni lo "
                          f"presentes como resultado.")

        if "generations" in args.sections and ancho is not None:
            resumen = escribir_generaciones(ancho, metricas, layout, etiqueta)
            print(f"Progreso      : {len(resumen)} generaciones de nacimiento "
                  f"en {layout.generations}")
            try:
                rutas_fig = grafico_convergencia(resumen, metricas, layout,
                                                 etiqueta)
                if rutas_fig:
                    print(f"Convergencia  : {len(rutas_fig)} figuras en "
                          f"{layout.figures}")
            except Exception as e:
                print(f"aviso: no se pudo generar figuras de convergencia: {e}",
                      file=sys.stderr)

        if "sdd" in args.sections:
            contextos = None
            if "HV" in (metricas or ()):
                contextos = fit_hv_contexts(
                    {nombre: experimento.individuals[nombre].fronts
                     for nombre in sorted(experimento.individuals)})

            rankings = calcular_sdd_por_evaluacion(
                experimento, args.sdd_metrics, referencias, contextos)

            if not rankings:
                if not args.sdd_metrics:
                    print("SDD           : omitida (usa --sdd-metrics)")
                elif not poblaciones_por_evaluacion(experimento):
                    print("SDD           : omitida; 'run_metrics/*.json' de "
                          "esta corrida no trae la seccion 'reference_fitness', "
                          "que es el registro de que individuos se midieron en "
                          "cada evaluacion. Sin el no se puede reconstruir la "
                          "SDD por evaluacion. Los indicadores si estan.")
                else:
                    print(f"SDD           : sin rankings para "
                          f"{args.sdd_metrics}")
            else:
                for metrica, tabla in rankings.items():
                    tabla.to_csv(layout.sub(layout.sdd,
                                            f"sdd_{etiqueta}_{metrica}.csv"),
                                 index=False)
                    mejores = (tabla[tabla["rank"] == 1]
                               .sort_values("evaluation")
                               .set_index("evaluation")["individual"])
                    print(f"SDD {metrica:<8}: {len(tabla)} filas en "
                          f"{tabla['evaluation'].nunique()} evaluaciones; "
                          f"mejor en las 3 primeras = "
                          f"{dict(mejores.head(3))}")

        if "complexity" in args.sections:
            complejidad = escribir_complejidad(
                experimento, layout, etiqueta, args.complexity_repeats,
                args.seed, args.matings)
            if complejidad is None:
                print("Complejidad   : sin poblaciones iniciales")
            else:
                print(f"Complejidad   : {len(complejidad)} filas en "
                      f"{layout.complexity}")
                try:
                    rutas_fig = grafico_complejidad(complejidad, layout,
                                                    etiqueta)
                    if rutas_fig:
                        print(f"Tiempos       : {len(rutas_fig)} boxplots en "
                              f"{layout.figures}")
                except Exception as e:
                    print(f"aviso: no se pudo generar el boxplot de "
                          f"complejidad: {e}", file=sys.stderr)

        if "operators" in args.sections:
            rutas = escribir_operadores(experimento, layout, etiqueta)
            print(f"Operadores    : {len(rutas)} modulos en {layout.operators}")

        resumen = {}
        if "statistics" in args.sections and largo is not None:
            resumen = escribir_estadisticas(
                largo, metricas, layout, etiqueta,
                con_figuras=not args.no_figures,
                unidades=args.stats_units,
                max_algoritmos=args.stats_max_columns,
                permitir=args.allow_unpaired_stats)
            for metrica, datos in sorted(resumen.items()):
                if "friedman_p" in datos:
                    print(f"Estadisticas  : {metrica} Friedman "
                          f"p={datos['friedman_p']:.3e} "
                          f"CD={datos['cd_nemenyi']:.3f} "
                          f"({datos['n_blocks']} instancias x "
                          f"{datos['n_candidates']} {datos['units']})")
                elif datos.get("post_hoc") == "omitido":
                    print(f"Estadisticas  : {metrica} post-hoc OMITIDO "
                          f"({datos['n_blocks']} instancias x "
                          f"{datos['n_candidates']} {datos['units']})")
                else:
                    print(f"Estadisticas  : {metrica} {datos['nota']}")

        omitidos = sorted(m for m, d in resumen.items()
                          if d.get("post_hoc") == "omitido")
        if omitidos:
            ruta = layout.sub(layout.statistics, "POST_HOC_SKIPED.txt")
            with open(ruta, "w", encoding="utf-8") as archivo:
                titulo = f"POST-HOC OMITIDO en {', '.join(omitidos)}"
                archivo.write(titulo + "\n" + "=" * len(titulo) + "\n\n")
                archivo.write(
                    "Friedman, Nemenyi y Wilcoxon NO se han calculado, y sus\n"
                    "p-valores no se deben citar. Motivos:\n\n")
                motivos = resumen[omitidos[0]]["motivos"]
                for i, motivo in enumerate(motivos, 1):
                    archivo.write(f"  {i}. {motivo}\n")
                archivo.write(
                    "\nQue hacer en su lugar:\n"
                    "  - La comparacion que si es valida entre experimentos la\n"
                    "    hace evaluate_31.py sobre el split de test, que tiene 16\n"
                    "    instancias: alli el suelo del Wilcoxon es 2/2^16 y si\n"
                    "    alcanza potencia.\n"
                    "  - Con 6 instancias de train, ningun test pareado puede\n"
                    "    resultar significativo; presenta las diferencias como\n"
                    "    descriptivas (medias por generacion) y no como\n"
                    "    significancia ni como evidencia estadistica.\n"
                    "  - Si necesitas la tabla aunque todo, ejecuta con\n"
                    "    --allow-unpaired-stats, aceptando que sus p-valores no\n"
                    "    respaldan ninguna conclusion.\n")
            print(f"Estadisticas  : post-hoc omitido; motivo en {ruta}")

        write_json(layout.sub(layout.config, "summary.json"), {
            "seconds": round(time.perf_counter() - inicio, 3),
            "n_individuals": len(experimento.individuals),
            "n_evaluations": len(experimento.generations),
            "n_birth_generations": len({individual.born_in_generation
                                       for individual
                                       in experimento.individuals.values()}),
            "sections": list(args.sections),
            "hv": {
                "asked": "HV" in metricas,
                "comparable_entre_experimentos": bool(hv_comparable),
                "motivo": motivo_hv,
                "ref_point": (escala_usada.ref_point
                              if escala_usada is not None else None),
                "origen": (escala_usada.source
                           if escala_usada is not None else "por_individuo"),
            } if "HV" in metricas else None,
            "saturacion": saturacion,
        })

        print(f"Completado en {time.perf_counter() - inicio:.2f}s")
        return 0

    except (AnalisisError, IndicatorsError, FileNotFoundError,
            ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
