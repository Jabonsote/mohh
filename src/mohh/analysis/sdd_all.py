"""
SDD (Success Difference Degree) para los cuatro indicadores, en analisis offline.

La SDD se calcula sobre una matriz `O_i` de resultados: filas = instancias,
columnas = los operadores (o individuos) que se comparan. Para cada celda se
mide el indicador (HV, IGD+, DELTA_P o R2) contra la referencia que
corresponde, y despues la SDD razona sobre esas medidas con la misma
definicion del protocolo:

    b_i : mejor valor conocido en la instancia i (max para HV, min para los
          indicadores de minimizacion)
    x_i : |O_i - b_i| / ((O_i + b_i) / 2) * 100
    x_b : media de las diferencias de cada operador
    SDD: desviacion muestral de {x_i} sobre N - 1 grados de libertad

Se apoya en `mohh.core.utilities.algorithm.sdd.SDD`, que ya valida entradas y
ordena el ranking, de modo que la convencion no se duplica aqui. Este modulo
solo aporta la construccion de la matriz por indicador y la escritura de los
artefactos (rankings, diferencias, texto legible).
"""

import numpy as np
import pandas as pd

from mohh.core.utilities.algorithm.sdd import SDD
from mohh.analysis.indicators_all import INDICATORS, METRIC_LABELS

__all__ = [
    "SDDError",
    "sdd_for_metric",
    "sdd_table",
    "write_sdd",
]


class SDDError(RuntimeError):
    """No hay datos suficientes para calcular una SDD."""


def sdd_table(
    indicators_frame: pd.DataFrame,
    metric: str = "HV",
    aggregation: str = "mean",
    candidate_order=None,
) -> pd.DataFrame:
    """
    Construye la matriz `O_i` de un indicador y calcula su ranking SDD.

    @param indicators_frame: DataFrame `instancia x candidato` con los valores
        del indicador. Puede venir en formato largo? No: debe ser ancho, con
        instancias en el indice y candidatos en las columnas.
    @param metric: indicador usado para la SDD.
    @param aggregation: como colapsar varias corridas por celda si las hay
        (`mean` o `median`); si solo hay una medida por celda no cambia nada.
    @param candidate_order: orden forzado de las columnas (operadores); por
        defecto se conserva el del DataFrame.
    @return: (ranking, b_i, x_i, matrix, score, mean_diff) con la tabla de
        ranking (`Operador`, `Mean_Diff`, `SDD`), la serie `b_i` por instancia,
        las diferencias `x_i`, la matriz original `O_i` colapsada, y las series
        `score` (la SDD, que es la aptitud que consume el GE) y `mean_diff`
        (`x_b`, la media de las diferencias) indexadas por candidato.
    """
    if metric not in INDICATORS:
        raise SDDError(f"indicador '{metric}' no soportado; usa uno de {INDICATORS}")

    matrix = indicators_frame.copy()

    if candidate_order is not None:
        faltan = [c for c in candidate_order if c not in matrix.columns]
        if faltan:
            raise SDDError(
                f"faltan candidatos en la matriz de indicadores: {faltan}")
        matrix = matrix[list(candidate_order)]

    if aggregation in ("mean", "median") and matrix.shape[1] and _has_duplicates(matrix):
        colapsar = np.mean if aggregation == "mean" else np.median
        matrix = matrix.groupby(level=0, sort=True).agg(colapsar)

    if matrix.shape[0] < 2:
        raise SDDError(
            f"la SDD muestral necesita N >= 2 instancias; hay {matrix.shape[0]}")

    if matrix.shape[1] == 0:
        raise SDDError("la matriz de indicadores no tiene candidatos (columnas)")

    sdd = SDD(metric, aggregation)
    ranking, b_i, x_i = sdd.compute(matrix)

    return ranking, b_i, x_i, matrix, sdd.score(matrix), sdd.mean_diff(matrix)


def _has_duplicates(frame: pd.DataFrame) -> bool:
    return bool(frame.columns.duplicated().any())


def sdd_for_metric(
    fronts_by_candidate: dict,
    metric: str,
    candidates=None,
    references: dict = None,
    hv_context=None,
    aggregation: str = "mean",
) -> dict:
    """
    Calcula la matriz de un indicador y su SDD para varios candidatos.

    @param fronts_by_candidate: `{candidato: {instancia: (n, m) ndarray}}`.
    @param metric: indicador a medir.
    @param candidates: orden de candidatos; por defecto, el de las claves.
    @param references: dict nombre -> `ReferenceSet` para los indicadores de
        minimizacion.
    @param hv_context: escala fija de HV compartida (opcional).
    @param aggregation: 'mean' o 'median'.
    @return: dict con 'indicators' (matriz `O_i` por candidato), 'ranking',
        'b_i', 'x_i', 'matrix', 'score' (SDD = aptitud) y 'mean_diff' (`x_b`).
    """
    from mohh.analysis.indicators_all import compute_indicators

    candidates = list(candidates or fronts_by_candidate.keys())
    instancias = sorted({i for c in candidates
                         for i in fronts_by_candidate.get(c, {})})

    matriz = {}
    for candidato in candidates:
        frentes = fronts_by_candidate[candidato]
        medidas = compute_indicators(
            {nombre: frentes[nombre] for nombre in instancias if nombre in frentes},
            references=references, metrics=(metric,), hv_context=hv_context)
        matriz[candidato] = medidas[metric]

    indicadores = pd.DataFrame(matriz)[candidates]
    indicadores.index.name = "instance"

    ranking, b_i, x_i, matrix, score, mean_diff = sdd_table(
        indicadores, metric=metric, aggregation=aggregation,
        candidate_order=candidates)

    return {"indicators": indicadores, "ranking": ranking, "b_i": b_i,
            "x_i": x_i, "matrix": matrix, "score": score,
            "mean_diff": mean_diff}


def write_sdd(result: dict, out_dir: str, metric: str, prefix: str = "",
              tag: str = "") -> list:
    """
    Escribe los artefactos de una SDD (CSV + texto) y devuelve sus rutas.

    @param result: salida de `sdd_for_metric`.
    @param out_dir: carpeta de destino (se crea si no existe).
    @param metric: nombre del indicador.
    @param prefix: prefijo de archivo (por ejemplo el nombre del experimento).
    @param tag: sufijo extra (por ejemplo el presupuesto).
    @return: lista de rutas escritas.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    etiqueta = METRIC_LABELS.get(metric, metric)
    base = "_".join(p for p in (prefix, etiqueta, tag) if p)

    rutas = []

    ranking_path = os.path.join(out_dir, f"sdd_ranking_{base}.csv")
    result["ranking"].to_csv(ranking_path, index=False)
    rutas.append(ranking_path)

    diff_path = os.path.join(out_dir, f"sdd_differences_{base}.csv")
    diferencias = result["x_i"].copy()
    diferencias.insert(0, "b_i", result["b_i"])
    diferencias.to_csv(diff_path)
    rutas.append(diff_path)

    indicadores_path = os.path.join(out_dir, f"indicators_{base}.csv")
    result["indicators"].to_csv(indicadores_path)
    rutas.append(indicadores_path)

    texto_path = os.path.join(out_dir, f"sdd_ranking_{base}.txt")
    with open(texto_path, "w") as handle:
        handle.write(f"SDD ranking - indicador {metric} - menor es mejor\n")
        handle.write("=" * 60 + "\n")
        handle.write(result["ranking"].to_string(index=False) + "\n")
    rutas.append(texto_path)

    return rutas
