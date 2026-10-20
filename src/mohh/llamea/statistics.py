"""Estadistica pareada para comparar operadores de cruce.

El diseno experimental de ``mohh/evaluation/experiment.py`` evalua cada
configuracion con las **mismas** poblaciones iniciales (``N_EXPERIMENTS``
repeticiones). Por eso la comparacion correcta es pareada: para cada repeticion
se resta el HV del clasico al del operador candidato y se prueba si la
distribucion de diferencias es distinta de cero.

Aqui viven las funciones que comparten ``scripts/compare_operators.py`` y la
evaluacion final de ``mohh.llamea.run``. Se mantienen fuera del nucleo GE.
"""

from __future__ import annotations

import math

import numpy as np


def finite_pairs(operator_values, baseline_values):
    """Pares ``(operador, clasico)`` donde ambos valores son finitos.

    Devuelve ``(diffs, n_missing)``. Un ``None``/``NaN`` en cualquiera de los dos
    lados rompe el pareado y se contabiliza como perdido; nunca se sustituye por
    cero de forma silenciosa (eso sesgaria la media a favor del operador con mas
    fallos).
    """
    if len(operator_values) != len(baseline_values):
        raise ValueError("operator_values and baseline_values must have the same length")
    diffs = []
    missing = 0
    for ours, base in zip(operator_values, baseline_values):
        ours_ok = ours is not None and np.isfinite(ours)
        base_ok = base is not None and np.isfinite(base)
        if not (ours_ok and base_ok):
            missing += 1
            continue
        diffs.append(float(ours) - float(base))
    return diffs, missing


def wilcoxon_signed_rank(differences):
    """Wilcoxon pareado de dos colas; ``(estadistico, p)`` o ``(None, None)``.

    ``zero_method="wilcox"`` descarta las diferencias exactamente cero, que es
    la convencion habitual cuando hay empates. Con menos de dos diferencias no
    nulas la distribucion no permite rechazar nada.
    """
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
    except ImportError:  # pragma: no cover - scipy es dependencia de pymoo
        return None, None
    nonzero = [float(value) for value in differences if value != 0]
    if len(nonzero) < 2:
        return None, None
    result = scipy_wilcoxon(
        differences, zero_method="wilcox", alternative="two-sided"
    )
    return float(result.statistic), float(result.pvalue)


def _t_critical(confidence, degrees_of_freedom):
    """Valor critico t de Student (o normal si scipy no esta disponible)."""
    alpha = 1.0 - float(confidence)
    try:
        from scipy.stats import t as student

        return float(student.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    except Exception:  # pragma: no cover - fallback sin scipy
        # Aproximacion normal; suficiente para un intervalo descriptivo.
        return 1.959963984540054 if confidence == 0.95 else 2.0


def paired_summary(operator_values, baseline_values, confidence=0.95):
    """Resumen pareado de un operador contra una linea base.

    ``operator_values`` y ``baseline_values`` deben estar alineados por
    repeticion (misma semilla / misma poblacion inicial). El resumen incluye
    victorias/derrotas/empates, medias calculadas **solo sobre los pares
    validos**, diferencia media, desviacion de las diferencias, ``dz`` de
    Cohen, intervalo de confianza t y p de Wilcoxon.

    Los valores faltantes se reportan en ``missing`` y no se imputan.
    """
    operator_values = list(operator_values)
    baseline_values = list(baseline_values)
    diffs, missing = finite_pairs(operator_values, baseline_values)
    ours = [float(a) for a, b in zip(operator_values, baseline_values)
            if a is not None and b is not None
            and np.isfinite(a) and np.isfinite(b)]
    base = [float(b) for a, b in zip(operator_values, baseline_values)
            if a is not None and b is not None
            and np.isfinite(a) and np.isfinite(b)]
    n = len(diffs)
    summary = {
        "pairs": n,
        "missing": missing,
        "wins": sum(1 for value in diffs if value > 0),
        "losses": sum(1 for value in diffs if value < 0),
        "ties": sum(1 for value in diffs if value == 0),
        "operator_mean": float(np.mean(ours)) if ours else None,
        "baseline_mean": float(np.mean(base)) if base else None,
        "mean_difference": float(np.mean(diffs)) if diffs else None,
        "sd_difference": float(np.std(diffs, ddof=1)) if n > 1 else None,
        "cohens_dz": None,
        "ci95_low": None,
        "ci95_high": None,
        "relative": None,
        "wilcoxon_statistic": None,
        "wilcoxon_p": None,
    }
    if n > 1:
        mean = float(np.mean(diffs))
        sd = float(np.std(diffs, ddof=1))
        if sd > 0:
            summary["cohens_dz"] = mean / sd
            half_width = _t_critical(confidence, n - 1) * sd / math.sqrt(n)
            summary["ci95_low"] = mean - half_width
            summary["ci95_high"] = mean + half_width
        else:
            # Diferencia constante: el intervalo es el punto (y dz no aplica).
            summary["ci95_low"] = mean
            summary["ci95_high"] = mean
    if summary["operator_mean"] is not None and summary["baseline_mean"]:
        summary["relative"] = (
            summary["operator_mean"] / summary["baseline_mean"] - 1.0
        )
    statistic, pvalue = wilcoxon_signed_rank(diffs)
    summary["wilcoxon_statistic"] = statistic
    summary["wilcoxon_p"] = pvalue
    return summary


def holm_adjust(pvalues):
    """Ajuste de Holm (familia de comparaciones).

    Devuelve una lista alineada con ``pvalues``. Los ``None`` se conservan.
    El ajuste controla la tasa de error por familia (FWER) sin suponer
    independencia entre comparaciones.
    """
    indexed = [(index, float(value)) for index, value in enumerate(pvalues)
               if value is not None and np.isfinite(value)]
    adjusted = [None] * len(pvalues)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        candidate = (total - rank) * value
        running = max(running, min(1.0, candidate))
        adjusted[index] = running
    return adjusted


def add_holm(rows, pkey="wilcoxon_p", adjusted_key="wilcoxon_p_holm"):
    """Agrega ``p_holm`` y una marca de significancia a una tabla de filas."""
    adjusted = holm_adjust([row.get(pkey) for row in rows])
    for row, value in zip(rows, adjusted):
        row[adjusted_key] = value
        row["significant_holm"] = (
            bool(value is not None and value < 0.05)
        )
    return rows


def minimum_detectable_difference(operator_values, baseline_values,
                                  confidence=0.95):
    """Diferencia minima detectable (MDD) para un contraste pareado.

    Es el semi-ancho del intervalo t al nivel de confianza pedido. No es una
    prueba de equivalencia: sirve para saber si una diferencia observada es
    siquiera distinguible con el numero de repeticiones disponible.
    """
    diffs, _missing = finite_pairs(operator_values, baseline_values)
    if len(diffs) < 2:
        return None
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(
        _t_critical(confidence, len(diffs) - 1) * sd / math.sqrt(len(diffs))
    )
