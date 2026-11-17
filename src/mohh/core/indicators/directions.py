"""
Direcciones de cada indicador: que se MAXIMIZA y que se MINIMIZA.

Registro unico de verdad para todo el proyecto. `stats.average_ranks`,
`stats.wilcoxon_vs_reference`, `sdd`, `analyze_search` y las etiquetas de las
figuras leen de aqui en vez de repetir la prueba `metric == "HV"`.

    HV        -> maximizar (mayor volumen dominado es mejor)
    IGD+      -> minimizar  (menor distancia al P* es mejor)
    DELTA_P   -> minimizar  (menos volumen ortogonal sin cubrir es mejor)
    R2        -> minimizar  (menor error de proyeccion es mejor)

La mayoria de los indicadores de `pymoo` devuelven el valor crudo sin invertir
el signo (IGD, IGD+, HV, Spacing, ...), asi que la direccion no la decide la
libreria: la decide aqui el proyecto, que trabaja en minimizacion de objetivos.
"""

from __future__ import annotations

__all__ = [
    "MAXIMIZATION_METRICS",
    "MINIMIZATION_METRICS",
    "DIRECTIONS",
    "direction_label",
    "axis_hint",
    "indicator_minimization",
    "aggregation_for_metric",
]

#: Indicadores que se MAXIMIZAN.
MAXIMIZATION_METRICS = frozenset({"HV"})

#: Indicadores que se MINIMIZAN.
MINIMIZATION_METRICS = frozenset({"IGD+", "DELTA_P", "R2"})

#: `metrica -> "maximizar" | "minimizar"`.
DIRECTIONS = {
    **{metrica: "maximizar" for metrica in MAXIMIZATION_METRICS},
    **{metrica: "minimizar" for metrica in MINIMIZATION_METRICS},
}


def indicator_minimization(metric: str) -> bool:
    """True si el indicador es de minimizacion (menor es mejor)."""
    return metric in MINIMIZATION_METRICS


def direction_label(metric: str) -> str:
    """
    Etiqueta legible de la direccion del indicador.

    @param metric: nombre del indicador (`"HV"`, `"IGD+"`, ...).
    @return: `"maximizar"` o `"minimizar"`.
    """
    if metric in DIRECTIONS:
        return DIRECTIONS[metric]
    # Un indicador nuevo sin registrar se trata como minimizacion, que es la
    # convencion por defecto del proyecto; se avisa en vez de fallar.
    return "minimizar"


def axis_hint(metric: str) -> str:
    """Sufijo para etiquetas de eje: `(mayor es mejor)` / `(menor es mejor)`."""
    if indicator_minimization(metric):
        return "(menor es mejor)"
    return "(mayor es mejor)"


def aggregation_for_metric(metric: str) -> str:
    """
    Agregacion para quedarse con el MEJOR valor de un grupo.

    @return: `"max"` para HV, `"min"` para el resto.
    """
    return "min" if indicator_minimization(metric) else "max"
