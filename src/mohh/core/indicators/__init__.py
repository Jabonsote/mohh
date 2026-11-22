"""
Indicadores multiobjetivo del proyecto (mQAP, minimizacion).

Direccion (ver `mohh.core.indicators.directions`): **HV se MAXIMIZA**;
**IGD+, DELTA_P y R2 se MINIMIZAN**.

Proveniencia de cada uno:

* :func:`igd`, :func:`igd_plus` -- implementacion **oficial de pymoo**
  (`pymoo.indicators.igd.IGD` y `pymoo.indicators.igd_plus.IGDPlus`, con
  calculo en `moocore`), envuelta con la validacion y la escala por objetivo
  del proyecto.
* HV -- tambien oficial de pymoo, via `pymoo.indicators.hv.HV` en
  `mohh.core.utilities.algorithm.MO.compute_hypervolume`.
* :func:`delta_p` -- DeltaP. **No existe** en pymoo ni en moocore, asi que se
  implementa aqui.
* :func:`r2` -- R2 de Hansen & Jaszkiewicz. **No existe** en pymoo; la unica
  version de moocore (`r2_exact`) soporta solo dos objetivos, asi que se
  implementa aqui.

Todos se reportan en la convencion "menor es mejor", salvo HV.
"""

from mohh.core.indicators.deltap import delta_p, non_dominated
from mohh.core.indicators.directions import (
    DIRECTIONS,
    MAXIMIZATION_METRICS,
    MINIMIZATION_METRICS,
    aggregation_for_metric,
    axis_hint,
    direction_label,
    indicator_minimization,
)
from mohh.core.indicators.igd import igd, igd_plus
from mohh.core.indicators.r2 import r2, uniform_weights
from mohh.core.indicators.reference_sets import (
    NADIR_MARGIN,
    ReferenceSet,
    ReferenceSetError,
    load_reference_sets,
    pooled_reference_sets,
    read_pof,
)

__all__ = [
    "DIRECTIONS",
    "MAXIMIZATION_METRICS",
    "MINIMIZATION_METRICS",
    "aggregation_for_metric",
    "axis_hint",
    "delta_p",
    "direction_label",
    "igd",
    "igd_plus",
    "indicator_minimization",
    "non_dominated",
    "r2",
    "uniform_weights",
    "NADIR_MARGIN",
    "ReferenceSet",
    "ReferenceSetError",
    "load_reference_sets",
    "pooled_reference_sets",
    "read_pof",
]
