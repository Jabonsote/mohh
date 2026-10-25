"""
Indicadores multiobjetivo del proyecto (mQAP, minimizacion).

Todos se reportan en la convencion "menor es mejor":

* :func:`igd`  -- distancia generacional invertida.
* :func:`igd_plus` -- IGD+ weakly preference-based.
* :func:`delta_p` -- DeltaP, volumen ortogonal no cubierto.
* :func:`r2` -- R2 de Hansen & Jaszkiewicz (2008).

`pymoo` solo trae IGD/IGD+ y en convencion de MAXIMIZACION; DeltaP y R2 no
estan disponibles, asi que se implementan aqui para minimizacion.
"""

from mohh.core.indicators.deltap import delta_p, non_dominated
from mohh.core.indicators.igd import igd, igd_plus
from mohh.core.indicators.r2 import r2, uniform_weights

__all__ = [
    "delta_p",
    "igd",
    "igd_plus",
    "non_dominated",
    "r2",
    "uniform_weights",
]
