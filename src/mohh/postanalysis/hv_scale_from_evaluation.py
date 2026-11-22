#!/usr/bin/env python
"""
Congela la escala de HV que uso la evaluacion final (`mohh-evaluate`).

`mohh-evaluate` mide el hipervolumen normalizando con los `vmin`/`vmax` del
frente no dominado consolidado de su propia invocacion y contra el nadir
`nadir_ref_point = [1.1] * n_obj`. Esos limites quedan escritos en
`evaluation_results/run_metrics/evaluation_<instancia>.json`
(`sections.best_fronts`), y son la UNICA fuente existente de una escala que
reproduce los numeros de la tabla de evaluacion.

Este script los lee de uno o mas experimentos, los une en una sola escala
compartible y la guarda en un JSON que `analyze_search.py --hv-scale` puede
reutilizar. Asi el HV del analisis de la busqueda es el mismo que el de la
evaluacion final, para test y (cuando la evaluacion no cubrio train) con la
misma receta aplicada a train: `analyze_search` rellena lo que falte con el
consolidado de la corrida y el MISMO `ref_point`.

Uso:

    python hv_scale_from_evaluation.py \\
        --experiment experimentos_001_002/Experiment_001 \\
        --experiment experimentos_001_002/Experiment_002 \\
        --budget 50k \\
        --output results/baseline/hv_scale_eval_shared.json
"""

from __future__ import annotations

import argparse
import os
import sys

_RAIZ = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_RAIZ, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from mohh.analysis.hv_scale import (  # noqa: E402
    HvScale,
    HvScaleError,
    load_hv_scale_from_evaluation,
    save_hv_scale,
)
from mohh.postanalysis.analyze_search import combinar_escalas  # noqa: E402


def unir_escalas(escalas):
    """
    Une las escalas tomando, por objetivo, el `vmin` minimo y el `vmax` maximo.

    Cada `mohh-evaluate` recalcula sus `boundaries` con el consolidado de su
    propia invocacion, asi que dos experimentos jamas dan los mismos
    `vmin/vmax` para la misma instancia: no se pueden usar por separado sin
    perder la comparacion entre corridas. La UNION da una caja que contiene a
    las de las dos corridas, es la MISMA para las dos (asi el HV es
    comparable entre experimentos) y sigue saliendo de la evaluacion final.

    @param escalas: escalas con el mismo `ref_point`.
    @return: `HvScale` unica.
    """
    if not escalas:
        raise HvScaleError("no hay escalas que unir")
    ref_point = escalas[0].ref_point
    bounds = {}
    for escala in escalas:
        if abs(escala.ref_point - ref_point) > 1e-12:
            raise HvScaleError(
                f"ref_point distinto: {escala.ref_point} vs {ref_point}")
        for nombre, (bajo, alto) in escala.bounds.items():
            if nombre in bounds:
                previo = bounds[nombre]
                bounds[nombre] = (np.minimum(previo[0], bajo),
                                  np.maximum(previo[1], alto))
            else:
                bounds[nombre] = (bajo, alto)

    fuentes = "+".join(dict.fromkeys(escala.source for escala in escalas))
    return HvScale(bounds, ref_point=ref_point,
                   source=f"union-eval:{fuentes}",
                   usados=[f"{len(escala)} instancias de {escala.source}"
                           for escala in escalas])


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Escala de HV tomada de la evaluacion final (mohh-evaluate).")
    parser.add_argument("--experiment", action="append", required=True,
                        help="raiz de un experimento; se puede repetir para "
                             "unir varias evaluaciones en una escala comun")
    parser.add_argument("--merge", default="union",
                        choices=["union", "strict"],
                        help="union: caja que contiene a las de todas las "
                             "corridas (comparable entre experimentos, por "
                             "defecto); strict: exige que todas den los "
                             "mismos vmin/vmax")
    parser.add_argument("--budget", default="50k",
                        help="presupuesto a usar (10k/30k/50k); los "
                             "presupuestos NO comparten vmin/vmax")
    parser.add_argument("--output", required=True,
                        help="JSON de salida para --hv-scale")
    return parser


def main(argv=None) -> int:
    args = construir_parser().parse_args(argv)

    escalas = []
    try:
        for raiz in args.experiment:
            escala = load_hv_scale_from_evaluation(raiz, args.budget)
            print(f"  {raiz}: {len(escala)} instancias, ref_point="
                  f"{escala.ref_point}, source={escala.source}")
            escalas.append(escala)
        if args.merge == "strict":
            combinada = combinar_escalas(escalas)
        else:
            combinada = unir_escalas(escalas)
    except (HvScaleError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 1

    if combinada is None:
        print("ERROR: ninguna escala cargada", file=sys.stderr)
        return 1

    ruta = save_hv_scale(combinada, args.output)
    print(f"Escala de la evaluacion final ({args.budget}): "
          f"{len(combinada)} instancias, ref_point={combinada.ref_point} "
          f"-> {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
