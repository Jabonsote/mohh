"""
Escala de hipervolumen congelada y compartible entre experimentos.

Por que existe
--------------
El HV sale de dos cosas: la escala `vmin/vmax` con la que se normaliza el frente
y el `ref_point` contra el que se mide el volumen. Si la escala se ajusta con los
propios individuos de cada corrida, dos experimentos con poblaciones distintas
quedan en escalas distintas y sus HV no significan nada al compararlos.

Que hacia `mohh-evaluate`
------------------------
`src/mohh/evaluation/experiment.py` normaliza con
`normalize_values(front, boundaries[label][0], boundaries[label][1])` y mide
contra `nadir_point = np.array([1.1] * n_obj)` fijo. Los `boundaries` son el
minimo y el maximo por objetivo del frente no dominado consolidado de TODAS las
configuraciones, solvers y experimentos de esa invocacion. Es decir: una sola
consolidacion fija la escala y a partir de ahi todo se mide en las mismas
unidades, y el `ref_point` es una constante que no depende de nadie.

Ese esquema es correcto DENTRO de una invocacion, pero no sobrevive a la
comparacion entre corridas: cada `mohh-evaluate` recalcula sus `boundaries` desde
su propio consolidado y no carga nada de disco, asi que `Experiment_001` y
`Experiment_002` medidos por separado no comparten escala. Eso es exactamente el
fallo que arrastraba el analisis offline.

Lo que hace este modulo
-----------------------
Congela la escala en un JSON reutilizable: se calcula una vez sobre el
consolidado de los experimentos que se quieran comparar, se guarda, y todos
miden con esos mismos `vmin/vmax` y el mismo `ref_point`. La escala queda ademas
escrita junto a los resultados, de modo que el HV se puede auditar y rehacer.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from mohh.analysis.indicators_all import (
    DEFAULT_HV_REF_POINT,
    HypervolumeContext,
    HypervolumeContexts,
)

__all__ = [
    "HvScaleError",
    "HvScale",
    "freeze_hv_scale",
    "save_hv_scale",
    "load_hv_scale",
    "contexts_from_scale",
    "load_hv_scale_from_evaluation",
    "instancias_sin_escala",
    "restringir_escala",
    "conflictos_escala",
    "diferencias_escala",
]


class HvScaleError(RuntimeError):
    """La escala de HV no se puede construir, cargar o aplicar."""


class HvScale:
    """
    Escala de HV por instancia: `vmin`, `vmax` y `ref_point`.

    @param bounds: `{instancia: (vmin, vmax)}` con vectores por objetivo.
    @param ref_point: punto de referencia normalizado, 1.1 como en
        `mohh-evaluate`.
    @param source: procedencia (ruta del JSON, 'consolidado', etc.), para poder
        auditar de donde sale la escala.
    """

    def __init__(self, bounds: Dict[str, tuple], ref_point: float = DEFAULT_HV_REF_POINT,
                 source: str = "", usados: Optional[List[str]] = None) -> None:
        self.bounds = {nombre: (np.asarray(lo, dtype=float),
                                np.asarray(hi, dtype=float))
                       for nombre, (lo, hi) in bounds.items()}
        self.ref_point = float(ref_point)
        self.source = source
        self.usados = list(usados or [])

        for nombre, (lo, hi) in self.bounds.items():
            if lo.shape != hi.shape:
                raise HvScaleError(
                    f"'{nombre}': vmin {lo.shape} y vmax {hi.shape} no coinciden")
            if np.any(hi < lo):
                raise HvScaleError(
                    f"'{nombre}': hay objetivos con vmax < vmin ({hi} vs {lo}); "
                    "la escala esta invertida")

    def __len__(self) -> int:
        return len(self.bounds)

    def __contains__(self, instance_name: str) -> bool:
        return instance_name in self.bounds

    def covers(self, instances: Iterable[str]) -> bool:
        """Si la escala cubre todas las instancias pedidas."""
        return set(instances) <= set(self.bounds)

    def missing(self, instances: Iterable[str]) -> List[str]:
        """Instancias pedidas que no tienen escala."""
        return sorted(set(instances) - set(self.bounds))

    def to_dict(self) -> dict:
        """Version serializable, estable y legible."""
        return {
            "ref_point": self.ref_point,
            "source": self.source,
            "usados": list(self.usados),
            "instances": {
                nombre: {"vmin": lo.tolist(), "vmax": hi.tolist()}
                for nombre, (lo, hi) in sorted(self.bounds.items())
            },
        }

    def contexts(self) -> HypervolumeContexts:
        """`HypervolumeContexts` con estos limites exactos, sin reajustar."""
        contexts = {}
        for nombre, (lo, hi) in self.bounds.items():
            contexto = HypervolumeContext(ref_point=self.ref_point)
            contexto.vmin = lo
            contexto.vmax = hi
            contexto._z_ref = np.array([self.ref_point] * lo.shape[0], dtype=float)
            contexts[nombre] = contexto
        return HypervolumeContexts(contexts)

    def __repr__(self) -> str:
        return (f"HvScale({len(self.bounds)} instancias, ref_point="
                f"{self.ref_point}, source={self.source!r})")


def _consolidado(frentes: Iterable[np.ndarray]) -> np.ndarray:
    """
    Une frentes y se queda con el frente no dominado conjunto.

    Es el `non_dominated_sorting_vectorized(consolidated)[0]` de
    `mohh-evaluate` (experiment.py:509), pero usando el NDS de pymoo en vez de
    su doble bucle en Python: el criterio es el mismo (minimizacion, no
    dominados) y con seis experimentos y 1100 individuos el doble bucle seria
    inviable.
    """
    bloques = [np.atleast_2d(np.asarray(f, dtype=float))
               for f in frentes if np.asarray(f).size]
    if not bloques:
        raise HvScaleError("no hay frentes con los que congelar la escala")

    puntos = np.vstack(bloques)
    if not np.all(np.isfinite(puntos)):
        raise HvScaleError("hay NaN o infinito en los frentes a consolidar")

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    indices = np.asarray(
        NonDominatedSorting().do(puntos, only_non_dominated_front=True))
    return puntos[indices]


def freeze_hv_scale(fronts_by_instance: Dict[str, Iterable[np.ndarray]],
                    ref_point: float = DEFAULT_HV_REF_POINT,
                    source: str = "consolidado") -> HvScale:
    """
    Congela la escala de HV con la misma regla que `mohh-evaluate`.

    Para cada instancia se toma el frente no dominado del consolidado de todos
    los frentes dados y se guardan su minimo y su maximo por objetivo. Ese
    `vmin/vmax` es el `boundaries[label]` del pipeline de evaluacion
    (experiment.py:515-519).

    @param fronts_by_instance: `{instancia: [frentes]}` a consolidar.
    @param ref_point: punto de referencia normalizado.
    @param source: procedencia para el JSON.
    @return: `HvScale`.
    """
    if not fronts_by_instance:
        raise HvScaleError("no hay instancias para congelar la escala")

    bounds = {}
    for nombre, frentes in fronts_by_instance.items():
        consolidado = _consolidado(frentes)
        bounds[nombre] = (np.min(consolidado, axis=0),
                          np.max(consolidado, axis=0))

    return HvScale(bounds, ref_point=ref_point, source=source)


def save_hv_scale(scale: HvScale, path: str) -> str:
    """
    Escribe la escala en un JSON.

    @param scale: escala a escribir.
    @param path: ruta destino; las carpetas se crean.
    @return: la ruta escrita.
    """
    carpeta = os.path.dirname(os.path.abspath(path))
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    with open(path, "w", encoding="utf-8") as archivo:
        json.dump(scale.to_dict(), archivo, indent=2, ensure_ascii=False)
        archivo.write("\n")
    return path


def load_hv_scale(path: str) -> HvScale:
    """
    Lee una escala de HV de un JSON.

    @param path: ruta del JSON.
    @return: `HvScale`.
    """
    if not os.path.isfile(path):
        raise HvScaleError(f"no existe el fichero de escala de HV: {path}")

    try:
        with open(path, encoding="utf-8") as archivo:
            datos = json.load(archivo)
    except (OSError, ValueError) as error:
        raise HvScaleError(f"'{path}' no se pudo leer: {error}") from error

    if "instances" not in datos:
        raise HvScaleError(
            f"'{path}' no tiene la clave 'instances'; no es una escala de HV "
            "de este formato")

    bounds = {}
    for nombre, entrada in datos["instances"].items():
        try:
            bounds[nombre] = (entrada["vmin"], entrada["vmax"])
        except (KeyError, TypeError) as error:
            raise HvScaleError(
                f"'{path}': la instancia '{nombre}' no tiene vmin/vmax") from error

    return HvScale(bounds, ref_point=float(datos.get("ref_point",
                                                     DEFAULT_HV_REF_POINT)),
                   source=str(datos.get("source", path)))


def contexts_from_scale(scale: HvScale,
                        instances: Optional[Iterable[str]] = None) -> HypervolumeContexts:
    """
    Contextos de HV de la escala, comprobando que cubra lo que se va a medir.

    @param scale: escala congelada.
    @param instances: instancias que se van a medir; si se dan, se exige que
        la escala las cubra todas.
    @return: `HypervolumeContexts`.
    """
    if instances is not None:
        faltan = scale.missing(instances)
        if faltan:
            raise HvScaleError(
                f"la escala de HV no cubre {len(faltan)} instancia(s): {faltan}. "
                "O se mide sin HV, o se congela una escala que las incluya.")

    return scale.contexts()


def load_hv_scale_from_evaluation(experiment_root: str,
                                  label: Optional[str] = None) -> HvScale:
    """
    Recupera la escala que `mohh-evaluate` ya dejo escrita en disco.

    `mohh-evaluate` guarda en
    `evaluation_results/run_metrics/evaluation_<instancia>.json`, dentro de
    `sections.best_fronts`, los `vmin`, `vmax` y `nadir_ref_point` que uso. Es la
    unica fuente existente de una escala congelada, asi que reutilizarla evita
    tener que recalcularla y garantiza que el analisis offline reproduce los
    numeros de la evaluacion.

    @param experiment_root: raiz del experimento (`results/.../Experiment_NNN`).
    @param label: presupuesto (`'10k'`, `'30k'`, `'50k'`). Hace falta porque
        los presupuestos NO comparten escala: en los datos reales de este
        repositorio 10 de las 16 instancias de baseline tienen `vmin`/`vmax`
        distintos segun el presupuesto, ya que cada poblacion da un consolidado
        distinto. El HV de `mohh-evaluate` es, por tanto, especifico del
        presupuesto, y con None se rechaza en vez de elegir uno al azar.
    @return: `HvScale`.
    """
    carpeta = os.path.join(experiment_root, "evaluation_results", "run_metrics")
    if not os.path.isdir(carpeta):
        raise HvScaleError(
            f"'{experiment_root}' no tiene evaluation_results/run_metrics; "
            "mohh-evaluate no llego a escribir la escala del HV")

    nombres = sorted(nombre for nombre in os.listdir(carpeta)
                     if nombre.startswith("evaluation_")
                     and nombre.endswith(".json"))
    if not nombres:
        raise HvScaleError(
            f"'{carpeta}' no tiene ningun evaluation_<instancia>.json")

    bounds = {}
    ref_point = None
    usados = []
    candidatos = {}

    for nombre in nombres:
        ruta = os.path.join(carpeta, nombre)
        try:
            with open(ruta, encoding="utf-8") as archivo:
                datos = json.load(archivo)
        except (OSError, ValueError):
            continue

        entradas = datos.get("sections", {}).get("best_fronts", [])
        for entrada in entradas:
            instancia = entrada.get("instance")
            if instancia is None:
                continue
            if label is not None and entrada.get("label") != label:
                continue
            try:
                limites = (entrada["vmin"], entrada["vmax"])
            except (KeyError, TypeError):
                continue
            candidatos.setdefault(instancia, {})[entrada.get("label")] = limites
            usados.append(f"{instancia}/{entrada.get('label')}")
            punto = entrada.get("nadir_ref_point")
            if punto is not None:
                ref_point = float(np.max(np.asarray(punto, dtype=float)))

    if not bounds and candidatos:
        # Varios presupuestos escriben la misma instancia. Si discrepan, elegir
        # uno en silencio daria una escala arbitraria: es exactamente la
        # inconsistencia que hace incomparables los HVs, asi que se rechaza.
        inconsistentes = []
        for instancia, por_label in sorted(candidatos.items()):
            distintos = {tuple(np.asarray(v).ravel().tolist())
                         for v in por_label.values()}
            if len(distintos) > 1:
                inconsistentes.append(
                    f"{instancia} (presupuestos "
                    f"{', '.join(sorted(str(k) for k in por_label))})")
        if inconsistentes:
            raise HvScaleError(
                f"los presupuestos dan limites distintos para "
                f"{len(inconsistentes)} instancia(s): "
                f"{'; '.join(inconsistentes[:3])}"
                f"{'...' if len(inconsistentes) > 3 else ''}. "
                f"Elige uno con label='10k' (o '30k', '50k') para que la escala "
                f"sea inequivoca.")
        for instancia, por_label in sorted(candidatos.items()):
            bounds[instancia] = next(iter(por_label.values()))

    if not bounds:
        raise HvScaleError(
            f"ningun evaluation_<instancia>.json de '{carpeta}' trae "
            "sections.best_fronts con vmin/vmax"
            + (f" para el presupuesto '{label}'" if label else ""))

    escala = HvScale(bounds,
                     ref_point=ref_point if ref_point is not None
                     else DEFAULT_HV_REF_POINT,
                     source=f"mohh-evaluate:{experiment_root}"
                            + (f"/{label}" if label else ""),
                     usados=sorted(set(usados)))
    return escala


def instancias_sin_escala(scale: "HvScale") -> List[str]:
    """Instancias cuyo marco de HV colapsa (`vmax == vmin` en algun objetivo).

    `mohh-evaluate` las invalida: `normalize_values` divide por
    `vmax - vmin`, asi que el objetivo constante deja el frente en NaN y la
    instancia no se puntua. Medirlas con `span = 1` (el relleno de
    `HypervolumeContext`) daria un HV que el pipeline de GE no reporta, por eso
    los analisis que quieren ser comparables con GE las excluyen y las dejan
    anotadas.

    @param scale: escala de HV.
    @return: nombres de instancia ordenados.
    """
    return sorted(
        nombre for nombre, (vmin, vmax) in scale.bounds.items()
        if np.any(np.asarray(vmax, dtype=float)
                  <= np.asarray(vmin, dtype=float))
    )


def restringir_escala(scale: "HvScale", instances: Iterable[str],
                      check: bool = True) -> "HvScale":
    """Copia de `scale` limitada a `instances`, conservando `ref_point`.

    Se usa para sanear una escala congelada antes de medir un split: quita las
    instancias degeneradas (`instancias_sin_escala`) sin cambiar los limites de
    las demas, de modo que todos los individuos y generaciones siguen midiendose
    con exactamente los mismos `vmin/vmax`.

    @param scale: escala de origen.
    @param instances: instancias que se quieren conservar.
    @param check: si True, exige que `scale` cubra todas las instancias.
    @return: nueva `HvScale`.
    """
    wanted = list(dict.fromkeys(instances))
    if check:
        missing = scale.missing(wanted)
        if missing:
            raise HvScaleError(
                f"la escala de HV no cubre {len(missing)} instancia(s): "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
    bounds = {nombre: scale.bounds[nombre]
              for nombre in wanted if nombre in scale.bounds}
    return HvScale(bounds, ref_point=scale.ref_point, source=scale.source,
                   usados=scale.usados)


def conflictos_escala(left: "HvScale", right: "HvScale",
                      atol: float = 0.0) -> List[str]:
    """Instancias **comunes** en las que dos escalas dan `vmin/vmax` distinto.

    Es la comprobacion para *combinar* escalas de splits distintos (train y
    test comparten pocas instancias, o ninguna): solo importa que, cuando una
    instancia aparece en las dos, sus limites coincidan.

    @param left: primera escala.
    @param right: segunda escala.
    @param atol: tolerancia absoluta al comparar los limites (0 = exacta).
    @return: nombres de instancia en conflicto, ordenados.
    """
    problemas = []
    for nombre in set(left.bounds) & set(right.bounds):
        lo_l, hi_l = left.bounds[nombre]
        lo_r, hi_r = right.bounds[nombre]
        lo_l, hi_l = np.asarray(lo_l, float), np.asarray(hi_l, float)
        lo_r, hi_r = np.asarray(lo_r, float), np.asarray(hi_r, float)
        if (lo_l.shape != lo_r.shape or hi_l.shape != hi_r.shape
                or not np.allclose(lo_l, lo_r, atol=atol, rtol=0.0)
                or not np.allclose(hi_l, hi_r, atol=atol, rtol=0.0)):
            problemas.append(nombre)
    return sorted(problemas)


def diferencias_escala(left: "HvScale", right: "HvScale",
                       atol: float = 0.0) -> List[str]:
    """Instancias en las que dos escalas NO son equivalentes.

    Incluye tanto las instancias presentes solo en una de las dos como las
    comunes con `vmin/vmax` distinto. Es la comprobacion que hace falta antes de
    comparar HV entre experimentos: dos escalas con el mismo `ref_point` y el
    mismo `source` ("consolidado") pueden tener limites distintos, y entonces
    sus HV no son comparables.

    @param left: primera escala.
    @param right: segunda escala.
    @param atol: tolerancia absoluta al comparar los limites (0 = exacta).
    @return: nombres de instancia que difieren, ordenados.
    """
    problemas = set(left.bounds) ^ set(right.bounds)
    problemas |= set(conflictos_escala(left, right, atol))
    return sorted(problemas)
