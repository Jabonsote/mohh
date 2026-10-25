"""
Calculo de los cuatro indicadores multiobjetivo sobre frentes ya guardados.

El repositorio trabaja en minimizacion (mQAP). Los cuatro indicadores:

* **HV**: hipervolumen de maximizacion, medido sobre el frente normalizado en
  `[0, 1]` con un punto de referencia `z_ref` fijo. Es la convencion que ya usa
  `mohh-evaluate`, y aqui la escala la fija el frente consolidado de la propia
  poblacion analizada (ver `HypervolumeContext`), de modo que todos los
  individuos de la misma corrida se miden con la MISMA escala y el HV es
  comparable entre ellos.
* **IGD+**, **DELTA_P**, **R2**: indicadores de minimizacion medidos contra el
  conjunto de referencia `P_i^*` con `ReferenceFitness`, que aplica la escala
  del propio `P_i^*` (ideal + span) y el nadir con margen de DeltaP.

Las funciones son puras: reciben frentes y referencias ya en memoria y devuelven
DataFrames de `instancia x indicador`. Ninguna genera datos.
"""

import numpy as np
import pandas as pd

from mohh.core.indicators.reference_sets import load_reference_sets
from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.generation.utilities.algorithm.reference_fitness import (
    REFERENCE_METRICS, ReferenceFitness)

__all__ = [
    "INDICATORS",
    "METRIC_LABELS",
    "HypervolumeContext",
    "HypervolumeContexts",
    "IndicatorsError",
    "compute_indicators",
    "fit_hv_contexts",
    "load_references",
]

#: Indicadores en el orden en que se reportan.
INDICATORS = ("HV", "IGD+", "DELTA_P", "R2")

#: Etiquetas legibles para nombres de archivo/carpeta.
METRIC_LABELS = {
    "HV": "HV",
    "IGD+": "IGD_plus",
    "DELTA_P": "DeltaP",
    "R2": "R2",
}

#: Indicadores de minimizacion (menor es mejor).
_MINIMIZATION = ("IGD+", "DELTA_P", "R2")

DEFAULT_HV_REF_POINT = 1.1


class IndicatorsError(RuntimeError):
    """Los datos de entrada no permiten calcular los indicadores."""


def load_references(directory: str, instance_names=None, n_objectives=None) -> dict:
    """
    Carga los conjuntos de referencia `P_i^*` de un directorio.

    @param directory: carpeta con un `.pof` por instancia (y su manifiesto).
    @param instance_names: si se indica, solo se quedan estas instancias y se
        exige que existan todas (fallar aqui es preferible a medir contra una
        referencia equivocada).
    @param n_objectives: si se indica, filtra por numero de objetivos.
    @return: dict nombre -> `ReferenceSet`.
    """
    references = load_reference_sets(directory, n_objectives=n_objectives)

    if instance_names is None:
        return references

    faltan = sorted(set(instance_names) - set(references))
    if faltan:
        raise IndicatorsError(
            f"faltan conjuntos de referencia en {directory} para "
            f"{len(faltan)} instancia(s): {faltan}. Genera los P* de ese "
            f"split antes de calcular IGD+/DELTA_P/R2; no se sustituyen por "
            f"referencias de otro split."
        )

    return {nombre: references[nombre] for nombre in instance_names}


class HypervolumeContext:
    """
    Escala fija del hipervolumen para un conjunto de frentes.

    La escala se deriva del frente consolidado de todos los frentes que se van
    a medir (min y max por objetivo) y se aplica con
    `normalize_values`, igual que hace `mohh-evaluate`. Al ser la misma para
    todos los individuos, el HV de cada uno queda en `[0, HV_max]` y es
    comparable; recalcularla por individuo produciria HVs sin sentido.

    @param fronts: lista de arrays (n, m) de los frentes a medir.
    @param ref_point: punto de referencia en la escala normalizada.
    """

    def __init__(self, fronts=None, ref_point: float = DEFAULT_HV_REF_POINT) -> None:
        self.ref_point = float(ref_point)
        self.vmin = None
        self.vmax = None
        self._z_ref = None

        if fronts is not None:
            self.fit(fronts)

    @property
    def n_objectives(self):
        return 0 if self.vmin is None else int(len(self.vmin))

    def fit(self, fronts) -> "HypervolumeContext":
        """
        Calcula la escala a partir de los frentes dados.

        @param fronts: lista de arrays (n, m), o un unico array (n, m).
        @return: self, para encadenar.
        """
        if isinstance(fronts, np.ndarray) and fronts.ndim == 2:
            bloques = [fronts]
        else:
            bloques = [np.atleast_2d(np.asarray(f, dtype=float)) for f in fronts]

        bloques = [b for b in bloques if b.size]
        if not bloques:
            raise IndicatorsError(
                "no hay frentes con los que fijar la escala del hipervolumen")

        apilado = np.vstack(bloques)
        vmin = np.min(apilado, axis=0)
        vmax = np.max(apilado, axis=0)

        # Un objetivo constante da vmax - vmin == 0 y produciria NaN; se deja
        # con span 1 para que la normalizacion sea la identidad en ese eje.
        span = np.where(vmax > vmin, vmax - vmin, 1.0)

        self.vmin = vmin
        self.vmax = vmax
        self._z_ref = np.array([self.ref_point] * apilado.shape[1], dtype=float)
        return self

    def scale(self, front) -> np.ndarray:
        """Lleva un frente a la escala fija del contexto."""
        if self.vmin is None:
            raise IndicatorsError(
                "el contexto de hipervolumen no esta ajustado; llama a fit()")
        front = np.atleast_2d(np.asarray(front, dtype=float))
        vmin = np.asarray(self.vmin, dtype=float)
        span = np.where(np.asarray(self.vmax) > vmin,
                        np.asarray(self.vmax) - vmin, 1.0)
        return (front - vmin) / span

    def value(self, front) -> float:
        """Hipervolumen (maximizacion) del frente en la escala del contexto."""
        if self._z_ref is None:
            raise IndicatorsError(
                "el contexto de hipervolumen no esta ajustado; llama a fit()")
        normalizado = self.scale(front)
        return float(compute_hypervolume(self._z_ref, normalizado))

    def bounds(self) -> dict:
        """Limites usados, para documentarlos junto a los resultados."""
        if self.vmin is None:
            return {}
        return {
            "ref_point": float(self.ref_point),
            "vmin": np.asarray(self.vmin, dtype=float).tolist(),
            "vmax": np.asarray(self.vmax, dtype=float).tolist(),
        }

    def __repr__(self) -> str:
        if self.vmin is None:
            return "HypervolumeContext(sin ajustar)"
        return f"HypervolumeContext(ref_point={self.ref_point})"


class HypervolumeContexts:
    """
    Una escala de hipervolumen por instancia.

    La escala del HV depende del numero de objetivos, y en este repositorio las
    instancias no lo comparten (las KC10/KC20 son de 2 objetivos y las KC30 de
    3). Por eso la escala se ajusta por instancia, igual que hace
    `mohh-evaluate`, que calcula `boundaries` dentro del bucle de instancias.

    @param contexts: `{instancia: HypervolumeContext}`.
    """

    def __init__(self, contexts: dict = None) -> None:
        self.contexts = dict(contexts or {})

    def __getitem__(self, instance_name: str) -> HypervolumeContext:
        return self.contexts[instance_name]

    def __contains__(self, instance_name: str) -> bool:
        return instance_name in self.contexts

    def __len__(self) -> int:
        return len(self.contexts)

    def fit(self, fronts_by_instance: dict, ref_point: float = DEFAULT_HV_REF_POINT):
        """
        Ajusta una escala por instancia a partir de los frentes dados.

        @param fronts_by_instance: `{instancia: frente o lista de frentes}`.
        @param ref_point: punto de referencia normalizado.
        @return: self.
        """
        self.contexts = {}
        for nombre, fronts in fronts_by_instance.items():
            if isinstance(fronts, np.ndarray) and fronts.ndim == 2:
                lista = [fronts]
            elif isinstance(fronts, dict):
                lista = [f for f in fronts.values()]
            else:
                lista = list(fronts)
            lista = [np.atleast_2d(np.asarray(f, dtype=float)) for f in lista]
            self.contexts[nombre] = HypervolumeContext(lista, ref_point=ref_point)
        return self

    def bounds(self) -> dict:
        return {nombre: contexto.bounds()
                for nombre, contexto in sorted(self.contexts.items())}

    def __repr__(self) -> str:
        return f"HypervolumeContexts({len(self.contexts)} instancias)"


def fit_hv_contexts(fronts_by_candidate: dict, ref_point: float = DEFAULT_HV_REF_POINT):
    """
    Ajusta la escala de HV por instancia con todos los candidatos a medir.

    @param fronts_by_candidate: `{candidato: {instancia: (n, m) ndarray}}`.
    @param ref_point: punto de referencia normalizado.
    @return: `HypervolumeContexts` con una escala por instancia, compartida por
        todos los candidatos (asi sus HVs son comparables).
    """
    por_instancia = {}

    for candidato in fronts_by_candidate.values():
        for nombre, front in candidato.items():
            por_instancia.setdefault(nombre, []).append(front)

    return HypervolumeContexts().fit(por_instancia, ref_point=ref_point)


def compute_indicators(
    fronts_by_instance: dict,
    references: dict = None,
    metrics=INDICATORS,
    hv_context=None,
    require_references: bool = True,
) -> pd.DataFrame:
    """
    Calcula los indicadores de un frente por instancia.

    @param fronts_by_instance: `{instancia: (n, m) ndarray}`.
    @param references: dict nombre -> `ReferenceSet`; necesario para IGD+,
        DELTA_P y R2.
    @param metrics: subconjunto de `INDICATORS` a calcular.
    @param hv_context: escala de HV, como `HypervolumeContexts` (una por
        instancia) o un unico `HypervolumeContext`. Si es None, cada instancia
        se normaliza con su propio frente, lo que solo tiene sentido si se
        compara un unico individuo.
    @param require_references: si True y falta la referencia de alguna
        instancia, lanza `IndicatorsError` en vez de devolver NaN (NaN
        silencioso es exactamente el fallo que hay que evitar).
    @return: DataFrame con indice = instancias y columnas = indicadores.
    """
    metrics = tuple(m for m in metrics if m in INDICATORS)
    if not metrics:
        raise IndicatorsError(f"ningun indicador valido en {tuple(metrics)}")

    instancias = sorted(fronts_by_instance)
    if not instancias:
        raise IndicatorsError("no hay instancias con frente")

    contextos = {}
    referencias_necesarias = [m for m in metrics if m in _MINIMIZATION]

    if referencias_necesarias:
        if references is None and require_references:
            raise IndicatorsError(
                f"hacen falta referencias P* para {referencias_necesarias}; "
                f"pasalas o desactiva require_references"
            )
        faltan = [nombre for nombre in instancias
                  if references is None or nombre not in references]
        if faltan and require_references:
            raise IndicatorsError(
                f"faltan referencias P* para {len(faltan)} instancia(s): "
                f"{faltan}"
            )

        # Los contextos se construyen siempre que haya alguna referencia, aun
        # si le faltan algunas instancias: en ese caso el indicador de la
        # instancia sin P* sale como NaN (mas abajo) en lugar de romper el
        # calculo de las demas. Antes solo se construian si no faltaba ninguna,
        # y con referencias parciales `contextos` quedaba vacio y la funcion
        # reventaba con KeyError.
        if references:
            for metrica in referencias_necesarias:
                contextos[metrica] = ReferenceFitness(references, metrica)

    columnas = {}
    for nombre in instancias:
        front = np.atleast_2d(np.asarray(fronts_by_instance[nombre], dtype=float))
        valores = {}

        if "HV" in metrics:
            contexto = _contexto_para(hv_context, nombre, front)
            valores["HV"] = contexto.value(front)

        for metrica in referencias_necesarias:
            if references is None or nombre not in references:
                valores[metrica] = np.nan
                continue
            valores[metrica] = contextos[metrica].value(front, nombre)

        columnas[nombre] = valores

    frame = pd.DataFrame.from_dict(columnas, orient="index")
    frame.index.name = "instance"

    return frame[list(metrics)]


def _contexto_para(hv_context, nombre: str, front):
    """Resuelve el contexto de HV de una instancia sea cual sea su forma."""
    if isinstance(hv_context, HypervolumeContexts):
        if nombre not in hv_context:
            raise IndicatorsError(
                f"no hay escala de HV ajustada para '{nombre}'")
        return hv_context[nombre]

    if isinstance(hv_context, HypervolumeContext):
        if hv_context.vmin is None:
            return HypervolumeContext(front)
        return hv_context

    return HypervolumeContext(front)


def indicators_long(frame: pd.DataFrame, individual=None, extra=None) -> pd.DataFrame:
    """
    Pasa la tabla `instancia x indicador` a formato largo.

    @param frame: salida de `compute_indicators`.
    @param individual: nombre del individuo (columna opcional).
    @param extra: dict de columnas constantes a anadir (split, solver, ...).
    @return: DataFrame con columnas `instance`, `indicator`, `value` (+ extra).
    """
    largo = frame.reset_index().melt(
        id_vars=["instance"], var_name="indicator", value_name="value")

    if individual is not None:
        largo.insert(0, "individual", individual)

    if extra:
        for clave, valor in extra.items():
            largo.insert(1, clave, valor)

    return largo


def indicator_minimization(metric: str) -> bool:
    """True si el indicador es de minimizacion (menor es mejor)."""
    return metric in _MINIMIZATION
