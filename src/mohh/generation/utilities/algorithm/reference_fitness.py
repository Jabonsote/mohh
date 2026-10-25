"""
Fitness del GE basada en los conjuntos de referencia fijos P_i^*.

Este modulo es el puente entre la fase de generacion y el protocolo de
evaluacion basado en SDD con indicadores multiobjetivo:

    1. Se cargan UNA SOLA VEZ los conjuntos de referencia P_i^* (ver
       `mohh.core.indicators.reference_sets`), generados con RSEHH/RSEIterative
       en `set-reference/`.
    2. Para cada instancia de entrenamiento y cada individual's se mide un
       indicador (IGD+, DeltaP o R2) entre su frente no dominado y P_i^*.
    3. La matriz resultante (instancias x individuos) se pasa a
       `mohh.core.utilities.algorithm.sdd.SDD`, que ya sabe que para estos
       indicadores el punto de sincronizacion es el MINIMO por instancia
       (b_i = min), tal y como establece el metodo.

Todos los indicadores son de MINIMIZACION, luego "menor es mejor": es la
direccion que espera la SDD para `IGD+`, `DELTA_P` y `R2`.

La media sobre K corridas independientes se aplica aqui, no en
`evaluation_step`: `evaluation_step` deja los K frentes de cada par
(candidato, instancia) y `matrix` promedia el indicador de los K. K = 11 es el
`N_EXPERIMENTS` del protocolo publicado y se activa con `N_REPLICATES`; con
K = 1 se guarda un unico frente y el resultado es el mismo.

Lo que se promedia es el INDICADOR, no el frente: unir los K frentes cambiaria
el valor del indicador y silenciaria el efecto de promediar.
"""

import numpy as np
import pandas as pd

from mohh.core.indicators import delta_p, igd_plus, r2, uniform_weights
from mohh.core.indicators.reference_sets import (
    ReferenceSet,
    ReferenceSetError,
    load_reference_sets,
)
from mohh.core.utilities.algorithm.sdd import SDD

__all__ = [
    "ReferenceFitness",
    "REFERENCE_METRICS",
    "FITNESS_AVERAGE_RANK",
    "FITNESS_SDD",
    "FITNESS_REFERENCE_SDD",
    "resolve_fitness",
    "compute_reference_fitness",
    "compute_sdd_hypervolume",
    "farthest_point_subset",
]

# Indicadores disponibles para la fitness por referencia.
REFERENCE_METRICS = {
    "IGD+": igd_plus,
    "DELTA_P": delta_p,
    "R2": r2,
}

_SEED_WEIGHTS = 20240617


def farthest_point_subset(points, size):
    """
    Subconjunto determinista de `size` puntos bien repartidos.

    Se empieza por el punto mas cercano al ideal y se sigue anadiendo el punto
    mas lejano al ya elegido (k-center / muestreo de Corey). Es determinista y
    no favorece ni castiga a ninguna zona del frente, a diferencia de un
    recorte por los primeros puntos del archivo.

    @param points: (n, m) frente.
    @param size: numero de puntos a conservar.
    @return: (size, m) subconjunto, en el orden de seleccion.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    n = points.shape[0]
    size = int(size)

    if size >= n:
        return points

    centro = np.min(points, axis=0)
    elegido = [int(np.argmin(np.sum((points - centro) ** 2, axis=1)))]
    distancias = np.sum((points - points[elegido[0]]) ** 2, axis=1)

    while len(elegido) < size:
        siguiente = int(np.argmax(distancias))
        elegido.append(siguiente)
        distancias = np.minimum(
            distancias, np.sum((points - points[siguiente]) ** 2, axis=1)
        )

    return points[elegido]


class ReferenceFitness:
    """
    Calcula la matriz de indicadores de la poblacion contra P_i^*.

    @param references: dict nombre -> ReferenceSet.
    @param metric: 'IGD+', 'DELTA_P' o 'R2'.
    @param subset: numero maximo de puntos de referencia (None = todos).
    @param n_weights: numero de vectores de peso para R2 (None => m + 1).
    """

    def __init__(self, references, metric: str, subset=None, n_weights=None):
        metric = str(metric).strip().upper()

        if metric not in REFERENCE_METRICS:
            raise ValueError(
                f"indicador de referencia '{metric}' no soportado. "
                f"Usa uno de: {sorted(REFERENCE_METRICS)}"
            )

        self.metric = metric
        self.references = dict(references)
        self.subset = subset
        self.n_weights = n_weights
        self._funcion = REFERENCE_METRICS[metric]
        self._pesos = {}

        if not self.references:
            raise ReferenceSetError("no hay conjuntos de referencia cargados")

    # ------------------------------------------------------------------
    # Preparacion
    # ------------------------------------------------------------------

    def _subconjunto(self, referencia: ReferenceSet):
        """Referencia recortada de forma determinista, cacheada."""
        if self.subset is None or referencia.size <= int(self.subset):
            return referencia.normalized

        clave = (referencia.name, int(self.subset))
        cache = getattr(self, "_cache_sub", None)
        if cache is None:
            cache = self._cache_sub = {}
        if clave not in cache:
            cache[clave] = farthest_point_subset(
                referencia.normalized, int(self.subset)
            )
        return cache[clave]

    def _pesos_de(self, nombre: str, m: int):
        if self.metric != "R2":
            return None
        if m not in self._pesos:
            self._pesos[m] = uniform_weights(m, self.n_weights, _SEED_WEIGHTS)
        return self._pesos[m]

    def check(self, instance_names, n_objectives=None):
        """
        Valida que todas las instancias de la generacion tengan referencia.

        @param instance_names: nombres de las instancias de entrenamiento.
        @param n_objectives: si se indica, se comprueba tambien que las
            referencias de esas instancias tengan ese numero de objetivos.
        """
        faltan = [n for n in instance_names if n not in self.references]
        if faltan:
            raise ReferenceSetError(
                f"faltan conjuntos de referencia para {len(faltan)} "
                f"instancia(s): {faltan}"
            )

        if n_objectives is not None:
            erroneas = [
                n for n in instance_names
                if self.references[n].n_objectives != n_objectives
            ]
            if erroneas:
                raise ReferenceSetError(
                    f"{len(erroneas)} referencia(s) no tienen "
                    f"{n_objectives} objetivos: {erroneas}"
                )

    # ------------------------------------------------------------------
    # Medicion
    # ------------------------------------------------------------------

    def value(self, front, instance_name):
        """
        Indicador de un frente de un individual's en una instancia.

        @param front: (n, m) frente no dominado del individual's, SIN
            normalizar (en la escala original de los objetivos).
        @param instance_name: nombre de la instancia.
        @return: escalar, menor es mejor.
        """
        referencia = self.references[instance_name]
        front = np.atleast_2d(np.asarray(front, dtype=float))

        if front.size == 0:
            raise ValueError(f"{instance_name}: frente vacio")
        if front.shape[1] != referencia.n_objectives:
            raise ValueError(
                f"{instance_name}: frente con {front.shape[1]} objetivos, la "
                f"referencia tiene {referencia.n_objectives}"
            )

        # Escala comun: la fija la referencia, nunca el frente del individual's.
        scaled = referencia.scale(front)
        objetivo = self._subconjunto(referencia)

        if self.metric == "R2":
            # `normalize=False` es obligatorio: la escala ya la fija la
            # referencia. Si R2 volviera a normalizar con el ideal y el nadir
            # del PROPIO frente, cada individual's se mediria en una escala
            # distinta y la matriz no seria comparable entre columnas.
            return r2(
                scaled,
                weights=self._pesos_de(referencia.name, referencia.n_objectives),
                ideal=np.zeros(referencia.n_objectives),
                normalize=False,
            )

        if self.metric == "DELTA_P":
            return delta_p(scaled, objetivo, nadir=referencia.nadir_normalized)

        return igd_plus(scaled, objetivo)

    def matrix(self, fronts_by_instance, instance_names, individuals):
        """
        Construye la matriz de indicadores (instancias x individuos).

        @param fronts_by_instance: dict nombre -> {individuo -> frente}. El
            frente de un individual's puede ser una lista de frentes (las K
            corridas independientes): en ese caso se promedian los indicadores
            de las K corridas, que es la media sobre K del protocolo.
        @param instance_names: orden de las filas.
        @param individuals: orden de las columnas.
        @return: DataFrame con indice = instancias, columnas = individuos.
        """
        columnas = {}

        for individuo in individuals:
            valores = []
            for nombre in instance_names:
                frentes = fronts_by_instance[nombre][individuo]
                if not isinstance(frentes, (list, tuple)):
                    frentes = [frentes]
                if not frentes:
                    raise ValueError(
                        f"{individuo} no tiene frente en {nombre}"
                    )
                medidas = [self.value(frente, nombre) for frente in frentes]
                valores.append(float(np.mean(medidas)))
            columnas[individuo] = valores

        return pd.DataFrame(
            columnas,
            index=list(instance_names),
            columns=list(individuals),
        )

    @classmethod
    def from_params(cls, metric, directory, subset=None, n_weights=None,
                    n_objectives=None):
        """
        Constructor desde un directorio de referencias.

        @param directory: directorio con un `.pof` por instancia.
        @param n_objectives: numero de objetivos esperado.
        @return: instancia de ReferenceFitness.
        """
        references = load_reference_sets(
            directory, n_objectives=n_objectives
        )
        return cls(references, metric, subset, n_weights)


# ----------------------------------------------------------------------
# Resolucion del fitness del GE (protocolo del pizarron)
# ----------------------------------------------------------------------

# Modos de aptitud de la hiperheuristica.
FITNESS_AVERAGE_RANK = "average_rank"        # protocolo original
FITNESS_SDD = "sdd"                          # SDD sobre el propio HV
FITNESS_REFERENCE_SDD = "reference_sdd"      # SDD sobre P_i^*

# Alias aceptados para nombrar un indicador directamente como fitness.
_FITNESS_ALIASES = {
    "hv": "HV",
    "hypervolume": "HV",
    "igd+": "IGD+",
    "igdplus": "IGD+",
    "delta_p": "DELTA_P",
    "deltap": "DELTA_P",
    "delta-p": "DELTA_P",
    "r2": "R2",
}


def resolve_fitness(fitness=None, sdd_metric=None):
    """
    Normaliza la configuracion de fitness en (modo, indicador).

    Acepta las formas explicitas (`FITNESS="reference_sdd"` con
    `SDD_METRIC="IGD+"`) y la directa, donde el indicador es el fitness
    (`FITNESS="IGD+"`). `sdd_metric` puede venir en cualquier caso.

    @param fitness: valor crudo de `FITNESS` (None => original).
    @param sdd_metric: valor crudo de `SDD_METRIC` (puede ser None).
    @return: (modo, indicador) con modo en
        {average_rank, sdd, reference_sdd}.
    """
    metrica = _FITNESS_ALIASES.get(
        str(sdd_metric or "").strip().lower()
    ) if sdd_metric else None

    if sdd_metric and metrica is None:
        raise ValueError(
            f"SDD_METRIC '{sdd_metric}' no soportado. Usa "
            f"{sorted(REFERENCE_METRICS)} o 'HV'."
        )

    bruto = str(fitness or FITNESS_AVERAGE_RANK).strip()
    tipo = bruto.lower().replace("-", "_")

    # `FITNESS = "HV"`: la SDD sobre el propio hipervolumen.
    if tipo == "hv":
        if metrica is not None and metrica != "HV":
            raise ValueError(
                f"FITNESS='HV' con SDD_METRIC='{sdd_metric}' no tiene sentido: "
                "la SDD sobre HV no usa otro indicador. Usa "
                f"'--fitness {metrica}' para un indicador multiobjetivo."
            )
        return FITNESS_SDD, "HV"

    # `FITNESS = "IGD+"` (o alias) => reference_sdd con ese indicador.
    alias = _FITNESS_ALIASES.get(tipo)
    if alias is not None:
        return FITNESS_REFERENCE_SDD, alias

    if tipo not in (FITNESS_AVERAGE_RANK, FITNESS_SDD, FITNESS_REFERENCE_SDD):
        raise ValueError(
            f"FITNESS '{fitness}' no soportado. Usa '{FITNESS_AVERAGE_RANK}', "
            f"'{FITNESS_SDD}', '{FITNESS_REFERENCE_SDD}' o un indicador: "
            f"{sorted(REFERENCE_METRICS) + ['HV']}."
        )

    if tipo == FITNESS_SDD and metrica is None:
        # La SDD clasica se define sobre HV; si no se da indicador se usa HV.
        return FITNESS_SDD, "HV"

    if tipo == FITNESS_REFERENCE_SDD and metrica not in REFERENCE_METRICS:
        raise ValueError(
            f"FITNESS='reference_sdd' necesita SDD_METRIC en "
            f"{sorted(REFERENCE_METRICS)}; recibido {sdd_metric!r}."
        )

    return tipo, metrica


# ----------------------------------------------------------------------
# Matrices O_i y aptitudes del fitness basado en SDD
# ----------------------------------------------------------------------

def compute_reference_fitness(individuals, instances, fitness, aggregation="mean"):
    """
    Matriz O_i de un indicador multiobjetivo contra P_i^* y SDD resultante.

    Para cada instancia y cada individuo mide el indicador entre su frente
    (o los K frentes de sus corri das independientes) y el conjunto de
    referencia P_i^*. La matriz se pasa a la SDD, que ya usa b_i = min
    para IGD+/DELTA_P/R2.

    @param individuals: poblacion actual (list).
    @param instances: instancias de entrenamiento (list).
    @param fitness: instancia de ReferenceFitness ya cargada.
    @param aggregation: 'mean' o 'median' sobre las K corridas.
    @return: (scores, x_i, indicadores) con x_i e indicadores alineados
        instancias x individuos.
    """
    names = [instance.instance_name for instance in instances]
    individuals = list(individuals)

    # Todas las instancias de la generacion tienen que tener SU conjunto de
    # referencia; el error lista las que falten.
    fitness.check(names)

    fronts = {}

    for instance in instances:
        nombre = instance.instance_name
        fronts[nombre] = {
            individual.name: (
                individual.pareto_fronts[nombre]
                if not getattr(individual, "replicate_fronts", None)
                else individual.replicate_fronts.get(nombre,
                                                     individual.pareto_fronts[nombre])
            )
            for individual in individuals
        }

    indicadores = fitness.matrix(fronts, names,
                                 [individual.name for individual in individuals])

    sdd = SDD(fitness.metric, aggregation)
    x_i = sdd.differences(indicadores)
    scores = sdd.score(indicadores).to_numpy(dtype=float)

    return scores, x_i, indicadores


def compute_sdd_hypervolume(individuals, instances, metric="HV", aggregation="mean"):
    """
    Matriz O_i de hipervolumenes ya medidos y SDD resultante.

    En el modo `sdd` el hipervolumen se mide en compute_metrics contra el
    z_ref congelado de la instancia; aqui solo se reune la matriz y se calcula
    la SDD, que para HV usa b_i = max.

    @param individuals: poblacion actual (list).
    @param instances: instancias de entrenamiento (list).
    @param metric: 'HV'.
    @param aggregation: 'mean' o 'median'.
    @return: (scores, x_i, frame) con x_i y frame alineados instancias x
        individuos.
    """
    individuals = list(individuals)

    frame = pd.DataFrame(
        {individual.name: [float(individual.hypervolumes[instance.instance_name])
                           for instance in instances]
         for individual in individuals},
        index=[instance.instance_name for instance in instances],
    )

    sdd = SDD(metric, aggregation)
    x_i = sdd.differences(frame)
    scores = sdd.score(frame).to_numpy(dtype=float)

    return scores, x_i, frame
