"""
Carga de los conjuntos de referencia P_i^* del protocolo de evaluacion.

Los conjuntos de referencia se generan UNA SOLA VEZ con `set-reference`
(RSEHH / RSEIterative) y luego se reutilizan tal cual durante toda la fase de
generacion ("train once, use indefinitely"): no se recalculan por generacion,
porque la SDD solo tiene sentido si la referencia es fija.

Escala de los objetivos
-----------------------
Los objetivos del mQAP son enteros grandes, asi que la distancia euclidea de
IGD/IGD+ mezclaria escalas incomparables. La escala la fija el propio conjunto
de referencia: `ideal` y `nadir` se toman de P_i^* y todos los valores (el
frente del individual's y la referencia) se llevan a [0, 1] con esa misma
transformacion. Es la unica forma de que las distancias de distintos problemas
sean comparables dentro de una misma matriz de la SDD.
"""

import os
from dataclasses import dataclass

import numpy as np

__all__ = [
    "NADIR_MARGIN",
    "ReferenceSet",
    "ReferenceSetError",
    "load_reference_sets",
    "pooled_reference_sets",
    "read_pof",
]


class ReferenceSetError(RuntimeError):
    """Los conjuntos de referencia no son utilizables."""


# Margen relativo que separa el nadir de la region de DeltaP del maximo de la
# referencia. DeltaP necesita una caja con volumen positivo, y el punto de
# referencia mas extremo de P_i^* coincide con su propio maximo.
NADIR_MARGIN = 0.1


@dataclass(frozen=True)
class ReferenceSet:
    """
    Conjunto de referencia de una instancia.

    Atributos
    ---------
    name: nombre de la instancia.
    front: (r, m) frente de referencia sin normalizar.
    ideal: (m,) minimo por objetivo de la referencia.
    nadir: (m,) tope de la region de DeltaP, estrictamente mayor que el maximo
        de la referencia en cada objetivo con rango no nulo.
    span: (m,) escala por objetivo (nadirCrudo - ideal); 1 donde el objetivo es
        constante, para no dividir por cero.
    """

    name: str
    front: np.ndarray
    ideal: np.ndarray
    nadir: np.ndarray
    span: np.ndarray

    @property
    def n_objectives(self) -> int:
        return int(self.front.shape[1])

    @property
    def size(self) -> int:
        return int(self.front.shape[0])

    def scale(self, values) -> np.ndarray:
        """Lleva `values` a la escala de la referencia."""
        values = np.atleast_2d(np.asarray(values, dtype=float))
        return (values - self.ideal) / self.span

    @property
    def normalized(self) -> np.ndarray:
        """La referencia en [0, 1]."""
        return self.scale(self.front)

    @property
    def nadir_normalized(self) -> np.ndarray:
        """El nadir de DeltaP en la escala de la referencia."""
        return (self.nadir - self.ideal) / self.span


def read_pof(path):
    """
    Lee un `.pof` de RSEHH: cabecera `# <n> <m>` y una fila por punto.

    La cabecera se usa, no se ignora: si el numero de puntos o de objetivos que
    declara el archivo no coincide con lo que hay debajo, el conjunto se
    rechaza. Un `.pof` truncado o con la cabecera de otra instancia daria
    referencias equivocadas en silencio, que es justo lo que no puede pasar en
    la fase de generacion.

    Se reimplementa aqui (en vez de importar de `setref.pof`) para que el
    paquete `mohh` no dependa del directorio hermano `set-reference/`.

    @param path: ruta de entrada.
    @return: array (n, m).
    """
    front = np.genfromtxt(path, comments="#")
    front = np.atleast_2d(np.asarray(front, dtype=float))

    if front.size == 0:
        raise ReferenceSetError(f"{path}: archivo de referencia vacio")
    if not np.all(np.isfinite(front)):
        raise ReferenceSetError(f"{path}: contiene valores no finitos")

    declarado = _leer_cabecera(path)
    if declarado is not None:
        puntos, objetivos = declarado
        if (puntos, objetivos) != front.shape:
            raise ReferenceSetError(
                f"{path}: la cabecera declara {puntos} puntos y {objetivos} "
                f"objetivos, pero el archivo tiene {front.shape[0]} puntos y "
                f"{front.shape[1]} objetivos"
            )

    return front


def _leer_cabecera(path):
    """
    Primera linea comentada del `.pof`, como (n_puntos, n_objetivos).

    @param path: ruta de entrada.
    @return: tupla de dos enteros, o None si la cabecera no se puede leer.
    """
    with open(path) as archivo:
        for linea in archivo:
            if not linea.lstrip().startswith("#"):
                break
            campos = linea.lstrip().lstrip("#").split()
            if len(campos) >= 2:
                try:
                    return int(campos[0]), int(campos[1])
                except ValueError:
                    return None
    return None


def _build(name: str, front, margin: float) -> ReferenceSet:
    front = np.atleast_2d(np.asarray(front, dtype=float))

    if front.ndim != 2:
        raise ReferenceSetError(f"{name}: el frente debe ser una matriz (r, m)")
    if front.shape[0] == 0:
        raise ReferenceSetError(f"{name}: el frente de referencia esta vacio")
    if not np.all(np.isfinite(front)):
        raise ReferenceSetError(f"{name}: el frente tiene valores no finitos")
    if np.any(front < 0):
        raise ReferenceSetError(
            f"{name}: el frente tiene valores negativos; los objetivos del "
            "proyecto son no negativos"
        )

    ideal = np.min(front, axis=0)
    top = np.max(front, axis=0)
    span = top - ideal
    # Un objetivo constante no aporta distancia: se deja con escala 1.
    span = np.where(span > 0, span, 1.0)

    # Nadir de la region de DeltaP: la referencia mas un margen relativo, de
    # modo que la caja sobre el punto de referencia mas extremo tenga volumen
    # positivo. Si el objetivo es constante, el margen es una unidad.
    width = np.where(top > ideal, top - ideal, 1.0)
    nadir = top + margin * width

    return ReferenceSet(
        name=name,
        front=front,
        ideal=ideal,
        nadir=nadir,
        span=span,
    )


def _como_frentes(frentes):
    """
    Normaliza a una lista de arrays (n, m) lo que se pase como frente medido.

    Hace falta porque `if array:` sobre un array de mas de un elemento lanza
    `ValueError: truth value is ambiguous`, y las dos formas de llamar son
    legitimas: el consolidado de un presupuesto es un unico array (n, m),
    mientras que por individuo hay una lista de arrays.

    La lista NO se convierte con un solo `np.asarray`: los frentes de
    individuos distintos tienen distinto numero de puntos no dominados, y
    numpy >= 1.24 rechaza esas listas irregulares. Se convierte elemento a
    elemento y se apilan despues, que es lo que si admite formas distintas.

    @param frentes: array (n, m), secuencia de arrays, o None.
    @return: lista de arrays (n, m), vacia si no hay nada.
    """
    if frentes is None:
        return []

    if isinstance(frentes, np.ndarray):
        if frentes.ndim == 2:
            return [frentes.astype(float)] if frentes.size else []
        if frentes.ndim == 3:
            return [bloque.astype(float) for bloque in frentes
                    if bloque.size]
        return []

    salida = []
    for front in frentes:
        bloque = np.atleast_2d(np.asarray(front, dtype=float))
        if bloque.size:
            salida.append(bloque)
    return salida


def pooled_reference_sets(
    references: dict,
    fronts_by_instance: dict,
    margin: float = NADIR_MARGIN,
) -> dict:
    """
    Reescala las referencias con la caja que abarcan P* y los frentes medidos.

    Por que hace falta
    -----------------
    La escala de `ReferenceSet` sale de P_i^*: `ideal` es su minimo y `span` su
    recorrido. Eso funciona mientras P* tenga recorrido. Cuando el frente
    verdadero es un UNICO punto (las instancias `KC10-2fl-2un` y
    `KC20-2fl-2un`, cuyos dos objetivos estan correlacionados con rho > 0.9),
    `ideal == max` y `span` cae al valor de convenio 1.0. Los valores
    normalizados quedan entonces en la escala CRUDA de los objetivos (~2e5), y
    de ahi sale todo lo que se observe:

    * IGD+ mide distancias de ~1e5 a ~1e8 en vez de ~0 a ~1, y sus valores no
      son comparables con los de una referencia con recorrido.
    * DeltaP vale exactamente 1 para cualquier solucion, porque la caja entre
      el frente y el nadir es la caja entera: no discrimina nada.
    * R2 crece con la lejania pero sobre una escala espuria.

    Que se hace
    -----------
    La caja de normalizacion se pasa a ser la de la UNION de P* con los
    frentes que se van a medir, en vez de la de P* sola. Los puntos de P*
    siguen siendo los que IGD+ y R2 usan como referencia, y la caja sigue
    siendo la misma para todos los individuos de la instancia, asi que los
    valores siguen siendo comparables entre individuos y entre generaciones.

    Ventaja adicional: la regla es la MISMA para todas las instancias, richa
    con P* de 15 o 100 puntos o con un solo punto. No hace falta ningun caso
    especial, y por eso el analisis puede calcular los cuatro indicadores en
    todas las instancias sin descartar ninguna.

    Es la misma idea que ya usa el HV (`HypervolumeContext` se ajusta a los
    frentes consolidados), pero para las metricas relativas a P*: aqui la caja
    se extiende con P* en lugar de sustituirse por ella.

    @param references: dict nombre -> `ReferenceSet` (los originales, sin
        modificar).
    @param fronts_by_instance: dict nombre -> frente o lista de frentes a
        medir en esa instancia. Se acepta cualquiera de las dos formas: un
        array (n, m) o una secuencia de ellos, como el consolidado por
        presupuesto y la lista por individuo que usan los dos modulos de
        analisis.
    @param margin: margen relativo del nadir de DeltaP.
    @return: dict nombre -> `ReferenceSet` con la escala poolingada y el
        `front` original intacto.
    """
    escaladas = {}

    for nombre, referencia in references.items():
        bloques = [referencia.front]
        medidos = fronts_by_instance.get(nombre)
        if medidos is not None:
            for front in _como_frentes(medidos):
                bloques.append(front)

        pooled = np.vstack(bloques)
        if pooled.shape[1] != referencia.n_objectives:
            raise ReferenceSetError(
                f"{nombre}: los frentes medidos tienen {pooled.shape[1]} "
                f"objetivos y P* tiene {referencia.n_objectives}"
            )

        ideal = np.min(pooled, axis=0)
        top = np.max(pooled, axis=0)
        span = top - ideal
        span = np.where(span > 0, span, 1.0)
        nadir = top + margin * span

        # El `front` se deja como estaba: los puntos de referencia que IGD+ y
        # R2 comparan son los de P*, no los de la caja poolingada.
        escaladas[nombre] = ReferenceSet(
            name=nombre,
            front=referencia.front,
            ideal=ideal,
            nadir=nadir,
            span=span,
        )

    return escaladas


# Sufijos que se quitan del nombre del archivo para obtener el nombre de la
# instancia. El manifiesto de `set-reference` es la via autoritativa; estos
# sufijos solo son el respaldo cuando no hay manifiesto.
SUFFIXES = ("_ref", "_reference", "_ground")

MANIFEST_NAME = "reference_manifest.json"


def instance_key(stem: str) -> str:
    """Nombre de la instancia a partir del nombre del archivo."""
    for sufijo in SUFFIXES:
        if stem.endswith(sufijo):
            return stem[: -len(sufijo)]
    return stem


def _discover(directory):
    """
    Devuelve {instancia: ruta del .pof} para un directorio de referencias.

    Si existe el manifiesto de `set-reference` se usa, porque es la fuente
    autoritativa del emparejamiento instancia -> archivo. Si no, se empareja por
    nombre quitando los sufijos conocidos.
    """
    import json

    manifest_path = os.path.join(directory, MANIFEST_NAME)

    if os.path.isfile(manifest_path):
        with open(manifest_path) as handle:
            manifest = json.load(handle)

        pares = {}
        for instancia, entrada in manifest.items():
            archivo = entrada["reference_file"] if isinstance(entrada, dict) \
                else entrada
            pares[instancia] = os.path.join(directory, archivo)
        if pares:
            return pares

    return {
        instance_key(os.path.splitext(f)[0]): os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.endswith(".pof")
    }


def load_reference_sets(directory, n_objectives=None, margin: float = NADIR_MARGIN):
    """
    Carga los conjuntos de referencia de un directorio.

    Se aceptan los `.pof` de RSEHH y los `.npy` por instancia. El emparejamiento
    instancia -> archivo sale del manifiesto si existe, y del nombre en caso
    contrario.

    El directorio puede contener referencias de mas (por ejemplo las de 3
    objetivos cuando la generacion actual es de 2): `n_objectives` no las
    rechaza, solo marca cuales son utilizables. La validacion dura de que las
    referencias que la generacion va a usar existan y coincidan en numero de
    objetivos la hace `ReferenceFitness.check`.

    @param directory: directorio con los conjuntos.
    @param n_objectives: si se indica, solo se devuelven las referencias con ese
        numero de objetivos (None = todas).
    @param margin: margen relativo del nadir de DeltaP.
    @return: dict nombre -> ReferenceSet.
    """
    if not os.path.isdir(directory):
        raise ReferenceSetError(
            f"directorio de referencias inexistente: {directory}"
        )

    pares = _discover(directory)

    if not pares:
        raise ReferenceSetError(
            f"no hay conjuntos de referencia (*.pof, *.npy o "
            f"{MANIFEST_NAME}) en {directory}"
        )

    referencias = {}
    for nombre, ruta in pares.items():
        if ruta.endswith(".npy"):
            front = np.load(ruta)
        else:
            front = read_pof(ruta)

        referencia = _build(nombre, front, margin)
        if n_objectives is not None and referencia.n_objectives != n_objectives:
            continue
        referencias[nombre] = referencia

    if not referencias:
        raise ReferenceSetError(
            f"ningun conjunto de referencia de {directory} tiene "
            f"{n_objectives} objetivos"
        )

    return referencias
