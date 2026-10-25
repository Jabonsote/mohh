"""
R2 para problemas de minimizacion multiobjetivo.

Referencia
----------
Hansen & Jaszkiewicz (2008), "A New Evaluation Method Comparing Performance of
Multiobjective Evolutionary Algorithms". M. Hansen y S. Jaszkiewicz, Fifth
European Congress on Intelligent Techniques and Soft Computing, EUCIS 2008.

Definicion
----------
Conjunto de vectores de peso W (simplex uniforme por defecto) e ideal z*:

    R(w)   = min_{a in A}  w^T (f(a) - z*) / ||w||_2
    R2(A)  = (1/|W|) * sum_{w in W} R(w)

Es una escalarizacion de Tchebycheff POR DIRECCION: para cada peso se toma la
MEJOR solucion del frente en esa direccion, y despues se promedia sobre las
direcciones. Menor es mejor, y vale 0 exactamente cuando el frente contiene un
punto en el ideal para cada direccion de peso.

OJO sobre el `min` y no el `sum`: R2 suma sobre las DIRECCIONES, no sobre las
soluciones. Si se sumara sobre el frente, R2 creceria con el numero de
soluciones y un individual's cuya corrida MOEA colapsase en un unico punto
bueno obtendria artificialmente el mejor R2, que es justo lo contrario de lo
deseado. Con el `min`, un frente que se extiende bien por todas las direcciones
consigue un R2 menor que uno que se concentra.

Con los objetivos normalizados a [0,1] y los pesos de norma L2 1, cada
R(w) cumple `0 <= R(w) <= sum(w) <= sqrt(m)`, de modo que

    0 <= R2 <= max_w sum(w) <= sqrt(m)

OJO: la cota NO es 1 cuando m > 1. Con pesos de norma 1, `w . x` alcanza
`sum(w)` en la esquina x = (1, ..., 1), y `sum(w) <= sqrt(m)` por Cauchy-Schwarz
(la igualdad daria un peso uniforme, que el muestreo de Dirichlet no produce).
Con los pesos por defecto (m+1 vectores) se mide, por ejemplo,
`max_w sum(w) = 1.391` para m = 2 y `1.681` para m = 3, frente a `sqrt(2) =
1.414` y `sqrt(3) = 1.732`. Por eso un R2 de 1.05 en un problema de 3
objetivos es un valor legitimo y no una cota rota.

Esto no afecta a la SDD: la SDD es una desviacion RELATIVA sobre la matriz de
indicadores, no un valor absoluto, asi que que el indicador se mueva en una escala
que no sea [0,1] no la afecta. Lo que si importa es que la escala la fije la
referencia y no el frente del individual's, que es como lo llama
`ReferenceFitness` (`normalize=False` e `ideal = 0`).

Ademas R2 vale 0 exactamente cuando el frente contiene un punto en el ideal para
cada direccion de peso. Convention: los problemas del proyecto (mQAP) son de
MINIMIZACION, luego menor es mejor, que es la convencion de esta
implementacion.
"""


import numpy as np

__all__ = ["r2", "uniform_weights"]


def uniform_weights(n_objectives, n_weights=None, seed=None):
    """
    Vectores de peso uniformes normalizados a norma L2 = 1.

    Se muestrean del simplex de Dirichlet (componentes positivas que suman 1) y
    despues se normalizan a norma 1, que es la convencion que hace que
    `w . x <= ||x||` para `x` en [0,1]^m y por tanto acota R2 por |A|.

    @param n_objectives: numero de objetivos m.
    @param n_weights: numero de vectores; None => m+1 (recomendado por Hansen).
    @param seed: semilla para reproducibilidad.
    @return: (n_weights, m) con filas de norma L2 1 y componentes > 0.
    """
    m = int(n_objectives)
    if m < 1:
        raise ValueError("n_objectives debe ser >= 1")

    n_weights = int(n_weights) if n_weights is not None else m + 1
    if n_weights < 1:
        raise ValueError("n_weights debe ser >= 1")

    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(m), size=n_weights)

    return weights / np.linalg.norm(weights, axis=1, keepdims=True)


def r2(front, weights=None, ideal=None, normalize=True):
    """
    R2 (minimizacion).

    @param front: (n, m) frente de aproximacion.
    @param weights: (s, m) vectores de peso; None => uniforme con semilla fija.
    @param ideal: (m,) punto ideal z*; None => minimo de cada objetivo del
        frente de aproximacion.
    @param normalize: si True, escala cada objetivo a [0,1] usando el ideal y el
        nadir *del propio frente* antes de escalarizar. Con `normalize=False` la
        escala la fija el llamante, que es lo que hace `ReferenceFitness`.
    @return: escalar en [0, sqrt(m)]. Menor es mejor.
    """
    front = np.atleast_2d(np.asarray(front, dtype=float))
    if front.size == 0:
        raise ValueError("front vacio")
    if front.ndim != 2:
        raise ValueError("front debe ser una matriz (n, m)")
    if not np.all(np.isfinite(front)):
        raise ValueError("front contiene valores no finitos")

    m = front.shape[1]

    if ideal is None:
        ideal = np.min(front, axis=0)
    else:
        ideal = np.asarray(ideal, dtype=float).reshape(-1)
        if ideal.shape[0] != m:
            raise ValueError(
                f"ideal con {ideal.shape[0]} componentes, se esperaban {m}"
            )

    if normalize:
        nadir = np.max(front, axis=0)
        span = nadir - ideal
        # Un objetivo constante no aporta al indicador: se deja con escala 1.
        span = np.where(span > 0, span, 1.0)
        front = (front - ideal) / span
        offset = np.zeros(m, dtype=float)
    else:
        front = front - ideal
        offset = np.zeros(m, dtype=float)

    if weights is None:
        weights = uniform_weights(m)
    else:
        weights = np.atleast_2d(np.asarray(weights, dtype=float))
        if weights.ndim != 2 or weights.shape[1] != m:
            raise ValueError(
                f"weights con forma {weights.shape}, se esperaba (s, {m})"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contienen valores no finitos")

    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("weights no puede contener el vector nulo")

    # (s, m) @ (m, n) -> (s, n): valor escalarizado de cada peso y cada punto.
    distances = weights @ (front - offset).T

    # Para cada direccion se toma la MEJOR solucion del frente: R2 promedia
    # sobre las direcciones, nunca sobre las soluciones.
    per_weight = np.min(distances, axis=1) / norms[:, 0]

    return float(np.mean(per_weight))
