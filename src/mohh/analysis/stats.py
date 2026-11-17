"""
Estadistica de comparacion multiple entre algoritmos.

Sin dependencias externas mas alla de numpy/scipy/matplotlib, este modulo
ofrece lo que pide el analisis:

* rangos promedio (average rank) por instancia, como en la democacia de Demsar;
* test de Friedman + diagrama de diferencias criticas (CD) de Nemenyi, con la
  constante de diferencia critica calculada a partir de la distribucion
  studentizada del rango (no un valor fijo de manual);
* boxplots de los valores por configuracion (en escala logaritmica cuando los
  HV abarcan varias ordenes de magnitud);
* test de Wilcoxon pareado entre cada configuracion y una de referencia, con
  correccion de Holm para la familia de comparaciones.

La DIRECCION de cada indicador sale de `mohh.core.indicators.directions`: HV
se maximiza, IGD+, DELTA_P y R2 se minimizan. Antes de rankear o comparar se
niegan los valores de maximizacion para que "mejor" sea siempre "menor", y
`metric` es quien dice que indicador se esta mirando.
"""

import numpy as np
import pandas as pd

from scipy import stats

from mohh.core.indicators.directions import indicator_minimization

__all__ = [
    "average_ranks",
    "friedman_test",
    "nemenyi_cd",
    "critical_difference_diagram",
    "boxplot_configs",
    "wilcoxon_vs_reference",
]


def _es_minimizacion(metric) -> bool:
    """Direccion del indicador, tolerando variantes de mayusculas."""
    return indicator_minimization(str(metric).strip().upper())


def average_ranks(frame: pd.DataFrame, metric: str = "HV") -> pd.DataFrame:
    """
    Rango promedio por configuracion, calculado instancia por instancia.

    @param frame: DataFrame `instancia x configuracion` con el valor del
        indicador.
    @param metric: nombre del indicador; fija la direccion con la que se
        rankea (1 = el mejor de esa instancia).
    @return: DataFrame con una fila por instancia y una columna por
        configuracion con su rango promedio (1 = el mejor de esa instancia).
    """
    valores = frame.copy()

    if not _es_minimizacion(metric):
        # Maximizacion: negar para que el menor valor sea el mejor.
        valores = -valores

    return valores.rank(axis=1, method="average")


def friedman_test(frame: pd.DataFrame, metric: str = "HV"):
    """
    Test de Friedman sobre las instancias (bloques).

    @param frame: DataFrame `instancia x configuracion`.
    @param metric: indicador; fija la direccion con la que se rankea.
    @return: (estadistico, p_valor) o (nan, nan) si no se puede calcular.
    """
    rangos = average_ranks(frame, metric)
    datos = rangos.to_numpy(dtype=float)
    n_bloques, n_algoritmos = datos.shape

    if n_algoritmos < 3 or n_bloques < 2:
        return float("nan"), float("nan")

    try:
        estadistico, p_valor = stats.friedmanchisquare(*datos.T)
    except ValueError:
        return float("nan"), float("nan")

    return float(estadistico), float(p_valor)


#: q de la distribucion studentizada del rango, tabla 5.7 de Demsar (2002,
#: "Statistical Comparisons of Classifiers over Multiple Data Sets"). Son los
#: valores que producen los CD publicados de referencia (k = 3, N = 10,
#: alpha = 0.05 -> CD = 1.674). La normalizacion de esta tabla NO coincide con
#: la de `scipy.stats.studentized_range`, asi que no se intenta derivarla: se
#: usan las tablas para los alphas habituales y se recurre a Scipy solo para
#: alphas poco comunes.
_DEMSAR_Q = {
    0.05: {
        3: 3.744, 4: 4.334, 5: 4.602, 6: 4.744, 7: 4.869, 8: 4.981,
        9: 5.106, 10: 5.216, 11: 5.317, 12: 5.410, 13: 5.498, 14: 5.579,
        15: 5.655, 16: 5.725, 17: 5.792, 18: 5.856, 19: 5.917, 20: 5.974,
    },
    0.10: {
        3: 3.182, 4: 3.639, 5: 3.958, 6: 4.207, 7: 4.412, 8: 4.589,
        9: 4.741, 10: 4.877, 11: 4.989, 12: 5.091, 13: 5.185, 14: 5.273,
        15: 5.354, 16: 5.432, 17: 5.505, 18: 5.575, 19: 5.640, 20: 5.702,
    },
}


def _q_studentized_range(k: int, alpha: float) -> float:
    """
    Cuantil q(alpha) de la distribucion studentizada del rango.

    Se consulta la tabla de Demsar para alpha = 0.05 y 0.10, que son los
    niveles con los que se publica el post-hoc de Nemenyi. Para otros alphas se
    usa SciPy, asumiendo la misma normalizacion que la tabla, que se calibra con
    el cociente de la tabla para k = 5.

    @param k: numero de algoritmos.
    @param alpha: nivel de significacion.
    @return: valor de q.
    """
    alpha = float(alpha)
    tabla = _DEMSAR_Q.get(round(alpha, 4))
    if tabla is not None and k in tabla:
        return float(tabla[k])

    referencia = _DEMSAR_Q[0.05].get(min(max(k, 3), 20), 4.602)

    try:
        # SciPy con df -> infinito da el rango de k normales; se escala con el
        # factor que lleva esa magnitud a la normalizacion de la tabla.
        q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf))
        escala = referencia / float(
            stats.studentized_range.ppf(0.95, k, np.inf))
        return q * escala
    except Exception:
        return referencia


def nemenyi_cd(n_instances: int, n_algorithms: int, alpha: float = 0.05) -> float:
    """
    Diferencia critica de Nemenyi para un post-hoc de Friedman.

    CD = q(alpha) * sqrt( k(k+1) / (6 n) ), con n el numero de bloques
    (instancias) y k el numero de algoritmos comparados.

    @param n_instances: numero de bloques (instancias).
    @param n_algorithms: numero de algoritmos comparados.
    @param alpha: nivel de significacion.
    @return: valor de CD.
    """
    k, n = int(n_algorithms), int(n_instances)
    if k < 2 or n < 2:
        return float("nan")

    q = _q_studentized_range(k, alpha)
    return float(q * np.sqrt(k * (k + 1) / (6 * n)))


def critical_difference_diagram(
    frame: pd.DataFrame,
    metric: str = "HV",
    alpha: float = 0.05,
    title: str = None,
    path: str = None,
):
    """
    Diagrama de diferencias criticas de Nemenyi.

    @param frame: DataFrame `instancia x configuracion` (bloques x metodos).
    @param metric: indicador; fija la direccion con la que se rankea.
    @param alpha: nivel de significacion.
    @param title: titulo del grafico.
    @param path: si se indica, se guarda la figura en esa ruta.
    @return: (fig, cd, friedman_p) con la figura de matplotlib, el CD y el
        p-valor de Friedman.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rangos = average_ranks(frame, metric)
    promedios = rangos.mean(axis=0).sort_values()
    n_bloques, k_metodos = rangos.shape
    cd = nemenyi_cd(n_bloques, k_metodos, alpha)
    _, p_friedman = friedman_test(frame, metric)

    # El eje de rangos va de 1 a k = numero de METODOS comparados (columnas),
    # nunca de bloques (filas): con `shape[0]` la figura salia estirada hasta
    # el numero de instancias/individuos (1100) y todos los puntos quedaban
    # apretados contra el borde izquierdo.
    etiquetas = list(promedios.index)
    valores = promedios.to_numpy(dtype=float)
    limite = max(float(valores.max()), float(k_metodos)) + 0.5
    # Los rangos son >= 1: sin este tope, `valores.min() - cd` podia bajar a
    # rangos negativos (p. ej. CD=30 con 6 bloques) y dejaba media figura
    # vacia a la izquierda.
    inferior = max(min(float(valores.min()) - cd, 1.0) - 0.5, 0.5)

    # La ALTURA crece con el numero de etiquetas (una por fila); el ancho se
    # mantiene razonable, que es lo contrario de lo que hacia antes.
    figura_alta = max(3.5, min(0.32 * len(etiquetas) + 1.5, 18.0))
    fig, ax = plt.subplots(figsize=(11.0, figura_alta))

    ax.set_xlim(inferior, limite)
    ax.set_ylim(0, len(etiquetas))

    for i, etiqueta in enumerate(etiquetas):
        ax.plot([valores[i], valores[i]], [i, i], color="black", linewidth=2)
        ax.plot([valores[i]], [i], "o", color="black", markersize=8)

    # Barras horizontales que muestran los rangos significativamente distintos.
    orden = np.argsort(valores)
    for a in range(len(orden)):
        for b in range(a + 1, len(orden)):
            i, j = orden[a], orden[b]
            if valores[j] - valores[i] > cd:
                ax.plot([valores[i], valores[j]], [j, j],
                        color="#888", linewidth=1.2, linestyle="-")
            else:
                break

    ax.axvline(1, color="gray", linestyle=":", linewidth=1)
    ax.axvline(k_metodos, color="gray", linestyle=":", linewidth=1)
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas, fontsize=max(6.0, min(10.0, 220 / max(
        len(etiquetas), 1))))
    ax.set_xlabel("Rango promedio (1 = mejor)")
    titulo = title or f"Diferencias criticas - {metric}"
    # Si el CD supera la diferencia maxima posible entre rangos (k - 1), no
    # puede haber NI UNA pareja significativa: se dice en el titulo en vez de
    # dejar una figura que parece dar un resultado.
    # Titulo super simple: solo QUE se compara. El CD y el p-valor de
    # Friedman van a consola, y en el titulo solo se queda el aviso cuando el
    # resultado no discrimina (CD >= k-1: imposible que haya pareja
    # significativa).
    inviable = cd >= (k_metodos - 1) and k_metodos > 1
    nota = " (sin parejas significativas)" if inviable else ""
    ax.set_title(f"{titulo}{nota}")
    if path:
        aviso = "; CD >= k-1, no hay pareja significativa" if inviable else ""
        # `p=0.00e+00` es underflow de la distribucion de Friedman, no un
        # p-valor de cero: se escribe como cota.
        texto_p = ("p<1e-16" if (not np.isfinite(p_friedman)
                                 or p_friedman < 1e-16)
                   else f"p={p_friedman:.2e}")
        print(f"  nota: CD {titulo}: CD={cd:.2f}, Friedman {texto_p} "
              f"({n_bloques} bloques x {k_metodos} metodos){aviso}",
              flush=True)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path, cd, p_friedman

    return fig, cd, p_friedman


def boxplot_configs(
    frame: pd.DataFrame,
    metric: str = "HV",
    title: str = None,
    path: str = None,
    log_scale: bool = None,
):
    """
    Boxplot de los valores del indicador por configuracion.

    @param frame: DataFrame `instancia x configuracion`.
    @param metric: indicador.
    @param title: titulo.
    @param path: si se indica, guarda la figura.
    @param log_scale: forzar escala logaritmica; si es None se decide sola
        (log cuando el rango max/min supera 100x, util para HV).
    @return: ruta de la figura guardada, o None si no se guardo.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [frame[col].dropna().to_numpy(dtype=float)
            for col in frame.columns]
    etiquetas = list(frame.columns)

    # Boxplot horizontal: las etiquetas van en el eje Y, asi que la ALTURA es
    # la que tiene que crecer con el numero de configuraciones (antes crecia
    # el ancho y con 89 configuraciones salia una figura de 60 pulgadas con
    # las etiquetas apiladas).
    figura_alta = max(4.0, min(0.34 * len(etiquetas) + 1.5, 18.0))
    fig, ax = plt.subplots(figsize=(10.0, figura_alta))
    # matplotlib >= 3.9 renombro `labels` a `tick_labels`; se usa el nombre
    # nuevo, que es el que sigue vivo en las versiones actuales del entorno.
    ax.boxplot(data, tick_labels=etiquetas, vert=False, patch_artist=True,
               medianprops=dict(color="black", linewidth=1.5))
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} por configuracion")

    if log_scale is None:
        positivos = [v for serie in data for v in serie if v > 0]
        if positivos:
            log_scale = (max(positivos) / min(positivos)) > 100
        else:
            log_scale = False

    if log_scale:
        ax.set_xscale("log")

    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    plt.close(fig)
    return None


def wilcoxon_vs_reference(
    frame: pd.DataFrame,
    reference: str,
    metric: str = "HV",
) -> pd.DataFrame:
    """
    Wilcoxon pareado de cada configuracion contra una de referencia.

    Los valores de maximizacion (HV) se niegan para que en todos los casos
    "menor" sea mejor y la hipotesis alternativa sea siempre la misma (la
    configuracion mejora). `metric` fija la direccion: sin el, todos los
    indicadores se tratarian como si fueran de maximizacion.

    @param frame: DataFrame `instancia x configuracion`.
    @param reference: nombre de la columna de referencia.
    @param metric: indicador; fija la direccion con la que se comparan.
    @return: DataFrame con una fila por configuracion: `n`, `statistic`,
        `p_raw`, `p_holm`, `better` (si la media de diferencias es negativa) y
        `mean_diff`.
    """
    if reference not in frame.columns:
        raise ValueError(f"la referencia '{reference}' no esta en el DataFrame")

    datos = -frame if not _es_minimizacion(metric) else frame
    base = datos[reference]

    filas = []
    nombres = list(frame.columns)
    p_valores = {}

    for columna in nombres:
        if columna == reference:
            continue
        try:
            estadistico, p_valor = stats.wilcoxon(datos[columna], base)
        except ValueError:
            # Diferencias todas identicas (p. ej. misma configuracion duplicada).
            estadistico, p_valor = float("nan"), 1.0
        filas.append({
            "config": columna,
            "n": int(datos[columna].notna().sum()),
            "statistic": float(estadistico),
            "p_raw": float(p_valor),
            "mean_diff": float((datos[columna] - base).mean()),
        })
        p_valores[columna] = float(p_valor)

    tabla = pd.DataFrame(filas)
    tabla["better"] = tabla["mean_diff"] < 0

    # Correccion de Holm sobre la familia de comparaciones.
    if not tabla.empty:
        orden = np.argsort(tabla["p_raw"].to_numpy(dtype=float))
        m = len(orden)
        holm = np.empty(m, dtype=float)
        running = 0.0
        for rank, idx in enumerate(orden):
            running = max(running, (m - rank) * tabla["p_raw"].iloc[idx])
            holm[idx] = min(1.0, running)
        tabla["p_holm"] = holm
        tabla = tabla.sort_values(["p_holm", "config"]).reset_index(drop=True)

    return tabla
