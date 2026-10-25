"""
SDD (Success Difference Degree) para evaluar y ordenar operadores candidatos.

Definicion (ecuaciones del pizatron):

    O_i : valor obtenido por el candidato h en el problema i
          K = 1  ->  O_i = HV(A_i^h, z_ref)
          K > 1  ->  O_i = (1 / K) * sum_l HV(A_{i,l}^h, z_ref)
    b_i : mejor valor conocido en el problema i
          indicadores de maximizacion (HV) -> b_i = max_h O_i
          indicadores de minimizacion (IGD+, Delta_p, R2) -> b_i = min_h O_i
    x_i : diferencia porcentual relativa
          x_i = 0                                  si O_i = 0 y b_i = 0
                |O_i - b_i| / ((O_i + b_i) / 2) * 100   en otro caso
    x_b : media de las diferencias,  x_b = (1 / N) * sum_i x_i
    SDD : desviacion estandar muestral de {x_1..x_N} sobre N - 1 grados de
          libertad:  SDD = sqrt( sum_i (x_i - x_b)^2 / (N - 1) )

    SDD(h) -> Fit.  Menor SDD = mayor estabilidad; es el valor de aptitud
    que consume el bucle evolutivo (menor es mejor).

El modulo es autocontenido: no importa `mohh` a nivel de modulo, de modo que
puede usarse como libreria independiente (fitness online) o para el analisis
posthoc de los CSV de `mohh-evaluate`.
"""

import os

import numpy as np
import pandas as pd


# Indicadores cuyo mejor valor conocido es el minimo.
_MINIMIZATION_METRICS = {"IGD+", "DELTA_P", "R2"}

# Alias aceptados para cada indicador.
_METRIC_ALIASES = {
    "HV": "HV",
    "HYPERVOLUME": "HV",
    "IGD+": "IGD+",
    "IGD_PLUS": "IGD+",
    "IGDPLUS": "IGD+",
    "DELTA_P": "DELTA_P",
    "DELTAP": "DELTA_P",
    "DP": "DELTA_P",
    "R2": "R2",
}

# Agregaciones disponibles para collapsing las K corridas independientes.
_AGGREGATORS = {"mean": np.mean, "median": np.median}

RANKING_COLUMNS = ["Operador", "Mean_Diff (x_bar %)", "SDD (Fitness)"]


def _parse_column(col: str) -> dict:
    """
    Descompone el nombre de una columna de combinacion.

    Formato: <modelo>_<combinacion>_<NM|WM>_<modelo_origen>
    Ejemplos: "MOEAD_Best_NM_MOEAD", "None_PMX_WM_NSGAII"
    """
    parts = col.split("_")

    if len(parts) < 3:
        return {"model": col, "combination": col, "mutation": None, "solver": None}

    return {
        "model": parts[0],
        "combination": parts[1],
        "mutation": parts[2],
        "solver": parts[-1],
    }


class SDD():
    """
    SDD reutilizable sobre una matriz de valores obtenidos O_i.

    La clase no mantiene estado entre llamadas, de modo que la misma
    instancia puede usarse en cada generacion del bucle evolutivo.

    @param metric_type: indicador de la familia {HV, R2, IGD+, Delta_p}.
    @param aggregation: como colapsar las K corridas independientes de cada
                        celda; "mean" (protocolo del pizatron) o "median".
    """

    def __init__(self, metric_type: str = "HV", aggregation: str = "mean") -> None:

        self.metric_type = self._normalize_metric(metric_type)
        self.aggregation = self._normalize_aggregation(aggregation)

    # ------------------------------------------------------------------
    # Validacion de la configuracion
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_metric(metric_type: str) -> str:

        if not isinstance(metric_type, str):
            raise ValueError(f"metric_type debe ser un str, recibido {type(metric_type)}.")

        key = metric_type.strip().upper().replace(" ", "")

        if key not in _METRIC_ALIASES:
            raise ValueError(
                f"Indicador '{metric_type}' no soportado. "
                f"Usa uno de: {sorted(set(_METRIC_ALIASES.values()))}."
            )

        return _METRIC_ALIASES[key]

    @staticmethod
    def _normalize_aggregation(aggregation: str) -> str:

        if not isinstance(aggregation, str):
            raise ValueError(f"aggregation debe ser un str, recibido {type(aggregation)}.")

        key = aggregation.strip().lower()

        if key not in _AGGREGATORS:
            raise ValueError(
                f"Agregacion '{aggregation}' no soportada. Usa 'mean' o 'median'."
            )

        return key

    @property
    def is_minimization(self) -> bool:
        """True si el mejor valor conocido es el minimo (IGD+, Delta_p, R2)."""
        return self.metric_type in _MINIMIZATION_METRICS

    # ------------------------------------------------------------------
    # Entrada / validacion
    # ------------------------------------------------------------------

    @staticmethod
    def _as_frame(results) -> pd.DataFrame:

        if isinstance(results, pd.DataFrame):
            return results.astype(float)

        array = np.asarray(results, dtype=float)

        if array.ndim != 2:
            raise ValueError(
                f"results debe ser 2D (problemas x operadores), recibido {array.ndim}D."
            )

        return pd.DataFrame(
            array,
            columns=[f"Op_{i + 1}" for i in range(array.shape[1])],
        )

    def _validated(self, results) -> pd.DataFrame:

        frame = self._as_frame(results)

        if frame.shape[1] == 0:
            raise ValueError("results no contiene ningun operador (columnas vacias).")

        n_problems = frame.shape[0]

        if n_problems < 2:
            raise ValueError(
                f"N debe ser >= 2 para calcular la SDD muestral (N-1); "
                f"recibidos N = {n_problems}."
            )

        # Ojo: no se usa `frame.stack()` para localizar la celda, porque
        # `stack()` descarta los NaN por defecto en pandas >= 1.3 y la mascara
        # saldria vacia. Se busca directamente sobre la matriz numerica.
        valores = frame.to_numpy(dtype=float)
        if not np.isfinite(valores).all():
            fila, columna = (int(i) for i in np.argwhere(~np.isfinite(valores))[0])
            raise ValueError(
                f"results contiene valores no finitos (NaN/inf) en la celda "
                f"({frame.index[fila]}, {frame.columns[columna]}). "
                "Revisa la ejecucion del MOEA antes de calcular la SDD."
            )

        return frame

    # ------------------------------------------------------------------
    # Ecuaciones
    # ------------------------------------------------------------------

    def best_known(self, results) -> pd.Series:
        """
        b_i: mejor valor conocido por problema (punto de sincronizacion).

        max para HV; min para IGD+, Delta_p y R2.
        """
        frame = self._validated(results)

        return frame.min(axis=1) if self.is_minimization else frame.max(axis=1)

    def differences(self, results) -> pd.DataFrame:
        """
        x_i: diferencia porcentual relativa por problema y operador.

        x_i = 0 si el denominador es 0 (caso O_i = b_i = 0); en otro caso
        x_i = |O_i - b_i| / ((O_i + b_i) / 2) * 100.
        """
        frame = self._validated(results)
        best = self.best_known(frame)

        # Se opera con numpy: frame - best alinearia la Serie con las COLUMNAS.
        best_values = best.to_numpy(dtype=float).reshape(-1, 1)

        # Se opera con numpy: frame - best alinearia la Serie con las COLUMNAS.
        values = frame.to_numpy(dtype=float)

        numerator = np.abs(values - best_values)
        denominator = (values + best_values) / 2.0

        # np.divide con where evita la division por cero (sin 0/0 -> NaN).
        x_values = np.zeros_like(numerator)
        np.divide(numerator, denominator, out=x_values, where=denominator != 0.0)

        return pd.DataFrame(
            x_values * 100.0,
            index=frame.index,
            columns=frame.columns,
        )

    def mean_diff(self, results) -> pd.Series:
        """x_b: media de las diferencias x_i de cada operador."""
        return self.differences(results).mean(axis=0)

    def score(self, results) -> pd.Series:
        """
        SDD(h): desviacion estandar muestral de x_i sobre N - 1 grados de
        libertad. Menor es mejor. Conserva el orden de las columnas de
        entrada, para poder alinearla con la poblacion del GE.
        """
        return self.differences(results).std(axis=0, ddof=1)

    def ranking(self, results) -> pd.DataFrame:
        """
        Tabla ordenada de menor a mayor SDD (fitness).

        Empates en SDD se desatan con la distancia media (menor x_b primero)
        y luego alfabeticamente, solo para que la tabla sea determinista; la
        aptitud que consume el GE sigue siendo la SDD pura.
        """
        frame = self._validated(results)
        x_i = self.differences(frame)

        table = pd.DataFrame({
            "Operador": list(frame.columns),
            "Mean_Diff (x_bar %)": x_i.mean(axis=0).to_numpy(dtype=float),
            "SDD (Fitness)": x_i.std(axis=0, ddof=1).to_numpy(dtype=float),
        })

        table = table.sort_values(
            by=["SDD (Fitness)", "Mean_Diff (x_bar %)", "Operador"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)

        table.index = pd.RangeIndex(1, len(table) + 1, name="Rank")

        return table

    def compute(self, results):
        """
        Pipeline completo: (ranking, b_i, x_i).
        """
        frame = self._validated(results)

        return self.ranking(frame), self.best_known(frame), self.differences(frame)


def calculate_sdd(df_results, metric_type: str = "HV", aggregation: str = "mean"):
    """
    Fachada de la ecuacion del pizatron sobre una matriz en memoria.

    @param df_results: DataFrame (problemas x operadores) o ndarray (N, M).
    @param metric_type: 'HV' (maximizacion) o 'IGD+', 'DELTA_P', 'R2' (min).
    @param aggregation: 'mean' (protocolo) o 'median'.

    @return: (ranking, b_i, x_i)
    """
    return SDD(metric_type, aggregation).compute(df_results)


def sdd_scores(results, metric_type: str = "HV", aggregation: str = "mean") -> np.ndarray:
    """
    Vector de fitness SDD en el mismo orden que las columnas de `results`.

    Pensado para el fitness online: evita construir el ranking y mantiene la
    alineacion 1:1 con la poblacion delIndividuals.
    """
    return SDD(metric_type, aggregation).score(results).to_numpy(dtype=float)


def compute_sdd(results_path: str, metric_type: str = "HV",
                aggregation: str = None) -> pd.DataFrame:
    """
    SDD (Success Difference Degree) sobre los CSV de `mohh-evaluate`.

    Por cada solver y presupuesto de evaluacion:
      O_i : promedio de las K corridas independientes (media aritmetica)
      b_i : mejor O_i entre todas las combinaciones comparadas en el problema i
      x_i : diferencia porcentual relativa |O_i - b_i| / ((O_i + b_i)/2) * 100
      SDD : desviacion estandar muestral de {x_1..x_N} con N - 1 g.l.

    Escribe <results_path>/sdd_summary.csv y <results_path>/sdd_differences.csv
    y devuelve el DataFrame resumen.
    """

    # Import perezoso: el modulo se puede usar sin el paquete `mohh` instalado.
    from mohh.evaluation.params import Params

    params = Params()
    solver_names = params["SOLVERS"] or []
    eval_steps = list((params["EVAL_BUDGETS"] or {}).keys())

    if aggregation is None:
        aggregation = params["SDD_AGGREGATION"] or "mean"

    metric = SDD(metric_type, aggregation)
    aggregate = _AGGREGATORS[metric.aggregation]

    rows = []
    diff_rows = []

    for solver in solver_names:

        for evals in eval_steps:

            solver_path = os.path.join(results_path, solver, evals)

            # Un presupuesto puede no alcanzarse si MO_GENERATIONS fue pequeno.
            if not os.path.isdir(solver_path):
                continue

            instance_files = sorted(f for f in os.listdir(solver_path) if f.endswith(".csv"))

            # La desviacion muestral necesita N - 1 >= 1.
            if len(instance_files) < 2:
                print(f"Skipping SDD for {solver}/{evals}: fewer than 2 instances.")
                continue

            # Por combinacion: {instance_name: O_i}
            combos = {}
            # Por instancia: b_i sobre las combinaciones presentes
            instance_best = {}

            for fname in instance_files:

                instance_name = fname.removesuffix(".csv")
                df = pd.read_csv(os.path.join(solver_path, fname))

                value_cols = [c for c in df.columns if c != "Experiment"]

                observed = {}
                for col in value_cols:
                    vals = df[col].dropna()
                    if vals.empty:
                        continue
                    observed[col] = float(aggregate(vals.to_numpy(dtype=float)))

                if not observed:
                    continue

                instance_best[instance_name] = max(observed.values())

                for col, o_i in observed.items():
                    combos.setdefault(col, {})[instance_name] = o_i

            # b_i debe compararse siempre contra el mismo conjunto de candidatos,
            # y cada operador debe medirse sobre las mismas N instancias.
            complete_cols = [c for c, per_inst in combos.items()
                             if set(per_inst) == set(instance_best)]

            if not complete_cols:
                print(f"Skipping SDD for {solver}/{evals}: no combination covers all instances.")
                continue

            dropped_cols = sorted(set(combos) - set(complete_cols))
            if dropped_cols:
                print(f"Skipping incomplete columns for {solver}/{evals}: {dropped_cols}")

            instances = sorted(instance_best)

            table = pd.DataFrame(
                {col: [combos[col][i] for i in instances] for col in complete_cols},
                index=instances,
            )

            ranking, b_i, x_i = metric.compute(table)

            for _, row in ranking.iterrows():

                parsed = _parse_column(row["Operador"])

                rows.append({
                    "Model": parsed["model"],
                    "Operator": parsed["combination"],
                    "Mutation": parsed["mutation"],
                    "Solver": solver,
                    "Budget": evals,
                    "Metric": metric.metric_type,
                    "Aggregation": metric.aggregation,
                    "N_instances": len(instances),
                    "Mean_Diff": float(row["Mean_Diff (x_bar %)"]),
                    "SDD": float(row["SDD (Fitness)"]),
                })

            for instance_name in instances:
                diff_rows.append({
                    "Solver": solver,
                    "Budget": evals,
                    "Instance": instance_name,
                    "b_i": float(b_i.loc[instance_name]),
                    **{col: float(x_i.loc[instance_name, col]) for col in x_i.columns},
                })

    summary = pd.DataFrame(rows)

    if not summary.empty:

        summary = summary.sort_values(
            by=["Solver", "Budget", "SDD", "Mean_Diff"],
            kind="mergesort",
        ).reset_index(drop=True)

        save_path = os.path.join(results_path, "sdd_summary.csv")
        summary.to_csv(save_path, index=False)

        differences = pd.DataFrame(diff_rows)
        diff_path = os.path.join(results_path, "sdd_differences.csv")
        differences.to_csv(diff_path, index=False)

        print("\n" + "-" * 40)
        print("SDD (Success Difference Degree) - lower is better")
        print("-" * 40)
        print(summary.to_string(index=False))
        print(f"\nSaved: {save_path}")
        print(f"Saved: {diff_path}")

    return summary
