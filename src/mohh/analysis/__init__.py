"""
Analisis offline de experimentos MOHH.

Este paquete agrupa las piezas que usan los dos scripts de la raiz:

* `layout`       -> resolver rutas y organizar la salida;
* `io_utils`     -> leer individuos, generaciones, operadores y CSV;
* `indicators_all` -> HV, IGD+, DELTA_P y R2 sobre frentes guardados;
* `sdd_all`      -> la SDD de cada uno de esos indicadores;
* `stats`        -> Friedman, Nemenyi, boxplots y Wilcoxon-Holm;
* `complexity`   -> tiempo real de los operadores sobre poblaciones reales;
* `phenotype`    -> conversion del operador fenotipico a codigo Python;
* `cache`        -> no repetir ninguna tarea ya calculada.

Ningun modulo genera datos de entrada: todos leen lo que las corridas
produjeron o lo que el usuario pidio ejecutar.
"""

__all__ = [
    "cache",
    "complexity",
    "indicators_all",
    "io_utils",
    "layout",
    "phenotype",
    "sdd_all",
    "stats",
]
