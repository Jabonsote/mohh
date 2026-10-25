"""
Traduccion de un operador fenotipico (GE) a codigo Python legible.

Un fenotipo como

    order_based(fill_first_occurring, preserve_elements, y,
                move_right(reverse(y)))

es una expresion de Python que el MOEA evalua con `eval`. Para poder leer,
revisar y reusar los operadores fuera del GE, este modulo genera un modulo
Python con:

1. las funciones auxiliares que el fenotipo usa, copiadas de
   `mohh.core.utilities.algorithm.HH_functions` (mismas fuentes, sin
   reimplementarlas);
2. una funcion `operate(x, y)` con el cuerpo del fenotipo, deducida del numero
   de aridad a partir de las variables libres (`x` y/o `y`).

No ejecuta el operador ni inventa comportamiento: el codigo generado es
equivalente al que el MOEA ya evaluo.
"""

import ast
import os
import inspect
import re

import mohh.core.utilities.algorithm.HH_functions as helpers

__all__ = [
    "arity_of",
    "used_functions",
    "to_python_module",
]


#: Nombres de las funciones auxiliares disponibles en `HH_functions`.
AVAILABLE_FUNCTIONS = {
    name for name, value in inspect.getmembers(helpers, inspect.isfunction)
    if not name.startswith("_")
}

#: Variables libres que puede usar un fenotipo.
VARIABLES = ("x", "y")


def _parse(phenotype: str):
    try:
        return ast.parse(phenotype.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(
            f"el fenotipo no es una expresion de Python valida: {error}") from error


def used_functions(phenotype: str) -> list:
    """
    Funciones auxiliares que aparecen en el fenotipo, en orden de aparicion.

    @param phenotype: expresion del operador.
    @return: lista de nombres presentes en `AVAILABLE_FUNCTIONS`.
    """
    arbol = _parse(phenotype)
    vistos, resultado = set(), []

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Name) and nodo.id in AVAILABLE_FUNCTIONS:
            if nodo.id not in vistos:
                vistos.add(nodo.id)
                resultado.append(nodo.id)

    return resultado


def _free_variables(phenotype: str) -> set:
    arbol = _parse(phenotype)

    argumentos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.arguments):
            for argumento in list(nodo.args) + list(nodo.kwonlyargs) + list(nodo.posonlyargs):
                argumentos.add(argumento.arg)

    return {nodo.id for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Name)} - argumentos


def arity_of(phenotype: str) -> int:
    """
    Cuantasvariables de entrada usa el operador (1 o 2).

    @param phenotype: expresion del operador.
    @return: numero de variables libres entre `x` e `y`.
    """
    libres = _free_variables(phenotype) & set(VARIABLES)
    return max(1, len(libres))


def to_python_module(phenotype: str, name: str = "operator") -> str:
    """
    Devuelve el codigo Python de un operador.

    @param phenotype: expresion del operador.
    @param name: nombre de la funcion principal.
    @return: texto del modulo.
    """
    funciones = used_functions(phenotype)
    aridad = arity_of(phenotype)

    firma = ", ".join(VARIABLES[:aridad])

    lineas = [
        '"""',
        f"Operador generado automaticamente a partir del fenotipo:",
        "",
        f"    {phenotype}",
        "",
        f"Aridad: {aridad}. Generado por mohh.analysis.phenotype.",
        '"""',
        "",
        "import numpy as np",
        "",
    ]

    # Funciones auxiliares, con su codigo original.
    for funcion in funciones:
        fuente = inspect.getsource(getattr(helpers, funcion)).rstrip()
        lineas.append(fuente)
        lineas.append("")
        lineas.append("")

    # Helpers de uso comun. `collapse` se emite siempre porque `verify_inputs`
    # lo llama cuando recibe una tupla, y las funciones del nucleo devuelven
    # tuplas.
    lineas.append("def collapse(x, y=None):")
    lineas.append("    return x")
    lineas.append("")
    lineas.append("")
    lineas.append("def verify_inputs(z):")
    lineas.append("    if isinstance(z, tuple):")
    lineas.append("        z = collapse(z[0], z[1])")
    lineas.append("    return z")
    lineas.append("")
    lineas.append("")
    lineas.append(f"def {name}({firma}):")
    lineas.append('    """')
    lineas.append(f"    Aplica el operador a ({firma}) y devuelve el hijo.")
    lineas.append('    """')

    for variable in VARIABLES[:aridad]:
        lineas.append(f"    {variable} = verify_inputs({variable})")

    # El cuerpo se indenta cuatro espacios mas y se DEVUELVE: sin `return` el
    # operador generado daria None.
    cuerpo = phenotype
    for linea in cuerpo.splitlines():
        lineas.append(f"    {linea}" if linea.strip() else "")
    lineas.append("")
    lineas.append(f"    return {phenotype}")

    lineas.append("")
    lineas.append("")
    lineas.append('if __name__ == "__main__":')
    if aridad == 1:
        lineas.append("    p1 = np.arange(10)")
        lineas.append(f"    print({name}(p1))")
    else:
        lineas.append("    p1 = np.arange(10)")
        lineas.append("    p2 = np.arange(10)[::-1].copy()")
        lineas.append(f"    print({name}(p1, p2))")

    return "\n".join(lineas) + "\n"


def write_python_module(phenotype: str, path: str, name: str = "operate") -> str:
    """
    Escribe el modulo Python del operador.

    @param phenotype: expresion del operador.
    @param path: ruta del `.py` de destino.
    @param name: nombre de la funcion principal.
    @return: la ruta escrita.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    codigo = to_python_module(phenotype, name=name)
    with open(path, "w") as handle:
        handle.write(codigo)
    return path


def sanitize_filename(name: str) -> str:
    """Convierte una etiqueta en un nombre de archivo seguro."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "operator"
