"""Siembra un operador: sustituye un generador propio por el RNG global.

El arnes de evaluacion siembra `np.random.seed(...)` antes de cada corrida
del MOEA, pero un operador que crea su propio generador
(`np.random.default_rng()`) ignora esa semilla: no es reproducible y la
comparacion pareada (mismas semillas) mediria ruido. La busqueda ya rechaza
esos operadores (`compile_operator`); este script sirve para los operadores ya
guardados, que hay que sembrar a mano antes de comparar.

Reescribe el generador por el RNG global de numpy (`np.random.*`). La
distribucion de los hijos no cambia, pero a partir de ahi el operador queda
atado a la semilla del MOEA.

    python scripts/seed_operator.py operador.py            # muestra el resultado
    python scripts/seed_operator.py operador.py salida.py  # lo escribe
"""
import argparse
import os
import re
import sys

#: `rng = np.random.default_rng()` (con o sin espacios, con o sin `np.`).
_GENERATOR_ASSIGN = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z_][A-Za-z_0-9]*)\s*=\s*"
    r"(?:np\.)?random\.default_rng\(\)\s*$",
    re.MULTILINE,
)
#: `np.random.default_rng().integers(...)`: generador en linea.
_INLINE_GENERATOR = re.compile(r"(?:np\.)?random\.default_rng\(\)\s*\.\s*")
#: Metodos de `Generator` que no existen en el RNG global.
_METHOD_MAP = {"integers": "randint"}


def _method_replacements(name):
    """Sustituye `name.metodo(...)` por `np.random.<equivalente>(...)`."""
    pattern = re.compile(rf"\b{re.escape(name)}\.(\w+)\s*\(")

    def replace(match):
        method = match.group(1)
        if method in ("integers", "integers_low"):
            return "np.random.randint("
        return f"np.random.{method}("

    return pattern, replace


def seed_operator(code: str):
    """Devuelve `(codigo_sembrado, avisos)`.

    Los avisos son problemas que no se pueden resolver solos (p. ej. un
    `endpoint=True` que `np.random.randint` no admite).
    """
    warnings = []
    names = set()

    def collect(match):
        names.add(match.group("name"))
        return f"{match.group('indent')}# RNG global: la semilla del MOEA lo controla"

    seeded = _GENERATOR_ASSIGN.sub(collect, code)
    seeded = _INLINE_GENERATOR.sub("np.random.", seeded)

    for name in names:
        pattern, replace = _method_replacements(name)
        seeded = pattern.sub(replace, seeded)

    # `Generator.integers` es half-open como `np.random.randint`, pero
    # `endpoint=True` no existe en el RNG global.
    if re.search(r"\bintegers\s*\([^)]*endpoint\s*=", seeded):
        warnings.append(
            "el operador usa `endpoint=True`, que `np.random.randint` no "
            "admite: revisa ese punto a mano")
    # Metodos de `Generator` que no tienen equivalente directo.
    for method in set(re.findall(r"\brng\.(\w+)\s*\(", seeded)):
        if method not in ("integers", "choice", "shuffle", "permutation",
                          "random", "uniform", "normal", "randint",
                          "standard_normal", "beta", "integers_low"):
            warnings.append(
                f"`rng.{method}` no tiene equivalente en el RNG global: "
                "revisalo a mano")
    return seeded, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Siembra un operador (RNG global en vez de generador propio).")
    parser.add_argument("source", help="archivo del operador")
    parser.add_argument("target", nargs="?", help="archivo de salida")
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        print(f"ERROR: no existe {args.source!r}", file=sys.stderr)
        return 1
    with open(args.source, encoding="utf-8") as handle:
        code = handle.read()

    seeded, warnings = seed_operator(code)
    for warning in warnings:
        print(f"AVISO: {warning}", file=sys.stderr)

    # Verificacion: el operador sembrado tiene que seguir siendo valido.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from mohh.llamea.fitness import compile_operator, uses_unseeded_rng

    if uses_unseeded_rng(seeded):
        print("ERROR: el codigo sembrado sigue usando un generador sin "
              "semilla.", file=sys.stderr)
        return 1
    try:
        compile_operator(seeded)
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: el operador sembrado no es valido: {error}",
              file=sys.stderr)
        return 1

    if args.target:
        with open(args.target, "w", encoding="utf-8") as handle:
            handle.write(seeded if seeded.endswith("\n") else seeded + "\n")
        print(f"Operador sembrado escrito en {args.target}")
    else:
        print(seeded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
