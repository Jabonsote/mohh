"""CLI unica para los scripts de post-analisis."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys

from mohh.postanalysis import COMMANDS


def _acepta_argv(func) -> bool:
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        p.kind is inspect.Parameter.VAR_POSITIONAL
        or p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for p in params
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mohh-post")
    sub = parser.add_subparsers(dest="command")
    for nombre in COMMANDS:
        sub.add_parser(nombre, add_help=False)
    args, rest = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    modulo, funcion = COMMANDS[args.command]
    entry = getattr(importlib.import_module(f"mohh.postanalysis.{modulo}"), funcion)
    if _acepta_argv(entry):
        return entry(rest)

    # Scripts que leen `sys.argv` directamente (analyze_operators).
    anterior = sys.argv
    sys.argv = [anterior[0], *rest]
    try:
        return entry()
    finally:
        sys.argv = anterior


if __name__ == "__main__":
    main()
