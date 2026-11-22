"""Guardia de capas: evaluation no depende de llamea directamente."""

import ast
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "src" / "mohh" / "evaluation"


def _llamea_imports(tree):
    """(linea, sentencia) de cada import directo de mohh.llamea en el AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "mohh.llamea" or node.module.startswith("mohh.llamea."):
                yield node.lineno, f"from {node.module} import ..."
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mohh.llamea" or alias.name.startswith("mohh.llamea."):
                    yield node.lineno, f"import {alias.name}"


def test_evaluation_no_importa_llamea():
    offenders = []
    for path in EVALUATION_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, statement in _llamea_imports(tree):
            offenders.append(
                f"{path.relative_to(EVALUATION_DIR)}:{lineno}: {statement}")

    assert not offenders, (
        "evaluation importa llamea directamente; usar "
        "mohh.shared.bridge.load_llm_operator:\n" + "\n".join(offenders))


def test_guardia_detecta_import_a_nivel_de_modulo():
    tree = ast.parse(
        "import mohh.llamea.fitness\n"
        "from mohh.llamea.operator import LLMOperator\n")
    assert len(list(_llamea_imports(tree))) == 2
