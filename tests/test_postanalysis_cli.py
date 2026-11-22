import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from mohh.postanalysis import COMMANDS
from mohh.postanalysis import cli

ROOT = Path(__file__).resolve().parents[1]

ESPERADOS = {
    "gen": "analyze_gen",
    "operators": "analyze_operators",
    "search": "analyze_search",
    "ast-diversity": "ast_diversity",
    "eval31": "evaluate_31",
    "generations": "evaluate_generations",
    "figures": "export_figure_values",
    "reference-sets": "generate_reference_sets_pop",
    "generation-comparison": "generation_comparison",
    "hv-scale": "hv_scale_from_evaluation",
    "indicators": "indicator_quality",
    "reference-report": "reference_sets_report",
}


def test_commands_todas_las_claves():
    assert set(COMMANDS) == set(ESPERADOS)
    for nombre, modulo in ESPERADOS.items():
        assert COMMANDS[nombre] == (modulo, "main")


def test_cli_despacha_y_pasa_argumentos(monkeypatch):
    vistos = []

    def falso_main(rest):
        vistos.append(rest)
        return 0

    monkeypatch.setattr(
        cli.importlib, "import_module",
        lambda name: SimpleNamespace(main=falso_main),
    )
    assert cli.main(["gen", "--experiment", "x", "--top", "5"]) == 0
    assert vistos == [["--experiment", "x", "--top", "5"]]


def test_cli_selecciona_modulo_correcto(monkeypatch):
    vistos = []
    monkeypatch.setattr(
        cli.importlib, "import_module",
        lambda name: vistos.append(name) or SimpleNamespace(main=lambda rest: None),
    )
    cli.main(["hv-scale", "--foo"])
    assert vistos == ["mohh.postanalysis.hv_scale_from_evaluation"]


@pytest.mark.parametrize("modulo", sorted(set(ESPERADOS.values())))
def test_wrapper_raiz_expone_main(modulo):
    ruta = ROOT / f"{modulo}.py"
    spec = importlib.util.spec_from_file_location(f"_wrapper_{modulo}", ruta)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    esperado = importlib.import_module(f"mohh.postanalysis.{modulo}").main
    assert mod.main is esperado
