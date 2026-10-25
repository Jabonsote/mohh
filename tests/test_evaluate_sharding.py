"""Selección de instancias de test para mohh-evaluate."""

from argparse import Namespace

from mohh.cli import _resolve_test_instances


def _args(dataset_path, instances=None, index=0, count=1):
    return Namespace(dataset_path=str(dataset_path), dataset=None,
                     instances=instances, shard_index=index,
                     shard_count=count)


def test_usa_todas_las_instancias_de_test(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for nombre in ("KC10-2fl-2rl", "KC20-2fl-2un", "KC30-3fl-2rl"):
        (test_dir / f"{nombre}.txt").write_text("x")

    rutas = _resolve_test_instances(_args(tmp_path))

    assert [p.split("/")[-1] for p in rutas] == [
        "KC10-2fl-2rl.txt", "KC20-2fl-2un.txt", "KC30-3fl-2rl.txt"]


def test_shard_reparte_sin_solapar(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for i in range(5):
        (test_dir / f"KC{i}-2fl-2rl.txt").write_text("x")

    vistos = []
    for index in range(3):
        parcial = _resolve_test_instances(_args(tmp_path, index=index, count=3))
        vistos.extend(p.split("/")[-1] for p in parcial)

    assert len(vistos) == 5
    assert len(set(vistos)) == 5


def test_instancias_concretas(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for nombre in ("KC10-2fl-2rl", "KC20-2fl-2un"):
        (test_dir / f"{nombre}.txt").write_text("x")

    rutas = _resolve_test_instances(_args(tmp_path, ["KC20-2fl-2un"]))

    assert [p.split("/")[-1] for p in rutas] == ["KC20-2fl-2un.txt"]


def test_extrae_operadores_de_una_generacion(tmp_path):
    """El modo intercalado lee population.json + general_info.json."""
    import json
    from mohh.run_evaluation import extract_generation_operators

    run = tmp_path / "MOEAD_test" / "generation_results"
    gen = run / "generations" / "generation_0002"
    gen.mkdir(parents=True)
    (run / "parameters.txt").write_text("ELITE_SIZE: 3\n")
    (gen / "population.json").write_text(json.dumps(["A", "B", "C"]))

    for nombre, fenotipo in (("A", "reverse(x)"),
                             ("B", "order_based(x,y)"),
                             ("C", "move_right(x)")):
        individual = run / "individuals" / nombre
        individual.mkdir(parents=True)
        (individual / "general_info.json").write_text(
            json.dumps({"phenotype": fenotipo}))

    operadores = extract_generation_operators(
        str(tmp_path), ["MOEAD"], generation=2)

    assert operadores == ["reverse(x)", "order_based(x,y)", "move_right(x)"]
