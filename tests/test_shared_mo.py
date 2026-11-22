"""Contrato del protocolo MOEA compartido por GE y LLaMEA."""

import numpy as np

from mohh.shared import mo


def test_fitness_reexporta_el_protocolo_mo():
    from mohh.llamea import fitness

    names = (
        "init_params", "_call_with_timeout", "_params_size", "_load_mutation",
        "solve_front_snapshots", "_solve_front_namespace",
        "_solve_front_protocols", "_TimeoutError",
    )
    for name in names:
        assert getattr(fitness, name) is getattr(mo, name)


def test_script_expone_solve_front_snapshots():
    from scripts.compare_operators import solve_front_snapshots

    assert solve_front_snapshots is mo.solve_front_snapshots


def test_solve_front_snapshots_devuelve_los_tres_presupuestos(monkeypatch):
    from mohh.constants import EVAL_BUDGETS
    from mohh.generation.algorithm import multiobjective

    class _Result:
        def __init__(self, generation):
            self.F = np.array([[generation, generation],
                               [generation + 1, generation - 1]], dtype=float)

    class _Algorithm:
        def __init__(self):
            self.n_gen = 0
            self.callback = None

        def result(self):
            return _Result(self.n_gen)

    class _FakeSolver:
        def __init__(self, seed=None):
            self.seed = seed
            self.algorithm = _Algorithm()
            self.generations = None
            self.crossover = None
            self.mutation = None

        def load_instance(self, instance):
            self.instance = instance

        def load_operator(self, name, operator):
            self.loaded = (name, operator)

        def start_model(self, model_name=None):
            self.model_name = model_name

        def solve_instance(self):
            for generation in range(1, self.generations + 1):
                self.algorithm.n_gen = generation
                self.algorithm.callback.notify(self.algorithm)

    monkeypatch.setattr(multiobjective, "MOSolver", _FakeSolver)
    monkeypatch.setattr(mo, "_params_size", lambda: (105, 300))

    fronts = mo.solve_front_snapshots(
        object(), {"crossover": lambda x, y: x}, seed=7,
        model_name="MOEAD", mutation=False, instance_timeout=None,
        crossover=object(),
    )
    assert set(fronts) == set(EVAL_BUDGETS)
    for label, budget in EVAL_BUDGETS.items():
        assert fronts[label].shape == (2, 2)
        assert fronts[label][0, 0] == budget // 105


def test_init_params_no_cambia_valores(monkeypatch):
    from mohh.core.params import Params
    from mohh.generation.params import params_dict

    monkeypatch.setattr(Params, "_params", {})
    first = mo.init_params()
    for key, value in params_dict.items():
        assert first[key] == value

    snapshot = dict(first.get_params())
    second = mo.init_params()
    assert second is first
    assert second.get_params() == snapshot

    custom = mo.init_params(MO_GENERATIONS=7, RANDOM_SEED=123)
    assert custom["MO_GENERATIONS"] == 7
    assert custom["RANDOM_SEED"] == 123
    assert custom["MO_POPULATION_SIZE"] == params_dict["MO_POPULATION_SIZE"]
