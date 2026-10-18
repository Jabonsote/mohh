
import tempfile

FALLBACK_SCORE = 0.0


def build_evaluator(train_instances, mo_generations, mo_population_size, seed=42):
    from mohh.generation.problem.instance import Instance
    from mohh.generation.algorithm.multiobjective import MOSolver
    from mohh.core.operators.operator_template import HH_Operator
    from mohh.core.utilities.algorithm.HH_auxiliars import test_individual
    from mohh.core.utilities.algorithm.MO import compute_hypervolume
    from mohh.core.params import Params

    params = Params()
    params.update_params({
        "RANDOM_SEED": seed,
        "MO_GENERATIONS": mo_generations,
        "MO_POPULATION_SIZE": mo_population_size,
        "MO_CROSSOVER_PROBABILITY": 1.0,
        "NUM_NEGIHBORS": 10,
        "PROB_NEIGHBORS": 0.9,
        "FILE_PATH_INITIAL_SOLUTIONS": tempfile.mkdtemp(prefix="llamea_init_"),
    })

    instances = [Instance(mo_population_size, "Natural") for _ in train_instances]
    for inst, path in zip(instances, train_instances):
        inst.load_problem("QAP", path)

    def evaluate(solution, logger=None):
        code = solution.code

        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            solution.set_scores(FALLBACK_SCORE, f"Code did not execute: {e}", e)
            return solution

        if "crossover" not in namespace or not callable(namespace["crossover"]):
            msg = "Code must define a callable 'crossover(x, y)' function."
            solution.set_scores(FALLBACK_SCORE, msg, ValueError(msg))
            return solution

        try:
            valid = test_individual("crossover(x, y)", "Natural")
        except Exception as e:
            solution.set_scores(FALLBACK_SCORE, f"Invalid operator: {e}", e)
            return solution

        if not valid:
            msg = "crossover must return a valid permutation of the same length as x."
            solution.set_scores(FALLBACK_SCORE, msg, ValueError(msg))
            return solution

        per_instance = []
        for inst in instances:
            solver = MOSolver(seed=seed)
            solver.load_operator(
                "operator_template", "HH_Operator",
                operator="crossover(x, y)",
                solution_type="Natural",
                prob=1.0,
                eval_namespace=namespace,
            )
            solver.load_operator("mutation", "NullMutation")
            solver.load_instance(inst)
            solver.start_model(model_name="MOEAD")
            _, pareto_front, *_ = solver.solve_instance()

            ref_point = pareto_front.max(axis=0) * 1.1
            ref_point[ref_point == 0] = 1.0
            per_instance.append(float(compute_hypervolume(ref_point, pareto_front)))

        score = sum(per_instance) / len(per_instance)
        feedback = (
            f"Mean hypervolume over {len(instances)} train instances: {score:.4f}. "
            f"Per-instance HV: {[round(hv, 4) for hv in per_instance]}. "
            "Improve the crossover operator to maximize hypervolume."
        )
        solution.set_scores(score, feedback)
        return solution

    return evaluate
