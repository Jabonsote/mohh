
from mohh.evaluation.params import Params
from mohh.evaluation.experiment import run_experiments
from mohh.evaluation.make_plots import make_figures

def execute_experiments(experiment_path, results_paths, operators,
                        overrides: dict = None, instances=None,
                        figures: bool = True):

    # First run the experiments and generate data
    evaluation_path = run_experiments(experiment_path, results_paths, operators,
                                      overrides, instances=instances)

    # Make figures in the results_path
    if figures:
        make_figures(evaluation_path)

    # Reset params
    params = Params()
    params.reset_instance()

    # Reset the metrics collector for a potential next phase
    from mohh.core.utilities.metrics import RunMetrics
    RunMetrics.reset_instance()

