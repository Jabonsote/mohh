"""Regresión: best_so_far debe ser el mejor global, no el de la última generación."""
import os


class _FakeIndividual:
    def __init__(self, generation, fitness, name, code):
        self.generation = generation
        self.fitness = fitness
        self.name = name
        self.description = name
        self.error = None
        self.code = code
        self.configspace = None

    def to_dict(self):
        return {
            "generation": self.generation,
            "fitness": self.fitness,
            "name": self.name,
            "description": self.description,
            "error": self.error,
            "code": self.code,
            "id": f"{self.generation}-{self.name}",
        }


def test_checkpoint_keeps_global_best(tmp_path):
    from mohh.llamea.reporting import ResultsExperimentLogger

    ResultsExperimentLogger.results_dir = str(tmp_path / "llamea_logs")
    ResultsExperimentLogger.verbose = False
    ResultsExperimentLogger.seed = 7
    ResultsExperimentLogger.solver = "MOEAD"
    ResultsExperimentLogger.mo_generations = 2
    ResultsExperimentLogger.mo_population_size = 6
    ResultsExperimentLogger.mutation = False

    logger = ResultsExperimentLogger("demo")
    seen = []
    logger._write_checkpoint = lambda best, record: seen.append(
        (best.code, record["best_fitness"]))

    good = _FakeIndividual(1, 0.8, "good", "GOOD")
    worse = _FakeIndividual(2, 0.2, "worse", "WORSE")

    logger.log_population([good])
    logger.log_population([worse])

    assert seen[-1][0] == "GOOD"
    assert seen[-1][1] == 0.8
    checkpoints = [record for _, record in seen]
    assert checkpoints[-1] == 0.8
    assert os.path.isdir(logger.dirname)
