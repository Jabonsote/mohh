
import copy
import time
import numpy as np
from tqdm import tqdm

# Classes
from mohh.generation.params import Params
from mohh.generation.saver import PopulationSaver, MyLogger

from mohh.generation.problem.instance import Instance
from mohh.generation.algorithm.multiobjective import MOSolver
from mohh.generation.representation.population import Population
from mohh.generation.representation.population import Individual

# Parallelization
from mohh.parallel import run_parallel

# Auxiliar functions
from mohh.core.utilities.print_utils import update_lines
from mohh.core.utilities.load_modules import find_module
from mohh.core.utilities.algorithm.MO import compute_nadir_point
from mohh.core.utilities.algorithm.MO import compute_hypervolume
from mohh.core.utilities.algorithm.MO import non_dominated_sorting_vectorized
from mohh.core.utilities.metrics import RunMetrics
from mohh.generation.utilities.algorithm.HH_auxiliars import compute_rank
from mohh.generation.utilities.algorithm.HH_auxiliars import test_individual


def _solve_individual_instance(task: dict) -> dict:
    """
    Worker del pool de procesos de la búsqueda: resuelve una instancia con el
    MOEA de un operador candidato. Conserva exactamente la lógica de
    evaluation_step; solo cambia el proceso en el que se ejecuta. El proceso
    principal aplica las mismas actualizaciones de estado con el resultado.
    """
    params_snapshot = task["params"]
    phenotype = task["phenotype"]
    solution_type = task["solution_type"]
    mo_cross_prob = task["mo_cross_prob"]
    seed = task["seed"]
    model_name = task["model_name"]
    instance = task["instance"]

    # Reconstruir el singleton de Params en este proceso (fork/spawn).
    Params().update_params(params_snapshot)

    # Semilla determinista para cada corrida independiente.
    np.random.seed(seed)

    solver = MOSolver(seed=seed)
    solver.load_operator("operator_template", "HH_Operator",
                         operator=phenotype,
                         solution_type=solution_type,
                         prob=mo_cross_prob)
    solver.load_operator("mutation", "NullMutation")

    # Problema / algoritmo / MOEA (idéntico al proceso principal).
    start = time.time()
    solver.load_instance(instance)
    problem_setup_s = time.time() - start

    start = time.time()
    solver.start_model(model_name=model_name)
    algorithm_setup_s = time.time() - start

    start = time.time()
    results = solver.solve_instance()
    solve_s = time.time() - start

    pareto_set, pareto_front = results[:2]
    weak_non_dominated, unique_solutions, strong_non_dominated = results[2:]

    return {
        "instance_name": instance.instance_name,
        "pareto_set": pareto_set,
        "pareto_front": pareto_front,
        "weak_non_dominated": weak_non_dominated,
        "unique_solutions": unique_solutions,
        "strong_non_dominated": strong_non_dominated,
        "problem_setup_s": problem_setup_s,
        "algorithm_setup_s": algorithm_setup_s,
        "solve_s": solve_s,
    }


class HyperHeuristic():

    def __init__(self) -> None:

        # Load General Parameters
        params = Params()

        # Parameters
        self.seed = params["RANDOM_SEED"]
        self.population_size = params["POPULATION_SIZE"]
        self.num_generations = params["GENERATIONS"]

        self.mo_model_name = params["MO_MODEL"]
        self.mo_cross_prob = params["MO_CROSSOVER_PROBABILITY"]
        self.mo_population_size = params["MO_POPULATION_SIZE"]

        # Genetic Operators
        self.selection = None
        self.crossover = None
        self.mutation = None

        # Population
        self.population = None
        self.num_offsprings = params["GENERATION_SIZE"]

        # Instances parameters
        self.problem_name = params["PROBLEM_NAME"]
        self.solution_type = params["SOLUTION_TYPE"]

        # Current generation counter (used by the metrics instrumentation).
        self.current_generation = 0

    def load_operator(self, operator_type: str, operator_name: str, **kwargs: dict):
        """
        Load a genetic operator to the algorithm.
        Choose between: selection, crossover, mutation,
        replacement and initialization.

        @param operator_type: Choose between selection, crossover,
        mutation, replacement and initialization
        @param operator_name: Operator name.
        """

        operator_type = operator_type.lower()

        module_name = f"mohh.generation.operators.{operator_type}"
        operator = find_module(module_name, operator_name, **kwargs)

        # Handle initialization
        if operator_type == "initialization" or operator_type == "initialization":
            self.population = Population(operator)
            return self

        # Dynamically set the attribute based on operator_type
        setattr(self, operator_type, operator)

        return self

    def load_instances(self, instance_paths: list) -> None:
        """
        Load instance paths, and create instance objects.

        @param instance_paths: List with instances file paths.
        """
        # Create an instance object per file.
        self.instances = []

        # Verify if MO population size is a list or integer
        if type(self.mo_population_size) == int:
            # Transform it into a list with repeated values
            self.mo_population_size = [self.mo_population_size]*len(instance_paths)

        for indx, instance_path in enumerate(instance_paths):
            # Initiate instance object
            instance = Instance(self.mo_population_size[indx], self.solution_type)

            # Load problem
            instance.load_problem(self.problem_name, instance_path)

            # Store it
            self.instances.append(instance)

    def evaluation_step(self):
        """
        Evaluates the individuals by solving the instances.

        Las corridas MOEA/D (candidato x instancia) son independientes, así
        que se resuelven en paralelo con un pool de procesos que usa todos
        los núcleos. La lógica de cada corrida es idéntica a la secuencial.
        """

        # Load logger
        logger = MyLogger().get_logger()

        params = Params()

        # List of non-evaluated individuals (mismo orden que la versión secuencial).
        individuals = list(self.population.non_evaluated())

        # Build the independent MOEA tasks (one per candidate x instance).
        tasks = [
            {
                "params": {k: v for k, v in params.get_params().items() if k != "BNF_GRAMMAR"},
                "phenotype": individual.phenotype,
                "solution_type": self.solution_type,
                "mo_cross_prob": self.mo_cross_prob,
                "seed": self.seed,
                "model_name": self.mo_model_name,
                "instance": instance,
            }
            for individual in individuals
            for instance in self.instances
        ]

        # Make a progress bar (avanza en vivo conforme cada individuo termina
        # todas sus instancias, igual que en la versión secuencial).
        pbar = tqdm(total=len(individuals), desc="Evaluating...",
                    bar_format="{l_bar}{bar:20}{r_bar}", leave=True)

        n_instances = len(self.instances)

        # Actualiza la barra desde el proceso principal por cada tarea que
        # completa: un individuo avanza cuando se resuelven todas sus instancias.
        def on_complete(task_index: int):
            if (task_index + 1) % n_instances == 0:
                pbar.update(1)

        results = run_parallel(_solve_individual_instance, tasks, on_complete=on_complete)
        results_iter = iter(results)

        # Collect the results and apply the same updates as before.
        for individual in individuals:

            for instance in self.instances:

                result = next(results_iter)

                metrics = RunMetrics()
                metrics.timing("moea_problem_setup", result["problem_setup_s"], "moea")
                metrics.timing("moea_algorithm_setup", result["algorithm_setup_s"], "moea")
                metrics.timing("moea_solve", result["solve_s"], "moea")

                # Extract results
                pareto_set = result["pareto_set"]
                pareto_front = result["pareto_front"]
                weak_non_dominated = result["weak_non_dominated"]
                unique_solutions = result["unique_solutions"]
                strong_non_dominated = result["strong_non_dominated"]

                metrics.record("moea_results", {
                    "generation": self.current_generation,
                    "individual": individual.name,
                    "instance": instance.instance_name,
                    "front_size": int(pareto_front.shape[0]) if hasattr(pareto_front, "shape") else len(pareto_front),
                    "pareto_set_size": int(len(pareto_set)),
                    "weak_non_dominated": int(weak_non_dominated),
                    "unique_solutions": int(unique_solutions),
                    "strong_non_dominated": int(strong_non_dominated),
                })
                metrics.increment("moea_runs", "count", 1)

                # Store results in the individual object
                instance_name = instance.instance_name
                individual.pareto_sets[instance_name] = pareto_set
                individual.pareto_fronts[instance_name] = pareto_front
                individual.unique_solutions[instance_name] = unique_solutions
                individual.weak_non_dominated[instance_name] = weak_non_dominated
                individual.strong_non_dominated[instance_name] = strong_non_dominated

                # Add front to the consolidated
                instance.non_dominated_front = \
                    instance.non_dominated_front.union([tuple(element) for element in pareto_front.tolist()])

        # Close progress bar
        pbar.close()

        # Extract elapsed time
        total_time = pbar.format_dict["elapsed"]

        # Add it to the log file
        logger.info(f"Evaluation took: {total_time} seconds")

    def compute_metrics(self):
        """
        Compute an indicator metric and rank the individuals.
        """

        # Load logger
        logger = MyLogger().get_logger()

        # Now that every individual has solved the instances.
        fitness_values = [[0]*len(self.instances) for _ in range(len(self.population.individuals))]

        # Make memory for instance times
        instance_sorting_time = [0.0]*len(self.instances)
        instance_hv_time = [0.0]*len(self.instances)

        # Display times
        sorting_text = ", ".join([f"{i:05.1f} min" for i in instance_sorting_time])
        hv_text = ", ".join([f"{i:05.1f} min" for i in instance_hv_time])
        print("\nSorting: ", sorting_text)
        print("Hypervolumes: ", hv_text)

        # Make a progress bar
        pbar = tqdm(self.instances, desc="Computing metrics...",
                     bar_format="{l_bar}{bar:20}{r_bar}", leave=True)

        # For each instance...
        for i, instance in enumerate(pbar):

            # Instance name
            instance_name = instance.instance_name

            metrics = RunMetrics()
            generation = self.current_generation

            # Measure non dominated sorting time
            sorting_time_init = time.time()
            metrics.start("metrics_nds", category="metrics")

            # Compute non-dominated-sorting on each instance non-dominated-front.
            instance.fronts = non_dominated_sorting_vectorized(instance.non_dominated_front)

            metrics.stop("metrics_nds")

            # Measure non dominated sorting end time [min]
            sorting_time = (time.time() - sorting_time_init) / 60

            # Set non-dominated front as the first front.
            instance.non_dominated_front = set(instance.fronts[0])

            # Compute Nadir Points
            instance.nadir_point = compute_nadir_point(instance.non_dominated_front) # TODO: Multiply by 1.1

            # Log whether the nadir changed (triggers HV recomputations).
            nadir_changed = instance.previous_nadir_point != instance.nadir_point
            metrics.record("nadir", {
                "generation": generation,
                "instance": instance_name,
                "nadir_point": list(instance.nadir_point),
                "changed": bool(nadir_changed),
                "consolidated_size": int(len(instance.non_dominated_front)),
            })

            # Measure Hypervolume computation time.
            hv_time_init = time.time()

            # If nadir point changed...
            if nadir_changed:

                # Re-compute the hypervolume of the actual population
                metrics.start("metrics_hv_recompute", category="metrics")
                for individual in self.population.evaluated():
                    front = individual.pareto_fronts[instance_name]
                    individual.hypervolumes[instance_name] = compute_hypervolume(instance.nadir_point, front)
                    metrics.record("hypervolumes", {
                        "generation": generation,
                        "instance": instance_name,
                        "individual": individual.name,
                        "hypervolume": float(individual.hypervolumes[instance_name]),
                        "front_size": int(front.shape[0]),
                        "recomputed": True,
                    })
                metrics.stop("metrics_hv_recompute")

            # Compute the hypervolume to all the new individuals
            for individual in self.population.non_evaluated():
                front = individual.pareto_fronts[instance_name]

                # Compute hypervolume to the new individual.
                individual.hypervolumes[instance_name] = compute_hypervolume(instance.nadir_point, front)
                metrics.record("hypervolumes", {
                    "generation": generation,
                    "instance": instance_name,
                    "individual": individual.name,
                    "hypervolume": float(individual.hypervolumes[instance_name]),
                    "front_size": int(front.shape[0]),
                    "recomputed": False,
                })

            # Get total time Hypervolume took [min]
            hv_time = (time.time() - hv_time_init) / 60

            # Save the nadir_points to verify changes.
            instance.previous_nadir_point = instance.nadir_point

            # Assign a rank for each individual based on their hypervolumes.
            ranks = compute_rank(self.population.get_hypervolumes(instance_name))

            # Store the rank to the individuals
            for j, individual in enumerate(self.population.individuals):
                individual.fitness_values[instance_name] = ranks[j]
                fitness_values[j][i] = ranks[j]

                metrics.record("ranks", {
                    "generation": generation,
                    "instance": instance_name,
                    "individual": individual.name,
                    "rank": ranks[j],
                    "hypervolume": float(individual.hypervolumes[instance_name]),
                })

            # Store computation times
            instance_sorting_time[i] = sorting_time
            instance_hv_time[i] = hv_time

            # Make new time strings
            sorting_text = ", ".join([f"{i:05.1f} min" for i in instance_sorting_time])
            hv_text = ", ".join([f"{i:05.1f} min" for i in instance_hv_time])

            # Update CLI prints
            update_lines(["Sorting: " + sorting_text,
                        "Hypervolumes: " + hv_text])

        # Once an individual computes its hypervolumes across all instances,
        # mark it as evaluated.
        for individual in self.population.non_evaluated():
            # Mark the individual as evaluated.
            individual.evaluated = True

        # Average the fitness values across all instances.
        scores = [sum(fitness_values[i]) / len(self.instances) for i in range(len(self.population.individuals))]

        # Log the final per-individual fitness (average rank across instances).
        for individual, score in zip(self.population.individuals, scores):
            RunMetrics().record("fitness", {
                "generation": self.current_generation,
                "individual": individual.name,
                "fitness": float(score),
            })

        # Close progress bar
        pbar.close()

        # Extract elapsed time
        total_time = pbar.format_dict["elapsed"]

        # Add it to the log file
        logger.info(f"Metrics took: {total_time} seconds")
        logger.info(f"Sorting: {sorting_text}")
        logger.info(f"Hypervolumes: {hv_text}")

        return scores

    def evolutionary_step(self, scores: list):
        """
        Compute an evolutionary step.
        Selection, Crossover, Mutation.
        """

        # Extract current genomes from population
        genomes = copy.deepcopy(self.population.get_genomes())

        # Extract current phenotypes from population
        phenotypes = copy.deepcopy(self.population.get_phenotypes())

        # Extract the genome usage of each individual
        genome_usage = copy.deepcopy(self.population.get_genome_usage())

        # Create variable for new individuals
        offspring = []

        # Metrics for the isolated grammatical-evolution cost.
        metrics = RunMetrics()
        metrics.increment("ge", "attempted_parent_pairs", 0)
        metrics.start("ge_operators_total", category=None)

        # Apply Genetic Operators
        while len(offspring) < self.num_offsprings:

            # Work with genomes only
            metrics.start("ge_selection", category="ge")
            parents, indices = self.selection(genomes, scores)
            metrics.stop("ge_selection")
            metrics.increment("ge", "attempted_parent_pairs", 1)
            relevant_genomes = [genome_usage[i] for i in indices]

            metrics.start("ge_crossover", category="ge")
            children = self.crossover(parents, relevant_genomes)
            metrics.stop("ge_crossover")

            metrics.start("ge_mutation", category="ge")
            children = self.mutation(children)
            metrics.stop("ge_mutation")

            for child in children:
                # Create new individual (includes grammar mapping cost).
                metrics.start("ge_mapping", category="ge")
                ind = Individual(child)
                metrics.stop("ge_mapping")

                # Verify if individual is invalid; or repeated
                if ind.invalid or ind.phenotype in phenotypes:
                    metrics.increment("ge", "rejected_invalid" if ind.invalid else "rejected_repeated", 1)
                    continue

                # Verify if it outputs valid values.
                metrics.start("ge_validation", category="ge")
                tested = test_individual(ind.phenotype, self.solution_type)
                metrics.stop("ge_validation")

                if tested == False:
                    metrics.increment("ge", "rejected_invalid_output", 1)
                    continue

                # Offspring is valid and outputs admisible results.
                offspring.append(ind)
                metrics.increment("ge", "offspring_created", 1)

                # Hard constraint, don't exceed num_offsprings.
                if len(offspring) == self.num_offsprings:
                    break

        metrics.stop("ge_operators_total")
        metrics.record("ge", {
            "generation": self.current_generation,
            "num_offsprings": self.num_offsprings,
            "offspring_created": len(offspring),
        })

        # Name individuals.
        for child in offspring:

            # Set child as offspring.
            child.is_offspring = True

            # Set individual name.
            child.name = f"Individual_{self.population.num_individuals_created+1:06d}"

            # Increase counter.
            self.population.num_individuals_created += 1

        # Append the new generation to the current one.
        self.population.individuals.extend(offspring)

    def replace_step(self, scores: list) -> list:
        """
        Apply replacement with elitism.
        """

        # Time the isolated replacement/elitism cost.
        metrics = RunMetrics()
        metrics.start("ge_replacement", category="ge")

        # Get the current population
        population = self.population.get_population()

        # Get the current offspring
        offspring = self.population.get_offspring()

        # Elitism size
        elitism_size = len(population) - len(offspring)

        # Assuming the order is conserved...
        paired_population = list(zip(population, scores[:len(population)]))

        paired_offspring = list(zip(offspring, scores[len(population):]))

        # Sort population
        sorted_population = sorted(paired_population, key=lambda x: x[1])

        # Keep the E individuals with best score
        preserved_population = sorted_population[:elitism_size]

        # Merge preserved population with offsprings
        preserved_population.extend(paired_offspring)

        # Sort final population
        final_pairs = sorted(preserved_population, key=lambda x: x[1])

        # Extract population and scores
        new_population = [ind for ind, _ in final_pairs] 

        scores = [score for _, score in final_pairs]

        # Update population
        self.population.individuals = new_population

        # Set individuals as not offsprings
        for individual in self.population.individuals:
            individual.is_offspring = False

        metrics.stop("ge_replacement")

        return scores

    def _metrics_save_consolidated(self, generation: int):
        """
        Persist the consolidated (non-dominated union) fronts + nadir points
        of every instance into run_metrics/. Timed as I/O, so it never
        inflates the "pure computation" cost.
        """
        metrics = RunMetrics()
        for instance in self.instances:
            metrics.save_json(
                f"fronts/consolidated/generation_{generation:04d}/instance_{instance.instance_name}.json",
                {
                    "generation": generation,
                    "instance": instance.instance_name,
                    "nadir_point": list(instance.nadir_point),
                    "size": int(len(instance.non_dominated_front)),
                    "front": sorted(instance.non_dominated_front),
                },
            )

    def run(self):
        """
        Execute the generation hyperheuristic.
        """

        # Load population saver
        population_saver = PopulationSaver()

        # Load logger
        logger_object = MyLogger()
        logger = logger_object.get_logger()

        logger.info(f"MO-Solver: {self.mo_model_name}")

        # Metrics collector (no-op unless LOG_METRICS was enabled at setup).
        metrics = RunMetrics()
        metrics.set_info("phase", "generation")
        metrics.set_info("mo_model", self.mo_model_name)
        metrics.set_info("instances", [i.instance_name for i in self.instances])
        metrics.record("run", {"event": "start", "model": self.mo_model_name,
                               "num_instances": len(self.instances)})

        # Show a summary of the run's key parameters
        params = Params()
        mo_pop_sizes = self.mo_population_size if isinstance(self.mo_population_size, list) else [self.mo_population_size]
        mo_pop_display = mo_pop_sizes[0] if len(set(mo_pop_sizes)) == 1 else mo_pop_sizes

        # Keep every effective parameter in the metrics metadata.
        metrics.set_info("params", params.get_params())

        print("\n" + "-" * 40)
        print("Starting Generation")
        print("-" * 40)
        print(f"MOEA model:         {self.mo_model_name}")
        print(f"Population size:    {self.population_size}")
        print(f"Generations:        {self.num_generations}")
        print(f"Elite size:         {params['ELITE_SIZE']}")
        print(f"MO population size: {mo_pop_display}")
        print(f"MO generations:     {params['MO_GENERATIONS']}")
        print("-" * 40)

        # Start timer
        start_time = time.time()

        print("\n" + "-" * 40)
        print("Generation 0.")
        print("-" * 40)
        print("Start initial population.")
        logger.info(f"Generation 0/{self.num_generations}")

        # Create initial population.
        self.current_generation = 0
        metrics.start("ge_initialization", category="ge")
        self.population.initialize_population(self.population_size)
        metrics.stop("ge_initialization")
        metrics.record("ge_initialization", {
            "generation": 0,
            "population_size": self.population_size,
            "individuals_created": self.population.num_individuals_created,
        })

        print("Evaluating initial population...")

        # Evaluate the initial population and compute metrics.
        metrics.start("evaluation_step", category=None)
        self.evaluation_step()
        metrics.stop("evaluation_step")

        metrics.start("compute_metrics_step", category=None)
        scores = self.compute_metrics()
        metrics.stop("compute_metrics_step")

        # Save population and instance data
        metrics.start("io_saver", category="io")
        population_saver.save_population(self.population, generation=0)
        population_saver.save_consolidated_fronts(self.instances, generation=0)
        metrics.stop("io_saver")

        self._metrics_save_consolidated(0)
        metrics.dump("generation_0000", {"generation": 0})

        for gen in range(1, self.num_generations + 1):

            print("\n" + "-" * 40)
            print(f"Generation {gen}/{self.num_generations}")
            print("-" * 40)
            logger.info(f"Generation {gen}/{self.num_generations}")

            self.current_generation = gen
            metrics.start("generation_compute", category=None)

            # Generate offsprings
            self.evolutionary_step(scores)

            # Note: Evolutionary step has increased the size from N to N + n.
            # Evaluation will skip the already evaluated N individuals,
            # and just evaluate the rest n.
            # If a nadir point is modified, compute metrics will recompute HVs
            # for all the individuals (N+n), else just n new individuals.

            # Evaluate offsprings
            ## Evaluation steps will test all offspring, and the consolidated will
            ## not fulfill the non-dominated condition after it, since new solutions are added.
            metrics.start("evaluation_step", category=None)
            self.evaluation_step()
            metrics.stop("evaluation_step")

            metrics.start("compute_metrics_step", category=None)
            scores = self.compute_metrics()
            metrics.stop("compute_metrics_step")

            metrics.stop("generation_compute")

            # Save offspring in disk (I/O is logged separately from the compute cost)
            metrics.start("io_saver", category="io")
            population_saver.save_population(self.population, generation=gen-1)

            # Save consolidated and front (Consolidated of current pop + offspring)
            population_saver.save_consolidated_fronts(self.instances, generation=gen)
            metrics.stop("io_saver")

            # Replace current population with offsprings.
            scores = self.replace_step(scores)

            # Note: Replace deletes the worst n individuals in the population,
            # therefore, population has now a size of N.

            # Save new population in disk
            metrics.start("io_saver", category="io")
            population_saver.save_population(self.population, generation=gen)
            metrics.stop("io_saver")

            self._metrics_save_consolidated(gen)
            metrics.set_info("best_individual", self.population.individuals[0].phenotype)
            metrics.dump(f"generation_{gen:04d}", {"generation": gen})

            print(f"\nBest Individual: {self.population.individuals[0].phenotype}")
            logger.info(f"Best Individual: {self.population.individuals[0].phenotype}")

        # Print total time
        end = time.time()
        seconds = round(end - start_time, 2)
        minutes = round(seconds / 60, 2)
        hours = round(minutes / 60, 2)
        print(f"\n\nTotal time: {seconds} seconds.")
        print(f"Minutes: {minutes}")
        print(f"Hours: {hours}")
        logger.info(f"Total time in seconds: {seconds}")
        logger.info(f"Total time in minutes: {minutes}")
        logger.info(f"Total time in hours: {hours}")

        # Finalize the run metrics (summary.json with the isolated
        # grammatical-evolution cost, MOEA cost and I/O, separated).
        metrics.record("run", {"event": "end", "total_seconds": seconds})
        metrics.write_summary()
        metrics.close()

        # Reset singletons
        population_saver.reset_instance()
        logger_object.reset_instance()
