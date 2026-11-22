
import os
import random

from datetime import datetime
from socket import gethostname

import numpy as np

from mohh.constants import (
    EVAL_BUDGETS,
    MO_GENERATIONS,
    MO_POPULATION_SIZE,
    N_EXPERIMENTS,
)
from mohh.core.params import Params
from mohh.core.utilities.paths import project_root

hostname = gethostname().split(".")
machine_name = hostname[0]

mo_params = {

    # Solvers to benchmark
    'SOLVERS': ["MOEAD", "NSGAII", "SMSEMOA"],

    # Number of repeated experiments per instance/combination
    'N_EXPERIMENTS': N_EXPERIMENTS,

    # Evaluation budgets to snapshot during a run (label: total evaluations).
    # From mohh.constants: 105 (MO_POPULATION_SIZE) * 100/300/500.
    'EVAL_BUDGETS': EVAL_BUDGETS,

    # Evolutionary parameters
    'MO_POPULATION_SIZE': MO_POPULATION_SIZE,
    'MO_GENERATIONS': MO_GENERATIONS,


    # ----------------
    # MOEAD Parameters
    'NUM_NEGIHBORS': 10,
    'PROB_NEIGHBORS': 0.9,

    # Crossover
    'MOEAD_CROSSOVER': None, # SBX
    'MOEAD_CROSSOVER_PROBABILITY': 1.0,

    # Mutation
    'MOEAD_MUTATION': "PM_Mutation", # PM
    'MOEAD_MUTATION_PROBABILITY': 0.05,
    'MOEAD_MUTATION_BOOL': False,


    # ----------------
    # NSGAII Parameters
    
    # Crossover
    'NSGA_CROSSOVER': None, # SBX
    'NSGA_CROSSOVER_PROBABILITY': 1.0,

    # Mutation
    'NSGA_MUTATION': "PM_Mutation", # PM
    'NSGA_MUTATION_PROBABILITY': 0.05,
    'NSGA_MUTATION_BOOL': False,

    # ----------------
    # SMS-EMOA Parameters

    # Crossover
    'SMS_CROSSOVER': None,
    'SMS_CROSSOVER_PROBABILITY': 1.0,

    # Mutation
    'SMS_MUTATION': "PM_Mutation",
    'SMS_MUTATION_PROBABILITY': 0.05,
    'SMS_MUTATION_BOOL': False

}

problem_params = {
    # ------------
    # Problem class
    'PROBLEM_NAME': "QAP",
    'SOLUTION_TYPE': "Real",
    # Define if maximize or minimize.
    # True for maximize, False for minimize
    'OPTIMIZATION_KIND': False,

    # ------------
    # Select problem dataset
    'DATASET': "mqap",
    # Ruta explícita al dataset (con train/ y test/). Si es None se usa
    # <project_root>/datasets/<DATASET>, que mantiene el comportamiento antiguo.
    'DATASET_PATH': None,
    # Carpeta de salida; None = <experimento>/evaluation_results.
    'EVALUATION_DIR': None,
    'DATASET_TRAIN': None,
    'DATASET_TEST': "test",
}
  
grammar_params = {
    # ------------
    # GRAMMAR
    'GRAMMAR_FILE': "naturals.bnf",

    # Set the number of depths permutations are calculated for
    # (starting from the minimum path of the grammar).
    # Mainly for use with the grammar analyser script.
    'PERMUTATION_RAMPS': 5,
}

misc_params = {
    # ------------
    # Set machine name
    'MACHINE': machine_name,

    "SAVE": True,

    # ------------
    # RUN BOOKKEEPING
    'EXPERIMENT_NAME': None,
    # Semilla maestra. None = una sola vez por reloj (no reproducible); para
    # un experimento publicable fijarla (CLI: --seed).
    'RANDOM_SEED': None,
    # Reutilizar las poblaciones iniciales y las semillas de una corrida
    # anterior (ruta a su carpeta `initial_solutions/`). Vacio = generar
    # nuevas. Es lo que hace reproducible la corrida sin repetir el MOEA.
    'INITIAL_POPULATION_PATH': "",
    # Re-evaluar instancias ya marcadas como terminadas en `.done/`.
    'FORCE_RERUN': False,
    # Segundos maximos por corrida del MOEA. None = sin limite (legado).
    'MOEA_TIMEOUT': 3600,

    # ------------
    # METRICS / LOGGING
    # Save detailed logs of the whole evaluation process in
    # <evaluation_results>/run_metrics: MOEA computation times per
    # experiment, front sizes, boundaries, normalization, HV per
    # experiment/label and the consolidated fronts.
    'LOG_METRICS': True,
    'METRICS_DIRNAME': "run_metrics",
    'METRICS_PATH': None,
}

params_dict = dict(mo_params, **problem_params, **grammar_params, **misc_params)

def set_params(overrides: dict = None):

    # Initialize singleton object with params
    params = Params()
    params.update_params(params_dict)

    if overrides:
        params.update_params(overrides)

    # Load Grammar & Saver
    from mohh.core.representation import grammar

    # Get actual time
    start = datetime.now()

    # Set random seed
    if params["RANDOM_SEED"] is None:
        params["RANDOM_SEED"] = int(start.microsecond)

    random.seed(params["RANDOM_SEED"])
    np.random.seed(params["RANDOM_SEED"])

    # Generate a timestamp to name folder
    hm = "%02d%02d" % (start.hour, start.minute)
    params["TIME_STAMP"] = "_".join([gethostname(),
                                     str(start.year),
                                     str(start.month),
                                     str(start.day), hm,
                                     str(params["RANDOM_SEED"])])

    print("\nStart:\t", start, "\n")

    # Parse grammar file and set grammar class
    params["BNF_GRAMMAR"] = grammar.Grammar(
        os.path.join(project_root(), "grammars", params["GRAMMAR_FILE"])
    )
