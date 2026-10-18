
import argparse
import os
import random

import numpy as np

from mohh.core.utilities.instance_utils import instance_paths
from mohh.llamea.fitness import build_evaluator
from mohh.llamea.prompts import (
    ROLE_PROMPT,
    TASK_PROMPT,
    EXAMPLE_PROMPT,
    OUTPUT_FORMAT_PROMPT,
)
from mohh.llamea.vllm import VLLM_LLM, ensure_model


def run_llamea(args):
    ensure_model(args.model)

    train_paths = instance_paths(train=True)
    if args.train_instances != "all":
        rng = random.Random(args.seed)
        train_paths = rng.sample(train_paths, min(args.train_instances, len(train_paths)))

    fitness = build_evaluator(
        train_paths,
        mo_generations=args.mo_generations,
        mo_population_size=args.mo_population_size,
        seed=args.seed,
    )

    llm = VLLM_LLM(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )

    from llamea import LLaMEA

    llamea = LLaMEA(
        f=fitness,
        llm=llm,
        n_parents=args.n_parents,
        n_offspring=args.n_offspring,
        role_prompt=ROLE_PROMPT,
        task_prompt=TASK_PROMPT,
        example_prompt=EXAMPLE_PROMPT,
        output_format_prompt=OUTPUT_FORMAT_PROMPT,
        budget=args.budget,
        elitism=not args.comma,
        minimization=False,
        max_workers=args.max_workers,
        experiment_name=args.experiment_name,
    )

    best = llamea.run()
    print(f"Best fitness: {best.fitness}")
    print(f"Best code:\n{best.code}")

    out_dir = os.path.join("results", args.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "llamea_best_operator.py"), "w") as f:
        f.write(best.code + "\n")
    with open(os.path.join(out_dir, "llamea_best_info.txt"), "w") as f:
        f.write(f"fitness: {best.fitness}\ndescription: {best.description}\n")
    print(f"Saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run LLaMEA search for mQAP crossover operators.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--n-parents", type=int, default=1)
    parser.add_argument("--n-offspring", type=int, default=1)
    parser.add_argument("--comma", action="store_true",
                        help="Use (1,1) comma strategy instead of (1+1) elitism.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--train-instances", default="8",
                        help="Number of train instances to evaluate on, or 'all'.")
    parser.add_argument("--mo-generations", type=int, default=30)
    parser.add_argument("--mo-population-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment-name", default="LLaMEA_Experiment")
    args = parser.parse_args()

    if args.train_instances != "all":
        args.train_instances = int(args.train_instances)

    run_llamea(args)


if __name__ == "__main__":
    main()
