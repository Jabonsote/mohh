# Referencias

Bibliografía usada para diseñar los prompts. Los PDFs están en
[`pdfs/`](pdfs/) (cuando son de acceso abierto).

## Papers principales

<a id="llamea"></a>
**[LLaMEA]** van Stein, N., & Bäck, T. (2024). *LLaMEA: A Large Language Model
Evolutionary Algorithm for Automatically Generating Metaheuristics.* IEEE
Transactions on Evolutionary Computation.
- arXiv: https://arxiv.org/abs/2405.20132
- PDF local: [`pdfs/llamea-2024-arxiv.pdf`](pdfs/llamea-2024-arxiv.pdf)
- Aporte al prompt: estructura `role + task + example + output format`, tarea
  precisa con contrato de código, ejemplo simple para no sesgar, y prompt de
  refinamiento "refine or redesign" con feedback (código, score, error).

<a id="funsearch"></a>
**[FunSearch]** Romera-Paredes, B., et al. (2024). *Mathematical discoveries
from program search with large language models.* Nature, 625, 468–475.
- https://www.nature.com/articles/s41586-023-06924-6
- PDF local: [`pdfs/funsearch-2024-nature.pdf`](pdfs/funsearch-2024-nature.pdf)
- Aporte: especificación del problema como función de evaluación + *skeleton*;
  *best-shot prompting*; islas para diversidad.

<a id="eoh"></a>
**[EoH]** Liu, F., et al. (2024). *Evolution of Heuristics: Towards Efficient
Automatic Algorithm Design Using Large Language Model.* ICML 2024.
- arXiv: https://arxiv.org/abs/2401.02051
- PDF local: [`pdfs/eoh-2024-icml.pdf`](pdfs/eoh-2024-icml.pdf)
- Aporte: coevolución de *thoughts* (lenguaje natural) y código; "primero
  describe, luego implementa"; estrategias de exploración/modificación.

<a id="reevo"></a>
**[ReEvo]** Ye, H., et al. (2024). *ReEvo: Large Language Models as
Hyper-Heuristics with Reflective Evolution.* NeurIPS 2024.
- arXiv: https://arxiv.org/abs/2402.01145
- PDF local: [`pdfs/reevo-2024-neurips.pdf`](pdfs/reevo-2024-neurips.pdf)
- Aporte: *system prompt* de experto, descripción de función, conocimiento de
  dominio en el prompt, reflexiones cortas/largas, mutación elitista.

<a id="evoprompt"></a>
**[EvoPrompt]** Guo, Q., et al. (2024). *Connecting Large Language Models with
Evolutionary Algorithms Yields Powerful Prompt Optimizers.* ICLR 2024.
- arXiv: https://arxiv.org/abs/2309.08532
- PDF local: [`pdfs/evoprompt-2024-iclr.pdf`](pdfs/evoprompt-2024-iclr.pdf)

<a id="opro"></a>
**[OPRO]** Yang, C., et al. (2024). *Large Language Models as Optimizers.* ICLR
2024.
- arXiv: https://arxiv.org/abs/2309.03409
- PDF local: [`pdfs/opro-2024-iclr.pdf`](pdfs/opro-2024-iclr.pdf)

<a id="prompt-report"></a>
**[Prompt Report]** Schulhoff, S., et al. (2024). *The Prompt Report: A
Systematic Survey of Prompting Techniques.*
- arXiv: https://arxiv.org/abs/2406.06608
- PDF local: [`pdfs/prompt-report-2024.pdf`](pdfs/prompt-report-2024.pdf)
- Aporte: taxonomía y respaldo de técnicas (role prompting, zero-shot CoT, etc.).

<a id="cot"></a>
**[CoT]** Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning
in Large Language Models.* NeurIPS 2022.
- arXiv: https://arxiv.org/abs/2201.11903
- PDF local: [`pdfs/chain-of-thought-2022.pdf`](pdfs/chain-of-thought-2022.pdf)

## Problema y operadores (mQAP / permutaciones)

<a id="knowles"></a>
**[mQAP]** Knowles, J., & Corne, D. (2003). *Instance Generators and Test Suites
for the Multiobjective Quadratic Assignment Problem.* EMO 2003, LNCS 2632.
- https://staff.cs.manchester.ac.uk/~jknowles/mQAP/
- PDF local: [`pdfs/knowles-corne-2003-mqap.pdf`](pdfs/knowles-corne-2003-mqap.pdf)
- Aporte: definición formal del mQAP, generadores de instancias y benchmark
  (las familias `KC` que usa MOHH-R).

<a id="crossover-review"></a>
**[Crossover review]** *Crossover Operators in Genetic Algorithms: A Review.*
ICTACT Journal on Soft Computing, 6(1), 2015.
- PDF local: [`pdfs/crossover-review-2015.pdf`](pdfs/crossover-review-2015.pdf)
- Aporte: descripción de OX, PMX, CX, ERX y qué estructura preserva cada una.

<a id="misevicius"></a>
**[Misevičius]** Misevičius, A. (2018). *Some Further Experiments with Crossover
Operators for the Quadratic Assignment Problem.* Informatica, 29(3).
- PDF local: [`pdfs/misevicius-qap-crossover-2018.pdf`](pdfs/misevicius-qap-crossover-2018.pdf)
- Aporte: comparación empírica de operadores de cruce para QAP.

## El método que sustituimos (referencia interna)

<a id="mohh"></a>
**[MOHH-R]** Morales-Paredes, A. I., Juárez, J., Falcón-Cardona, J. G.,
Terashima-Marín, H., & Coello Coello, C. A. (2025). *Automatic Design of
Specialized Variation Operators for the Multi-Objective Quadratic Assignment
Problem.* GECCO 2025.
- DOI: https://doi.org/10.1145/3712256.3726456 (acceso cerrado)
- Abstract: [`mohh-2025-abstract.md`](mohh-2025-abstract.md)
- Aporte: métrica (hipervolumen), datasets (`KC`), MOEA interno y la observación
  de que el mejor operador en train no siempre generaliza.

## Relacionados

- Liu, T., et al. (2024). *LLAMBO: Large Language Models to Enhance Bayesian
  Optimization.* ICLR 2024. https://arxiv.org/abs/2402.03921 —
  [`pdfs/llamabo-2024-icml.pdf`](pdfs/llamabo-2024-icml.pdf)

## BibTeX

```bibtex
@article{vanstein2024llamea,
  title   = {LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics},
  author  = {van Stein, Niki and B{\"a}ck, Thomas},
  journal = {IEEE Transactions on Evolutionary Computation},
  year    = {2024}
}

@article{romera2024funsearch,
  title   = {Mathematical discoveries from program search with large language models},
  author  = {Romera-Paredes, Bernardino and Barekatain, Mohammadamin and Novikov, Alexander and Balog, Matej and Kumar, M. Pawan and Dupont, Emilien and Ruiz, Francisco J. R. and Ellenberg, Jordan S. and Wang, Pengming and Fawzi, Omar and Kohli, Pushmeet and Fawzi, Alhussein},
  journal = {Nature},
  volume  = {625},
  pages   = {468--475},
  year    = {2024}
}

@inproceedings{liu2024eoh,
  title     = {Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model},
  author    = {Liu, Fei and Tong, Xialiang and Yuan, Mingxuan and Lin, Xi and Luo, Fu and Wang, Zhenkun and Lu, Zhichao and Zhang, Qingfu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2024}
}

@inproceedings{ye2024reevo,
  title     = {ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution},
  author    = {Ye, Haoran and Wang, Jiarui and Cao, Zhiguang and Berto, Federico and Hua, Chuanbo and Kim, Haeyeon and Park, Jinkyoo and Song, Guojie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{guo2024evoprompt,
  title     = {Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers},
  author    = {Guo, Qingyan and Wang, Rui and Guo, Junliang and Li, Bei and Song, Kaitao and Tan, Xu and Liu, Guoqing and Bian, Jiang and Yang, Yujiu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

@inproceedings{yang2024opro,
  title     = {Large Language Models as Optimizers},
  author    = {Yang, Chengrun and Wang, Xuezhi and Lu, Yifeng and Liu, Hanxiao and Le, Quoc V. and Zhou, Denny and Chen, Xinyun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}

@article{schulhoff2024promptreport,
  title   = {The Prompt Report: A Systematic Survey of Prompting Techniques},
  author  = {Schulhoff, Sander and others},
  journal = {arXiv preprint arXiv:2406.06608},
  year    = {2024}
}

@inproceedings{wei2022cot,
  title     = {Chain-of-Thought Prompting Elicits Reasoning in Large Language Models},
  author    = {Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Ichter, Brian and Xia, Fei and Chi, Ed and Le, Quoc V. and Zhou, Denny},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}

@inproceedings{knowles2003mqap,
  title     = {Instance Generators and Test Suites for the Multiobjective Quadratic Assignment Problem},
  author    = {Knowles, Joshua D. and Corne, David W.},
  booktitle = {Evolutionary Multi-Criterion Optimization (EMO)},
  series    = {LNCS},
  volume    = {2632},
  pages     = {845--858},
  year      = {2003}
}

@article{misevicius2018qap,
  title   = {Some Further Experiments with Crossover Operators for the Quadratic Assignment Problem},
  author  = {Misevi{\v{c}}ius, Alfonsas},
  journal = {Informatica},
  volume  = {29},
  number  = {3},
  year    = {2018}
}

@article{crossoverreview2015,
  title   = {Crossover Operators in Genetic Algorithms: A Review},
  journal = {ICTACT Journal on Soft Computing},
  volume  = {6},
  number  = {1},
  pages   = {1083--1092},
  year    = {2015}
}

@inproceedings{morales2025mohh,
  title     = {Automatic Design of Specialized Variation Operators for the Multi-Objective Quadratic Assignment Problem},
  author    = {Morales-Paredes, Adri{\'a}n Isa{\'i} and Ju{\'a}rez, Julio and Falc{\'o}n-Cardona, Jes{\'u}s Guillermo and Terashima-Mar{\'i}n, Hugo and Coello Coello, Carlos A.},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year      = {2025},
  doi       = {10.1145/3712256.3726456}
}
```
