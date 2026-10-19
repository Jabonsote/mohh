# MOHH-R (paper del método que se sustituye)

**Título:** Automatic Design of Specialized Variation Operators for the
Multi-Objective Quadratic Assignment Problem

**Autores:** Adrián Isaí Morales-Paredes, Julio Juárez, Jesús Guillermo
Falcón-Cardona, Hugo Terashima-Marín, Carlos A. Coello Coello.

**Venue:** GECCO 2025.
**DOI:** [10.1145/3712256.3726456](https://doi.org/10.1145/3712256.3726456)

> **Nota:** el PDF es de **acceso cerrado** (sin versión abierta), por eso no se
> incluye en `pdfs/`. El abstract se reproduce abajo para tener el contexto a
> mano; el PDF se obtiene vía ACM con la suscripción institucional.

## Abstract

> The development of specialized, domain-specific operators has significantly
> enhanced the performance of evolutionary algorithms for solving optimization
> problems. However, creating such operators often requires substantial effort
> from human experts, making the process slow, resource-intensive, and heavily
> reliant on domain knowledge. To overcome these limitations, generation
> hyper-heuristics provide a framework for automating the design of variation
> operators by evolving combinations of heuristic components without direct
> expert input. In this work, we propose a generation hyper-heuristic method
> based on grammatical evolution to automatically design variation operators
> (crossover and mutation) tailored to the multi-objective quadratic assignment
> problem (mQAP)—a challenging combinatorial optimization problem with many
> real-world applications. Using the proposed method, variation operators were
> generated considering six mQAP instances with two and three objectives,
> leveraging MOEA/D as a multi-objective optimizer. For validation, the
> generated operators were evaluated on unseen instances. Our experimental
> results indicate that the evolved operators enhance the performance of
> MOEA/D compared to standard crossover operators. Furthermore, the
> top-performing operator in training did not always generalize best to larger
> instances, while some lower-ranked operators showed better adaptability. These
> results highlight the potential of automated operator design in effectively
> tackling complex optimization problems like the mQAP.

## Qué retomamos de aquí

- **Métrica:** hipervolumen del frente de Pareto (la misma en LLaMEA).
- **Datos:** instancias mQAP de las familias `KC` en `datasets/mqap/{train,test}`.
- **MOEA interno:** MOEA/D (y, por la misma lógica, NSGA-II y SMS-EMOA).
- **Hallazgo clave para el prompt:** el mejor operador en entrenamiento no
  siempre generaliza; por eso el prompt pide robustez en lugar de ajuste fino.
