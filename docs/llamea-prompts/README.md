# Prompts para generar operadores de variación con LLaMEA (mQAP)

Esta carpeta documenta el **diseño basado en evidencia** de los prompts que usa
la integración LLaMEA de MOHH-R para generar operadores de cruce (`crossover`)
sobre el problema multiobjetivo de asignación cuadrática (mQAP).

El objetivo es sustituir la búsqueda por gramática por una búsqueda con un LLM
local, **reutilizando la misma evaluación** (mismo `MOSolver` interno y mismo
hipervolumen que la búsqueda GE), pero con prompts diseñados a partir de
literatura revisada y no derivados de la gramática.

## Contenido

| Archivo | Qué contiene |
|---|---|
| [`diseno-de-prompts.md`](diseno-de-prompts.md) | Razonamiento de diseño: cada elemento del prompt con su respaldo y citas. |
| [`prompt-final.md`](prompt-final.md) | El texto **exacto** de los prompts finales y su mapeo a los slots de LLaMEA. |
| [`pruebas-locales.md`](pruebas-locales.md) | Corridas reales con un modelo local pequeño: resultados, fallas típicas y los arreglos de robustez que salieron de ellas. |
| [`mejoras-prompt.md`](mejoras-prompt.md) | Áreas de mejora identificadas, su justificación (literatura + evidencia local) y el plan de ablación. |
| [`areas-de-mejora.md`](areas-de-mejora.md) | Mejoras del bucle completo (modelo, evaluación, recompensa, paralelismo, seguridad y observabilidad), priorizadas por impacto sobre la validez. |
| [`referencias.md`](referencias.md) | Bibliografía con enlaces, PDF local y BibTeX. |
| [`mohh-2025-abstract.md`](mohh-2025-abstract.md) | El paper de MOHH-R (acceso cerrado): DOI + abstract. |
| [`pdfs/`](pdfs/) | Los PDFs descargados (ver `pdfs/README.md`). |

## Resumen en una tabla

| Slot de LLaMEA | Prompt | Evidencia principal |
|---|---|---|
| `role_prompt` | Rol de experto en EC + operadores de permutación | LLaMEA (van Stein & Bäck 2024); ReEvo (Ye et al. 2024) |
| `task_prompt` | Problema + contrato + objetivo + conocimiento de dominio + restricciones negativas + feedback de ranking | LLaMEA 2024; FunSearch (Romera-Paredes et al. 2024); ReEvo 2024; Knowles & Corne 2003; evidencia local |
| `example_prompt` | Un ejemplo simple de OX para fijar el contrato, pidiendo ir más allá | LLaMEA 2024; FunSearch 2024 |
| `output_format_prompt` | "Idea de diseño" + descripción de una línea + código | EoH (Liu et al. 2024); CoT (Wei et al. 2022); LLaMEA 2024 |
| operadores (`refine`/`redesign`) | Refinar (explotar) o rediseñar (explorar), guiados por el ranking | LLaMEA 2024; EoH 2024; ReEvo 2024 |

## Cómo usarlos

Los prompts viven en el código, en `src/mohh/llamea/prompts.py`, y se inyectan
en LLaMEA desde `src/mohh/llamea/run.py`:

```python
LLaMEA(
    f=fitness,
    llm=llm,
    n_parents=1, n_offspring=1,
    role_prompt=ROLE_PROMPT,
    task_prompt=make_task_prompt(solver),   # adaptado al MOEA interno
    example_prompt=EXAMPLE_PROMPT,
    output_format_prompt=OUTPUT_FORMAT_PROMPT,
    operators=make_operators(),             # refine / redesign
    ...
)
```

Para lanzar una búsqueda (servidor vLLM ya levantado):

```bash
python -m mohh.llamea.run --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://localhost:8000/v1 --solver MOEAD --budget 50 --eval-test
```
