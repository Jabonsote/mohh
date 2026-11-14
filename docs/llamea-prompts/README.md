# LLaMEA para mQAP

LLaMEA sustituye la gramática de GE por un LLM local que escribe
`crossover(x, y)`. Reutiliza el mismo `MOSolver` y el mismo ranking GE:
hipervolumen de la población, nadir recalculado y rango medio (1 = mejor). El
núcleo GE no se modifica.

## Cluster

Guía mínima, perfiles y solución de problemas:

[`CLUSTER.md`](CLUSTER.md)

```bash
cd /LUSTRE/$USER/MOOHH-R
PROFILE=coder30b-awq GENERATIONS=1000 EVAL_TEST=1 \
  bash slurm/llamea-all.sh
```

## Documentación detallada

Solo para diseño, evidencia o análisis; no hace falta para lanzar una corrida:

| Archivo | Contenido |
|---|---|
| [`diseno-de-prompts.md`](diseno-de-prompts.md) | Diseño de los prompts y evidencia. |
| [`prompt-final.md`](prompt-final.md) | Prompts finales y mapeo a slots. |
| [`pruebas-locales.md`](pruebas-locales.md) | Corridas locales y fallas típicas. |
| [`comparacion-justa.md`](comparacion-justa.md) | Evaluación pareada y resultados. |
| [`mejoras-prompt.md`](mejoras-prompt.md) | Mejoras y plan de ablación. |
| [`areas-de-mejora.md`](areas-de-mejora.md) | Mejoras del bucle completo. |
| [`referencias.md`](referencias.md) | Bibliografía y BibTeX. |
| [`mohh-2025-abstract.md`](mohh-2025-abstract.md) | Paper de MOHH-R. |
