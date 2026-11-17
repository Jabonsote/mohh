# Comparación por generación (train)

Protocolo congelado, igual para los dos experimentos:

- **IGD+, DELTA_P y R2**: contra el `P*` fijo de train; sin recalcular la referencia y sin agrandar la caja con frentes medidos.
- **HV**: misma escala `vmin/vmax` por instancia y `ref_point = 1.1` en los dos experimentos.
- Cada fila es la media de todos los individuos de esa generación.

Experimentos: Experiment_001, Experiment_002.

## Escala HV

| experimento | ref_point | origen |
|---|---:|---|
| Experiment_001 | 1.1 | consolidado:Experiment_001 |
| Experiment_002 | 1.1 | consolidado:Experiment_001 |

> La escala HV es la misma en los dos experimentos.

Ficheros:

```text
generation_mean.csv          # largo: experimento/generación/métrica
generation_wide.csv          # ancho: una columna por experimento y métrica
generation_by_instance.csv   # instancia/generación/métrica
generation_comparison.png    # media ± desviación por generación
```
