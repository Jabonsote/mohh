# cambios.md — Integración LLaMEa + vLLM

Objetivo: usar LLaMEa (LLM como operador de variación evolutivo) para el **mismo
problema** que MOHH-R: diseñar operadores de variación (crossover) para MOEAs,
con los mismos conjuntos `datasets/mqap/train` y `test`, evaluados por
hipervolumen. Sin gramática: el LLM genera código Python directamente.

## ¿Qué se agregó?

### 1. Nuevo paquete `src/mohh/llamea/`

| Archivo | Contenido |
|---|---|
| `vllm.py` | `VLLM_LLM` (conector OpenAI-compatible → servidor vLLM local), `ensure_model()` (verifica en cache HF y descarga si falta) |
| `fitness.py` | Función de fitness LLaMEa: valida el código, lo ejecuta y evalúa el operador con el MOEA/D interno de MOHH-R sobre instancias train → score = hipervolumen medio |
| `prompts.py` | `ROLE_PROMPT`, `TASK_PROMPT` (interface `crossover(x, y)` para mQAP), `EXAMPLE_PROMPT` (OX), `OUTPUT_FORMAT_PROMPT` (Description + Code) |
| `run.py` | CLI `python -m mohh.llamea.run` (1 algoritmo por generación por defecto, opción poblacional) |

### 2. Cambio mínimo en núcleo MOHH-R (`src/mohh/core/operators/operator_template.py`)

`HH_Operator` acepta `eval_namespace=None` y `_do` hace
`eval(self.expression, self.eval_namespace)` cuando está presente. Permite que
el código generado por el LLM (función `crossover` definida por el modelo) se
evalúe dentro del operador de MOHH-R sin tocar nada más. **Compatible con la
versión anterior** (default `None` = comportamiento original).

### 3. Scripts SLURM (`slurm/`)

| Script | Descripción |
|---|---|
| `llamea-vllm-server.slrm` | Carga el modelo en GPU (partición `gpu`) y lo sirve por `/v1` (API OpenAI-compatible) hasta el time limit |
| `llamea-generate.slrm` | Corre la búsqueda LLaMEa en `ixachi` contra el servidor vLLM (`LLAMEA_SERVER_HOST` apunta al nodo del job vLLM) |

### 4. `scripts/ensure_model.py`

Verifica/descarga el modelo en el cache de HuggingFace antes de servir.

### 5. `modelos.md`

Catálogo de modelos con requerimientos de VRAM (ver ese archivo).

## Flujo de uso

```bash
# 1. Servidor vLLM con el modelo en GPU (cluster):
MODEL=Qwen/Qwen2.5-1.5B-Instruct sbatch slurm/llamea-vllm-server.slrm
squeue --user=$USER   # anotar el nodo asignado

# 2. Búsqueda LLaMEa (cluster o login con GPU local):
LLAMEA_SERVER_HOST=<nodo> sbatch slurm/llamea-generate.slrm
# o sin SLURM, contra localhost:
python -m mohh.llamea.run --model Qwen/Qwen2.5-1.5B-Instruct --base-url http://localhost:8000/v1
```

## Decisiones de diseño

- **Un solo algoritmo por generación por defecto**: `(1+1)-EA`
  (`n_parents=1, n_offspring=1, elitism=True`). Para poblacional: flags
  `--n-parents/--n-offspring` o variables `N_PARENTS/N_OFFSPRING` en el `.slrm`.
- **Fitness = hipervolumen medio** sobre un subconjunto semilla de instancias
  train (default 8; `--train-instances all` para el protocolo completo de
  MOHH-R). Misma métrica que la generación GE: comparable.
- **Modelo siempre verificado**: `ensure_model()` corre antes de cada búsqueda y
  el servidor puede auto-descargar; `scripts/ensure_model.py` lo hace
  explícito.
- **vLLM, no llamadas por request**: el modelo queda cargado en GPU y atiende
  OpenAI-compatible `/v1`, así el bucle evolutivo no paga load por iteración.
- **Sin gramática**: el LLM genera la función `crossover(x, y)` directa; la
  gramática BNF de MOHH-R queda como vía alternativa (`mohh-generate`).
- **Conexión simple entre proyectos**: el fitness reutiliza `MOSolver` +
  `HH_Operator` de MOHH-R, por lo que los operadores descubiertos por LLaMEa
  pueden evaluarse después con `mohh-evaluate` (mismos frentes `.pof`, mismos
  CSVs) — ver `modelos.md` para el paso de integración.
