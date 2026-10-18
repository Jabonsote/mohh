# modelos.md — Catálogo de modelos para LLaMEa + vLLM

El modelo se sirve local con vLLM (`slurm/llamea-vllm-server.slrm`) y LLaMEa
se conecta por API OpenAI-compatible. **Antes de correr, verificar que el
modelo esté descargado** (el modelo es local, no hay API externa):

```bash
python scripts/ensure_model.py <model_id>   # verifica en cache HF, descarga si falta
```

Cache: `~/.cache/huggingface/hub/models--<org>--<modelo>`. vLLM auto-descarga
en el primer `serve` si falta, pero la verificación explícita evita sorpresas
en jobs SLURM.

## VRAM de referencia (fp16, sin KV cache extra)

| Modelo | Parámetros | VRAM fp16 | Caso de uso |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | ~1 GB | Smoke test / CI |
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | **Default de pruebas** (cabe en 12 GB) |
| Qwen/Qwen2.5-7B-Instruct | 7B | ~15 GB | 24 GB: buen balance calidad/costo |
| Qwen/Qwen2.5-14B-Instruct | 14B | ~29 GB | 24 GB (justo, vigilar KV) o 32+ |
| Qwen/Qwen2.5-32B-Instruct | 32B | ~63 GB | 50 GB: usar cuantificación (AWQ/GPTQ ~18 GB) |
| Meta-Llama-3.1-8B-Instruct | 8B | ~16 GB | 24 GB |
| Meta-Llama-3.1-70B-Instruct | 70B | ~140 GB | 50 GB: solo con cuantificación agresiva o `--tensor-parallel-size 2` |
| Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 | 32B | ~18 GB | 24-50 GB: calidad alta, memoria baja |

Notas:

- vLLM reserva memoria para el KV cache; con prompts largos (el bucle evolutivo
  manda código + feedback de varias generaciones) vigilar `--gpu-memory-utilization`
  (default 0.9). Si OOM: bajarlo o usar `EXTRA_VLLM_ARGS="--gpu-memory-utilization 0.8"`.
- Para la partición `gpu` de ixachi (A10/A100/H100): 7B-14B en fp16 o 32B
  cuantificado entran sin problema en A100/H100 (40-80 GB). En A10 (24 GB):
  hasta 14B fp16 o 32B GPTQ-Int4.
- En esta máquina (RTX 3060, 12 GB): pruebas con Qwen2.5-0.5B/1.5B.
- Los modelos con chat template (Instruct) funcionan mejor: el prompt de LLaMEa
  va como mensaje de usuario y la respuesta se parsea por bloques ```python```.

## Configuración de calidad del modelo (LLaMEa)

- `temperature=0.8` por default en `VLLM_LLM` (balance exploración/explotación;
  subir para más diversidad, bajar a 0.2-0.4 para operadores más conservadores).
- `budget` = número de llamadas LLM/generaciones (default 20; corridas serias
  50-200).
- `n_parents`/`n_offspring`: 1/1 = un algoritmo por generación (por ahora);
  subir cuando se quiera búsqueda poblacional.
