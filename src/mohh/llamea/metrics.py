"""Métricas de costo del modelo: tokens, tiempo de GPU y energía.

Dos fuentes, complementarias:

1. **Cliente** (`VLLM_LLM.usage_history`): por petición, tokens de entrada
   (`prompt_tokens`), de salida (`completion_tokens`), total y **latencia de
   pared** (`seconds`). Siempre está disponible (vLLM, Ollama, llama.cpp).
2. **Servidor** (solo vLLM): el endpoint Prometheus `/metrics` publica el
   **tiempo real de GPU** de cada petición, que es la medida honesta del costo:

   | Métrica vLLM | Qué mide |
   |---|---|
   | `vllm:request_inference_time_seconds_sum` | segundos de GPU (prefill + decode) |
   | `vllm:request_prefill_time_seconds_sum` | segundos de GPU en prefill (procesar el prompt) |
   | `vllm:request_decode_time_seconds_sum` | segundos de GPU generando tokens |
   | `vllm:time_to_first_token_seconds_sum` | tiempo hasta el primer token |
   | `vllm:e2e_request_latency_seconds_sum` | latencia extremo a extremo |
   | `vllm:prompt_tokens_total` / `vllm:generation_tokens_total` | tokens contados por el servidor |
   | `vllm:request_success_total` | peticiones terminadas |
   | `vllm:gpu_cache_usage_perc` / `vllm:kv_cache_usage_perc` | ocupación del KV cache |

   Se toman dos fotos (antes y después de la corrida) y se usan los **deltas**:
   el proceso que sirve el modelo puede llevar otras peticiones y lo que importa
   es lo que consumió *esta* corrida.

3. **GPU local** (`nvidia-smi`): VRAM usada, utilización, potencia y
   temperatura. La potencia integrada en el tiempo da una **energía** estimada
   (Wh); en el cluster el job del servidor deja además una traza
   `vllm-gpu-<jobid>.csv` muestreada con `nvidia-smi` (ver
   `slurm/llamea-vllm-server.slrm`).

Con eso se derivan: `tokens_per_second` (salida/segundo de pared),
`gpu_seconds` (tiempo de GPU consumido), `gpu_hours`, `gpu_seconds_per_request`,
`cost_units` (GPU-hora) y `energy_wh` (si hay potencia).
"""

import json
import subprocess
import time
import urllib.error
import urllib.request

#: Métricas de vLLM que interesan, con su descripción (para el reporte).
VLLM_COST_METRICS = {
    "vllm:request_inference_time_seconds_sum": "segundos de GPU (prefill + decode)",
    "vllm:request_prefill_time_seconds_sum": "segundos de GPU en prefill",
    "vllm:request_decode_time_seconds_sum": "segundos de GPU generando",
    "vllm:time_to_first_token_seconds_sum": "tiempo hasta el primer token",
    "vllm:e2e_request_latency_seconds_sum": "latencia extremo a extremo",
    "vllm:request_queue_time_seconds_sum": "tiempo en cola",
    "vllm:prompt_tokens_total": "tokens de entrada (servidor)",
    "vllm:generation_tokens_total": "tokens de salida (servidor)",
    "vllm:request_success_total": "peticiones terminadas",
    "vllm:gpu_cache_usage_perc": "ocupacion del KV cache (%)",
    "vllm:kv_cache_usage_perc": "ocupacion del KV cache (%)",
    "vllm:num_requests_running": "peticiones en ejecucion",
    "vllm:num_requests_waiting": "peticiones en cola",
}

#: Métricas que son contadores acumulados (se usan por diferencia).
COUNTER_METRICS = {name for name in VLLM_COST_METRICS if name.endswith("_total")} | {
    "vllm:request_inference_time_seconds_sum",
    "vllm:request_prefill_time_seconds_sum",
    "vllm:request_decode_time_seconds_sum",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:request_queue_time_seconds_sum",
    "vllm:request_prompt_tokens_sum",
    "vllm:request_generation_tokens_sum",
}


def parse_prometheus(text: str) -> dict:
    """Convierte el formato Prometheus en `{metrica: valor}`.

    Las series con etiquetas se **suman** (es lo correcto para contadores e
    histogramas: `request_success_total{finished_reason="stop"}` y
    `{finished_reason="length"}` son el total de peticiones terminadas).
    """
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        name, _, raw = line.rpartition(" ")
        name = name.split("{", 1)[0].strip()
        try:
            value = float(raw)
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + value
    return values


def scrape_prometheus(base_url: str, timeout: float = 5.0) -> dict | None:
    """Lee `/metrics` del servidor (vLLM). `None` si no existe o no responde.

    Ollama y llama.cpp no publican `/metrics`: ahí la única medida es la del
    cliente (tokens y latencia), y esta función devuelve `None` sin romper nada.
    """
    # vLLM publica `/metrics` en la raiz, no bajo `/v1` (que es el prefijo de la
    # API OpenAI). Si el usuario pasa `.../v1`, hay que quitarlo.
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = base + "/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return parse_prometheus(response.read().decode(errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return None


def server_version(base_url: str, timeout: float = 5.0) -> dict | None:
    """Version del servidor (vLLM expone `/version`; Ollama `/api/version`).

    Registrarla es requisito de reproducibilidad: el mismo modelo servido por
    versiones distintas de vLLM puede dar resultados distintos.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    for endpoint in ("/version", "/api/version"):
        try:
            with urllib.request.urlopen(base + endpoint, timeout=timeout) as response:
                return json.loads(response.read().decode(errors="replace"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            continue
    return None


def nvidia_smi(timeout: float = 10.0) -> dict | None:
    """Foto de la GPU local (`None` si no hay `nvidia-smi` ni GPU)."""
    query = "name,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        rows.append({
            "name": parts[0],
            "memory_used_mb": _number(parts[1]),
            "memory_total_mb": _number(parts[2]),
            "utilization_percent": _number(parts[3]),
            "power_w": _number(parts[4]),
            "temperature_c": _number(parts[5]),
        })
    if not rows:
        return None
    return {"gpus": rows, "time": time.time()}


def _number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def counter_delta(before: dict | None, after: dict | None) -> dict:
    """Diferencia de los contadores entre dos fotos de `/metrics`."""
    if not before or not after:
        return {}
    delta = {}
    for name in sorted(set(after) & set(before)):
        if name in COUNTER_METRICS or name.endswith("_sum") or name.endswith("_count"):
            delta[name] = round(float(after[name]) - float(before[name]), 6)
    return delta


def gauges(values: dict | None) -> dict:
    """Gauges de interés (ocupación de KV cache, peticiones en vuelo)."""
    if not values:
        return {}
    return {name: values[name] for name in VLLM_COST_METRICS
            if name in values and name not in COUNTER_METRICS}


def cost_summary(usage: dict, before: dict | None = None, after: dict | None = None,
                 gpu: dict | None = None, wall_seconds: float | None = None) -> dict:
    """Resumen de costo de la corrida (cliente + servidor + GPU local).

    @param usage: `VLLM_LLM.usage_snapshot()` (tokens y latencia del cliente).
    @param before, after: fotos de `/metrics` antes y después de la corrida.
    @param gpu: foto de `nvidia_smi()` (potencia para estimar energía).
    @param wall_seconds: duración total de la corrida.
    """
    delta = counter_delta(before, after)
    summary = {
        "source": "vllm" if before or after else "client-only",
        "requests": usage.get("requests"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "inference_seconds": _round(usage.get("inference_seconds")),
        "failed_requests": usage.get("failed_requests"),
        "wall_seconds": _round(wall_seconds),
    }
    completion = usage.get("completion_tokens") or 0
    inference = usage.get("inference_seconds") or 0.0
    if inference > 0:
        summary["tokens_per_second"] = _round(completion / inference, 3)
        summary["seconds_per_request"] = _round(inference / max(1, usage.get("requests", 1)), 3)

    # Tiempo de GPU real (servidor vLLM).
    gpu_seconds = delta.get("vllm:request_inference_time_seconds_sum")
    if gpu_seconds is not None:
        summary["gpu_seconds"] = _round(gpu_seconds)
        summary["gpu_prefill_seconds"] = _round(delta.get("vllm:request_prefill_time_seconds_sum"))
        summary["gpu_decode_seconds"] = _round(delta.get("vllm:request_decode_time_seconds_sum"))
        summary["gpu_hours"] = _round(gpu_seconds / 3600.0, 6)
        requests = usage.get("requests") or 0
        if requests:
            summary["gpu_seconds_per_request"] = _round(gpu_seconds / requests, 4)
        if gpu_seconds > 0:
            summary["gpu_tokens_per_second"] = _round(
                (delta.get("vllm:generation_tokens_total") or completion) / gpu_seconds, 3
            )
        summary["cost_units_gpu_hours"] = summary["gpu_hours"]
    summary["server_tokens_prompt"] = delta.get("vllm:prompt_tokens_total")
    summary["server_tokens_generation"] = delta.get("vllm:generation_tokens_total")
    summary["server_requests"] = delta.get("vllm:request_success_total")
    summary["ttft_mean_seconds"] = _mean_per_request(
        delta.get("vllm:time_to_first_token_seconds_sum"),
        delta.get("vllm:time_to_first_token_seconds_count"),
    )
    summary["e2e_mean_seconds"] = _mean_per_request(
        delta.get("vllm:e2e_request_latency_seconds_sum"),
        delta.get("vllm:e2e_request_latency_seconds_count"),
    )
    summary["kv_cache_usage_percent"] = gauges(after).get("vllm:gpu_cache_usage_perc",
                                                        gauges(after).get("vllm:kv_cache_usage_perc"))

    # Energía estimada: potencia media de la GPU x tiempo de GPU.
    if gpu and gpu.get("gpus") and gpu_seconds:
        power = gpu["gpus"][0].get("power_w")
        if power:
            summary["gpu_power_w"] = power
            summary["energy_wh"] = _round(power * (gpu_seconds / 3600.0), 4)
    if gpu and gpu.get("gpus"):
        summary["gpu_memory_used_mb"] = gpu["gpus"][0].get("memory_used_mb")
        summary["gpu_memory_total_mb"] = gpu["gpus"][0].get("memory_total_mb")
        summary["gpu_name"] = gpu["gpus"][0].get("name")
    return summary


def _mean_per_request(total, count):
    if total is None or not count:
        return None
    return _round(total / count, 4)


def _round(value, digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def write_metrics(path: str, summary: dict, before: dict | None = None,
                  after: dict | None = None, history: list | None = None,
                  extra: dict | None = None) -> str:
    """Guarda `metrics.json` (resumen + fotos crudas + historial por petición)."""
    payload = {
        "summary": summary,
        "metrics_before": before,
        "metrics_after": after,
        "per_request": history or [],
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    return path
