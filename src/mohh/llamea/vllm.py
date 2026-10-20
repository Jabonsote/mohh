"""Conector con un servidor de inferencia local OpenAI-compatible (vLLM).

El modelo **queda cargado en la GPU** del servidor (vLLM) y LLaMEA solo manda
peticiones HTTP a `/v1`; asi el bucle evolutivo no paga la carga del modelo en
cada iteracion. El modelo es local: `ensure_model` verifica que este descargado
y, si falta, lo descarga al cache de HuggingFace.

Pensado para el cluster: el servidor corre en un job con GPU y la busqueda en
otro nodo apuntando a `http://<nodo-del-servidor>:8000/v1`.
"""

import json
import os
import re
import threading
import time

from llamea.llm import OpenAI_LLM

from mohh.llamea.parsing import strip_thinking, repair_code, extract_code_block


#: Descripcion de una linea que LLaMEA pide en el formato de salida.
_DESCRIPTION_RE = re.compile(r"#\s*(?:Description|Design idea)\s*:\s*(.+)",
                             re.IGNORECASE)

#: Marcador del bloque de formato; la memoria se inserta justo antes para no
#: dejar instrucciones nuevas despues del formato de salida.
_OUTPUT_FORMAT_MARKER = "Answer in exactly this format:"

#: Una descripcion es una linea del LLM; se acota para que 12 recuerdos no
#: consuman el contexto ni inflen el costo por generacion.
_MAX_DESCRIPTION_CHARS = 240
_MAX_MEMORY_BLOCK_CHARS = 4000


def _json_safe(value):
    """Convierte valores del SDK/OpenAI a algo serializable sin perder texto."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _message_field(message, *names):
    """Campo extra de un mensaje (vLLM usa `reasoning_content`)."""
    if message is None:
        return None
    for name in names:
        value = getattr(message, name, None)
        if value:
            return value
    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        for name in names:
            if extra.get(name):
                return extra[name]
    return None


def extract_description(text: str) -> str:
    """Primera linea de `# Description:` / `# Design idea:` de una respuesta.

    Se usa para construir la memoria de mecanismos ya intentados. Devuelve ""
    si la respuesta no trae descripcion (p. ej. la reescritura adaptativa del
    task prompt, que no es un operador).
    """
    match = _DESCRIPTION_RE.search(text or "")
    return match.group(1).strip()[:_MAX_DESCRIPTION_CHARS] if match else ""


def memory_block(descriptions, limit: int = 12) -> str:
    """Bloque de prompt con los mecanismos recientes, para evitar anclaje.

    El fallo observado en `b200` fue generar 108 veces el mismo operador. El
    `--repeat-penalty` castiga la repeticion en el ranking, pero el modelo sigue
    gastando generaciones en ella; recordarle explicitamente lo ya intentado es
    mas barato que descartar el candidato despues. Cada descripcion y el bloque
    completo estan acotados para que la memoria no infle el contexto ni el costo.
    """
    items = [d.strip()[:_MAX_DESCRIPTION_CHARS] for d in descriptions
             if d and d.strip()]
    if not items or limit <= 0:
        return ""
    # Ultimos `limit` mecanismos, sin pasarse del presupuesto de caracteres.
    lines = []
    used = 0
    for description in reversed(items[-limit:]):
        line = f"- {description}"
        if lines and used + len(line) + 1 > _MAX_MEMORY_BLOCK_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    lines.reverse()
    return (
        "\n\n# Mechanisms already generated in this run\n"
        "Do NOT repeat or lightly paraphrase these operators. Deliberately "
        "choose a different combination of order-, position-, adjacency- or "
        "cycle-based ideas:\n" + "\n".join(lines) + "\n"
    )


def _hf_cache_dir() -> str:
    """Directorio del cache de HuggingFace hub (respeta HF_HOME)."""
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return hub_cache
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def model_is_available(model_id: str) -> bool:
    """True si el modelo ya esta en el cache de HuggingFace, en la raiz
    persistente de `mohh.llamea.models`, o es una ruta local."""
    if os.path.isdir(model_id):
        return True
    cache = _hf_cache_dir()
    if os.path.isdir(os.path.join(cache, f"models--{model_id.replace('/', '--')}")):
        return True
    try:
        from mohh.llamea.models import model_dir

        if os.path.exists(os.path.join(model_dir(model_id), "config.json")):
            return True
    except Exception:  # noqa: BLE001 - la comprobacion es informativa
        pass
    return False


def ensure_model(model_id: str, allow_download: bool = True,
                 local_dir: str | None = None) -> str:
    """Verifica que el modelo este descargado; si falta y `allow_download`,
    lo descarga al cache de HuggingFace.

    vLLM puede auto-descargar al servir, pero la verificacion explicita evita
    fallos silenciosos (y reintentos de red) dentro de un job SLURM.
    """
    if model_is_available(model_id):
        print(f"Modelo presente: {model_id}")
        return model_id

    if not allow_download:
        raise FileNotFoundError(
            f"Modelo no encontrado en {_hf_cache_dir()} ni como ruta local: "
            f"{model_id}. Descargalo con: python scripts/mohh_models.py download "
            f"{model_id}"
        )

    print(f"Descargando modelo: {model_id} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(model_id, local_dir=local_dir)
    print(f"Modelo descargado: {model_id}")
    return model_id


def wait_for_server(base_url: str, timeout: float = 1800, interval: float = 3.0) -> bool:
    """Espera a que el servidor OpenAI-compatible responda en `/models`.

    En SLURM el servidor vLLM tarda en cargar el modelo; la busqueda puede
    arrancar en otro nodo y quedarse esperando en lugar de fallar.

    @return True si respondio antes del timeout; False en caso contrario.
    """
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        time.sleep(interval)
    return False


class VLLM_LLM(OpenAI_LLM):
    """Conector OpenAI-compatible para un servidor vLLM local.

    vLLM expone `/v1` con la misma API que OpenAI; solo hay que apuntar
    `base_url` al servidor y usar una api_key dummy (vLLM no la valida).

    Ademas del parseo base de LLaMEA, aplica el mecanismo de reparacion de
    respuestas (ver `mohh.llamea.parsing`): quita razonamiento `<think>`,
    repara bloques truncados y recurre a un extractor tolerante si el modelo
    no cierra el bloque de codigo. La validacion de `crossover(x, y)` la hace
    el fitness (`compile_operator`).

    Lleva la cuenta de **tokens** consumidos (prompt, completion y total) y de
    peticiones, leyendo el campo `usage` que devuelve el servidor; si falta, lo
    estima por longitud. `usage_snapshot()` devuelve un resumen acumulado.
    """

    def __init__(self, model, base_url="http://localhost:8000/v1",
                 api_key="EMPTY", temperature=0.8, timeout=600,
                 max_tokens=None, top_p=None, top_k=None,
                 sampling_repetition_penalty=None, operator_memory=0,
                 thinking=None, request_log_path=None, request_context=None,
                 **kwargs):
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.sampling_repetition_penalty = sampling_repetition_penalty
        self.operator_memory = max(0, int(operator_memory or 0))
        # None = no tocar el template del servidor; True/False fuerza
        # `chat_template_kwargs.enable_thinking` (Qwen3 y familia).
        self.thinking = thinking
        #: JSONL con la peticion exacta (incluida la memoria), respuesta cruda,
        #: `reasoning_content`, parametros de muestreo, usage y latencia.
        self.request_log_path = request_log_path
        self.request_context = dict(request_context or {})
        self._request_lock = threading.Lock()
        self._request_index = 0
        self._last_response = None
        self._last_elapsed = 0.0
        self._memory = []
        self._client_kwargs["base_url"] = base_url
        self._client_kwargs.setdefault("timeout", timeout)
        from openai import OpenAI
        self.client = OpenAI(**self._client_kwargs)
        self.reset_usage()
        self._install_usage_capture()

    # -- contabilidad de tokens ------------------------------------------- #
    def reset_usage(self):
        self.usage = {"requests": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "total_tokens": 0,
                      "reasoning_tokens": 0, "cached_prompt_tokens": 0,
                      "inference_seconds": 0.0, "failed_requests": 0}
        self.usage_history = []
        self._last_message_chars = 0
        self._last_response = None
        self._last_elapsed = 0.0

    def _install_usage_capture(self):
        """Envuelve `chat.completions.create` para registrar tokens, `max_tokens`
        y la **latencia** de cada peticion, sin reimplementar el bucle de
        reintentos de LLaMEA."""
        raw = self.client.chat.completions
        owner = self

        class _CapturingCompletions:
            def create(self, *args, **kwargs):
                if owner.max_tokens is not None and "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = owner.max_tokens
                if owner.top_p is not None and "top_p" not in kwargs:
                    kwargs["top_p"] = owner.top_p
                # top_k y repetition_penalty son extensiones de vLLM: van en
                # extra_body para que el cliente OpenAI no las rechace.
                extra = dict(kwargs.get("extra_body") or {})
                if owner.top_k is not None:
                    extra.setdefault("top_k", owner.top_k)
                if owner.sampling_repetition_penalty is not None:
                    extra.setdefault("repetition_penalty",
                                     owner.sampling_repetition_penalty)
                if owner.thinking is not None:
                    # vLLM/Qwen3: el template recibe enable_thinking. No se
                    # manda a servidores que no lo entiendan salvo pedido
                    # explicito (--thinking on/off).
                    template = dict(extra.get("chat_template_kwargs") or {})
                    template.setdefault("enable_thinking", bool(owner.thinking))
                    extra["chat_template_kwargs"] = template
                if extra:
                    kwargs["extra_body"] = extra
                start = time.perf_counter()
                try:
                    response = raw.create(*args, **kwargs)
                except Exception:
                    # Peticion fallida (timeout, 5xx, conexion): se cuenta y se
                    # deja que LLaMEA aplique su propio reintento.
                    owner.usage["failed_requests"] += 1
                    raise
                elapsed = time.perf_counter() - start
                owner._last_response = response
                owner._last_elapsed = elapsed
                owner._record_usage(response, elapsed, kwargs=kwargs)
                return response

            def __getattr__(self, name):
                return getattr(raw, name)

        self.client.chat.completions = _CapturingCompletions()

    def _record_usage(self, response, elapsed: float = 0.0, kwargs=None):
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage else None
        completion = getattr(usage, "completion_tokens", None) if usage else None
        total = getattr(usage, "total_tokens", None) if usage else None
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        reasoning = (getattr(details, "reasoning_tokens", None) if details else None)
        if reasoning is None and usage is not None:
            reasoning = getattr(usage, "reasoning_tokens", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage else None
        cached = (getattr(prompt_details, "cached_tokens", None)
                  if prompt_details else None)
        if cached is None and usage is not None:
            cached = getattr(usage, "cached_tokens", None)

        if prompt is None:
            # Estimacion gruesa (~4 caracteres por token) si el servidor no
            # reporta usage (p. ej. algunos proxies OpenAI-compatibles).
            prompt = max(0, round(self._last_message_chars / 4))
        if completion is None:
            text = ""
            try:
                text = response.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                text = ""
            completion = max(0, round(len(text) / 4))
        if total is None:
            total = int(prompt) + int(completion)

        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += int(prompt)
        self.usage["completion_tokens"] += int(completion)
        self.usage["total_tokens"] += int(total)
        self.usage["reasoning_tokens"] += int(reasoning or 0)
        self.usage["cached_prompt_tokens"] += int(cached or 0)
        self.usage["inference_seconds"] += float(elapsed)

        message = None
        finish_reason = None
        try:
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
        except Exception:  # noqa: BLE001
            pass
        reasoning_content = _message_field(
            message, "reasoning_content", "reasoning", "thinking"
        )
        self.usage_history.append({
            "request": self.usage["requests"],
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
            "reasoning_tokens": int(reasoning or 0),
            "cached_prompt_tokens": int(cached or 0),
            "seconds": round(float(elapsed), 4),
            "time": time.time(),
            # Parametros efectivos y respuesta: sin esto, el usage no permite
            # reconstruir que se le pidio al modelo ni que devolvio.
            "model": getattr(response, "model", None) or getattr(self, "model", None),
            "response_id": getattr(response, "id", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "finish_reason": finish_reason,
            "response_chars": len(getattr(message, "content", None) or ""),
            "reasoning_chars": len(reasoning_content or ""),
            "params": self._effective_params(kwargs or {}),
        })

    def _effective_params(self, kwargs):
        """Parametros de muestreo efectivos de una peticion (serializables)."""
        effective = {
            "model": kwargs.get("model") or getattr(self, "model", None),
            "temperature": kwargs.get("temperature", getattr(self, "temperature", None)),
            "max_tokens": kwargs.get("max_tokens", getattr(self, "max_tokens", None)),
            "top_p": kwargs.get("top_p", getattr(self, "top_p", None)),
            "top_k": kwargs.get("top_k", getattr(self, "top_k", None)),
            "repetition_penalty": kwargs.get(
                "repetition_penalty",
                getattr(self, "sampling_repetition_penalty", None),
            ),
            "thinking": getattr(self, "thinking", None),
        }
        extra = kwargs.get("extra_body")
        if extra:
            effective["extra_body"] = _json_safe(extra)
        return _json_safe(effective)

    def _response_reasoning(self):
        """`reasoning_content` de la ultima respuesta, si el servidor lo expone."""
        try:
            message = self._last_response.choices[0].message
        except Exception:  # noqa: BLE001
            return ""
        return _message_field(message, "reasoning_content", "reasoning", "thinking") or ""

    def _response_raw(self):
        try:
            message = self._last_response.choices[0].message
            return getattr(message, "content", None) or ""
        except Exception:  # noqa: BLE001
            return ""

    def _write_request_log(self, request_index, messages, kwargs,
                           raw="", cleaned="", error=None):
        """Escribe una linea JSONL con la peticion y la respuesta completas.

        Se guarda aparte de `conversationlog.jsonl` porque LLaMEA registra la
        respuesta ya limpia (sin razonamiento) y no conoce los parametros
        efectivos de muestreo. Si el archivo no esta configurado, no hace nada.
        """
        if not self.request_log_path:
            return
        response = self._last_response
        message = None
        finish_reason = None
        if response is not None:
            try:
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason
            except Exception:  # noqa: BLE001
                pass
        reasoning = _message_field(
            message, "reasoning_content", "reasoning", "thinking"
        ) or ""
        usage = getattr(response, "usage", None) if response is not None else None
        usage_payload = None
        if usage is not None:
            usage_payload = _json_safe(
                usage.model_dump() if hasattr(usage, "model_dump") else usage
            )
        prompt_text = "\n\n".join(
            str(item.get("content", "")) for item in messages
            if isinstance(item, dict)
        )
        record = {
            "request": request_index,
            "time": time.time(),
            "model": (getattr(response, "model", None)
                      if response is not None else None) or getattr(self, "model", None),
            "base_url": self.base_url,
            "context": _json_safe(self.request_context),
            "messages": _json_safe(messages),
            "prompt": prompt_text,
            "params": self._effective_params(kwargs or {}),
            "response": {
                "raw_content": raw or "",
                "cleaned_content": cleaned or "",
                "reasoning_content": reasoning,
                "finish_reason": finish_reason,
                "id": getattr(response, "id", None) if response is not None else None,
                "system_fingerprint": (
                    getattr(response, "system_fingerprint", None)
                    if response is not None else None
                ),
                "usage": usage_payload,
                "seconds": round(float(self._last_elapsed), 4),
            },
            "error": f"{type(error).__name__}: {error}" if error else None,
        }
        try:
            directory = os.path.dirname(os.path.abspath(self.request_log_path))
            os.makedirs(directory, exist_ok=True)
            with self._request_lock:
                with open(self.request_log_path, "a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False,
                                          default=str) + "\n")
                    file.flush()
        except Exception:  # noqa: BLE001 - el log no debe tumbar la busqueda
            pass

    def usage_snapshot(self):
        """Copia del consumo acumulado (segura para loguear/serializar)."""
        snapshot = dict(self.usage)
        snapshot["request_log"] = getattr(self, "request_log_path", None)
        snapshot["thinking"] = getattr(self, "thinking", None)
        return snapshot

    def query(self, session_messages, **kwargs):
        messages = session_messages
        if self.operator_memory and self._memory:
            # Copia superficial: no mutar la lista que arma LLaMEA (la usa para
            # loguear la conversacion).
            messages = [dict(message) for message in session_messages]
            if messages:
                content = str(messages[-1].get("content", ""))
                note = memory_block(self._memory, self.operator_memory)
                if _OUTPUT_FORMAT_MARKER in content:
                    content = content.replace(
                        _OUTPUT_FORMAT_MARKER, note + _OUTPUT_FORMAT_MARKER, 1
                    )
                else:
                    content += note
                messages[-1]["content"] = content
        # Guardar el tamano del prompt para estimar tokens si el servidor no
        # devuelve `usage`.
        self._last_message_chars = sum(
            len(str(message.get("content", ""))) for message in messages
        )
        with self._request_lock:
            self._request_index += 1
            request_index = self._request_index
        self._last_response = None
        self._last_elapsed = 0.0
        try:
            raw = super().query(messages, **kwargs)
        except Exception as error:  # noqa: BLE001 - se registra y se relanza
            self._write_request_log(request_index, messages, kwargs, error=error)
            raise
        raw = raw or ""
        cleaned = strip_thinking(raw)
        # Algunos servidores devuelven el codigo solo en reasoning_content y
        # dejan content vacio. Se usa como fallback solo si no hay respuesta.
        if not cleaned.strip():
            reasoning = self._response_reasoning()
            if reasoning:
                cleaned = strip_thinking(reasoning)
        self._remember(cleaned)
        self._write_request_log(
            request_index, messages, kwargs,
            raw=self._response_raw() or raw, cleaned=cleaned,
        )
        return cleaned

    def _remember(self, response):
        """Guarda la descripcion del operador recien generado (memoria acotada)."""
        if not self.operator_memory:
            return
        description = extract_description(response or "")
        if not description or description in self._memory:
            return
        self._memory.append(description)
        if len(self._memory) > self.operator_memory:
            del self._memory[0]

    def operator_memory_snapshot(self):
        """Estado de la memoria de mecanismos (para tokens.json/reportes)."""
        return {
            "enabled": bool(self.operator_memory),
            "size": len(self._memory),
            "limit": self.operator_memory,
            "descriptions": list(self._memory),
        }

    def __getstate__(self):
        # LLaMEA serializa el objeto completo para sus checkpoints. Un
        # `threading.Lock` y la ultima respuesta del SDK no son serializables;
        # se recrean/limpian en `__setstate__`.
        state = super().__getstate__()
        state.pop("_request_lock", None)
        state.pop("_last_response", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        # Estados antiguos (pickles/checkpoints) no traen los campos nuevos.
        if not hasattr(self, "thinking"):
            self.thinking = None
        if not hasattr(self, "request_log_path"):
            self.request_log_path = None
        if not hasattr(self, "request_context"):
            self.request_context = {}
        if not hasattr(self, "_request_lock"):
            self._request_lock = threading.Lock()
        if not hasattr(self, "_request_index"):
            self._request_index = 0
        if not hasattr(self, "_last_response"):
            self._last_response = None
        if not hasattr(self, "_last_elapsed"):
            self._last_elapsed = 0.0
        # El cliente se recreo en __setstate__; volver a enganchar la captura.
        self._install_usage_capture()

    def extract_algorithm_code(self, message):
        # 1. Intentar el regex de LLaMEA (bloque ```python ... ```).
        try:
            code = super().extract_algorithm_code(message)
        except Exception:
            # 2. Fallback: nuestro extractor tolera fences sin cerrar y
            #    respuestas sin fence que aun parecen codigo.
            from llamea.utils import NoCodeException
            try:
                code = extract_code_block(message)
            except ValueError as e:
                raise NoCodeException from e
        code = repair_code(code)

        # 3. Exigir el contrato ANTES de devolver: si no hay un `crossover` a
        #    nivel de modulo, LLaMEA haria `re.findall(...)[0]` y lanzaria un
        #    IndexError opaco. Mejor fallar aqui con NoCodeException (feedback
        #    claro al LLM y score 0 controlado).
        if "def crossover" not in code:
            from llamea.utils import NoCodeException
            raise NoCodeException(
                "Response has no module-level 'def crossover(x, y)' function."
            )
        return code
