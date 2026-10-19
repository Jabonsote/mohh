"""Conector con un servidor de inferencia local OpenAI-compatible (vLLM).

El modelo **queda cargado en la GPU** del servidor (vLLM) y LLaMEA solo manda
peticiones HTTP a `/v1`; asi el bucle evolutivo no paga la carga del modelo en
cada iteracion. El modelo es local: `ensure_model` verifica que este descargado
y, si falta, lo descarga al cache de HuggingFace.

Pensado para el cluster: el servidor corre en un job con GPU y la busqueda en
otro nodo apuntando a `http://<nodo-del-servidor>:8000/v1`.
"""

import os
import time

from llamea.llm import OpenAI_LLM

from mohh.llamea.parsing import strip_thinking, repair_code, extract_code_block


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
    """True si el modelo ya esta en el cache de HuggingFace o es ruta local."""
    if os.path.isdir(model_id):
        return True
    cache = _hf_cache_dir()
    return os.path.isdir(os.path.join(cache, f"models--{model_id.replace('/', '--')}"))


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
            f"{model_id}. Descargalo con: python scripts/ensure_model.py {model_id}"
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
                 max_tokens=None, **kwargs):
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._client_kwargs["base_url"] = base_url
        self._client_kwargs.setdefault("timeout", timeout)
        from openai import OpenAI
        self.client = OpenAI(**self._client_kwargs)
        self.reset_usage()
        self._install_usage_capture()

    # -- contabilidad de tokens ------------------------------------------- #
    def reset_usage(self):
        self.usage = {"requests": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "total_tokens": 0}
        self.usage_history = []
        self._last_message_chars = 0

    def _install_usage_capture(self):
        """Envuelve `chat.completions.create` para registrar tokens y
        `max_tokens`, sin reimplementar el bucle de reintentos de LLaMEA."""
        raw = self.client.chat.completions
        owner = self

        class _CapturingCompletions:
            def create(self, *args, **kwargs):
                if owner.max_tokens is not None and "max_tokens" not in kwargs:
                    kwargs["max_tokens"] = owner.max_tokens
                response = raw.create(*args, **kwargs)
                owner._record_usage(response)
                return response

            def __getattr__(self, name):
                return getattr(raw, name)

        self.client.chat.completions = _CapturingCompletions()

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) if usage else None
        completion = getattr(usage, "completion_tokens", None) if usage else None
        total = getattr(usage, "total_tokens", None) if usage else None

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
        self.usage_history.append({
            "request": self.usage["requests"],
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
        })

    def usage_snapshot(self):
        """Copia del consumo acumulado (segura para loguear/serializar)."""
        return dict(self.usage)

    def query(self, session_messages, **kwargs):
        # Guardar el tamano del prompt para estimar tokens si el servidor no
        # devuelve `usage`.
        self._last_message_chars = sum(
            len(str(message.get("content", ""))) for message in session_messages
        )
        response = super().query(session_messages, **kwargs)
        return strip_thinking(response)

    def __setstate__(self, state):
        super().__setstate__(state)
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
