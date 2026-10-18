
import os
import shutil
import subprocess

from llamea.llm import OpenAI_LLM


class VLLM_LLM(OpenAI_LLM):
    """Conector OpenAI-compatible para un servidor vLLM local.

    vLLM expone /v1 con la misma API que OpenAI; solo hay que apuntar
    base_url al servidor y usar una api_key dummy (vLLM no la valida).
    """

    def __init__(self, model, base_url="http://localhost:8000/v1",
                 api_key="EMPTY", temperature=0.8, **kwargs):
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)
        self.base_url = base_url
        self._client_kwargs["base_url"] = base_url
        from openai import OpenAI
        self.client = OpenAI(**self._client_kwargs)


def model_is_available(model_id: str) -> bool:
    """True si el modelo ya esta en el cache de HuggingFace o es ruta local."""
    if os.path.isdir(model_id):
        return True
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    return os.path.isdir(os.path.join(cache, f"models--{model_id.replace('/', '--')}"))


def ensure_model(model_id: str) -> str:
    """Verifica que el modelo este descargado; si no, lo descarga.

    vLLM puede auto-descargar al servir, pero verificamos/Descargamos
    explicitamente para que el servidor arranque siempre con el modelo listo.
    """
    if model_is_available(model_id):
        print(f"Modelo presente: {model_id}")
        return model_id

    print(f"Descargando modelo: {model_id} ...")
    from huggingface_hub import snapshot_download
    snapshot_download(model_id)
    print(f"Modelo descargado: {model_id}")
    return model_id
