"""Pesos de modelos en disco **persistente** (no en `$HOME`, que se borra).

En el cluster ixachi todo trabajo vive dentro de `/LUSTRE/<usuario>/` (ver
`cluster.md` §2): `$HOME` y `/tmp` no son visibles desde los nodos de cómputo y
se limpian. Los pesos de un modelo de 14B-32B son 8-60 GB, así que se descargan
**una sola vez** a un directorio persistente y de ahí los lee vLLM (y
`transformers`, para la verificación local).

Raíz de modelos (`models_root()`), por orden de prioridad:

1. `$MOHH_MODELS_DIR` (explícito; es lo que usan los scripts `.slrm`),
2. `$SCRATCH/models` si `$SCRATCH` existe,
3. `$LUSTRE/$USER/mohh-models` si `$LUSTRE` existe,
4. `/LUSTRE/$USER/mohh-models` si existe (ixachi no siempre exporta `$LUSTRE`),
5. `~/mohh-models` (último recurso en una máquina personal).

Cada modelo vive en `<raiz>/<nombre-seguro>/` con el layout de HuggingFace
(`config.json`, `*.safetensors`, tokenizer) más un manifiesto `mohh-model.json`
(id, revisión, fecha, archivos, tamaños, hashes opcionales, total). Al usar la
raíz se exportan `HF_HOME` y `HUGGINGFACE_HUB_CACHE` hacia ella para que ninguna
descarga caiga en el cache de `$HOME`.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime

#: Sufijos que cuentan como "pesos" de un modelo.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf", ".npz")

#: Archivos que deben existir para considerar el modelo servible.
REQUIRED_FILES = ("config.json",)

#: Tokenizer: basta con uno de estos.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                   "tokenizer.model", "vocab.json")

#: Nombres de archivos que NO son pesos aunque acaben en `.bin`/`.pt`.
_NON_WEIGHT = ("training_args.bin", "optimizer.bin", "scheduler.bin", "pytorch_model.bin.index.json")


def models_root(explicit: str | None = None) -> str:
    """Directorio raíz persistente para los pesos (ver docstring del módulo)."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    if os.environ.get("MOHH_MODELS_DIR"):
        return os.path.abspath(os.path.expanduser(os.environ["MOHH_MODELS_DIR"]))
    scratch = os.environ.get("SCRATCH")
    if scratch and os.path.isdir(scratch):
        return os.path.join(scratch, "models")
    lustre = os.environ.get("LUSTRE")
    if lustre and os.path.isdir(lustre):
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
        return os.path.join(lustre, user, "mohh-models")
    # ixachi define /LUSTRE pero no siempre exporta $LUSTRE en el login.
    # Preferir la ruta real antes de caer en $HOME, que no es persistente.
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
    cluster_lustre = os.path.join("/LUSTRE", user)
    if os.path.isdir(cluster_lustre):
        return os.path.join(cluster_lustre, "mohh-models")
    return os.path.expanduser("~/mohh-models")


def configure_hf_env(root: str | None = None) -> str:
    """Fija `HF_HOME`/`HUGGINGFACE_HUB_CACHE` a la raíz persistente.

    Devuelve la raíz. Llamar **antes** de cualquier descarga: así el cache de
    HuggingFace (blobs temporales, locks, revisiones) queda en el mismo disco
    persistente y no en `$HOME`.
    """
    root = models_root(root)
    os.makedirs(root, exist_ok=True)
    os.environ["HF_HOME"] = root
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(root, "hub"))
    return root


def safe_name(model_id: str) -> str:
    """Nombre de directorio seguro para un id de modelo (`a/b` -> `a--b`)."""
    return model_id.replace("/", "--").replace(":", "-").strip("-")


def model_dir(model_id: str, root: str | None = None) -> str:
    """Ruta local donde vive (o debe vivir) `model_id`."""
    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "config.json")):
        return os.path.abspath(model_id)
    return os.path.join(models_root(root), safe_name(model_id))


def _iter_files(path: str):
    for base, _dirs, files in os.walk(path):
        if os.path.basename(base) in (".hf-cache", "hub"):
            continue
        for name in sorted(files):
            yield os.path.join(base, name)


def _sha256(path: str, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(path: str, hash_files: bool = False) -> dict:
    """Inventario de archivos: tamaño y (opcional) sha256 de cada uno."""
    files = {}
    for full in _iter_files(path):
        relative = os.path.relpath(full, path)
        entry = {"bytes": os.path.getsize(full)}
        if hash_files:
            entry["sha256"] = _sha256(full)
        files[relative] = entry
    return files


def total_bytes(files: dict) -> int:
    return int(sum(entry.get("bytes", 0) for entry in files.values()))


def verify_model(path: str, hash_files: bool = False) -> dict:
    """Comprueba que los pesos están completos y sirven para cargar el modelo.

    Sin `hash_files` es barato (existe, no vacío, shards del índice, tokenizer);
    con `hash_files` calcula el sha256 de cada archivo (minutos en un 30B).
    """
    problems = []
    if not os.path.isdir(path):
        return {"ok": False, "path": path, "problems": ["no existe el directorio"]}

    files = file_inventory(path, hash_files=hash_files)
    names = set(files)

    for required in REQUIRED_FILES:
        if required not in names:
            problems.append(f"falta {required}")

    weights = [name for name in names
               if name.endswith(WEIGHT_SUFFIXES) and os.path.basename(name) not in _NON_WEIGHT]
    if not weights:
        problems.append("no hay archivos de pesos (.safetensors/.bin/.gguf)")
    for name in weights:
        if files[name]["bytes"] <= 0:
            problems.append(f"peso vacio: {name}")

    if not any(name in names for name in TOKENIZER_FILES):
        problems.append("falta el tokenizer (tokenizer.json / tokenizer_config.json)")

    # Si hay índice de shards, todos los shards referenciados deben existir.
    for name in names:
        if name.endswith(".safetensors.index.json"):
            try:
                with open(os.path.join(path, name)) as handle:
                    index = json.load(handle)
                for shard in sorted(set(index.get("weight_map", {}).values())):
                    if shard not in names:
                        problems.append(f"falta el shard {shard} (referenciado por {name})")
            except (OSError, ValueError) as error:
                problems.append(f"indice ilegible {name}: {error}")

    return {
        "ok": not problems,
        "path": path,
        "problems": problems,
        "files": len(files),
        "weights": len(weights),
        "total_bytes": total_bytes(files),
        "inventory": files,
    }


def write_manifest(path: str, model_id: str, revision: str | None = None,
                   verification: dict | None = None, extra: dict | None = None) -> str:
    """Escribe `mohh-model.json` con la procedencia y el inventario."""
    verification = verification or verify_model(path)
    manifest = {
        "model_id": model_id,
        "revision": revision,
        "local_path": os.path.abspath(path),
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "ok": verification["ok"],
        "problems": verification["problems"],
        "files": verification["files"],
        "weights": verification["weights"],
        "total_bytes": verification["total_bytes"],
        "human_size": human_size(verification["total_bytes"]),
        "inventory": verification.get("inventory", {}),
    }
    if extra:
        manifest.update(extra)
    destination = os.path.join(path, "mohh-model.json")
    with open(destination, "w") as handle:
        json.dump(manifest, handle, indent=2)
    return destination


def read_manifest(path: str) -> dict | None:
    manifest = os.path.join(path, "mohh-model.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest) as handle:
        return json.load(handle)


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def list_models(root: str | None = None) -> list:
    """Modelos presentes en la raíz, con tamaño y estado de verificación."""
    root = models_root(root)
    if not os.path.isdir(root):
        return []
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or name in (".hf-cache", "hub"):
            continue
        if not os.path.exists(os.path.join(path, "config.json")):
            continue
        manifest = read_manifest(path) or {}
        verification = verify_model(path)
        found.append({
            "name": name,
            "model_id": manifest.get("model_id", name.replace("--", "/")),
            "path": path,
            "ok": verification["ok"],
            "problems": verification["problems"],
            "total_bytes": verification["total_bytes"],
            "human_size": human_size(verification["total_bytes"]),
            "verified_at": manifest.get("verified_at"),
        })
    return found


def remote_size(model_id: str, revision: str | None = None) -> int:
    """Tamano total (bytes) que ocupan los archivos del modelo en el Hub.

    Se consulta antes de descargar para fallar temprano si no hay cuota: un
    30B AWQ son ~18 GB y un 80B FP8 ~80 GB. Si la API no esta disponible
    (p. ej. tests sin red), el llamador debe continuar con un aviso.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    return int(sum((sibling.size or 0) for sibling in info.siblings))


def check_free_space(root: str, needed: int, margin: float = 1.15) -> None:
    """Falla con un mensaje accionable si `root` no tiene espacio para `needed`."""
    free = shutil.disk_usage(root).free
    required = int(needed * margin)
    if required > free:
        raise RuntimeError(
            f"Espacio insuficiente en {root}: se necesitan ~{human_size(required)} "
            f"(modelo + margen) y hay {human_size(free)}. Libera cuota con "
            f"`python scripts/mohh_models.py remove <modelo>` o define "
            f"MOHH_MODELS_DIR/SCRATCH en otro disco persistente."
        )


def download_model(model_id: str, revision: str | None = None,
                   root: str | None = None, allow_download: bool = True,
                   hash_files: bool = False, max_workers: int = 8) -> str:
    """Descarga (o reutiliza) `model_id` en la raíz persistente y la verifica.

    @return la ruta local del modelo.
    @raise FileNotFoundError si falta y `allow_download` es False.
    """
    root = configure_hf_env(root)
    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "config.json")):
        path = os.path.abspath(model_id)
    else:
        path = model_dir(model_id, root)

    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        verification = verify_model(path, hash_files=hash_files)
        if verification["ok"]:
            print(f"Modelo ya presente y verificado: {model_id} -> {path}")
            write_manifest(path, model_id, revision, verification)
            return path
        print(f"Modelo incompleto en {path}: {verification['problems']}")

    if not allow_download:
        raise FileNotFoundError(
            f"Modelo no encontrado en {root}: {model_id}. "
            f"Descargalo con: python scripts/mohh_models.py download {model_id}"
        )

    from huggingface_hub import snapshot_download

    os.makedirs(path, exist_ok=True)
    try:
        needed = remote_size(model_id, revision)
    except Exception as error:  # noqa: BLE001 - la estimacion es best-effort
        print(f"AVISO: no se pudo estimar el tamano remoto de {model_id}: {error}")
    else:
        check_free_space(root, needed)

    print(f"Descargando {model_id} en {path} (raiz persistente {root}) ...")
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=path,
        cache_dir=os.path.join(root, ".hf-cache"),
        max_workers=max_workers,
    )
    verification = verify_model(path, hash_files=hash_files)
    write_manifest(path, model_id, revision, verification)
    if not verification["ok"]:
        raise RuntimeError(
            f"El modelo quedo incompleto en {path}: {verification['problems']}"
        )
    print(f"Listo: {model_id} ({human_size(verification['total_bytes'])}) en {path}")
    return path


def remove_model(model_id: str, root: str | None = None) -> bool:
    """Borra un modelo de la raíz persistente (para liberar cuota de LUSTRE)."""
    path = model_dir(model_id, root)
    if os.path.isdir(path):
        shutil.rmtree(path)
        return True
    return False


def load_check(path: str, deep: bool = False) -> dict:
    """Comprueba que el modelo se puede **cargar** (no solo que los bytes están).

    Carga el tokenizer (barato) y, con `deep`, construye el modelo en el
    dispositivo `meta` de PyTorch: valida config + índice de shards sin reservar
    VRAM ni leer los 30 GB de pesos.
    """
    result = {"ok": False, "path": path, "tokenizer": None, "model": None}
    try:
        from transformers import AutoTokenizer
    except ImportError:
        result["problems"] = ["transformers no esta instalado; no se puede cargar"]
        return result

    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        result["tokenizer"] = f"{type(tokenizer).__name__} (vocab {tokenizer.vocab_size})"
    except Exception as error:  # noqa: BLE001 - se reporta, no se lanza
        result["problems"] = [f"tokenizer: {error}"]
        return result

    if deep:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM

            config = AutoConfig.from_pretrained(path)
            with _meta_device():
                AutoModelForCausalLM.from_pretrained(path, config=config,
                                                     low_cpu_mem_usage=True)
            result["model"] = f"{config.model_type} ({config.num_hidden_layers} capas)"
        except Exception as error:  # noqa: BLE001
            result["problems"] = [f"modelo: {error}"]
            return result

    result["ok"] = True
    return result


def _meta_device():
    """Contexto que fuerza `device_map='meta'` (0 VRAM) si torch está disponible."""
    import contextlib

    try:
        import torch  # noqa: F401
    except ImportError:  # pragma: no cover - torch no es dependencia de la busqueda
        return contextlib.nullcontext()

    original = {}

    @contextlib.contextmanager
    def _context():
        import torch

        original["from_pretrained"] = torch.nn.Module.from_pretrained

        def patched(cls, *args, **kwargs):
            kwargs.setdefault("device_map", "meta")
            return original["from_pretrained"](cls, *args, **kwargs)

        torch.nn.Module.from_pretrained = classmethod(patched)
        try:
            yield
        finally:
            torch.nn.Module.from_pretrained = original["from_pretrained"]

    return _context()


def smoke_test(base_url: str, model: str, timeout: float = 60.0,
               prompt: str = "Reply with the single word: ok") -> dict:
    """Prueba de extremo a extremo contra el servidor OpenAI-compatible.

    Es la verificación de "el modelo cargó y responde" (lo que `verify_model`
    no puede saber): manda una petición mínima a `/chat/completions` y devuelve
    el texto y la latencia.
    """
    import time
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 8,
    }).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as error:
        raise RuntimeError(f"El servidor {url} no respondio correctamente: {error}")
    elapsed = time.perf_counter() - start
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Respuesta inesperada del servidor: {payload}")
    usage = payload.get("usage") or {}
    return {
        "ok": True,
        "model": payload.get("model", model),
        "text": text.strip(),
        "seconds": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }
