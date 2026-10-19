"""Servidor OpenAI-compatible minimo para DESARROLLO LOCAL (sin vLLM/GPU grande).

Cumple la misma interfaz que vLLM (`/v1/models`, `/v1/chat/completions`), asi
que `mohh.llamea` se puede probar de extremo a extremo en una maquina pequena
(por ejemplo una RTX 3060) sin instalar vLLM. En el cluster usa vLLM
(`slurm/llamea-vllm-server.slrm`), que es mucho mas eficiente y deja el modelo
cargado en GPU.

Requiere el extra `local-server`:
    pip install -e ".[local-server]"

Uso:
    python scripts/local_openai_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000

Luego, en otra terminal:
    python -m mohh.llamea.run --model Qwen/Qwen2.5-0.5B-Instruct \
        --base-url http://localhost:8000/v1 --budget 3
"""

import argparse
import time
import uuid

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Local OpenAI-compatible dev server")

STATE = {"model": None, "tokenizer": None, "name": None}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    temperature: float = 0.8
    max_tokens: int | None = None
    top_p: float = 1.0
    stream: bool = False


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": STATE["name"], "object": "model", "owned_by": "local"}],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]

    inputs = tokenizer.apply_chat_template(
        req.messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    max_new_tokens = req.max_tokens or 512
    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if req.temperature and req.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=req.temperature, top_p=req.top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    generated = output[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    prompt_tokens = int(inputs["input_ids"].shape[-1])
    completion_tokens = int(generated.shape[-1])
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": STATE["name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Servidor OpenAI-compatible local (dev).")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Cargando {args.model} en {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=args.dtype)
    model.to(args.device)
    model.eval()

    STATE.update(model=model, tokenizer=tokenizer, name=args.model)
    print(f"Servidor listo en http://{args.host}:{args.port}/v1")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
