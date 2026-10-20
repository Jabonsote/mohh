
import re

THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>",
                      re.DOTALL | re.IGNORECASE)
STRAY_THINK_RE = re.compile(r"</?(?:think|thinking|reasoning)>", re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:python|py|diff)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FENCE_OPEN_RE = re.compile(r"```(?:python|py)?\s*\n(.*)", re.DOTALL | re.IGNORECASE)
MAIN_GUARD_RE = re.compile(
    r"^\s*if __name__\s*={1,2}\s*['\"]__main__['\"]\s*:.*$",
    re.MULTILINE | re.DOTALL,
)

OPENERS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = {")", "]", "}"}


def strip_thinking(text: str) -> str:
    """Quita bloques <think>...</think> y sueltos (modelos con razonamiento)."""
    if not text:
        return text
    text = THINK_RE.sub("", text)
    text = STRAY_THINK_RE.sub("", text)
    return text.strip()


def extract_code_block(text: str) -> str:
    """Extrae el bloque de codigo de una respuesta LLM.

    - Con fence ```python / ```py / ```: toma el bloque mas grande.
    - Fence sin cerrar (truncamiento): toma lo que haya.
    - Sin fences: si parece codigo (tiene 'def' o 'import'), devuelve el texto.
    """
    text = strip_thinking(text)
    blocks = FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()

    m = FENCE_OPEN_RE.search(text)
    if m:
        return m.group(1).strip()

    stripped = text.strip()
    if "def " in stripped or "import " in stripped:
        return stripped

    raise ValueError("No code block found in LLM response")


def repair_code(code: str) -> str:
    """Repara malformaciones comunes del codigo generado.

    - Qua prosa alrededor del bloque.
    - Elimina bloques if __name__ == '__main__'.
    - Cierra brackets/parentesis/llaves sin balancear (truncamiento).
    """
    lines = [ln for ln in code.splitlines()]

    # Quitar lineas de prosa fuera de strings (heuristica: fuera de la funcion).
    code_lines = []
    started = False
    for ln in lines:
        if not started and not ln.strip():
            continue
        if not started and not (ln.startswith(("import ", "from ", "def ", "class ", "#", "@", "    ", "\t"))):
            continue
        started = True
        code_lines.append(ln)
    code = "\n".join(code_lines).strip()

    # Quitar bloque if __name__ == '__main__'
    code = MAIN_GUARD_RE.sub("", code).strip()

    # Balancear brackets sin cerrar (generacion truncada).
    code = close_unbalanced(code)

    return code


def close_unbalanced(code: str) -> str:
    """Appends missing closers; returns code unchanged if it looks balanced."""
    stack = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        # Triple-quoted strings: skip the whole literal.
        if code.startswith('"""', i) or code.startswith("'''", i):
            quote = code[i:i + 3]
            j = code.find(quote, i + 3)
            i = n if j == -1 else j + 3
            continue
        # Single/double quoted strings (with escapes).
        if ch in ('"', "'"):
            j = i + 1
            while j < n:
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == ch:
                    break
                j += 1
            i = j + 1
            continue
        # Comments: skip to end of line.
        if ch == "#":
            j = code.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if ch in OPENERS:
            stack.append(ch)
        elif ch in CLOSERS:
            if stack and OPENERS[stack[-1]] == ch:
                stack.pop()
        i += 1
    closers = "".join(OPENERS[c] for c in reversed(stack))
    return code + closers if closers else code


def looks_like_operator(code: str) -> bool:
    """True si el codigo compila y define una funcion crossover."""
    if "def crossover" not in code:
        return False
    try:
        compile(code, "<llm-response>", "exec")
        return True
    except SyntaxError:
        return False


def parse_llm_response(text: str) -> str:
    """Pipeline completo: thinking -> bloque -> reparacion -> validacion.

    Devuelve el codigo listo para ejecutar. Levanta ValueError si no hay
    codigo usable (el fitness la rechaza con feedback claro).
    """
    code = extract_code_block(text)
    code = repair_code(code)
    if not looks_like_operator(code):
        raise ValueError("LLM response has no valid 'def crossover' code")
    return code


def main():
    """CLI: prueba el parseo contra un archivo o texto guardado.

    Uso: python -m mohh.llamea.parsing respuesta_llm.txt
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Parsea/repara una respuesta LLM.")
    parser.add_argument("file", help="Archivo con la respuesta cruda del LLM.")
    args = parser.parse_args()

    with open(args.file) as f:
        raw = f.read()

    try:
        code = parse_llm_response(raw)
    except ValueError as e:
        print(f"FALLO: {e}", file=sys.stderr)
        sys.exit(1)

    print(code)


if __name__ == "__main__":
    main()
