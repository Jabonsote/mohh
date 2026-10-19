"""Exporta los prompts finales a un Markdown reproducible.

Uso:
    python -m mohh.llamea.export_prompts [salida.md]

Sin argumento imprime en stdout. Sirve para mantener
`docs/llamea-prompts/prompt-final.md` sincronizado con `prompts.py`.
"""

import sys

from mohh.llamea import prompts


def _md_code(title, text, lang=""):
    # Elegir un fence mas largo que cualquier racha de backticks del contenido,
    # para poder mostrar prompts que a su vez contienen ```python`.
    longest = 0
    run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"### {title}\n\n{fence}{lang}\n{text}\n{fence}\n"


def render() -> str:
    parts = [
        "# Prompts finales (generados)\n",
        "> Generado por `python -m mohh.llamea.export_prompts`. "
        "Edita `src/mohh/llamea/prompts.py`, no este archivo.\n",
        "Los MOEA disponibles son MOEAD, NSGAII y SMSEMOA.\n",
        _md_code("ROLE_PROMPT", prompts.ROLE_PROMPT),
    ]

    for solver in ("MOEAD", "NSGAII", "SMSEMOA"):
        parts.append(
            _md_code(f"TASK_PROMPT ({solver})", prompts.make_task_prompt(solver))
        )

    parts.append(_md_code("EXAMPLE_PROMPT", prompts.EXAMPLE_PROMPT, "text"))
    parts.append(_md_code("OUTPUT_FORMAT_PROMPT", prompts.OUTPUT_FORMAT_PROMPT, "text"))
    parts.append(_md_code("OPERATOR_REFINE", prompts.OPERATOR_REFINE, "text"))
    parts.append(_md_code("OPERATOR_REDESIGN", prompts.OPERATOR_REDESIGN, "text"))
    return "\n".join(parts)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    text = render()
    if argv:
        with open(argv[0], "w") as f:
            f.write(text)
        print(f"Escrito {argv[0]}")
    else:
        print(text)


if __name__ == "__main__":
    main()
