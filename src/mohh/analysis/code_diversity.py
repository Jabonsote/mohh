"""Diversidad de código con CrystalBLEU.

CrystalBLEU (Eghbali & Pradel, ASE 2022) mide similitud de código ignorando
los n-gramas más frecuentes del corpus. La diversidad se define como
``1 - similitud``: 0 = código idéntico, 1 = sin n-gramas en común.

Requiere ``crystalbleu`` y ``nltk``:
    pip install crystalbleu nltk
"""
from __future__ import annotations

import io
import re
import tokenize
from collections import Counter

from crystalbleu import sentence_bleu
from nltk.util import ngrams

_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")


def code_tokens(code: str) -> list:
    """Tokens de código: nombres, números, cadenas y operadores."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(str(code)).readline):
            if token.type in (tokenize.NAME, tokenize.NUMBER,
                              tokenize.STRING, tokenize.OP):
                tokens.append(token.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = _TOKEN_RE.findall(str(code))
    return tokens


def crystal_ignoring(codes, n_max: int = 4, top_fraction: float = 0.05) -> set:
    """n-gramas más frecuentes del corpus (los que CrystalBLEU ignora).

    Se cuentan los n-gramas de 1..``n_max`` en todos los códigos y se ignora
    el ``top_fraction`` más frecuente. Es la parte "crystal" del métrico:
    patrones ubicuos del lenguaje/dominio no cuentan como similitud.
    """
    counter = Counter()
    for code in codes:
        tokens = code_tokens(code)
        for n in range(1, n_max + 1):
            counter.update(ngrams(tokens, n))
    if not counter:
        return set()
    keep = max(1, int(len(counter) * float(top_fraction)))
    return {ngram for ngram, _ in counter.most_common(keep)}


def crystalbleu_similarity(code_a: str, code_b: str, ignoring=None) -> float:
    """Similitud simétrica CrystalBLEU entre dos códigos, en [0, 1]."""
    tokens_a = code_tokens(code_a)
    tokens_b = code_tokens(code_b)
    if not tokens_a or not tokens_b:
        return 0.0
    forward = sentence_bleu([tokens_a], tokens_b, ignoring=ignoring)
    backward = sentence_bleu([tokens_b], tokens_a, ignoring=ignoring)
    return float(max(forward, backward))


def crystalbleu_diversity(codes, ignoring=None) -> float:
    """Diversidad media por pares: ``1 - CrystalBLEU``.

    Una población con menos de dos códigos tiene diversidad 0.
    """
    codes = list(codes)
    if len(codes) < 2:
        return 0.0
    if ignoring is None:
        ignoring = crystal_ignoring(codes)
    total = 0.0
    pairs = 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            total += 1.0 - crystalbleu_similarity(codes[i], codes[j], ignoring)
            pairs += 1
    return total / pairs if pairs else 0.0
