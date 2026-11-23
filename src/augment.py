"""Funciones de aumentación de datos de texto (copiadas también en el notebook final).

- split_sentences: segmenta una reseña en oraciones (para traducir oración por oración).
- code_switch: mezcla oraciones originales (es) con su traducción (en) -> code-switching sintético.
- eda_noise: borrado aleatorio + intercambio de palabras adyacentes, protegiendo negaciones y
  conectores de contraste (cambiarían el sentimiento).
"""
import re
import random

PROTECTED = set("""no not nunca never jamás nada nothing ni nor sin without pas jamais rien não nem nunca
pero but sin_embargo however aunque although mais mas porém embora devolví returned devolvi rendu""".split())


def split_sentences(text):
    parts = re.split(r'(?<=[.!?…])\s+', str(text).strip())
    return [p for p in parts if p]


def code_switch(src_sents, tgt_sents, rng, p=0.5):
    """Alterna oraciones entre idioma original y traducción, garantizando al menos una de cada."""
    n = min(len(src_sents), len(tgt_sents))
    if n < 2:
        return None
    pick = [rng.random() < p for _ in range(n)]
    if all(pick):
        pick[rng.randrange(n)] = False
    if not any(pick):
        pick[rng.randrange(n)] = True
    return ' '.join(tgt_sents[i] if pick[i] else src_sents[i] for i in range(n))


def eda_noise(text, rng, p_del=0.1, n_swap=1):
    words = str(text).split()
    if len(words) < 6:
        return text
    kept = [w for w in words if re.sub(r'\W', '', w.lower()) in PROTECTED or rng.random() > p_del]
    for _ in range(n_swap):
        if len(kept) > 3:
            i = rng.randrange(len(kept) - 1)
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
    return ' '.join(kept)
