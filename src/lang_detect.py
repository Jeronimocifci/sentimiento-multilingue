"""Detección de idioma ligera (sin dependencias) basada en stopwords.

Devuelve 'es', 'en', 'pt', 'fr' o 'es-en' (code-switching español/inglés).
Se usa para estratificar los folds y para el análisis de errores por idioma.
"""
import re

STOP = {
    'es': set('el la los las de del que y en un una es por con para lo pero muy me se mi su al como más esta este está fue ya sí porque también hace'.split()),
    'en': set('the and is it to of i was for my but with this that very not have has had so just after it\'s they you are on at its'.split()),
    'pt': set('o os as do da dos das que e em um uma é não para com mas muito eu pra meu minha isso ele ela foi já tem também bem'.split()),
    'fr': set('le les de des du que et en un une est pas pour avec mais très je ce il elle c\'est j\'ai au aux sur qui ne plus'.split()),
}
_WORD = re.compile(r"[a-záéíóúñãõçâêôàèùœüï']+")


def lang_scores(text):
    words = _WORD.findall(str(text).lower())
    n = max(len(words), 1)
    return {l: sum(w in s for w in words) / n for l, s in STOP.items()}


def detect_lang(text, cs_thr=0.05):
    s = lang_scores(text)
    if s['es'] >= cs_thr and s['en'] >= cs_thr:
        return 'es-en'
    return max(s, key=s.get)
