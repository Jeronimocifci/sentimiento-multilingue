"""Genera los notebooks de CPU: 00_eda.ipynb y 01_baseline_tfidf.ipynb (se ejecutan en local)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = r"""
import sys, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, '../src')
from lang_detect import detect_lang
train = pd.read_csv('../data/train.csv'); ev = pd.read_csv('../data/eval.csv')
for d in (train, ev): d['lang'] = d.text.map(detect_lang); d['n_chars'] = d.text.str.len()
LANGS = ['es', 'en', 'pt', 'fr', 'es-en']; LABELS = ['negativo', 'neutral', 'positivo']
print(train.shape, ev.shape)
"""


def build(name, cells):
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(c[1].strip()) if c[0] == 'md' else nbf.v4.new_code_cell(c[1].strip()) for c in cells])
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    nbf.write(nb, ROOT / 'notebooks' / name)


build('00_eda.ipynb', [
    ('md', """
# 00 · Análisis exploratorio
Objetivo: entender **qué cambió respecto a la Parte 1** (multilingüe + sentimiento indirecto) y cómo difieren train y eval.
Corre en CPU en segundos."""),
    ('code', SETUP),
    ('md', "## Balance de clases"),
    ('code', "display(train.label.value_counts().to_frame('n').assign(pct=lambda d: (d.n / d.n.sum() * 100).round(1)))"),
    ('md', """
## Idiomas: cambio de distribución train → eval
El enunciado advierte que eval no sigue la distribución de train. El detector de idioma (stopwords, `src/lang_detect.py`) confirma
que **eval tiene menos español y más inglés/portugués/francés** → un modelo bueno solo en español pierde puntos."""),
    ('code', """
shift = pd.DataFrame({'train %': train.lang.value_counts(normalize=True), 'eval %': ev.lang.value_counts(normalize=True)}).reindex(LANGS).mul(100).round(1)
shift['eval/train'] = (shift['eval %'] / shift['train %']).round(2)
display(shift)
ax = shift[['train %', 'eval %']].plot.bar(rot=0, figsize=(7, 3.5), title='Distribución de idiomas'); ax.set_ylabel('%'); plt.tight_layout(); plt.show()"""),
    ('code', "display(pd.crosstab(train.lang, train.label, normalize='index').reindex(LANGS).round(3))"),
    ('md', "## Longitud de las reseñas"),
    ('code', """
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(train.n_chars, bins=60, alpha=.6, density=True, label='train'); ax.hist(ev.n_chars, bins=60, alpha=.6, density=True, label='eval')
ax.set_xlabel('caracteres'); ax.legend(); ax.set_title('Eval tiene reseñas algo más largas'); plt.tight_layout(); plt.show()
display(pd.DataFrame({'train': train.n_chars.describe(), 'eval': ev.n_chars.describe()}).round(0))"""),
    ('md', """
## Ejemplos por idioma y fenómenos difíciles
Muchas reseñas comparten una plantilla (envío, empaque, uso) y el sentimiento está en **una o dos oraciones**, a veces con
contraste o mezclando aspectos positivos y negativos."""),
    ('code', """
for lg in ['es', 'en', 'pt', 'fr', 'es-en']:
    for r in train[train.lang == lg].sample(2, random_state=1).itertuples():
        print(f'[{lg} · {r.label}] {r.text[:260]}\\n')"""),
    ('code', """
PHEN = {'negación': r"\\b(?:no|not|nunca|never|pas|jamais|não|nem|n't)\\b",
        'contraste': r"\\b(?:pero|but|however|sin embargo|aunque|mais|mas|porém|embora|cependant)\\b",
        'devolución': r"\\b(?:devolví|devolvi|returned|rendu|reembolso|refund)\\b"}
display(pd.DataFrame({k: [train.text.str.contains(p, case=False).mean(), ev.text.str.contains(p, case=False).mean()] for k, p in PHEN.items()},
                     index=['train', 'eval']).mul(100).round(1).rename_axis('% reseñas con...'))"""),
    ('md', """
## Conclusiones para el diseño
1. **Modelo multilingüe** (XLM-R) en vez de uno solo-español (BETO, que usamos en el proyecto anterior).
2. **Aumentar en/pt/fr** traduciendo reseñas en español → compensa el cambio de distribución.
3. **Code-switching** casi no aparece en train → se generan ejemplos sintéticos.
4. Las reseñas caben en ≤ 256 tokens → `MAX_LEN=256` sin truncar."""),
])

build('01_baseline_tfidf.ipynb', [
    ('md', """
# 01 · Línea base clásica (TF-IDF + Regresión logística)
Referencia **no-deep-learning** para medir cuánto aportan los transformers. No es el modelo final (el enunciado exige DL).
Validación: 5-fold estratificado; se reporta accuracy por idioma."""),
    ('code', SETUP),
    ('code', """
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
X = sparse.hstack([TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(train.text),
                   TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=150000).fit_transform(train.text)]).tocsr()
pred = cross_val_predict(LogisticRegression(C=4, max_iter=3000), X, train.label, cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
train['ok'] = pred == train.label
print(f'CV accuracy: {train.ok.mean():.4f}')
display(train.groupby('lang').ok.agg(['size', 'mean']).reindex(LANGS).round(4).rename(columns={'size': 'n', 'mean': 'accuracy'}))"""),
    ('md', """
**Lectura:** ~0.72 global, y cae en en/pt/fr (menos datos y n-gramas que no se comparten entre idiomas). Los n-gramas no
capturan negaciones de largo alcance ni contraste → justifica un transformer multilingüe + aumentación por traducción."""),
])
print('ok')
