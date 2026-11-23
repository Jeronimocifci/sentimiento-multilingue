"""TF-IDF posicional + LogReg (modelo clásico auxiliar, NO es el modelo DL entregable).

Hallazgo de EDA: la última oración concentra el veredicto (solo-última-oración 0.758 > reseña completa 0.699).
Features: n-gramas de la reseña completa, última, penúltima y primera oración + char n-gramas (completa y última).
CV 5-fold = 0.799. Guarda probs OOF (para stacking con el transformer) y probs de eval.

Uso: python src/tfidf_posicional.py --data data --out outputs
"""
import argparse, os, re
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

LABELS = ['negativo', 'neutral', 'positivo']
ap = argparse.ArgumentParser(); ap.add_argument('--data', default='data'); ap.add_argument('--out', default='outputs'); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tr, ev = pd.read_csv(f'{a.data}/train.csv'), pd.read_csv(f'{a.data}/eval.csv')
split = lambda t: [p for p in re.split(r'(?<=[.!?…])\s+', str(t).strip()) if p]
def parts(df):
    S = df.text.map(split)
    return {'full': S.map(' '.join), 'last': S.map(lambda s: s[-1]), 'last2': S.map(lambda s: s[-2] if len(s) > 1 else ''), 'first': S.map(lambda s: s[0])}
P_tr, P_ev = parts(tr), parts(ev)
vecs = [(k, TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)) for k in P_tr] + \
       [(k, TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100000)) for k in ('full', 'last')]
X_tr = sparse.hstack([v.fit_transform(P_tr[k]) for k, v in vecs]).tocsr()
X_ev = sparse.hstack([v.transform(P_ev[k]) for k, v in vecs]).tocsr()
y = tr.label.map({l: i for i, l in enumerate(LABELS)}).values
oof, pe = np.zeros((len(tr), 3)), np.zeros((len(ev), 3))
for f, (i, j) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X_tr, y)):
    m = LogisticRegression(C=4, max_iter=4000).fit(X_tr[i], y[i]); oof[j] = m.predict_proba(X_tr[j]); pe += m.predict_proba(X_ev) / 5
print(f'CV accuracy: {(oof.argmax(1) == y).mean():.4f}')
np.save(f'{a.out}/probs_oof_tfidfpos.npy', oof.astype(np.float32)); np.save(f'{a.out}/probs_eval_tfidfpos.npy', pe.astype(np.float32))
pd.DataFrame({'id': ev.id, 'answer': [LABELS[k] for k in pe.argmax(1)]}).to_csv(f'{a.out}/submission_tfidf_posicional.csv', index=False)
print('ok')
