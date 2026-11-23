"""Promedia probabilidades de varias corridas (p. ej. 2 semillas del notebook 02) y genera una submission.

Uso:
    python src/ensemble.py runs/seed42/probs_eval.npy runs/seed7/probs_eval.npy --weights 0.5 0.5 \
        --eval data/eval.csv --out submissions/submission_ens.csv
"""
import argparse
import numpy as np
import pandas as pd

LABELS = ['negativo', 'neutral', 'positivo']

ap = argparse.ArgumentParser()
ap.add_argument('probs', nargs='+')
ap.add_argument('--weights', nargs='*', type=float)
ap.add_argument('--eval', default='data/eval.csv')
ap.add_argument('--out', default='submissions/submission_ens.csv')
a = ap.parse_args()
w = np.array(a.weights or [1.0] * len(a.probs), dtype=np.float64); w /= w.sum()
P = sum(wi * np.load(p) for wi, p in zip(w, a.probs))
ev = pd.read_csv(a.eval)
assert P.shape == (len(ev), 3), P.shape
for p in a.probs:
    print(f'{p}: acuerdo con ensemble {(np.load(p).argmax(1) == P.argmax(1)).mean():.4f}')
pd.DataFrame({'id': ev.id, 'answer': [LABELS[i] for i in P.argmax(1)]}).to_csv(a.out, index=False)
print('escrito', a.out)
