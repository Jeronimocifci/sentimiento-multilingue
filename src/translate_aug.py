"""Aumentación por traducción automática en GPU (PyTorch, MarianMT) -> data/train_aug.csv

Traduce oración por oración (conserva el orden -> conserva la posición del veredicto):
  - N reseñas en español (estratificadas): es->en (opus-mt-es-en) y en->fr / en->pt (opus-mt-en-ROMANCE)
  - todas las reseñas en inglés: en->fr, en->pt, en->es (opus-mt-en-ROMANCE con >>fr<<, >>pt_br<<, >>es<<)
  - todas las reseñas en francés/portugués: ->en (opus-mt-ROMANCE-en)
  - code-switching es/en: alterna oraciones originales y traducidas (p=0.5, al menos una de cada idioma)
Cada fila guarda src_id (id de la reseña original) -> train_torch.py solo usa aumentos de reseñas del fold de train.

Uso: python src/translate_aug.py --data data --n_es 4000
     (--dry_run usa traducción identidad para probar la tubería sin descargar modelos)
"""
import argparse, os, random, re, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_detect import detect_lang

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='data'); ap.add_argument('--out', default=None)
ap.add_argument('--n_es', type=int, default=4000); ap.add_argument('--n_cs', type=int, default=3000)
ap.add_argument('--bs', type=int, default=96); ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--dry_run', action='store_true')
a = ap.parse_args()
OUT = a.out or f'{a.data}/train_aug.csv'
T0 = time.time(); mins = lambda: f'{(time.time() - T0) / 60:.1f} min'
rng = random.Random(a.seed)

tr = pd.read_csv(f'{a.data}/train.csv')
tr['text'] = tr.text.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip(); tr['lang'] = tr.text.map(detect_lang)
split = lambda t: [p for p in re.split(r'(?<=[.!?…])\s+', str(t).strip()) if p]

if not a.dry_run:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def translate(sents, name, prefix=''):
    uniq = list(dict.fromkeys(sents)); pos = {s: i for i, s in enumerate(uniq)}
    if a.dry_run:
        out = [s for s in uniq]
    else:
        tok = MarianTokenizer.from_pretrained(name); model = MarianMTModel.from_pretrained(name).to(dev).eval()
        if dev == 'cuda': model = model.half()
        order = np.argsort([len(s) for s in uniq]); out = [None] * len(uniq)
        with torch.no_grad():
            for b in range(0, len(order), a.bs):
                idx = order[b:b + a.bs]
                enc = tok([prefix + uniq[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=200).to(dev)
                gen = model.generate(**enc, num_beams=1, max_new_tokens=220)
                for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)): out[i] = t.strip()
                if (b // a.bs) % 50 == 0: print(f'  {name.split("/")[-1]} {prefix.strip()} {b + len(idx)}/{len(uniq)} ({mins()})', flush=True)
        del model; torch.cuda.empty_cache() if dev == 'cuda' else None
    print(f'  {name.split("/")[-1]} {prefix.strip()}: {len(uniq)} oraciones únicas ({mins()})', flush=True)
    return [out[pos[s]] for s in sents]

def reviews(df, name, prefix, tag, lang):
    ss = [split(t) for t in df.text]; flat = [s for x in ss for s in x]; of = np.cumsum([0] + [len(x) for x in ss])
    try:
        o = translate(flat, name, prefix)
    except Exception as e:
        print(f'⚠️ {tag} omitido: {e}'); return [], None, None
    return [(r.id, ' '.join(o[of[k]:of[k + 1]]), r.label, tag, lang) for k, r in enumerate(df.itertuples())], (ss, of, flat), o

rows = []
es = tr[tr.lang == 'es']; n = min(a.n_es // 3, es.label.value_counts().min())
src = es.groupby('label').sample(n=n, random_state=a.seed).reset_index(drop=True)
print(f'Español: {len(src)} reseñas · inglés: {(tr.lang == "en").sum()} · fr/pt: {tr.lang.isin(["fr", "pt"]).sum()} ({mins()})')
r_en, meta, flat_en = reviews(src, 'Helsinki-NLP/opus-mt-es-en', '', 'mt_es2en', 'en'); rows += r_en
if flat_en is not None:
    ss, of, flat = meta
    for pre, tag, lg in (('>>fr<< ', 'mt_es2fr', 'fr'), ('>>pt_br<< ', 'mt_es2pt', 'pt')):
        try:
            o = translate(flat_en, 'Helsinki-NLP/opus-mt-en-ROMANCE', pre)
            rows += [(r.id, ' '.join(o[of[k]:of[k + 1]]), r.label, tag, lg) for k, r in enumerate(src.itertuples())]
        except Exception as e:
            print(f'⚠️ {tag} omitido: {e}')
    for k in rng.sample(range(len(src)), min(a.n_cs, len(src))):   # code-switching
        b0, b1 = of[k], of[k + 1]
        if b1 - b0 < 2: continue
        pick = [rng.random() < 0.5 for _ in range(b1 - b0)]
        if all(pick): pick[rng.randrange(b1 - b0)] = False
        if not any(pick): pick[rng.randrange(b1 - b0)] = True
        rows.append((src.id[k], ' '.join(flat_en[b0 + i] if p else flat[b0 + i] for i, p in enumerate(pick)), src.label[k], 'codeswitch', 'es-en'))
en = tr[tr.lang == 'en']
for pre, tag, lg in (('>>fr<< ', 'mt_en2fr', 'fr'), ('>>pt_br<< ', 'mt_en2pt', 'pt'), ('>>es<< ', 'mt_en2es', 'es')):
    rows += reviews(en, 'Helsinki-NLP/opus-mt-en-ROMANCE', pre, tag, lg)[0]
rows += reviews(tr[tr.lang.isin(['fr', 'pt'])], 'Helsinki-NLP/opus-mt-ROMANCE-en', '', 'mt_rom2en', 'en')[0]

aug = pd.DataFrame(rows, columns=['src_id', 'text', 'label', 'aug_type', 'lang'])
aug = aug[aug.text.str.len() > 20].reset_index(drop=True)
aug.to_csv(OUT, index=False)
print(aug.aug_type.value_counts().to_string()); print(f'✅ {len(aug)} aumentos -> {OUT} ({mins()})')
for t in aug.aug_type.unique():
    print(f'[{t}] {aug[aug.aug_type == t].text.iloc[0][:200]}')
