"""Entrenamiento con PyTorch (Mac Apple Silicon/MPS, GPU CUDA o CPU) — alternativa a la TPU.

Misma receta que el notebook v2: entrada en par (últimas 2 oraciones, reseña completa), LLRD, label smoothing,
warmup lineal, aumentos opcionales (train_aug*.csv de los notebooks TPU, filtrados por fold -> sin fuga).

Ejemplos (desde la raíz del repo, con data/train.csv y data/eval.csv):
    # Mac, rápido (~30-45 min): XLM-R base, holdout 10%
    python src/train_torch.py --model FacebookAI/xlm-roberta-base --epochs 3 --bs 16 --lr 3e-5
    # Mac, fuerte (~2-3 h): XLM-R large
    python src/train_torch.py --model FacebookAI/xlm-roberta-large --epochs 3 --bs 8 --lr 1e-5
    # con aumentos generados por la TPU
    python src/train_torch.py --aug_csv outputs/train_aug_v2.csv

Salidas en --out: probs_eval.npy, probs_val.npy + val_idx.npy, submission_*.csv, modelo/ (save_pretrained), log.json
"""
import argparse, json, math, os, random, re, sys, time, copy
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_detect import detect_lang

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='FacebookAI/xlm-roberta-base')
ap.add_argument('--data', default='data'); ap.add_argument('--out', default=None)
ap.add_argument('--epochs', type=int, default=3); ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--lr', type=float, default=3e-5); ap.add_argument('--llrd', type=float, default=0.925)
ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=float, default=0.1)
ap.add_argument('--smooth', type=float, default=0.05); ap.add_argument('--max_len', type=int, default=320)
ap.add_argument('--no_pair', action='store_true'); ap.add_argument('--n_last', type=int, default=2)
ap.add_argument('--folds', type=int, default=0, help='0 = holdout de --val_frac; k>1 = k-fold')
ap.add_argument('--val_frac', type=float, default=0.1)
ap.add_argument('--aug_csv', default=None); ap.add_argument('--aug_ratio', type=float, default=1.0)
ap.add_argument('--seed', type=int, default=42); ap.add_argument('--grad_accum', type=int, default=1)
ap.add_argument('--limit', type=int, default=0, help='(debug) usar solo N filas de train')
ap.add_argument('--tiny_random', action='store_true', help='(debug) modelo aleatorio diminuto, sin internet')
a = ap.parse_args()

LABELS = ['negativo', 'neutral', 'positivo']; L2I = {l: i for i, l in enumerate(LABELS)}
LANGS = ['es', 'en', 'pt', 'fr', 'es-en']
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
use_amp = dev.type == 'cuda'   # fp16 solo en CUDA; en MPS fp32 (más estable)
OUT = a.out or f"outputs/torch_{a.model.split('/')[-1]}_s{a.seed}"
os.makedirs(OUT, exist_ok=True)
T0 = time.time(); mins = lambda: f'{(time.time() - T0) / 60:.1f} min'
print(f'Dispositivo: {dev} | modelo: {a.model} | salida: {OUT}')

# ---------------- datos ----------------
tr = pd.read_csv(f'{a.data}/train.csv'); ev = pd.read_csv(f'{a.data}/eval.csv')
if a.limit: tr = tr.sample(a.limit, random_state=a.seed).reset_index(drop=True); ev = ev.head(64)
for d in (tr, ev):
    d['text'] = d.text.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip(); d['lang'] = d.text.map(detect_lang)
tr['y'] = tr.label.map(L2I)
aug = None
if a.aug_csv and os.path.exists(a.aug_csv):
    aug = pd.read_csv(a.aug_csv).dropna(subset=['text']); aug['y'] = aug.label.map(L2I); print('Aumentos:', len(aug))

split = lambda t: [p for p in re.split(r'(?<=[.!?…])\s+', str(t).strip()) if p]
def last_sents(t): return ' '.join(split(t)[-a.n_last:])[-300:]

# ---------------- modelo/tokenizer ----------------
if a.tiny_random:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaForSequenceClassification
    _t = Tokenizer(models.BPE(unk_token='<unk>')); _t.pre_tokenizer = pre_tokenizers.Whitespace()
    _t.train_from_iterator(tr.text.tolist(), trainers.BpeTrainer(vocab_size=800, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>']))
    _t.post_processor = processors.TemplateProcessing(single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>', special_tokens=[('<s>', 0), ('</s>', 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=_t, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>', cls_token='<s>', sep_token='</s>')
    make_model = lambda: XLMRobertaForSequenceClassification(XLMRobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                                                              intermediate_size=64, max_position_embeddings=a.max_len + 4, pad_token_id=1, num_labels=3))
else:
    tok = AutoTokenizer.from_pretrained(a.model)
    make_model = lambda: AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=L2I)

def collate(batch):
    texts = [b[0] for b in batch]
    if a.no_pair: enc = tok(texts, truncation=True, max_length=a.max_len, padding=True, return_tensors='pt')
    else: enc = tok([last_sents(t) for t in texts], texts, truncation='longest_first', max_length=a.max_len, padding=True, return_tensors='pt')
    enc['labels'] = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return enc

def predict(model, texts, bs=64):
    model.eval(); out = []
    order = np.argsort([len(t) for t in texts]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    st = [texts[i] for i in order]
    with torch.no_grad():
        for i in range(0, len(st), bs):
            enc = collate([(t, 0) for t in st[i:i + bs]]); enc.pop('labels')
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)[inv]

def param_groups(model):
    n_layers = model.config.num_hidden_layers; groups = {}
    for n, p in model.named_parameters():
        m = re.search(r'encoder\.layer\.(\d+)\.', n)
        f = a.llrd ** (n_layers - 1 - int(m.group(1))) if m else (a.llrd ** n_layers if 'embeddings' in n else 1.0)
        wd = 0.0 if (n.endswith('bias') or 'LayerNorm' in n) else a.wd
        groups.setdefault((f, wd), []).append(p)
    return [{'params': ps, 'lr': a.lr * f, 'weight_decay': wd} for (f, wd), ps in groups.items()]

def run(tr_idx, va_idx, tag):
    model = make_model().to(dev)
    base = list(zip(tr.text.values[tr_idx], tr.y.values[tr_idx]))
    pool = []
    if aug is not None:
        sub = aug[aug.src_id.isin(set(tr.id.values[tr_idx]))]; pool = list(zip(sub.text.values, sub.y.values))
    n_aug = min(len(pool), int(a.aug_ratio * len(base)))
    steps = math.ceil((len(base) + n_aug) / a.bs / a.grad_accum) * a.epochs
    opt = torch.optim.AdamW(param_groups(model)); sch = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp); lossf = torch.nn.CrossEntropyLoss(label_smoothing=a.smooth)
    va_texts = list(tr.text.values[va_idx]); yv = tr.y.values[va_idx]; best = (-1, None, None)
    print(f'[{tag}] {len(base)} originales + {n_aug}/{len(pool)} aumentos por época · {steps} pasos ({mins()})')
    for ep in range(a.epochs):
        model.train(); data = base + (random.sample(pool, n_aug) if n_aug else [])
        dl = DataLoader(data, batch_size=a.bs, shuffle=True, collate_fn=collate, drop_last=True)
        tot, t_ep = 0.0, time.time()
        for i, enc in enumerate(dl):
            enc = {k: v.to(dev) for k, v in enc.items()}; y = enc.pop('labels')
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                loss = lossf(model(**enc).logits.float(), y) / a.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % a.grad_accum == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
            tot += loss.item() * a.grad_accum
            if i % 200 == 0 and i:
                eta = (time.time() - t_ep) / i * (len(dl) - i) / 60
                print(f'  ep{ep + 1} {i}/{len(dl)} loss {tot / (i + 1):.4f} · ETA época {eta:.1f} min', flush=True)
        pv = predict(model, va_texts); acc = (pv.argmax(1) == yv).mean()
        per = pd.Series(pv.argmax(1) == yv).groupby(tr.lang.values[va_idx]).mean().reindex(LANGS).round(4).to_dict()
        print(f'  ep{ep + 1} loss {tot / max(1, len(dl)):.4f} val_acc {acc:.4f} {per} ({mins()})', flush=True)
        if acc > best[0]:
            best = (acc, pv, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2]); pe = predict(model, list(ev.text.values))
    return best[0], best[1], pe, model

# ---------------- folds ----------------
strat = tr.label + '_' + tr.lang
vc = strat.value_counts(); strat = strat.where(strat.map(vc) >= 10, tr.label + '_es')   # estratos raros -> estrato grande de su etiqueta
if a.folds > 1:
    splits = list(StratifiedKFold(a.folds, shuffle=True, random_state=a.seed).split(tr, strat))
else:
    i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=a.val_frac, stratify=strat, random_state=a.seed); splits = [(i_tr, i_va)]
oof = np.full((len(tr), 3), np.nan, np.float32); pe_sum = np.zeros((len(ev), 3)); accs = []; best_model = (-1, None)
for f, (i_tr, i_va) in enumerate(splits):
    acc, pv, pe, model = run(i_tr, i_va, f'fold{f}'); oof[i_va] = pv; pe_sum += pe; accs.append(float(acc))
    if acc > best_model[0]:
        best_model = (acc, f); model.save_pretrained(f'{OUT}/modelo'); tok.save_pretrained(f'{OUT}/modelo')
    del model
    if dev.type == 'mps': torch.mps.empty_cache()
    np.save(f'{OUT}/probs_eval.npy', (pe_sum / (f + 1)).astype(np.float32))
pe = pe_sum / len(splits); cov = ~np.isnan(oof[:, 0])
np.save(f'{OUT}/probs_eval.npy', pe.astype(np.float32)); np.save(f'{OUT}/probs_val.npy', oof[cov]); np.save(f'{OUT}/val_idx.npy', np.where(cov)[0])
name = f"submission_torch_{a.model.split('/')[-1]}{'' if a.no_pair else '_par'}{'_aug' if aug is not None else ''}.csv"
pd.DataFrame({'id': ev.id, 'answer': [LABELS[i] for i in pe.argmax(1)]}).to_csv(f'{OUT}/{name}', index=False)
res = tr[cov].assign(ok=oof[cov].argmax(1) == tr.y.values[cov])
log = {'args': vars(a), 'val_accs': accs, 'val_acc_mean': float(np.mean(accs)), 'por_idioma': res.groupby('lang').ok.mean().reindex(LANGS).round(4).to_dict(), 'minutos': (time.time() - T0) / 60}
json.dump(log, open(f'{OUT}/log.json', 'w'), indent=2, default=str)
print(json.dumps({k: log[k] for k in ('val_acc_mean', 'por_idioma')}, indent=2)); print(f'✅ {OUT}/{name} ({mins()})')
