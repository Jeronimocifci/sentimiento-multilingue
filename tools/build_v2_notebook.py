"""Genera notebooks/03_tpu_v2_max_accuracy.ipynb (v2: entrada en par, más traducción, LLRD, stacking, presupuesto de tiempo).

Se mantiene como script para poder versionar y regenerar el notebook de forma reproducible:
    python tools/build_main_notebook.py
"""
import nbformat as nbf
from pathlib import Path

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Sentimiento multilingüe — v2 · máximo accuracy en TPU (Parte 2)
**Curso Aprendizaje de Máquina 2026-20 · Uniandes** · 3 clases (`negativo` / `neutral` / `positivo`) · métrica: **accuracy**

### Novedades v2 (cada una justificada con evidencia)
| Cambio | Evidencia |
|---|---|
| **Entrada en par**: `últimas 2 oraciones </s></s> reseña completa` | EDA: con TF-IDF, solo la última oración (0.758) predice mejor que la reseña completa (0.699); features posicionales suben el clásico de 0.717 a 0.798 |
| **Traducción en todas las direcciones**: es→en→fr/pt, en→fr/pt/es, fr/pt→en | en/fr/pt son los idiomas más débiles (~0.68–0.71) y pesan más en eval |
| **Muestreo dinámico de aumentos**: cada época usa todos los originales + una muestra distinta de aumentos (1.5×) | Más diversidad sin que los datos sintéticos dominen ni multipliquen el tiempo |
| **LLRD** (tasa de aprendizaje decreciente por capa, 0.925) | Técnica estándar de fine-tuning de transformers grandes (Sun et al. 2019; Zhang et al. 2021) |
| **Stacking OOF** con TF-IDF posicional | Familias con errores distintos; pesos aprendidos sobre OOF, no sobre el leaderboard |
| **Presupuesto de tiempo** | Si no alcanza para los 5 folds, se detiene y promedia los folds terminados |

### Enfoque
| Componente | Elección | Por qué |
|---|---|---|
| Modelo base | **XLM-RoBERTa large** (`FacebookAI/xlm-roberta-large`, MIT) | Preentrenado en 100 idiomas (CC-100): cubre es/en/pt/fr y code-switching con un solo modelo |
| Arquitectura | Transformer encoder + cabeza de clasificación, *fine-tuning* completo | Deep learning + transferencia de aprendizaje (requisitos 4.1 y 4.2) |
| Aumentación | **(1)** traducción automática es→en→fr/pt, **(2)** code-switching sintético es/en, **(3)** ruido EDA protegiendo negaciones | Eval tiene **más** en/pt/fr que train; la aumentación reequilibra idiomas |
| Validación | K-fold estratificado por **etiqueta × idioma**, OOF para análisis | Estimación honesta por idioma |
| Hardware | **TPU v5e-8** con JAX/Flax + `pmap` (8 cores), bf16 | Ver lecciones del proyecto anterior (TF no soporta v5e en Kaggle; torch_xla se colgaba) |

### Cumplimiento del enunciado
- ✅ Deep learning (transformer) · ✅ modelo base y fuente declarados
- ✅ Aumentación + **comparación con/sin** (sección 4, misma partición) · ✅ datos sintéticos descritos en detalle (sección 2)
- ✅ **Eval nunca se usa para entrenar** (ni pseudo-labels, ni traducción de eval) · ✅ análisis de errores por idioma (sección 6)
- ✅ Modelo guardado (Flax `save_pretrained` + conversión a PyTorch) · ✅ semillas fijas → replicable

### Cómo correr en Kaggle
1. *Add Input* → datos de la competencia. 2. *Settings → Accelerator →* **TPU v5e-8**, **Internet On**.
3. **Save Version → Save & Run All (Commit)**. Tiempo estimado: **~4–6 h** (tope `TIME_BUDGET_H`).
""")

code(r"""
# Dependencias: transformers 5.x eliminó Flax -> forzamos 4.x. No tocamos jax (rompería la TPU).
import subprocess, sys, importlib.util
from importlib.metadata import version, PackageNotFoundError
def _pip(*args): subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args], check=False)
try:
    if int(version('transformers').split('.')[0]) >= 5: _pip('transformers>=4.44,<5')
except PackageNotFoundError:
    _pip('transformers>=4.44,<5')
_jax = [f'jax=={version("jax")}'] if importlib.util.find_spec('jax') else []
for mod in ['flax', 'optax', 'sentencepiece', 'sacremoses']:
    if importlib.util.find_spec(mod) is None: _pip(mod, *_jax)
print('transformers', version('transformers'))
""")

code(r"""
import os, re, gc, time, json, random, functools
from collections import Counter
SMOKE = os.environ.get('SMOKE', '0') == '1'   # prueba rápida en CPU con modelos aleatorios (solo desarrollo)

import numpy as np, pandas as pd
import jax, jax.numpy as jnp, optax, flax
from flax import jax_utils, traverse_util, serialization
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

NUM_DEVICES = jax.device_count()
print('JAX', jax.__version__, '| backend:', jax.default_backend(), '| devices:', NUM_DEVICES)
if not SMOKE:
    assert jax.default_backend() == 'tpu', 'Activa Settings → Accelerator → TPU v5e-8'
print('✅ listo')
""")

code(r"""
# ============================ CONFIGURACIÓN ============================
BACKBONE   = 'FacebookAI/xlm-roberta-large'     # https://huggingface.co/FacebookAI/xlm-roberta-large (MIT)
MT_ES_EN   = 'Helsinki-NLP/opus-mt-es-en'       # https://huggingface.co/Helsinki-NLP/opus-mt-es-en (Apache-2.0)
MT_EN_ROM  = 'Helsinki-NLP/opus-mt-en-ROMANCE'  # https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE (tokens >>fr<<, >>pt_br<<)

SEED          = 42
N_FOLDS       = 5
EPOCHS        = 3           # con 2.5x datos por época (originales + aumentos), 3 épocas bastan
LR            = 1.5e-5      # LR de la capa superior; las inferiores decaen con LLRD
LLRD          = 0.925       # capa k recibe LR * LLRD**(24-1-k); embeddings LR * LLRD**24
USE_PAIR      = True        # entrada (últimas N_LAST oraciones, reseña completa)
N_LAST        = 2
WARMUP        = 0.1
WEIGHT_DECAY  = 0.01
LABEL_SMOOTH  = 0.05
MAX_LEN       = 320         # par (últimas 2 oraciones + completa) < 320 tokens
BUCKETS       = (192, 320)  # longitudes fijas -> solo 2 compilaciones XLA
PER_DEV_BATCH = 2           # batch global = 2 x 8 cores = 16 (lección notebooks 12/16: batch chico generaliza mejor)
EVAL_PER_DEV  = 32

# Aumentación
N_TRANSLATE   = 8000        # reseñas en español (estratificadas) -> en -> fr, pt
N_CODESWITCH  = 4000        # reseñas code-switching es/en sintéticas
AUG_RATIO     = 1.5         # por época: todos los originales + AUG_RATIO x (n originales) aumentos muestreados
MT_ROM_EN     = 'Helsinki-NLP/opus-mt-ROMANCE-en'  # fr/pt -> en
EDA_FRAC      = 0.3         # fracción de originales con ruido EDA
MT_PER_DEV    = 32          # batch por core para traducir
MT_BUCKETS    = (64, 128)

RUN_ABLATION   = True       # fold 0: sin aug vs con aug (requisito del enunciado)
SAVE_ALL_FOLDS = True       # guarda los pesos (bf16) de los folds -> reproduce la submission exacta
TIME_BUDGET_H  = 8.0        # Kaggle corta sesiones TPU a las ~9 h
STACK_TFIDF    = True       # stacking OOF con TF-IDF posicional

if SMOKE:   # parámetros diminutos para validar el flujo en CPU
    N_FOLDS, EPOCHS, N_TRANSLATE, N_CODESWITCH, MT_PER_DEV, EVAL_PER_DEV = 3, int(os.environ.get('SMOKE_EP', 1)), 40, 20, 4, 4
    LR = float(os.environ.get('SMOKE_LR', LR))
    TIME_BUDGET_H = float(os.environ.get('SMOKE_TB', TIME_BUDGET_H))

GLOBAL_BATCH = PER_DEV_BATCH * NUM_DEVICES
LABELS = ['negativo', 'neutral', 'positivo']; L2I = {l: i for i, l in enumerate(LABELS)}; NUM_LABELS = 3
OUT = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.abspath('outputs')
os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
START = time.time()
def mins(): return f'{(time.time() - START) / 60:.1f} min'
print(f'Batch global {GLOBAL_BATCH} | salida en {OUT}')
""")

md(r"""
## 1. Datos y detección de idioma
El idioma no viene en los datos. Usamos un detector ligero por *stopwords* (es/en/pt/fr) que además marca
`es-en` cuando ambos idiomas tienen presencia fuerte (code-switching). Sirve para **estratificar los folds** y para el
**análisis por idioma**; el modelo en sí no recibe el idioma.
""")

code(r"""
import glob
cands = [p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True)] + glob.glob('data/train.csv') + glob.glob('../data/train.csv')
assert cands, 'Conecta los datos de la competencia (Add Input)'
DATA_DIR = os.path.dirname(cands[0]); print('Datos:', DATA_DIR)
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
eval_df  = pd.read_csv(f'{DATA_DIR}/eval.csv')
if SMOKE:
    train_df = train_df.sample(int(os.environ.get('SMOKE_N', 400)), random_state=SEED).reset_index(drop=True); eval_df = eval_df.head(100)

STOP = {
    'es': set('el la los las de del que y en un una es por con para lo pero muy me se mi su al como más esta este está fue ya sí porque también hace'.split()),
    'en': set("the and is it to of i was for my but with this that very not have has had so just after it's they you are on at its".split()),
    'pt': set('o os as do da dos das que e em um uma é não para com mas muito eu pra meu minha isso ele ela foi já tem também bem'.split()),
    'fr': set("le les de des du que et en un une est pas pour avec mais très je ce il elle c'est j'ai au aux sur qui ne plus".split()),
}
_W = re.compile(r"[a-záéíóúñãõçâêôàèùœüï']+")
def detect_lang(text, cs_thr=0.05):
    w = _W.findall(str(text).lower()); n = max(len(w), 1)
    s = {l: sum(x in st for x in w) / n for l, st in STOP.items()}
    return 'es-en' if (s['es'] >= cs_thr and s['en'] >= cs_thr) else max(s, key=s.get)

for df in (train_df, eval_df):
    df['text'] = df['text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['lang'] = df['text'].map(detect_lang)
train_df['y'] = train_df['label'].map(L2I).astype(np.int32)
LANGS = ['es', 'en', 'pt', 'fr', 'es-en']

shift = pd.DataFrame({'train %': train_df.lang.value_counts(normalize=True), 'eval %': eval_df.lang.value_counts(normalize=True)}).reindex(LANGS).mul(100).round(1)
print('Distribución de idiomas (cambio de distribución train → eval):'); display(shift)
display(pd.crosstab(train_df.lang, train_df.label, normalize='index').reindex(LANGS).round(3))
""")

md(r"""
## 2. Aumentación de datos (datos sintéticos)
Tres estrategias, todas aplicadas **solo a train** y siempre ligadas a su reseña de origen (`src_id`), de modo que en
cada fold **solo se usan aumentos de reseñas del fold de entrenamiento** (sin fuga hacia validación):

1. **Traducción automática, oración por oración** (conserva el orden de las oraciones → conserva la posición del
   veredicto). Decodificación *greedy* con `pmap` sobre los 8 cores; oraciones deduplicadas.
   - `N_TRANSLATE` reseñas en español (estratificadas): es→en (`opus-mt-es-en`) y en→fr / en→pt (`opus-mt-en-ROMANCE`).
   - **Todas** las reseñas en inglés: en→fr, en→pt, en→es (`opus-mt-en-ROMANCE`, `>>fr<<`, `>>pt_br<<`, `>>es<<`).
   - **Todas** las reseñas en francés y portugués: →en (`opus-mt-ROMANCE-en`).
   La etiqueta se conserva (la traducción preserva la polaridad global).
2. **Code-switching sintético es/en**: con las oraciones alineadas (original es + su traducción en) se alterna
   aleatoriamente oración por oración (p=0.5, al menos una de cada idioma).
3. **Ruido EDA**: borrado aleatorio de palabras (10%) + 1 intercambio de palabras adyacentes, **protegiendo
   negaciones y conectores de contraste** (*no, not, pas, não, pero, but, mais…*), que cambiarían el sentimiento.
""")

code(r"""
from transformers import AutoTokenizer, FlaxMarianMTModel, MarianConfig

def split_sentences(text):
    return [p for p in re.split(r'(?<=[.!?…])\s+', str(text).strip()) if p]

def load_mt(name):
    if SMOKE:   # modelo aleatorio diminuto (sin internet)
        tok = SMOKE_TOK
        cfg = MarianConfig(vocab_size=len(tok), d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                           decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=256,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, decoder_start_token_id=tok.pad_token_id)
        return tok, FlaxMarianMTModel(cfg, seed=0)
    tok = AutoTokenizer.from_pretrained(name)
    try:    model = FlaxMarianMTModel.from_pretrained(name)
    except Exception: model = FlaxMarianMTModel.from_pretrained(name, from_pt=True)
    return tok, model

def translate(sentences, name, prefix=''):
    # Traduce una lista de oraciones con pmap (8 cores). Ordena por longitud y usa buckets fijos.
    tok, model = load_mt(name)
    params = jax_utils.replicate(model.params)
    max_out = MT_BUCKETS[-1] + 32
    @functools.partial(jax.pmap, axis_name='d')
    def p_gen(params, ids, mask):
        return model.generate(ids, attention_mask=mask, params=params, max_length=max_out, num_beams=1).sequences
    gb = MT_PER_DEV * NUM_DEVICES
    uniq = list(dict.fromkeys(sentences)); pos = {s: i for i, s in enumerate(uniq)}   # muchas oraciones se repiten
    texts = [prefix + s for s in uniq]
    enc = tok(texts, truncation=True, max_length=MT_BUCKETS[-1], padding='max_length', return_tensors='np')
    ids, mask = enc['input_ids'].astype(np.int32), enc['attention_mask'].astype(np.int32)
    lens = mask.sum(1); order = np.argsort(lens); out = [None] * len(texts)
    for b in range(0, len(order), gb):
        idx = order[b:b + gb]; L = next(x for x in MT_BUCKETS if x >= lens[idx].max())
        bi, bm = np.full((gb, L), tok.pad_token_id, np.int32), np.zeros((gb, L), np.int32)
        bi[:len(idx)], bm[:len(idx)] = ids[idx, :L], mask[idx, :L]
        bm[len(idx):, 0] = 1   # filas de relleno
        seqs = np.asarray(p_gen(params, bi.reshape(NUM_DEVICES, -1, L), bm.reshape(NUM_DEVICES, -1, L))).reshape(gb, -1)
        for j, t in zip(idx, tok.batch_decode(seqs[:len(idx)], skip_special_tokens=True)): out[j] = t.strip()
        if (b // gb) % 20 == 0: print(f'  {name.split("/")[-1]}{prefix.strip()}: {b + len(idx)}/{len(texts)} ({mins()})')
    del params, model; gc.collect()
    print(f'  {len(uniq)} oraciones únicas traducidas ({mins()})')
    return [out[pos[s]] for s in sentences]
""")

code(r"""
if SMOKE:   # tokenizer diminuto entrenado en local (solo para la prueba en CPU sin internet)
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    _t = Tokenizer(models.BPE(unk_token='<unk>')); _t.pre_tokenizer = pre_tokenizers.Whitespace()
    _t.train_from_iterator(train_df.text.tolist(), trainers.BpeTrainer(vocab_size=800, special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>']))
    _t.post_processor = processors.TemplateProcessing(single='<s> $A </s>', pair='<s> $A </s> </s> $B </s>', special_tokens=[('<s>', 0), ('</s>', 2)])
    SMOKE_TOK = PreTrainedTokenizerFast(tokenizer_object=_t, bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>', mask_token='<mask>', cls_token='<s>', sep_token='</s>')

AUG_CSV = f'{OUT}/train_aug_v2.csv'
prev = glob.glob('/kaggle/input/**/train_aug_v2.csv', recursive=True)
if prev or os.path.exists(AUG_CSV):
    aug_df = pd.read_csv(prev[0] if prev else AUG_CSV); print('Aumentos reutilizados:', len(aug_df))
else:
    rng = random.Random(SEED)
    es_pool = train_df[train_df.lang == 'es']
    src = es_pool.groupby('label').sample(n=min(N_TRANSLATE // 3, es_pool.label.value_counts().min()), random_state=SEED).reset_index(drop=True)
    sents_es = [split_sentences(t) for t in src.text]
    flat = [s for ss in sents_es for s in ss]; offs = np.cumsum([0] + [len(ss) for ss in sents_es])
    print(f'Traduciendo {len(src)} reseñas = {len(flat)} oraciones ({mins()})')
    flat_en = translate(flat, MT_ES_EN)
    flat_fr = translate(flat_en, MT_EN_ROM, '>>fr<< ')
    flat_pt = translate(flat_en, MT_EN_ROM, '>>pt_br<< ')
    rows = []
    for k, r in enumerate(src.itertuples()):
        a, b = offs[k], offs[k + 1]
        for lang, fl in (('en', flat_en), ('fr', flat_fr), ('pt', flat_pt)):
            rows.append((r.id, ' '.join(fl[a:b]), r.label, f'mt_{lang}', lang))

    def translate_reviews(df, name, prefix, tag, lang):
        # Traduce reseñas completas oración por oración; si el modelo falla, se omite sin romper el notebook.
        try:
            ss = [split_sentences(t) for t in df.text]; fl = [s for x in ss for s in x]; of = np.cumsum([0] + [len(x) for x in ss])
            out = translate(fl, name, prefix)
            return [(r.id, ' '.join(out[of[k]:of[k + 1]]), r.label, tag, lang) for k, r in enumerate(df.itertuples())]
        except Exception as e:
            print(f'⚠️ {tag} omitido: {e}'); return []
    en_src = train_df[train_df.lang == 'en']; frpt_src = train_df[train_df.lang.isin(['fr', 'pt'])]
    rows += translate_reviews(en_src, MT_EN_ROM, '>>fr<< ', 'mt_en2fr', 'fr')
    rows += translate_reviews(en_src, MT_EN_ROM, '>>pt_br<< ', 'mt_en2pt', 'pt')
    rows += translate_reviews(en_src, MT_EN_ROM, '>>es<< ', 'mt_en2es', 'es')
    rows += translate_reviews(frpt_src, MT_ROM_EN, '', 'mt_rom2en', 'en')
    # code-switching es/en a partir de pares alineados
    for k in rng.sample(range(len(src)), min(N_CODESWITCH, len(src))):
        a, b = offs[k], offs[k + 1]; r = src.iloc[k]
        t = None
        if b - a >= 2:
            pick = [rng.random() < 0.5 for _ in range(b - a)]
            if all(pick): pick[rng.randrange(b - a)] = False
            if not any(pick): pick[rng.randrange(b - a)] = True
            t = ' '.join(flat_en[a + i] if p else flat[a + i] for i, p in enumerate(pick))
        if t: rows.append((r.id, t, r.label, 'codeswitch', 'es-en'))
    aug_df = pd.DataFrame(rows, columns=['src_id', 'text', 'label', 'aug_type', 'lang'])
    aug_df = aug_df[aug_df.text.str.len() > 20].reset_index(drop=True)
    aug_df.to_csv(AUG_CSV, index=False)
    print(f'Aumentos generados: {len(aug_df)} ({mins()})')

# EDA (barato, se genera siempre igual por la semilla)
PROTECTED = set("no not nunca never jamás nada nothing ni nor sin without pas jamais rien não nem pero but aunque although mais mas porém embora".split())
def eda_noise(text, rng, p_del=0.1):
    w = text.split()
    if len(w) < 6: return text
    k = [x for x in w if re.sub(r'\W', '', x.lower()) in PROTECTED or rng.random() > p_del]
    i = rng.randrange(len(k) - 1); k[i], k[i + 1] = k[i + 1], k[i]
    return ' '.join(k)
rng = random.Random(SEED + 1)
eda_src = train_df.sample(frac=EDA_FRAC, random_state=SEED)
eda_df = pd.DataFrame({'src_id': eda_src.id.values, 'text': [eda_noise(t, rng) for t in eda_src.text], 'label': eda_src.label.values, 'aug_type': 'eda', 'lang': eda_src.lang.values})
aug_all = pd.concat([aug_df, eda_df], ignore_index=True); aug_all['y'] = aug_all['label'].map(L2I).astype(np.int32)
display(aug_all.aug_type.value_counts().rename('n'))
for t in aug_all.aug_type.unique():
    s = aug_all[aug_all.aug_type == t]
    if len(s): print(f'[{t}] ({s.label.iloc[0]}) {s.text.iloc[0][:220]}')
""")

md(r"""
## 3. Modelo y entrenamiento en TPU (JAX/Flax + `pmap`)
Optimizaciones para TPU v5e-8:
- **`pmap` sobre 8 cores** con promedio de gradientes (`pmean`); parámetros replicados, batch global 16 (2 por core).
- **bf16** en el cómputo (los pesos y el optimizador quedan en fp32 → estable).
- **Buckets de longitud fijos (128/256)** + lotes agrupados por longitud: menos padding y solo 2 compilaciones XLA.
- **El learning rate entra como argumento** del paso de entrenamiento (AdamW manual con `optax.chain`): la función se
  compila **una sola vez** y se reutiliza en todos los folds y configuraciones (en el proyecto 1 cada fold recompilaba).
- `donate_argnums` para reutilizar memoria HBM de parámetros/estado del optimizador.
- **LLRD** implementado como un árbol de factores constante que multiplica la actualización (sin recompilar).
- **Entrada en par**: el tokenizer recibe `(últimas 2 oraciones, reseña completa)` (primer segmento con tope de 300 caracteres),
  así el veredicto final nunca se pierde y el modelo lo ve dos veces (cerca del token `<s>` y en contexto).
""")

code(r"""
from transformers import FlaxAutoModelForSequenceClassification, XLMRobertaConfig, FlaxXLMRobertaForSequenceClassification

if SMOKE:
    tokenizer = SMOKE_TOK
    cfg = XLMRobertaConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                           max_position_embeddings=MAX_LEN + 4, pad_token_id=tokenizer.pad_token_id, num_labels=NUM_LABELS)
    model = FlaxXLMRobertaForSequenceClassification(cfg, seed=SEED, dtype=jnp.bfloat16)
else:
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = FlaxAutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=NUM_LABELS, dtype=jnp.bfloat16,
                                                                   id2label=dict(enumerate(LABELS)), label2id=L2I)
INIT_PARAMS = jax.device_get(model.params)   # copia en host para reiniciar cada fold
print(f'Parámetros: {sum(x.size for x in jax.tree_util.tree_leaves(INIT_PARAMS)) / 1e6:.0f} M ({mins()})')

def last_sents(t, n=N_LAST, max_chars=300):
    return ' '.join(split_sentences(t)[-n:])[-max_chars:]   # tope por si la reseña no tiene puntuación

def encode(texts):
    texts = [str(t) for t in texts]
    if USE_PAIR:
        e = tokenizer([last_sents(t) for t in texts], texts, truncation='longest_first', max_length=MAX_LEN, padding='max_length', return_tensors='np')
    else:
        e = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding='max_length', return_tensors='np')
    return e['input_ids'].astype(np.int32), e['attention_mask'].astype(np.int32)

tr_ids, tr_mask = encode(train_df.text); ev_ids, ev_mask = encode(eval_df.text); au_ids, au_mask = encode(aug_all.text)
lens = tr_mask.sum(1); print('Tokens por reseña: p50 %d · p99 %d · max %d' % (np.median(lens), np.percentile(lens, 99), lens.max()))
print(f'% truncadas: {(lens >= MAX_LEN).mean():.2%}')

# --- AdamW con LR externo: update = -lr * (adam(g) + wd * p) ---
def decay_mask(p):
    flat = traverse_util.flatten_dict(p)
    return traverse_util.unflatten_dict({k: (k[-1] not in ('bias', 'scale')) for k in flat})
TX = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.add_decayed_weights(WEIGHT_DECAY, decay_mask))

N_LAYERS = model.config.num_hidden_layers
def llrd_factors(p):
    flat = traverse_util.flatten_dict(p); out = {}
    for k in flat:
        if 'layer' in k: f = LLRD ** (N_LAYERS - 1 - int(k[k.index('layer') + 1]))
        elif 'embeddings' in k: f = LLRD ** N_LAYERS
        else: f = 1.0   # cabeza de clasificación
        out[k] = np.float32(f)
    return traverse_util.unflatten_dict(out)
LR_FACTORS = llrd_factors(INIT_PARAMS)
print('LLRD: embeddings x%.3f · capa 0 x%.3f · capa %d x1.0' % (LLRD ** N_LAYERS, LLRD ** (N_LAYERS - 1), N_LAYERS - 1))

@functools.partial(jax.pmap, axis_name='d', donate_argnums=(0, 1))
def p_train_step(params, opt_state, ids, mask, y, lr, rng):
    def loss_fn(p):
        logits = model(input_ids=ids, attention_mask=mask, params=p, train=True, dropout_rng=rng).logits.astype(jnp.float32)
        target = optax.smooth_labels(jax.nn.one_hot(y, NUM_LABELS), LABEL_SMOOTH)
        return optax.softmax_cross_entropy(logits, target).mean()
    loss, g = jax.value_and_grad(loss_fn)(params)
    g, loss = jax.lax.pmean(g, 'd'), jax.lax.pmean(loss, 'd')
    upd, opt_state = TX.update(g, opt_state, params)
    upd = jax.tree_util.tree_map(lambda u, f: -lr * f * u, upd, LR_FACTORS)
    return optax.apply_updates(params, upd), opt_state, loss

@functools.partial(jax.pmap, axis_name='d')
def p_eval_step(params, ids, mask):
    return jax.nn.softmax(model(input_ids=ids, attention_mask=mask, params=params, train=False).logits.astype(jnp.float32), -1)

def bucket_len(m): return next(b for b in BUCKETS if b >= m)

def make_batches(lengths, bs, rng):
    # Lotes agrupados por longitud (dentro de bloques de 50 lotes) y barajados -> menos padding.
    idx = rng.permutation(len(lengths)); idx = idx[: (len(idx) // bs) * bs]
    chunks = []
    for c in range(0, len(idx), bs * 50):
        blk = idx[c:c + bs * 50]; blk = blk[np.argsort(lengths[blk])]
        chunks += [blk[i:i + bs] for i in range(0, len(blk), bs)]
    rng.shuffle(chunks)
    return chunks

def predict(params_r, ids, mask):
    gb = EVAL_PER_DEV * NUM_DEVICES; lens_ = mask.sum(1); order = np.argsort(lens_)
    out = np.zeros((len(ids), NUM_LABELS), np.float32)
    for b in range(0, len(order), gb):
        idx = order[b:b + gb]; L = bucket_len(lens_[idx].max())
        bi, bm = np.zeros((gb, L), np.int32), np.zeros((gb, L), np.int32); bm[:, 0] = 1
        bi[:len(idx)], bm[:len(idx)] = ids[idx, :L], mask[idx, :L]
        pr = p_eval_step(params_r, bi.reshape(NUM_DEVICES, -1, L), bm.reshape(NUM_DEVICES, -1, L))
        out[idx] = np.asarray(pr).reshape(gb, -1)[:len(idx)]
    return out

def run_fold(tr_idx, va_idx, use_aug, aug_types=None, tag='', lr=LR, seed=SEED, keep_params=False):
    # Entrena un fold. Devuelve (mejor acc val, probs val, probs eval, params host del mejor epoch).
    ids, mask, y = tr_ids[tr_idx], tr_mask[tr_idx], train_df.y.values[tr_idx]
    n_orig = len(ids); aug_pool = np.array([], dtype=int)
    if use_aug:
        keep = aug_all.src_id.isin(set(train_df.id.values[tr_idx]))      # solo aumentos de reseñas del fold de train
        if aug_types: keep &= aug_all.aug_type.isin(aug_types)
        aug_pool = np.where(keep.values)[0]
        ids, mask = np.concatenate([ids, au_ids[aug_pool]]), np.concatenate([mask, au_mask[aug_pool]])
        y = np.concatenate([y, aug_all.y.values[aug_pool]])
    n_aug_ep = min(len(aug_pool), int(AUG_RATIO * n_orig))   # aumentos muestreados por época (distintos cada época)
    lengths = mask.sum(1); rng_np = np.random.RandomState(seed)
    steps_ep = (n_orig + n_aug_ep) // GLOBAL_BATCH; total = steps_ep * EPOCHS; warm = max(1, int(WARMUP * total))
    lr_at = lambda s: lr * min((s + 1) / warm, max(0.0, (total - s) / max(1, total - warm)))
    params = jax_utils.replicate(INIT_PARAMS); opt_state = jax_utils.replicate(TX.init(INIT_PARAMS))
    key = jax.random.PRNGKey(seed); step = 0; best = (-1, None, None, None)
    print(f'[{tag}] {n_orig} originales + {n_aug_ep}/{len(aug_pool)} aumentos por época · {steps_ep} pasos/época ({mins()})')
    for ep in range(EPOCHS):
        tl, nb = 0.0, 0
        ep_idx = np.concatenate([np.arange(n_orig), n_orig + rng_np.choice(len(aug_pool), n_aug_ep, replace=False)]) if n_aug_ep else np.arange(n_orig)
        for bb in make_batches(lengths[ep_idx], GLOBAL_BATCH, rng_np):
            bidx = ep_idx[bb]
            L = bucket_len(lengths[bidx].max()); key, sk = jax.random.split(key)
            sh = lambda a: a.reshape(NUM_DEVICES, -1, *a.shape[1:])
            params, opt_state, loss = p_train_step(params, opt_state, sh(ids[bidx, :L]), sh(mask[bidx, :L]), sh(y[bidx]),
                                                   jnp.full((NUM_DEVICES,), lr_at(step), jnp.float32), jax.random.split(sk, NUM_DEVICES))
            step += 1; nb += 1; tl = tl + loss[0]   # se queda en el device: no bloquea el pipeline asíncrono
        pv = predict(params, tr_ids[va_idx], tr_mask[va_idx]); acc = accuracy_score(train_df.y.values[va_idx], pv.argmax(1))
        print(f'  ep{ep + 1} loss {float(tl) / max(nb, 1):.4f} val_acc {acc:.4f} ({mins()})')
        if ep == 0 and acc < 0.45 and not SMOKE and lr > LR / 4:   # colapso a clase mayoritaria -> reintento con LR menor
            print('  ⚠️ colapso detectado, reintentando con LR/2'); del params, opt_state
            return run_fold(tr_idx, va_idx, use_aug, aug_types, tag, lr / 2, seed + 7, keep_params)
        if acc > best[0]:
            best = (acc, pv, predict(params, ev_ids, ev_mask), jax.device_get(jax_utils.unreplicate(params)) if keep_params else None)
    del params, opt_state; gc.collect()
    return best

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
FOLDS = list(skf.split(train_df, train_df.label + '_' + train_df.lang))
print('Folds listos (estratificados por etiqueta × idioma)')
""")

md(r"""
## 4. Ablación en el fold 0: **con vs sin aumentación** (requisito del enunciado)
Misma partición, mismo modelo, mismos hiperparámetros y semilla; solo cambian los datos de entrenamiento.
Criterio de selección: accuracy de validación **re-ponderada con la distribución de idiomas de eval**, porque el
enunciado advierte que eval no sigue la distribución de train.
""")

code(r"""
def lang_acc(va_idx, probs):
    d = train_df.iloc[va_idx].assign(ok=(probs.argmax(1) == train_df.y.values[va_idx]))
    return d.groupby('lang').ok.mean().reindex(LANGS)

CONFIGS = {   # en v1 se compararon también las variantes parciales; aquí: sin vs con aumentación completa
    'A_sin_aug':       dict(use_aug=False),
    'B_con_aug_todo':  dict(use_aug=True, aug_types=None),
}
ablation, fold0_cache = {}, {}
tr0, va0 = FOLDS[0]
if RUN_ABLATION:
    for name, kw in CONFIGS.items():
        acc, pv, pe, prm = run_fold(tr0, va0, tag=name, keep_params=True, **kw)
        ablation[name] = {'val_acc': acc, **lang_acc(va0, pv).to_dict()}
        fold0_cache[name] = (acc, pv, pe, prm) if name != 'A_sin_aug' else (acc, pv, pe, None)
        if name == 'A_sin_aug': del prm
        gc.collect()
    abl = pd.DataFrame(ablation).T
    # accuracy re-ponderado con la mezcla de idiomas de EVAL (eval tiene más en/pt/fr que train)
    w_eval = eval_df.lang.value_counts(normalize=True).reindex(LANGS).fillna(0)
    filled = abl[LANGS].astype(float).apply(lambda col: col.fillna(abl['val_acc'].astype(float)))
    abl['acc_ponderada_eval'] = (filled * w_eval.values).sum(axis=1)
    abl = abl.round(4); display(abl); abl.to_csv(f'{OUT}/ablacion_aumentacion.csv')
    # criterio: accuracy ponderada por idiomas de eval; empate (<0.2 pts) -> con aumentación (robustez al cambio de distribución)
    BEST_CFG = 'B_con_aug_todo' if abl.loc['B_con_aug_todo', 'acc_ponderada_eval'] >= abl.loc['A_sin_aug', 'acc_ponderada_eval'] - 0.002 else 'A_sin_aug'
else:
    BEST_CFG = 'B_con_aug_todo'
print('Configuración elegida para el K-fold:', BEST_CFG)
""")

code(r"""
if RUN_ABLATION:
    ax = abl[LANGS].T.plot.bar(figsize=(9, 4), rot=0, title='Accuracy por idioma — fold 0 (con vs sin aumentación)')
    ax.set_ylim(max(0, abl[LANGS].min().min() - 0.05), 1); ax.set_ylabel('accuracy'); plt.tight_layout(); plt.savefig(f'{OUT}/ablacion.png', dpi=120); plt.show()
""")

md(r"""
## 5. K-fold completo con la mejor configuración
El fold 0 se reutiliza de la ablación (no se re-entrena). Las probabilidades de eval se promedian entre folds.
""")

code(r"""
oof = np.zeros((len(train_df), NUM_LABELS), np.float32); test_probs = np.zeros((len(eval_df), NUM_LABELS), np.float64)
fold_accs, fold_eval = [], []; os.makedirs(f'{OUT}/models', exist_ok=True); best_fold = (-1, None)
done_folds, t_fold = [], []
for f, (tr, va) in enumerate(FOLDS):
    elapsed_h = (time.time() - START) / 3600; est_h = (np.mean(t_fold) if t_fold else 0) / 3600
    if f > 0 and elapsed_h + est_h > TIME_BUDGET_H - 0.4:   # margen para análisis y guardado
        print(f'⏱️ Presupuesto de tiempo: se detiene tras {f} folds ({elapsed_h:.2f} h)'); break
    t0 = time.time()
    if f == 0 and BEST_CFG in fold0_cache and fold0_cache[BEST_CFG][3] is not None:
        acc, pv, pe, prm = fold0_cache.pop(BEST_CFG)
    else:
        acc, pv, pe, prm = run_fold(tr, va, tag=f'fold{f}', keep_params=True, **CONFIGS[BEST_CFG])
    oof[va] = pv; test_probs += pe; fold_accs.append(acc); fold_eval.append(pe); done_folds.append(f)
    if not (f == 0 and RUN_ABLATION): t_fold.append(time.time() - t0)
    if SAVE_ALL_FOLDS:   # pesos bf16 (~1.1 GB/fold) para reproducir exactamente la submission
        with open(f'{OUT}/models/fold{f}.msgpack', 'wb') as fh:
            fh.write(serialization.to_bytes(jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=jnp.bfloat16), prm)))
    if acc > best_fold[0]: best_fold = (acc, prm)
    else: del prm
    np.save(f'{OUT}/probs_eval_partial.npy', test_probs / len(done_folds)); gc.collect()
    print(f'== fold {f}: {acc:.4f} ({mins()})')
fold0_cache.clear(); gc.collect()
test_probs /= len(done_folds)
COVERED = np.concatenate([FOLDS[f][1] for f in done_folds])   # filas con predicción OOF
print(f'Folds: {done_folds} | CV accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}  |  OOF: {accuracy_score(train_df.y.values[COVERED], oof[COVERED].argmax(1)):.4f}')
""")

md(r"""
## 6. Análisis de errores por idioma (OOF sobre train)
""")

code(r"""
res = train_df.iloc[COVERED].assign(pred=[LABELS[i] for i in oof[COVERED].argmax(1)], conf=oof[COVERED].max(1))
res['ok'] = res.pred == res.label
by_lang = res.groupby('lang').agg(n=('ok', 'size'), accuracy=('ok', 'mean')).reindex(LANGS)
by_lang['peso en eval %'] = (eval_df.lang.value_counts(normalize=True) * 100).reindex(LANGS).round(1)
display(by_lang.round(4))
print(classification_report(res.label, res.pred, digits=4, zero_division=0))

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, lg in zip(axes, ['es', 'en', 'pt', 'fr']):
    s = res[res.lang == lg]; cm = confusion_matrix(s.label, s.pred, labels=LABELS, normalize='true')
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1); ax.set_title(f'{lg} (acc {s.ok.mean():.3f})')
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='white' if cm[i, j] > .5 else 'black')
axes[0].set_ylabel('real'); plt.suptitle('Matrices de confusión normalizadas por idioma (filas = real)'); plt.tight_layout()
plt.savefig(f'{OUT}/confusion_por_idioma.png', dpi=120); plt.show()
""")

code(r"""
# Fenómenos difíciles descritos en el enunciado: ¿dónde falla el modelo?
PHEN = {
    'negación':      r"\b(?:no|not|nunca|never|pas|jamais|não|nem|n't)\b",
    'contraste':     r"\b(?:pero|but|however|sin embargo|aunque|mais|mas|porém|embora|cependant)\b",
    'comparación':   r"\b(?:esperaba|expected|esperava|attendais|más que|better than|worse than|mejor que|peor que|que mon ancien)\b",
    'consecuencia':  r"\b(?:devolví|devolvi|returned|rendu|reembolso|refund|garantía|warranty)\b",
    'ironía (marcas)': r"(?:genial\.\.\.|claro que sí|qué maravilla|yeah right|great\.\.\.|bravo|oh,? (?:sí|yes))",
}
rows = []
for k, pat in PHEN.items():
    m = res.text.str.contains(pat, case=False, regex=True)
    rows.append({'fenómeno': k, 'n': int(m.sum()), 'acc con fenómeno': res.ok[m].mean(), 'acc sin fenómeno': res.ok[~m].mean()})
display(pd.DataFrame(rows).round(4))

print('\nErrores con mayor confianza (revisar posibles etiquetas ruidosas / ironía):')
for r in res[~res.ok].sort_values('conf', ascending=False).head(8).itertuples():
    print(f'[{r.lang}] real={r.label} pred={r.pred} conf={r.conf:.2f} :: {r.text[:230]}')
""")

md(r"""
## 6b. Stacking OOF con TF-IDF posicional (familia clásica complementaria)
El TF-IDF posicional (n-gramas de la reseña completa, última, penúltima y primera oración) llega a 0.798 en CV.
Se combina con el transformer mediante una **regresión logística sobre las probabilidades OOF** (validación cruzada
interna para estimar su accuracy honestamente). Solo se usa si mejora el OOF del transformer solo.
""")

code(r"""
FINAL_PROBS, FINAL_NAME = test_probs, 'transformer'
if STACK_TFIDF:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    def parts(df):
        S = df.text.map(split_sentences)
        return {'full': S.map(' '.join), 'last': S.map(lambda s: s[-1] if s else ''), 'last2': S.map(lambda s: s[-2] if len(s) > 1 else ''), 'first': S.map(lambda s: s[0] if s else '')}
    Ptr, Pev = parts(train_df), parts(eval_df)
    vecs = [(k, TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)) for k in Ptr] + \
           [(k, TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100000)) for k in ('full', 'last')]
    Xtr = sparse.hstack([v.fit_transform(Ptr[k]) for k, v in vecs]).tocsr(); Xev = sparse.hstack([v.transform(Pev[k]) for k, v in vecs]).tocsr()
    oof_tf, ev_tf = np.zeros((len(train_df), 3)), np.zeros((len(eval_df), 3)); yv = train_df.y.values
    for tr, va in FOLDS:   # mismos folds que el transformer
        m = LogisticRegression(C=4, max_iter=4000).fit(Xtr[tr], yv[tr]); oof_tf[va] = m.predict_proba(Xtr[va]); ev_tf += m.predict_proba(Xev) / len(FOLDS)
    print(f'TF-IDF posicional CV: {(oof_tf.argmax(1) == yv).mean():.4f} ({mins()})')
    lg = lambda p: np.log(np.clip(p, 1e-6, 1))
    Z, Zev = np.hstack([lg(oof[COVERED]), lg(oof_tf[COVERED])]), np.hstack([lg(test_probs), lg(ev_tf)])
    stacker = LogisticRegression(C=1.0, max_iter=2000)
    st_oof = cross_val_predict(stacker, Z, yv[COVERED], cv=5, method='predict_proba')
    acc_tr, acc_st = accuracy_score(yv[COVERED], oof[COVERED].argmax(1)), accuracy_score(yv[COVERED], st_oof.argmax(1))
    print(f'OOF transformer {acc_tr:.4f} | OOF stacking {acc_st:.4f}')
    stack_probs = stacker.fit(Z, yv[COVERED]).predict_proba(Zev)
    np.save(f'{OUT}/probs_eval_tfidfpos.npy', ev_tf.astype(np.float32)); np.save(f'{OUT}/probs_eval_stack.npy', stack_probs.astype(np.float32))
    pd.DataFrame({'id': eval_df.id, 'answer': [LABELS[i] for i in stack_probs.argmax(1)]}).to_csv(f'{OUT}/submission_stack.csv', index=False)
    if acc_st > acc_tr + 0.001: FINAL_PROBS, FINAL_NAME = stack_probs, 'stacking transformer + TF-IDF posicional'
print('Submission principal:', FINAL_NAME)
""")

md(r"""
## 7. Guardado del modelo y submission
- `modelo_final/` → mejor fold en formato HuggingFace **Flax** (`flax_model.msgpack`) + tokenizer.
- `modelo_final_pytorch/` → el mismo modelo convertido a **PyTorch** (`model.safetensors`).
- `models/fold*.msgpack` → pesos de los 5 folds (la submission es su promedio).
""")

code(r"""
MODEL_DIR, MODEL_DIR_PT = f'{OUT}/modelo_final', f'{OUT}/modelo_final_pytorch'
model.save_pretrained(MODEL_DIR, params=best_fold[1]); tokenizer.save_pretrained(MODEL_DIR)
print(f'✅ Flax guardado en {MODEL_DIR} (fold val acc {best_fold[0]:.4f})')
if not SMOKE:
    try:
        from transformers import AutoModelForSequenceClassification
        AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, from_flax=True).save_pretrained(MODEL_DIR_PT)
        tokenizer.save_pretrained(MODEL_DIR_PT); print(f'✅ PyTorch guardado en {MODEL_DIR_PT}')
    except Exception as e:
        print('Conversión a PyTorch omitida:', e)

np.save(f'{OUT}/probs_eval.npy', test_probs.astype(np.float32)); np.save(f'{OUT}/probs_oof.npy', oof)
pd.DataFrame({'id': eval_df.id, 'answer': [LABELS[i] for i in test_probs.argmax(1)]}).to_csv(f'{OUT}/submission_transformer.csv', index=False)
sub = pd.DataFrame({'id': eval_df.id, 'answer': [LABELS[i] for i in FINAL_PROBS.argmax(1)]})
sub.to_csv(f'{OUT}/submission.csv', index=False)
assert list(sub.columns) == ['id', 'answer'] and len(sub) == len(eval_df) and sub.answer.isin(LABELS).all()
json.dump({'backbone': BACKBONE, 'config': BEST_CFG, 'final': FINAL_NAME, 'folds_done': done_folds, 'use_pair': USE_PAIR, 'llrd': LLRD, 'cv_mean': float(np.mean(fold_accs)), 'fold_accs': [float(a) for a in fold_accs],
           'ablation': ablation, 'epochs': EPOCHS, 'lr': LR, 'global_batch': GLOBAL_BATCH, 'seed': SEED},
          open(f'{OUT}/resultados.json', 'w'), indent=2, default=float)
print(sub.answer.value_counts(normalize=True).round(3)); print(f'\n✅ submission.csv ({FINAL_NAME}) + submission_transformer.csv ({mins()} total)')
""")

code(r"""
# Chequeo de replicabilidad: recargar los pesos guardados del fold 0 y re-predecir una muestra de eval
if SAVE_ALL_FOLDS:
    with open(f'{OUT}/models/fold0.msgpack', 'rb') as fh:
        p0 = serialization.from_bytes(INIT_PARAMS, fh.read())
    p0 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float32), p0)
    n = min(512, len(eval_df)); re_pred = predict(jax_utils.replicate(p0), ev_ids[:n], ev_mask[:n]).argmax(1)
    print(f'Concordancia recargado vs original (fold 0, {n} reseñas): {(re_pred == fold_eval[0][:n].argmax(1)).mean():.4f}')
    del p0
""")

md(r"""
## 8. Conclusiones y registro de iteraciones
Llenar con los scores públicos de Kaggle de cada envío (se necesitan ≥ 5 envíos distintos):

| # | Cambio | CV (OOF) | Público Kaggle | Por qué mejoró / empeoró |
|---|---|---|---|---|
| 1 | TF-IDF + LogReg (`01_baseline_tfidf`) — referencia no-DL | 0.717 | | |
| 2 | XLM-R large sin aumentación (fold 0, config A) | ver tabla sección 4 | | |
| 3 | + traducción es→en/fr/pt + code-switching (config B) | ver tabla sección 4 | | |
| 4 | + EDA (config C) | ver tabla sección 4 | | |
| 5 | K-fold (5) promedio con la mejor config | ver sección 5 | | |

**Uso de IA generativa:** ver `docs/ENTREGABLE_referencias_y_IA.md`. La IA no etiquetó textos de eval; todas las
predicciones provienen del modelo entrenado en este notebook.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"},
               "kaggle": {"accelerator": "tpuV5e8", "isInternetEnabled": True}}
out = Path(__file__).resolve().parents[1] / 'notebooks' / '03_tpu_v2_max_accuracy.ipynb'
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print('escrito', out)
