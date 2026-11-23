"""Genera notebooks/ENTREGA_modelo_final.ipynb — notebook entregable (Bloque Neón).

Dos modos:
  ENTRENAR = False -> carga los resultados ya obtenidos (carpeta resultados/) y hace todo el análisis (minutos, CPU).
  ENTRENAR = True  -> reproduce desde cero en Colab/Kaggle GPU: traducción + 2 entrenamientos (~2.5 h en T4).
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Clasificación de sentimiento multilingüe de reseñas — Modelo final (Parte 2)
**Aprendizaje de Máquina 2026-20 · Universidad de los Andes** · Competencia Kaggle *parte-2-competencia-aprendizaje-de-maquina-2026-20*

**Tarea:** dada una reseña (español, inglés, portugués, francés o *code-switching* es/en), predecir `negativo`, `neutral` o `positivo`. Métrica: *accuracy*.

### Resumen de la solución
| Componente | Elección |
|---|---|
| Modelo base (transferencia de aprendizaje) | **XLM-RoBERTa large** — `FacebookAI/xlm-roberta-large` (MIT), transformer encoder de 24 capas preentrenado en 100 idiomas (CC-100) |
| Representación del texto | Tokenización SentencePiece (250k subpalabras compartidas entre idiomas) y **entrada en par**: `(últimas 2 oraciones, reseña completa)` |
| Arquitectura | Encoder XLM-R + cabeza de clasificación (dense 1024 → tanh → dropout → 3 clases) sobre el token `<s>`; *fine-tuning* completo |
| Aumentación de datos | **Traducción automática** con MarianMT (es→en→fr/pt, en→fr/pt/es, fr/pt→en) + **code-switching sintético** es/en |
| Entrenamiento | AdamW, LR 1e-5 con *warmup* lineal 10 % y decaimiento por capa (LLRD 0.925), *label smoothing* 0.05, 3 épocas, batch 16, *gradient clipping* 1.0 |
| Validación | *Holdout* estratificado por **etiqueta × idioma** (8 %), semilla 7. **El conjunto eval nunca se usa para entrenar** |
| Hardware | GPU NVIDIA T4 (Google Colab), PyTorch + Hugging Face Transformers |

### Cumplimiento del enunciado
- ✅ Deep learning (transformer) con transferencia de aprendizaje; modelo base y fuente declarados.
- ✅ Aumentación de datos y **comparación con vs. sin aumentación** con la misma partición y semilla (sección 6).
- ✅ Datos sintéticos descritos en detalle (sección 5). No se usaron datos externos.
- ✅ **Análisis de errores por idioma** sobre la partición de validación (sección 7).
- ✅ Evidencia de iteración con su impacto en la métrica (sección 8).
- ✅ Ningún LLM etiquetó textos de eval; eval no se usa para entrenar ni para elegir hiperparámetros.
- ✅ Modelo guardado con PyTorch (`model.safetensors`) y semillas fijas → replicable.
""")

code(r"""
# ===== CONFIGURACIÓN =====
ENTRENAR = False   # False: carga resultados ya obtenidos (resultados/) · True: reproduce todo en GPU (~2.5 h en T4)
RES_SIN  = 'resultados/xlmr_large_s7'       # XLM-R large, semilla 7, SIN aumentación
RES_CON  = 'resultados/xlmr_large_s7_aug'   # XLM-R large, semilla 7, CON aumentación (modelo final)
AUG_CSV  = 'resultados/train_aug.csv'

# Scores públicos de Kaggle de cada envío (actualizar si se hacen más envíos)
PUBLIC = {
    'DistilBERT multilingüe (CPU, sin aug)': 0.82416,
    'MiniLM-L12 multilingüe (CPU, sin aug)': 0.856,
    'XLM-R large s42 (Kaggle GPU, sin aug)': 0.89096,
    'XLM-R large s42 (Colab, sin aug)': 0.89538,
    'XLM-R large s7 (Colab, sin aug)': 0.89980,
    'XLM-R large s2026 3-fold (compañero, sin aug)': 0.89833,
    'Ensemble 2× XLM-R large': 0.89096,
    'Ensemble 3× XLM-R large': 0.89636,
    'Ensemble 4× XLM-R large': 0.8934,
    'XLM-R large s7 CON aumentación (modelo final)': None,   # <- completar con el score público
}
""")

code(r"""
import os, sys, re, json, glob, subprocess
import numpy as np, pandas as pd, matplotlib.pyplot as plt
IN_COLAB = 'google.colab' in sys.modules
if os.path.basename(os.getcwd()) == 'notebooks' and os.path.exists('../src/train_torch.py'): os.chdir('..')
if not os.path.exists('src/train_torch.py'):          # en Colab/Kaggle: clonar el repositorio
    if not os.path.exists('sentimiento-multilingue'):
        subprocess.run(['git', 'clone', '-q', 'https://github.com/Jeronimocifci/sentimiento-multilingue.git'])
    os.chdir('sentimiento-multilingue')
sys.path.insert(0, 'src')
from lang_detect import detect_lang
DATA = next(d for d in ['data', '/kaggle/input/parte-2-competencia-aprendizaje-de-maquina-2026-20'] + glob.glob('/kaggle/input/**/') if os.path.exists(f'{d}/train.csv'))
train = pd.read_csv(f'{DATA}/train.csv'); ev = pd.read_csv(f'{DATA}/eval.csv')
for d in (train, ev):
    d['text'] = d.text.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip(); d['lang'] = d.text.map(detect_lang)
LANGS = ['es', 'en', 'pt', 'fr', 'es-en']; LABELS = ['negativo', 'neutral', 'positivo']; L2I = {l: i for i, l in enumerate(LABELS)}
train['y'] = train.label.map(L2I)
print('train', train.shape, '| eval', ev.shape, '| datos en', DATA)
""")

md(r"""
## 1. Datos: qué cambió respecto a la Parte 1
El idioma no viene en los datos; lo detectamos con un detector ligero por *stopwords* (`src/lang_detect.py`) que además marca
`es-en` cuando español e inglés tienen presencia fuerte (code-switching). Lo usamos para **estratificar la validación** y para
el **análisis por idioma** (el modelo no recibe el idioma).
""")

code(r"""
shift = pd.DataFrame({'train %': train.lang.value_counts(normalize=True), 'eval %': ev.lang.value_counts(normalize=True)}).reindex(LANGS).mul(100).round(1)
display(shift)
display(train.label.value_counts(normalize=True).round(3).to_frame('proporción en train'))
ax = shift.plot.bar(rot=0, figsize=(7, 3.2), title='Distribución de idiomas: train vs eval'); ax.set_ylabel('%'); plt.tight_layout(); plt.show()
""")

md(r"""
**Lectura.** Eval tiene **menos español y más inglés/portugués/francés** que train. Un modelo bueno solo en español pierde
puntos → justifica (i) un modelo **multilingüe** y (ii) **aumentar** los idiomas minoritarios con traducción.

## 2. Hallazgo clave: el veredicto está al final de la reseña
Muchas reseñas mezclan aspectos positivos y negativos (concesiones, contraste). Para ver **dónde** está la señal, entrenamos
un clasificador lineal TF-IDF usando distintas partes de cada reseña (5-fold CV):
""")

code(r"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
split = lambda t: [p for p in re.split(r'(?<=[.!?…])\s+', str(t).strip()) if p]
S = train.text.map(split); cv5 = StratifiedKFold(5, shuffle=True, random_state=0)
def cv_acc(texts):
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(texts)
    return (cross_val_predict(LogisticRegression(C=4, max_iter=3000), X, train.label, cv=cv5, n_jobs=-1) == train.label).mean()
pos = {'primeras 2 oraciones': S.map(lambda s: ' '.join(s[:2])), 'reseña completa': S.map(' '.join), 'solo la última oración': S.map(lambda s: s[-1])}
pos_res = pd.Series({k: cv_acc(v) for k, v in pos.items()}).round(4).rename('accuracy CV')
display(pos_res.to_frame())
""")

md(r"""
**Lectura.** La última oración sola predice **mejor** que la reseña completa, y el inicio casi no sirve: el veredicto suele
estar al final (*"…pero al final lo devolví"*). Decisión de diseño: alimentar el transformer con el **par
`(últimas 2 oraciones, reseña completa)`**, de modo que el veredicto aparece cerca del token `<s>` y además en contexto.

## 3. Línea base clásica (referencia, no es el modelo entregado)
TF-IDF posicional (n-gramas de la reseña completa, última, penúltima y primera oración + n-gramas de caracteres) + regresión logística.
""")

code(r"""
from scipy import sparse
P = {'full': S.map(' '.join), 'last': S.map(lambda s: s[-1]), 'last2': S.map(lambda s: s[-2] if len(s) > 1 else ''), 'first': S.map(lambda s: s[0])}
vecs = [(k, TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)) for k in P] + \
       [(k, TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=100000)) for k in ('full', 'last')]
X = sparse.hstack([v.fit_transform(P[k]) for k, v in vecs]).tocsr()
pred_tf = cross_val_predict(LogisticRegression(C=4, max_iter=4000), X, train.label, cv=cv5, n_jobs=-1)
ok_tf = pred_tf == train.label
print(f'TF-IDF posicional · CV accuracy {ok_tf.mean():.4f}'); display(ok_tf.groupby(train.lang).mean().reindex(LANGS).round(4).to_frame('accuracy'))
""")

md(r"""
## 4. Modelo: XLM-RoBERTa large con entrada en par
- **Representación:** el tokenizer SentencePiece de XLM-R convierte el texto en subpalabras de un vocabulario de 250k
  compartido por 100 idiomas, lo que permite transferir lo aprendido en español a pt/fr/en. La entrada es
  `<s> últimas_2_oraciones </s></s> reseña_completa </s>` (máx. 320 tokens; ninguna reseña se trunca en la práctica).
- **Red:** 24 capas transformer (hidden 1024, 16 cabezas de atención, 560M parámetros). La representación contextual del
  token `<s>` pasa por la cabeza de clasificación (dense 1024 + tanh + dropout + lineal a 3 clases).
- **Por qué XLM-R large:** en nuestras iteraciones, a mayor capacidad y mejor preentrenamiento multilingüe, mejor score
  (DistilBERT 0.824 → MiniLM 0.856 → XLM-R large 0.89–0.90, ver sección 8).
- **Optimización:** AdamW (wd 0.01), LR máx. 1e-5 en la capa superior con decaimiento 0.925 por capa hacia abajo (LLRD: las capas
  bajas, más genéricas, cambian menos), *warmup* 10 % + decaimiento lineal, *label smoothing* 0.05 (reduce sobreconfianza ante
  etiquetas ruidosas), *clipping* 1.0, fp16 en GPU. Se guarda el modelo de la época con mejor accuracy de validación.
- **Código:** `src/train_torch.py` (entrenamiento) y `src/translate_aug.py` (aumentación), versionados en el repositorio.

## 5. Aumentación de datos (datos sintéticos)
Todos los aumentos se derivan **solo de train** y guardan `src_id` (id de la reseña original): en el entrenamiento se usan
únicamente aumentos de reseñas que están en la partición de entrenamiento → **sin fuga hacia validación**.

1. **Traducción automática oración por oración** (conserva el orden de las oraciones → conserva la posición del veredicto),
   decodificación *greedy* con MarianMT en GPU (fp16):
   - 4 000 reseñas en español (estratificadas por etiqueta): es→en (`Helsinki-NLP/opus-mt-es-en`) y en→fr / en→pt
     (`Helsinki-NLP/opus-mt-en-ROMANCE`, tokens `>>fr<<`, `>>pt_br<<`).
   - Todas las reseñas en inglés: en→fr, en→pt, en→es (`opus-mt-en-ROMANCE`).
   - Todas las reseñas en francés y portugués: →en (`Helsinki-NLP/opus-mt-ROMANCE-en`).
2. **Code-switching sintético es/en:** con las oraciones alineadas (original en español + su traducción al inglés) se toma cada
   oración en inglés con p = 0.5, garantizando al menos una de cada idioma (3 000 reseñas).
3. En cada época se usan **todos los originales + una muestra aleatoria distinta de aumentos** del mismo tamaño
   (`--aug_ratio 1.0`), para que los datos sintéticos no dominen.

La etiqueta del aumento es la de la reseña original (la traducción preserva la polaridad global).
""")

code(r"""
if ENTRENAR:
    subprocess.run([sys.executable, 'src/translate_aug.py', '--data', DATA, '--n_es', '4000', '--out', AUG_CSV], check=True)
aug = pd.read_csv(AUG_CSV)
display(aug.aug_type.value_counts().rename('n aumentos').to_frame())
for t in aug.aug_type.unique():
    print(f'[{t}] {aug[aug.aug_type == t].text.iloc[0][:230]}\n')
""")

md(r"""
## 6. Comparación con vs. sin aumentación (mismo modelo, misma partición, misma semilla)
Ambas corridas usan XLM-R large, semilla 7 y el mismo *holdout* estratificado de 8 % (1 638 reseñas). Lo único que cambia son
los datos de entrenamiento.
""")

code(r"""
CMD = 'python src/train_torch.py --model FacebookAI/xlm-roberta-large --epochs 3 --bs 16 --lr 1e-5 --seed 7 --val_frac 0.08 --data {data} --out {out}'
if ENTRENAR:
    subprocess.run(CMD.format(data=DATA, out=RES_SIN).split(), check=True)
    subprocess.run((CMD.format(data=DATA, out=RES_CON) + f' --aug_csv {AUG_CSV} --aug_ratio 1.0').split(), check=True)

def load_run(d):
    vi = np.load(f'{d}/val_idx.npy'); pv = np.load(f'{d}/probs_val.npy')
    r = train.iloc[vi].assign(pred=[LABELS[i] for i in pv.argmax(1)], conf=pv.max(1)); r['ok'] = r.pred == r.label
    return r, np.load(f'{d}/probs_eval.npy'), json.load(open(f'{d}/log.json'))
r_sin, pe_sin, log_sin = load_run(RES_SIN); r_con, pe_con, log_con = load_run(RES_CON)
assert (r_sin.index == r_con.index).all(), 'las dos corridas deben tener la misma partición de validación'

w_eval = ev.lang.value_counts(normalize=True).reindex(LANGS).fillna(0)
def fila(r, pub):
    acc = r.groupby('lang').ok.mean().reindex(LANGS)
    return {'val accuracy': r.ok.mean(), **acc.to_dict(), 'val ponderada (mezcla eval)': (acc.fillna(r.ok.mean()) * w_eval).sum(), 'Kaggle público': pub}
comp = pd.DataFrame({'Sin aumentación': fila(r_sin, PUBLIC['XLM-R large s7 (Colab, sin aug)']),
                     'Con aumentación': fila(r_con, PUBLIC['XLM-R large s7 CON aumentación (modelo final)'])}).T
comp.loc['Diferencia (con − sin)'] = comp.iloc[1] - comp.iloc[0]
display(comp.astype(float).round(4))
ax = comp.iloc[:2][LANGS].T.plot.bar(rot=0, figsize=(8, 3.5), title='Accuracy de validación por idioma: con vs. sin aumentación')
ax.set_ylim(0.6, 1); ax.set_ylabel('accuracy'); plt.tight_layout(); plt.show()
print(f"n validación por idioma: {r_con.lang.value_counts().reindex(LANGS).to_dict()}  (es-en tiene muy pocas reseñas: su accuracy es muy ruidosa)")
""")

md(r"""
**Lectura.** *(Completar tras ver la tabla.)* La aumentación por traducción apunta a los idiomas minoritarios, que son los que más
pesan en eval respecto a train; por eso además del accuracy de validación (distribución de train) reportamos la accuracy
**re-ponderada con la mezcla de idiomas de eval** y el score público de Kaggle.

## 7. Análisis de errores por idioma (partición de validación, modelo final)
""")

code(r"""
from sklearn.metrics import confusion_matrix, classification_report
R = r_con   # modelo final
print(classification_report(R.label, R.pred, digits=4, zero_division=0))
display(R.groupby('lang').agg(n=('ok', 'size'), accuracy=('ok', 'mean')).reindex(LANGS).round(4))
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, lg in zip(axes, ['es', 'en', 'pt', 'fr']):
    s = R[R.lang == lg]; cm = confusion_matrix(s.label, s.pred, labels=LABELS, normalize='true')
    ax.imshow(cm, cmap='Blues', vmin=0, vmax=1); ax.set_title(f'{lg} · acc {s.ok.mean():.3f} · n={len(s)}')
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS); ax.set_yticks(range(3)); ax.set_yticklabels(LABELS)
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', color='white' if cm[i, j] > .5 else 'black')
axes[0].set_ylabel('real'); plt.suptitle('Matrices de confusión normalizadas por idioma (filas = clase real)'); plt.tight_layout(); plt.show()
""")

code(r"""
# Fenómenos difíciles que menciona el enunciado
PHEN = {'negación': r"\b(?:no|not|nunca|never|pas|jamais|não|nem|n't)\b",
        'contraste / concesión': r"\b(?:pero|but|however|sin embargo|aunque|mais|mas|porém|embora|cependant)\b",
        'comparación / expectativa': r"\b(?:esperaba|expected|esperava|attendais|mejor que|peor que|better than|worse than)\b",
        'solo consecuencias (devolución)': r"\b(?:devolví|devolvi|returned|rendu|reembolso|refund)\b"}
rows = []
for k, p in PHEN.items():
    m = R.text.str.contains(p, case=False, regex=True)
    rows.append({'fenómeno': k, 'n': int(m.sum()), 'acc con fenómeno': R.ok[m].mean(), 'acc sin fenómeno': R.ok[~m].mean()})
display(pd.DataFrame(rows).round(4))
cm_all = pd.crosstab(R.label, R.pred).reindex(index=LABELS, columns=LABELS, fill_value=0); display(cm_all)
print('Errores con mayor confianza:')
for r in R[~R.ok].sort_values('conf', ascending=False).head(8).itertuples():
    print(f'[{r.lang}] real={r.label} · pred={r.pred} · conf={r.conf:.2f} :: {r.text[:240]}\n')
""")

md(r"""
**Lectura.** *(Completar tras ver las tablas.)* Puntos a revisar: (i) qué idioma tiene menor accuracy y si coincide con los que
más pesan en eval; (ii) si los errores se concentran en `neutral` (frontera difusa con reseñas mixtas); (iii) si los errores con
alta confianza son ironía o posibles etiquetas ruidosas.

## 8. Evidencia de iteración: envíos a Kaggle y su impacto
""")

code(r"""
it = pd.DataFrame([
    ('TF-IDF + LogReg (Parte 1, no-DL)', 'referencia', f'CV {0.717:.3f}', None, 'n-gramas no comparten información entre idiomas'),
    ('TF-IDF posicional (no-DL)', 'referencia', f'CV {ok_tf.mean():.3f}', None, 'features de última/penúltima oración (+8 pts): el veredicto está al final'),
    ('DistilBERT multilingüe, entrada en par', 'DL', '—', PUBLIC['DistilBERT multilingüe (CPU, sin aug)'], 'primer transformer; modelo chico (CPU)'),
    ('MiniLM-L12 multilingüe, entrada en par', 'DL', 'val 0.887', PUBLIC['MiniLM-L12 multilingüe (CPU, sin aug)'], 'mejor preentrenamiento multilingüe'),
    ('XLM-R large s42', 'DL', '—', PUBLIC['XLM-R large s42 (Colab, sin aug)'], 'modelo grande multilingüe: +4 pts, sobre todo en en/pt/fr'),
    ('XLM-R large s7 (holdout 8 %)', 'DL', f"val {log_sin['val_acc_mean']:.3f}", PUBLIC['XLM-R large s7 (Colab, sin aug)'], 'otra semilla/partición: varianza entre corridas ~0.5 pts'),
    ('XLM-R large s2026, 3 folds', 'DL', '—', PUBLIC['XLM-R large s2026 3-fold (compañero, sin aug)'], 'promedio de 3 folds'),
    ('Ensembles de 2/3/4 XLM-R large', 'DL', '—', PUBLIC['Ensemble 3× XLM-R large'], 'en el público quedan entre los individuales (diferencias dentro del ruido)'),
    ('XLM-R large s7 + aumentación por traducción', 'DL (final)', f"val {log_con['val_acc_mean']:.3f}", PUBLIC['XLM-R large s7 CON aumentación (modelo final)'], 'misma partición que s7 sin aug → comparación directa'),
], columns=['iteración', 'tipo', 'validación', 'Kaggle público', 'qué cambió / por qué'])
display(it)
pub = pd.Series({k: v for k, v in PUBLIC.items() if v}).sort_values()
ax = pub.plot.barh(figsize=(8, 4.5), title='Score público por envío'); ax.set_xlim(0.8, 0.91); plt.tight_layout(); plt.show()
print('Nota: el público usa solo una parte de eval; diferencias < 0.005 equivalen a ~15 reseñas y no son concluyentes.')
""")

md(r"""
## 9. Modelo final: guardado, recarga y submission
El modelo (pesos + tokenizer) se guarda con `save_pretrained` de PyTorch/Transformers en `RES_CON/modelo/`
(`model.safetensors`, ~2.2 GB en fp32). Abajo se verifica que la submission corresponde a ese modelo.
""")

code(r"""
sub = pd.DataFrame({'id': ev.id, 'answer': [LABELS[i] for i in pe_con.argmax(1)]})
assert len(sub) == len(ev) and sub.answer.isin(LABELS).all()
sub.to_csv('submission_final.csv', index=False)
print(sub.answer.value_counts(normalize=True).round(3).to_dict())
mdir = f'{RES_CON}/modelo'
if os.path.exists(f'{mdir}/config.json'):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(mdir); mod = AutoModelForSequenceClassification.from_pretrained(mdir).eval()
    last2 = lambda t: ' '.join(split(t)[-2:])[-300:]
    smp = ev.text.iloc[:16].tolist()
    with torch.no_grad():
        enc = tok([last2(t) for t in smp], smp, truncation='longest_first', max_length=320, padding=True, return_tensors='pt')
        p = mod(**enc).logits.argmax(-1).numpy()
    print('Concordancia modelo recargado vs submission (16 reseñas):', (p == pe_con[:16].argmax(1)).mean())
else:
    print('Pesos no incluidos en esta carpeta (se entregan aparte por tamaño):', mdir)
""")

md(r"""
## 10. Conclusiones
1. **El modelo multilingüe grande es la decisión que más pesa:** XLM-R large supera a los transformers chicos por 4–7 puntos en el
   público, con la mayor ganancia en inglés, portugués y francés.
2. **La posición importa:** el veredicto suele estar en la última oración; la entrada en par lo hace explícito.
3. **Aumentación:** ver sección 6 (efecto medido con la misma partición y semilla).
4. **Varianza:** entre semillas el público varía ~0.5 pts; los ensembles no mejoraron el público, pero se mantienen como
   alternativa más estable para el privado.

**Referencias y uso de IA generativa:** ver `docs/ENTREGABLE_referencias_y_IA.md`.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}, "language_info": {"name": "python"}}
nbf.write(nb, ROOT / 'notebooks' / 'ENTREGA_modelo_final.ipynb'); print('ok')
