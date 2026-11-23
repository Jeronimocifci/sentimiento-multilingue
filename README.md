# Sentimiento multilingüe de reseñas — Proyecto Parte 2

Competencia Kaggle del curso **Aprendizaje de Máquina 2026-20 (Uniandes)**: dada una reseña (es / en / pt / fr o
code-switching es-en), predecir `negativo`, `neutral` o `positivo`. Métrica: **accuracy**.

**Solución:** XLM-RoBERTa large con *fine-tuning* en **TPU v5e-8 (JAX/Flax + `pmap`)**, aumentación por **traducción
automática + code-switching sintético + EDA**, K-fold (5) estratificado por etiqueta × idioma.

## Estructura

```
├── notebooks/
│   ├── 00_eda.ipynb                 # CPU · cambio de distribución de idiomas train→eval (ejecutado)
│   ├── 01_baseline_tfidf.ipynb      # CPU · línea base no-DL: 0.717 CV (ejecutado)
│   └── 02_tpu_solucion_final.ipynb  # TPU · ENTREGABLE: modelo final, ablación con/sin aug, análisis por idioma
├── src/
│   ├── lang_detect.py               # detector de idioma por stopwords (+ code-switching)
│   ├── augment.py                   # traducción por oraciones, code-switching, EDA
│   └── ensemble.py                  # promedia probs de varias corridas → submission
├── tools/
│   ├── build_main_notebook.py       # genera el notebook 02 (versionable y reproducible)
│   └── build_cpu_notebooks.py       # genera los notebooks 00 y 01
├── docs/ENTREGABLE_referencias_y_IA.md   # entregable 3: modelos, datos, licencias, uso de IA
├── submissions/                     # CSVs enviados (ignorados por git)
└── data/                            # train.csv, eval.csv (ignorados por git)
```

## Cómo correr el modelo final (Kaggle)
1. Subir `notebooks/02_tpu_solucion_final.ipynb` a Kaggle → *Add Input* → datos de la competencia.
2. *Settings*: **Accelerator TPU v5e-8**, **Internet On**.
3. **Save Version → Save & Run All**. Tiempo estimado: ~1.5–2 h.
4. Output: `submission.csv`, `modelo_final/` (Flax), `modelo_final_pytorch/`, `models/fold*.msgpack`,
   `ablacion_aumentacion.csv`, `probs_eval.npy`, `probs_oof.npy`, `resultados.json`, gráficas.

**Para re-correr más rápido** (p. ej. otra semilla): añade el Output de la primera corrida como Input; el notebook
reutiliza `train_aug.csv` y se salta la traducción.

## Optimizaciones TPU (lecciones del proyecto anterior)
| Problema en el proyecto 1 | Solución aquí |
|---|---|
| TF no soporta TPU v5e en Kaggle; torch_xla se colgaba | JAX + Flax nativo |
| `pmap` con batch grande generalizaba peor | `pmap` con 2 ejemplos por core = **batch global 16** |
| Cada fold recompilaba (schedule dentro del optimizador) | LR como **argumento** del paso → una sola compilación por longitud |
| Padding a 512 en todo | **Buckets 128/256** + lotes agrupados por longitud |
| fp32 | Cómputo en **bf16**, pesos/optimizador en fp32 |
| Traducción lenta | `generate` dentro de `pmap` (8 cores) + oraciones deduplicadas |

## Validación local del código (sin TPU)
El notebook 02 tiene un modo `SMOKE` con modelos aleatorios diminutos para comprobar el flujo completo en CPU,
simulando 8 dispositivos:
```bash
pip install -r requirements.txt
cd notebooks && SMOKE=1 XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  jupyter nbconvert --to notebook --execute 02_tpu_solucion_final.ipynb --output /tmp/smoke.ipynb
```

## Resultados (llenar tras cada envío — se necesitan ≥ 5 envíos distintos)
| # | Envío | CV | Público | Notas |
|---|---|---|---|---|
| 1 | TF-IDF + LogReg (no-DL, referencia) | 0.717 | | |
| 2 | XLM-R large, K-fold, mejor config de la ablación | | | |
| 3 | `submission` de una sola config (A/B/C) para comparar | | | |
| 4 | 2.ª semilla del notebook 02 | | | |
| 5 | Ensemble de semillas (`src/ensemble.py`) | | | |

## Reglas que respeta
- Eval **nunca** se usa para entrenar: ni pseudo-labels, ni traducción de eval, ni en el split.
- Ningún LLM etiqueta textos de eval: todas las predicciones salen del modelo entrenado.
- Las traducciones son datos sintéticos descritos en la sección 2 del notebook.
