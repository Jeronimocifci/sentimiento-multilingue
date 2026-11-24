# Entregable 3 — Referencias a modelos y datos, y uso de IA generativa

**Proyecto:** Clasificación de sentimiento multilingüe de reseñas — Parte 2
**Curso:** Aprendizaje de Máquina 2026-20 · Universidad de los Andes
**Competencia:** Kaggle *parte-2-competencia-aprendizaje-de-maquina-2026-20*
**Integrantes:**
| Nombre | Código |
|---|---|
| Ali Cifci | 202210698 |
| Daniel Felipe Torres López | 202110365 |
| Santiago Tenjo Venegas | 202113965 |
| Johan Alejandro Charry Acevedo | 202111151 |

**Repositorio:** https://github.com/Jeronimocifci/sentimiento-multilingue

---

## 0. Modelo final entregado

**Ensemble de 3 modelos XLM-RoBERTa large** ajustados por el grupo (promedio de probabilidades). Score público en Kaggle: **0.89833**.

| Componente | Entrenamiento | Kaggle público |
|---|---|---|
| XLM-R large, semilla 7 | Colab T4, holdout 8 %, sin aumentación | 0.89980 |
| XLM-R large, semilla 7 **con aumentación por traducción** | Colab T4, misma partición | 0.89587 |
| XLM-R large, semilla 2026, 3 folds | Kaggle GPU P100 | 0.89833 |

Pesos guardados con PyTorch / Transformers (`save_pretrained` → `model.safetensors` + tokenizer). Enlace a los pesos: *(pendiente: carpeta compartida de Google Drive)*.

## 1. Modelos preentrenados utilizados

| Uso en el proyecto | Modelo | Fuente (enlace) | Licencia |
|---|---|---|---|
| **Modelo final** (fine-tuning para clasificación de 3 clases) | XLM-RoBERTa large (560M parámetros) | https://huggingface.co/FacebookAI/xlm-roberta-large — Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale*, ACL | MIT |
| Iteración (modelo comparativo) | paraphrase-multilingual-MiniLM-L12-v2 | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 — Reimers & Gurevych (2020) | Apache-2.0 |
| Iteración (modelo comparativo) | distiluse-base-multilingual-cased-v2 | https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2 | Apache-2.0 |
| Aumentación: traducción es→en | MarianMT opus-mt-es-en | https://huggingface.co/Helsinki-NLP/opus-mt-es-en — Tiedemann & Thottingal (2020), *OPUS-MT*, EAMT | Apache-2.0 |
| Aumentación: traducción en→fr / pt / es | MarianMT opus-mt-en-ROMANCE | https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE | Apache-2.0 |
| Aumentación: traducción fr/pt→en | MarianMT opus-mt-ROMANCE-en | https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en | Apache-2.0 |

**Librerías:** PyTorch (BSD-3), Hugging Face Transformers y Tokenizers (Apache-2.0), scikit-learn (BSD-3), pandas,
NumPy, Matplotlib. **Hardware:** GPU NVIDIA T4 (Google Colab) y GPU P100 (Kaggle); los modelos pequeños, en CPU.

## 2. Datos externos

**No se usaron datos externos.** Solo `train.csv` (para entrenar y validar) y `eval.csv` (solo para generar las
predicciones enviadas). El conjunto eval **no** se usó para entrenar, ni para pseudo-etiquetado, ni para elegir hiperparámetros.

## 3. Datos sintéticos (aumentación) — cómo se construyeron

Todos los datos sintéticos se generaron **a partir de `train.csv`** con el script `src/translate_aug.py`:

1. **Segmentación en oraciones** con una expresión regular sobre `.`, `!`, `?`, `…` y **traducción oración por
   oración**, conservando el orden (así se conserva la posición del veredicto, que suele estar al final).
   Decodificación *greedy* (`num_beams=1`), fp16 en GPU, oraciones repetidas traducidas una sola vez.
2. **Direcciones de traducción:**
   - 4 000 reseñas en español, ~1 333 por clase, muestreadas con semilla 42: es→en (`opus-mt-es-en`), luego
     en→fr y en→pt (`opus-mt-en-ROMANCE` con los tokens `>>fr<<` y `>>pt_br<<`).
   - Todas las reseñas detectadas en inglés: en→fr, en→pt y en→es (`opus-mt-en-ROMANCE`, `>>es<<`).
   - Todas las reseñas detectadas en francés y portugués: →en (`opus-mt-ROMANCE-en`).
3. **Code-switching sintético es/en:** para 3 000 de las reseñas en español traducidas se alternan oraciones
   originales y su traducción al inglés (cada oración en inglés con p = 0.5, al menos una de cada idioma).
4. **Etiqueta:** cada reseña sintética hereda la etiqueta de la reseña original.
5. **Sin fuga de información:** cada fila sintética guarda el `src_id` de la reseña original; al entrenar solo se
   usan aumentos cuyo original está en la partición de entrenamiento, nunca en la de validación.
6. **Uso en el entrenamiento:** en cada época se usan todos los originales más una muestra aleatoria de aumentos del
   mismo tamaño (`--aug_ratio 1.0`).

Total generado: ≈26 000 reseñas sintéticas. Conteo por tipo y ejemplos: sección 5 del notebook entregable.

## 4. Descripción del uso de IA generativa en el desarrollo del proyecto

Se usó **Claude (Anthropic)** como asistente de programación y análisis durante el proyecto, para:

- Leer el enunciado y proponer la estructura del repositorio y del pipeline (detección de idioma, validación
  estratificada por etiqueta × idioma, estrategia de aumentación).
- Escribir y depurar el código: `src/train_torch.py`, `src/translate_aug.py`, `src/lang_detect.py`,
  `src/tfidf_posicional.py`, `src/ensemble.py` y los notebooks, incluida la versión para TPU (JAX/Flax).
- Analizar los datos: el experimento que mostró que la última oración concentra el veredicto, que motivó la entrada en par.
- Combinar las predicciones de varias corridas (ensembles) y verificar el formato de las submissions.
- Redactar la documentación (README, este documento y los textos del notebook).

Validación por parte del grupo: el código se probó con pruebas rápidas en CPU antes de las corridas completas, y los
resultados reportados (validación y Kaggle) son los obtenidos en nuestras ejecuciones. El grupo revisó, ejecutó y es
responsable de las decisiones y del código entregado.

**La IA generativa no etiquetó ningún texto de eval** ni produjo predicciones. Todas las predicciones enviadas a Kaggle
provienen de los modelos entrenados por el grupo. Los modelos de traducción (MarianMT) se usaron solo para generar datos
sintéticos de entrenamiento a partir de train, como permite el enunciado.
