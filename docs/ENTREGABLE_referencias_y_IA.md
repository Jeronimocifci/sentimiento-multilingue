# Entregable 3 — Referencias a modelos y datos, y uso de IA generativa

**Proyecto:** Sentimiento multilingüe de reseñas — Parte 2 · Aprendizaje de Máquina 2026-20 · Universidad de los Andes

## 1. Modelos preentrenados

| Uso | Modelo | Fuente | Licencia |
|---|---|---|---|
| Clasificador (*fine-tuning*) | XLM-RoBERTa large | https://huggingface.co/FacebookAI/xlm-roberta-large · Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at Scale*, ACL | MIT |
| Traducción es→en (datos sintéticos) | MarianMT opus-mt-es-en | https://huggingface.co/Helsinki-NLP/opus-mt-es-en · Tiedemann & Thottingal (2020), *OPUS-MT*, EAMT | Apache-2.0 |
| Traducción fr/pt→en (v2) | MarianMT opus-mt-ROMANCE-en | https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en | Ver tarjeta del modelo (OPUS-MT) — **verificar** |
| Traducción en→fr / en→pt / en→es (datos sintéticos) | MarianMT opus-mt-en-ROMANCE | https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE | Ver tarjeta del modelo (OPUS-MT) — **verificar antes de entregar** |

Librerías: JAX, Flax, Optax (Apache-2.0), Hugging Face Transformers (Apache-2.0), scikit-learn (BSD-3).

## 2. Datos externos y datos sintéticos

- **Datos externos descargados:** ninguno. Solo se usan `train.csv` (para entrenar) y `eval.csv` (solo para predecir).
- **Datos sintéticos (aumentación)**, todos derivados exclusivamente de `train.csv`:
  1. **Traducción pivot:** 5 000 reseñas en español (≈1 667 por clase, semilla 42) → se segmentan en oraciones →
     es→en con `opus-mt-es-en` → en→fr y en→pt con `opus-mt-en-ROMANCE` (`>>fr<<`, `>>pt_br<<`). Decodificación
     greedy. Se conserva la etiqueta de la reseña original. ≈15 000 reseñas nuevas.
  2. **Code-switching es/en:** con las oraciones alineadas (original + traducción al inglés), cada oración se toma en
     inglés con p = 0.5, garantizando al menos una de cada idioma. 3 000 reseñas.
  3. **EDA:** al 30 % de las reseñas de train se les borra cada palabra con p = 0.1 y se intercambia un par de palabras
     adyacentes, **sin tocar negaciones ni conectores de contraste**.
- **v2 (notebook 03):** 8 000 reseñas en español (es→en→fr/pt), **todas** las reseñas en inglés (→fr, pt, es) y
  **todas** las en francés/portugués (→en); 4 000 code-switching. Cada época usa todos los originales + una muestra
  aleatoria distinta de aumentos (1.5× el número de originales).
- **Modelo clásico auxiliar (stacking):** TF-IDF posicional + regresión logística, combinado con el transformer por
  una regresión logística sobre probabilidades OOF. El modelo de deep learning es el componente principal.
- **Sin fuga:** cada aumento guarda `src_id`; en cada fold solo se usan aumentos de reseñas del fold de entrenamiento.
- **Efecto medido:** ablación en el fold 0 (sin aug / traducción + code-switching / todo), sección 4 del notebook
  (`ablacion_aumentacion.csv`).

## 3. Uso de IA generativa

Se usó **Claude (Anthropic)** como asistente de programación para:
- Diseñar la estructura del repositorio y del notebook a partir del enunciado y del proyecto anterior del grupo.
- Escribir y depurar el código JAX/Flax para TPU (pmap, buckets, optimizador) y la tubería de traducción.
- Redactar la documentación.

El código se validó con una prueba en CPU (8 dispositivos simulados) y un test de sobreajuste del paso de entrenamiento.
Las decisiones finales y los resultados son responsabilidad del grupo.

**La IA no etiquetó ningún texto de eval** ni produjo predicciones: todas las predicciones enviadas provienen del modelo
entrenado en `02_tpu_solucion_final.ipynb`.
