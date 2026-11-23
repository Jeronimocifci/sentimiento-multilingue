# Correr el entrenamiento en otro computador (PyTorch)

Sirve para cualquier PC: **Windows o Linux con GPU NVIDIA** (lo ideal), **Mac Apple Silicon** (MPS) o solo CPU (muy lento).
Script: `src/train_torch.py`. Es la misma receta que el notebook v2: entrada en par, LLRD y label smoothing.

> Los bloques de comandos no llevan comentarios: cópialos tal cual.

---

## 1. Clonar el repo

Si en ese PC tienes `gh` instalado (https://cli.github.com):
```
gh auth login
gh repo clone Jeronimocifci/sentimiento-multilingue
cd sentimiento-multilingue
```

Sin `gh`, con git (te pedirá usuario y un *Personal Access Token* como contraseña, porque el repo es privado):
```
git clone https://github.com/Jeronimocifci/sentimiento-multilingue.git
cd sentimiento-multilingue
```

## 2. Datos de la competencia

Los CSV **no** están en git. Tienes dos opciones.

**Opción A (más simple):** descarga `train.csv`, `eval.csv` y `sample_submission.csv` desde la pestaña *Data* de la
competencia en Kaggle y ponlos en la carpeta `data/` del repo.

**Opción B (Kaggle CLI):** en kaggle.com ve a *Settings → API → Create New Token*. Eso baja `kaggle.json`; muévelo a
`~/.kaggle/kaggle.json` (Linux/Mac) o a `C:\Users\TU_USUARIO\.kaggle\kaggle.json` (Windows). Luego:
```
pip install kaggle
kaggle competitions download -c parte-2-competencia-aprendizaje-de-maquina-2026-20 -p data
```
Descomprime el zip dentro de `data/`. Al final deben quedar `data/train.csv` y `data/eval.csv`.

## 3. Entorno de Python (3.10–3.12)

**Windows (PowerShell):**
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Si PowerShell bloquea el script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repite la activación.

**Linux / Mac:**
```
python3 -m venv .venv
source .venv/bin/activate
```

**Instalar PyTorch:**
- **GPU NVIDIA** (Windows o Linux):
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
  Si tu driver es viejo y falla, usa `cu121` en lugar de `cu124`. Para ver tu GPU y la versión de CUDA del driver: `nvidia-smi`.
- **Mac o solo CPU:**
  ```
  pip install torch
  ```

**Resto de dependencias:**
```
pip install -r requirements-torch.txt
```

**Verificar que vea la GPU** (debe imprimir `cuda` o `mps`):
```
python -c "import torch; print('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

## 4. Entrenar

Elige el comando según la GPU. El script imprime cada 200 pasos la loss y un **ETA de la época**.

| GPU | Comando recomendado | Tiempo aprox. |
|---|---|---|
| NVIDIA ≥ 16 GB (RTX 4080/4090, 3090, A4000…) | XLM-R **large**, 5 folds | 2–4 h |
| NVIDIA 10–12 GB (RTX 3060/3080/4070) | XLM-R **large**, holdout | 1–2 h |
| NVIDIA 6–8 GB / Mac | XLM-R **base**, holdout | 30–60 min |

**Large con 5 folds (lo más fuerte):**
```
python src/train_torch.py --model FacebookAI/xlm-roberta-large --epochs 3 --bs 16 --lr 1e-5 --folds 5
```

**Large en holdout de 10% (un solo modelo):**
```
python src/train_torch.py --model FacebookAI/xlm-roberta-large --epochs 3 --bs 8 --grad_accum 2 --lr 1e-5
```

**Base (rápido):**
```
python src/train_torch.py --model FacebookAI/xlm-roberta-base --epochs 3 --bs 16 --lr 3e-5
```

Si sale `CUDA out of memory`: baja `--bs` a la mitad y sube `--grad_accum` al doble. El batch efectivo queda igual.

**Con los aumentos de la TPU:** cuando el notebook v2 termine, descarga `train_aug_v2.csv` de su Output, ponlo en `data/`
y agrega `--aug_csv data/train_aug_v2.csv --aug_ratio 1.5` al comando.

## 5. Resultados

Todo queda en `outputs/torch_<modelo>_s42/`:
- `submission_torch_*.csv` → **subir a Kaggle** (cuenta como envío).
- `probs_eval.npy` → para ensamblar con otras corridas:
  ```
  python src/ensemble.py outputs/torch_xlm-roberta-large_s42/probs_eval.npy RUTA/OTRA/probs_eval.npy --out submissions/submission_ens.csv
  ```
- `modelo/` → modelo PyTorch (`model.safetensors` + tokenizer): entregable #2 del enunciado.
- `log.json` → accuracy de validación global y por idioma (para el notebook y la sustentación).

## Problemas comunes

| Error | Solución |
|---|---|
| `Torch not compiled with CUDA enabled` | Se instaló el torch de CPU: `pip uninstall torch` y reinstala con la línea `--index-url .../cu124` |
| `CUDA out of memory` | `--bs 4 --grad_accum 4` |
| Error de tokenizer / `sentencepiece` | `pip install sentencepiece protobuf` |
| Operación no soportada en MPS (Mac) | Antepón `PYTORCH_ENABLE_MPS_FALLBACK=1` al comando |
| El PC se suspende | Windows: Configuración → Energía → Nunca suspender. Mac: antepón `caffeinate -i` |
