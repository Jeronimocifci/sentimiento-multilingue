"""Genera notebooks/04_kaggle_gpu_torch.ipynb: corre src/train_torch.py en una GPU de Kaggle (P100/T4)."""
import nbformat as nbf
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
train_src = (ROOT / 'src' / 'train_torch.py').read_text()
lang_src = (ROOT / 'src' / 'lang_detect.py').read_text()
cells = [
 nbf.v4.new_markdown_cell("""# XLM-R en GPU de Kaggle (PyTorch) — alternativa a la TPU
Misma receta que el notebook v2: entrada en par (últimas 2 oraciones + reseña completa), LLRD, label smoothing,
análisis por idioma. Modelo guardado en **PyTorch** (`modelo/model.safetensors`).

**Cómo correr:** *Add Input* → datos de la competencia · *Settings → Accelerator →* **GPU P100** (o T4) · **Internet On** ·
*Save Version → Save & Run All*.

| Config (celda 4) | Tiempo aprox. en P100/T4 |
|---|---|
| `MODO = 'large_holdout'` (por defecto) | ~1–1.5 h |
| `MODO = 'large_3folds'` | ~3–4 h |
| `MODO = 'base_holdout'` | ~20–30 min |
"""),
 nbf.v4.new_code_cell("!nvidia-smi\nimport torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"),
 nbf.v4.new_code_cell("%%writefile lang_detect.py\n" + lang_src),
 nbf.v4.new_code_cell("%%writefile train_torch.py\n" + train_src),
 nbf.v4.new_code_cell("""import glob, os
MODO = 'large_holdout'   # 'large_holdout' | 'large_3folds' | 'base_holdout'
data_dir = os.path.dirname(glob.glob('/kaggle/input/**/train.csv', recursive=True)[0])
aug = glob.glob('/kaggle/input/**/train_aug_v2.csv', recursive=True)   # opcional: Output del notebook TPU v2 como Input
ARGS = {
    'large_holdout': '--model FacebookAI/xlm-roberta-large --epochs 3 --bs 16 --lr 1e-5',
    'large_3folds':  '--model FacebookAI/xlm-roberta-large --epochs 3 --bs 16 --lr 1e-5 --folds 3',
    'base_holdout':  '--model FacebookAI/xlm-roberta-base --epochs 3 --bs 32 --lr 3e-5',
}[MODO]
if aug: ARGS += f' --aug_csv {aug[0]} --aug_ratio 1.5'
cmd = f'python train_torch.py --data {data_dir} --out /kaggle/working/run {ARGS}'
print(cmd)"""),
 nbf.v4.new_code_cell("!{cmd}"),
 nbf.v4.new_code_cell("""import json, shutil, glob
print(json.dumps(json.load(open('/kaggle/working/run/log.json')), indent=2, default=str)[:1500])
sub = glob.glob('/kaggle/working/run/submission_*.csv')[0]; shutil.copy(sub, '/kaggle/working/submission.csv')
print('✅ submission.csv lista ←', sub)"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}
nbf.write(nb, ROOT / 'notebooks' / '04_kaggle_gpu_torch.ipynb'); print('ok')
