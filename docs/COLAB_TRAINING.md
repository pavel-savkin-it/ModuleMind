# Training modules on Google Colab / Kaggle (free)

The code is deliberately tiny: demo_math trains in ~10 seconds even on CPU,
so free quotas are plenty. GPU is optional but speeds up bigger modules.

## Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) →
   "New notebook".
2. First cell:

```python
!git clone https://github.com/pavel-savkin-it/ModuleMind.git
%cd ModuleMind
!pip install -q pytest
!python -m pytest -q          # verify everything is green
```

3. Train your module — a cell with your script modeled on
   `modules/demo_math/train.py` (replace the facts with your data):

```python
from modules.demo_math.train import train_demo
import json

res = train_demo(steps=400, seed=0, out_dir="weights_out")
print(json.dumps(res, ensure_ascii=False, indent=2))
```

4. Download `weights_out/lora.npz` (Files panel) and put it into your module
   directory.

## Kaggle Notebook

The same: New Notebook → Settings → Internet → ON → the same commands. Free
CPU/GPU quotas cover dozens of training runs.

## Pretraining the core (stage 1b — plan)

Stage 1a (done, local CPU) proved the pipeline: `core_v1.npz` was trained
from scratch on a toy corpus, and modules (`english`, `demo_math`) were
trained under it.

Stage 1b plan: a real text corpus (10–100 MB), a bigger TinyCore
(d_model 256–512), trained on Colab T4 GPU; the checkpoint is released as
'core release #2' (config + weight hash fixed) — and all community modules
train under that release.
