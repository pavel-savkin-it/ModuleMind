# Обучение модулей на Google Colab / Kaggle (бесплатно)

Наш код специально крошечный: demo_math обучается за ~10 секунд даже на CPU,
поэтому бесплатных лимитов хватает с запасом. GPU не обязателен, но ускоряет
крупные модули.

## Google Colab

1. Открой [colab.research.google.com](https://colab.research.google.com) →
   «Создать блокнот».
2. В первой ячейке:

```python
!git clone https://github.com/pavel-savkin-it/ModuleMind.git
%cd ModuleMind
!pip install -q pytest
!python -m pytest -q          # убедиться, что всё зелёное
```

3. Обучение своего модуля — ячейкой с твоим скриптом по образцу
   `modules/demo_math/train.py` (замени факты на свои пары):

```python
from modules.demo_math.train import train_demo
import json

res = train_demo(steps=400, seed=0, out_dir="weights_out")
print(json.dumps(res, ensure_ascii=False, indent=2))
```

4. Скачай `weights_out/lora.npz` и положи в каталог своего модуля
   (панель «Файлы» → скачать).

## Kaggle Notebook

Аналогично: New Notebook → включить интернет (Settings → Internet → ON) →
те же команды. Лимитов бесплатного CPU/GPU хватает на десятки обучений.

## Предобучение ядра (этап 1 — план)

Пока ядро не предобучено, модули обучаются «поверх случайного» ядра — этого
достаточно для запоминания фактов (demo_math), но не для языка.

План этапа 1: собрать небольшой корпус (10–100 МБ текста), обучить TinyCore
(увеличив размеры) на Colab GPU T4, сохранить чекпоинт ядра как
«релиз ядра №1» (фиксируются seed, конфиг, hash весов) — и все модули
сообщества обучаются уже под этот релиз.
