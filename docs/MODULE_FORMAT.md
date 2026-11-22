# Формат модуля-нейрона (v0.1)

Модуль — это каталог:

```
modules/<имя>/
├── manifest.json          манифест (ниже)
└── weights/lora.npz       веса адаптера (float32, маленький файл)
```

Имя каталога = имя модуля: строчные латинские буквы, цифры, `_`.

## manifest.json

```json
{
  "name": "demo_math",
  "version": "0.1.0",
  "type": "lora_adapter",
  "description": "Что модуль даёт ядру (по-русски, кратко)",
  "targets": ["attn.q", "attn.v", "lm_head"],
  "rank": 8,
  "alpha": 16.0,
  "weights": "weights/lora.npz",
  "core_compat": {"d_model": 64, "n_layers": 2, "vocab_size": 14}
}
```

| Поле | Обяз. | Правила |
|---|---|---|
| `name` | да | `^[a-z0-9_]+$`, совпадает с именем каталога |
| `version` | да | семантическая версия, строки вида `1.2.0` |
| `type` | да | пока только `lora_adapter` |
| `description` | да | что ИИ «знает» с этим модулем |
| `targets` | да | непустой список имён слоёв ядра (`attn.q`, `attn.v`, `attn.k`, `attn.o`, `mlp.fc`, `mlp.proj`, `lm_head`) |
| `rank` | да | целое 1..256 (размер адаптера) |
| `alpha` | нет | масштаб (по умолчанию = rank); scaling = alpha/rank |
| `weights` | нет | путь ВНУТРИ каталога модуля; без `..` |
| `core_compat` | нет | ожидаемые параметры ядра; при расхождении — warning |

Битый манифест не ломает систему: ModuleHub пропускает модуль и пишет warning
в лог. Два модуля с пересекающимися `targets` одновременно подключить нельзя.

## weights/lora.npz

Ключи: `<путь слоя>.A` и `<путь слоя>.B`.

- `A` — форма `(rank, in_features)`; `B` — форма `(out_features, rank)`.
- Все массивы `float32`.
- `B = 0` → модуль ничего не меняет (безопасное подключение).

Вклад модуля: `base(x) + (alpha/rank) · x · Aᵀ · Bᵀ`.

## Совместимость с ядром

Веса обязаны совпадать по формам с текущим ядром, иначе `ModuleError`
(например, веса под `d_model=64` не встанут в ядро `d_model=32`).
ВНИМАНИЕ: совпадение форм — необходимое, но не достаточное условие:
адаптер обучен под конкретные веса ядра (см. инвариант 4 в ARCHITECTURE.md).
Указывай `core_compat` и seed ядра, под которым обучался, в description.

## Правка весов ИИ-агентом (например, агентом ChatGPT)

Веса — обычный numpy-архив, правится без torch:

```python
import numpy as np

npz = np.load("modules/demo_math/weights/lora.npz")
w = {k: npz[k].copy() for k in npz.files}     # ключи: *.A, *.B
w["lm_head.B"] *= 1.1                          # пример правки
np.savez_compressed("modules/demo_math/weights/lora.npz", **w)
```

Правила для агентов: только float32, только существующие ключи и формы,
после правки обязательно `python -m pytest tests/test_e2e.py -v` —
приёмочный тест должен остаться зелёным.
