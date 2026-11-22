# ModuleMind

**Модульный ИИ: ядро-«сознание» + подключаемые модули-нейроны.**

Ядро само по себе не знает ничего конкретного. Подключили модуль «математика» —
ИИ математик. Подключили модуль «английский язык» — говорит по-английски.
Отключили — знание исчезло. **ИИ знает ровно то, что к нему подключили.**

```
запрос → ЯДРО (TinyCore, заморожено) → [LoRA-модули: ✅ математика ✅ английский …] → ответ
```

- **Ядро (TinyCore)** — крошечный трансформер, неприкосновенен: градиенты
  разрешены только модулям.
- **Модуль-нейрон** = `manifest.json` + маленький файл весов `lora.npz`
  (LoRA-адаптер). Сообщество обучает модули на Colab/Kaggle по нашей
  [инструкции](docs/MODULE_GUIDE.md).
- **ModuleHub** — подключает/отключает модули; каждое действие пишется в лог.

## Быстрый старт (CPU, слабый ПК — ок)

```bash
git clone git@github.com:pavel-savkin-it/ModuleMind.git
cd ModuleMind
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# обучить демо-модуль (~10 секунд на CPU)
python -m modules.demo_math.train

# все мини-тесты (31 шт., ~7 секунд)
python -m pytest -v
```

## Подключить модуль — 6 строк

```python
import torch
from modulemind.core import ModuleHub, TinyCore, CoreConfig, CharTokenizer

core = TinyCore(CoreConfig(vocab_size=14, d_model=64, n_layers=2, n_heads=4, max_seq=16, seed=0))
tok = CharTokenizer("0123456789+= ")
hub = ModuleHub(core, "modules")

hub.connect("demo_math")            # ← ИИ стал математиком
print(hub.describe())               # что ИИ знает прямо сейчас
ids = torch.tensor([tok.encode("2+1=")])
print(tok.decode([int(core(ids)[0, -1].argmax())]))   # → «3»
```

## Структура

```
modulemind/core/   ядро: tiny_core, lora, hub, tokenizer, config, log_setup
modules/           подключаемые модули (demo_math — образец для сообщества)
tests/             31 мини-тест (тесты всегда пишутся ПЕРЕД кодом)
docs/              архитектура, формат модуля, гайд сообщества, Colab, HANDOFF
logs/              modulemind.log (в git не входит)
```

## Документация

| Документ | О чём |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | как всё устроено и почему |
| [docs/MODULE_FORMAT.md](docs/MODULE_FORMAT.md) | формат модуля (для людей и ИИ-агентов) |
| [docs/MODULE_GUIDE.md](docs/MODULE_GUIDE.md) | как сообществу создать и обучить свой модуль |
| [docs/COLAB_TRAINING.md](docs/COLAB_TRAINING.md) | обучение модулей на Colab/Kaggle |
| [docs/HANDOFF.md](docs/HANDOFF.md) | документ продолжения работы (при обрыве связи) |

## Статус

- [x] Этап 0 (готово): MVP — ядро + LoRA-модули + демо-модуль «сложение 0..4» + 31 тест
- [ ] Этап 1: предобучение ядра на Colab/Kaggle (свой корпус)
- [ ] Этап 2: модуль «английский язык» на обученном ядре
- [ ] Этап 3: CI для PR сообщества, слияние/конфликт модулей

Референс подхода: [minimind](https://github.com/jingyaogong/minimind) — мы идём тем же путём, но меньше и с модульностью.
