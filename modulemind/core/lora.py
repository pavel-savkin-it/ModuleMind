"""LoRA-адаптеры — механизм «нейронов» ModuleMind.

Модуль = маленькие обучаемые матрицы A (rank×d) и B (d×rank), вставленные
поверх выбранных слоёв ядра. Свежий модуль инициализируется так, что его
вклад равен нулю: подключение необученного модуля не меняет поведение ядра.
"""
import logging

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("modulemind.lora")


class LoRALinear(nn.Module):
    """Обёртка над nn.Linear: base(x) + scaling * (x @ Aᵀ @ Bᵀ)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA вставляется только поверх nn.Linear")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


def _replace(root: nn.Module, path: str, new_module: nn.Module):
    parent = root
    parts = path.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _matches(name: str, targets) -> bool:
    return any(name == t or name.endswith("." + t) for t in targets)


def inject_lora(model, targets, rank: int, alpha=None):
    """Вставить LoRA поверх слоёв с именами из targets.

    Возвращает dict {путь_слоя: LoRALinear}.
    """
    if alpha is None:
        alpha = float(rank)
    handles = {}
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and _matches(name, targets):
            lora = LoRALinear(mod, rank, alpha)
            _replace(model, name, lora)
            handles[name] = lora
    if not handles:
        raise ValueError(f"LoRA: ни одна из целей не найдена в ядре: {targets}")
    # Ядро неприкосновенно: градиенты разрешены только адаптерам.
    for n, p in model.named_parameters():
        if "lora" not in n:
            p.requires_grad_(False)
    log.info("LoRA вставлена в %d слоёв: %s", len(handles), sorted(handles))
    return handles


def remove_lora(model, handles) -> int:
    """Аккуратно вынуть свои LoRA-обёртки, вернув исходные слои."""
    for path, lora in handles.items():
        _replace(model, path, lora.base)
    log.info("LoRA снята с %d слоёв", len(handles))
    return len(handles)


def detach_lora(model) -> int:
    """Снять ВСЕ LoRA-обёртки с модели. Возвращает количество."""
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            _replace(model, name, mod.base)
            n += 1
    return n


def lora_state_dict(handles) -> dict:
    """Веса всех адаптеров -> {«путь.A»/«путь.B»: float32 ndarray}."""
    out = {}
    for path, h in handles.items():
        out[f"{path}.A"] = h.lora_A.detach().cpu().numpy().astype(np.float32)
        out[f"{path}.B"] = h.lora_B.detach().cpu().numpy().astype(np.float32)
    return out


def load_lora_state(handles, state):
    """Загрузить веса в существующие адаптеры с проверкой форм."""
    for path, h in handles.items():
        for key, param, shape in (
            (f"{path}.A", h.lora_A, h.lora_A.shape),
            (f"{path}.B", h.lora_B, h.lora_B.shape),
        ):
            if key not in state:
                raise KeyError(f"в весах нет ключа «{key}»")
            arr = np.asarray(state[key])
            if tuple(arr.shape) != tuple(shape):
                raise ValueError(
                    f"{key}: форма {arr.shape} не совпадает с ожидаемой {tuple(shape)}"
                )
            with torch.no_grad():
                param.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
    log.info("загружены веса для %d LoRA-слоёв", len(handles))
