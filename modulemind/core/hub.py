"""ModuleHub — «сознание»-диспетчер: подключает модули-нейроны к ядру.

ИИ знает РОВНО то, что подключено: список подключённых модулей
(dcx.describe()) — это и есть текущее «знание» системы.
"""
import json
import re
from pathlib import Path

import numpy as np

from .lora import inject_lora, remove_lora, load_lora_state
from .log_setup import get_logger

REQUIRED_FIELDS = ("name", "version", "type", "description", "targets", "rank")
SUPPORTED_TYPES = {"lora_adapter"}
_NAME_RE = re.compile(r"^[a-z0-9_]+$")


class ModuleError(Exception):
    """Ошибка манифеста или жизненного цикла модуля."""


class ModuleConflictError(ModuleError):
    """Два модуля претендуют на одни и те же слои ядра."""


class ModuleHub:
    def __init__(self, core, modules_dir, log_dir="logs"):
        self.core = core
        self.modules_dir = Path(modules_dir)
        self.log = get_logger("modulemind.hub", log_dir=log_dir)
        self._catalog = {}    # name -> мета модуля
        self._connected = {}  # name -> {"meta": ..., "handles": ...}
        self.rescan()

    # ---------- обнаружение ----------

    def rescan(self):
        """Пересканировать каталог модулей; битые манифесты пропускаем с warning."""
        self._catalog.clear()
        if not self.modules_dir.exists():
            self.log.warning("каталог модулей %s не существует", self.modules_dir)
            return []
        for child in sorted(self.modules_dir.iterdir()):
            mf = child / "manifest.json"
            if not child.is_dir() or not mf.exists():
                continue
            try:
                meta = self._validate(
                    json.loads(mf.read_text(encoding="utf-8")), child
                )
            except (json.JSONDecodeError, ModuleError) as e:
                self.log.warning("пропущен модуль %s: %s", child.name, e)
                continue
            self._catalog[meta["name"]] = meta
            self.log.info("найден модуль «%s» v%s", meta["name"], meta["version"])
        return list(self._catalog)

    # синоним: короче в примерах
    def scan(self):
        return self.rescan()

    def _validate(self, data, mod_dir: Path) -> dict:
        if not isinstance(data, dict):
            raise ModuleError("манифест должен быть JSON-объектом")
        for k in REQUIRED_FIELDS:
            if k not in data:
                raise ModuleError(f"нет обязательного поля «{k}»")
        if data["type"] not in SUPPORTED_TYPES:
            raise ModuleError(f"неизвестный тип модуля: {data['type']}")
        if not isinstance(data["name"], str) or not _NAME_RE.match(data["name"]):
            raise ModuleError(
                "имя модуля — строчные латинские буквы, цифры и «_»"
            )
        t = data["targets"]
        if not isinstance(t, list) or not t or not all(isinstance(x, str) for x in t):
            raise ModuleError("targets должен быть непустым списком строк")
        rank = data["rank"]
        if not isinstance(rank, int) or not 1 <= rank <= 256:
            raise ModuleError("rank должен быть целым 1..256")
        alpha = data.get("alpha", rank)
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ModuleError("alpha должен быть положительным числом")
        w = data.get("weights")
        if w is not None:
            if ".." in str(w):
                raise ModuleError("weights не может выходить за каталог модуля")
        compat = data.get("core_compat")
        if compat is not None and not isinstance(compat, dict):
            raise ModuleError("core_compat должен быть объектом")
        return {**data, "alpha": alpha, "dir": mod_dir}

    # ---------- жизненный цикл ----------

    def connect(self, name: str):
        """Подключить модуль: вставить LoRA и (если есть) загрузить веса."""
        if name not in self._catalog:
            raise ModuleError(f"модуль «{name}» не найден в {self.modules_dir}")
        if name in self._connected:
            raise ModuleError(f"модуль «{name}» уже подключён")
        meta = self._catalog[name]

        for other, info in self._connected.items():
            overlap = set(meta["targets"]) & set(info["meta"]["targets"])
            if overlap:
                raise ModuleConflictError(
                    f"слои {sorted(overlap)} уже заняты модулем «{other}»"
                )

        compat = meta.get("core_compat", {})
        mism = {
            k: (v, getattr(self.core.config, k, None))
            for k, v in compat.items()
            if getattr(self.core.config, k, None) != v
        }
        if mism:
            self.log.warning(
                "модуль «%s» собран под другое ядро: %s", name, mism
            )

        try:
            handles = inject_lora(
                self.core, meta["targets"], meta["rank"], meta["alpha"]
            )
        except ValueError as e:
            raise ModuleError(str(e)) from e

        w = meta.get("weights")
        if w:
            wfile = meta["dir"] / w
            if wfile.exists():
                try:
                    load_lora_state(handles, np.load(wfile))
                except (KeyError, ValueError) as e:
                    remove_lora(self.core, handles)
                    raise ModuleError(f"веса модуля «{name}» несовместимы: {e}") from e
            else:
                self.log.warning(
                    "модуль «%s» без весов (необученный): поведение ядра не меняется",
                    name,
                )
        self._connected[name] = {"meta": meta, "handles": handles}
        self.log.info(
            "подключён модуль «%s» v%s targets=%s rank=%s",
            name, meta["version"], meta["targets"], meta["rank"],
        )

    def disconnect(self, name: str):
        """Отключить модуль: ядро возвращается в исходное состояние."""
        if name not in self._connected:
            raise ModuleError(f"модуль «{name}» не подключён")
        remove_lora(self.core, self._connected[name]["handles"])
        del self._connected[name]
        self.log.info("отключён модуль «%s»", name)

    # ---------- «знание» системы ----------

    def connected(self):
        return list(self._connected)

    def describe(self):
        """Что ИИ знает прямо сейчас = список подключённых модулей."""
        return [
            {
                "name": m["meta"]["name"],
                "version": m["meta"]["version"],
                "description": m["meta"]["description"],
                "targets": m["meta"]["targets"],
                "rank": m["meta"]["rank"],
            }
            for m in self._connected.values()
        ]
