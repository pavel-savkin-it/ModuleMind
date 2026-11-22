"""ModuleHub — the dispatcher 'consciousness': plugs modules into the core.

The AI knows EXACTLY what is connected: the list of connected modules
(hub.describe()) is the current knowledge of the system.
"""
import json
import re
from pathlib import Path

import numpy as np

from .lora import inject_lora, remove_lora, load_lora_state, lora_state_dict, detach_lora
from .log_setup import get_logger

REQUIRED_FIELDS = ("name", "version", "type", "description", "targets", "rank")
SUPPORTED_TYPES = {"lora_adapter"}
_NAME_RE = re.compile(r"^[a-z0-9_]+$")


class ModuleError(Exception):
    """Manifest error or module lifecycle error."""


class ModuleConflictError(ModuleError):
    """Reserved for a future per-layer exclusivity mode (not raised now:
    modules are allowed to stack on the same layers)."""


class ModuleHub:
    def __init__(self, core, modules_dir, log_dir="logs"):
        self.core = core
        self.modules_dir = Path(modules_dir)
        self.log = get_logger("modulemind.hub", log_dir=log_dir)
        self._catalog = {}    # name -> module meta
        self._connected = {}  # name -> {"meta": ..., "handles": ...}
        self.rescan()

    # ---------- discovery ----------

    def rescan(self):
        """Rescan the modules directory; broken manifests are skipped with
        a warning and never crash the system."""
        self._catalog.clear()
        if not self.modules_dir.exists():
            self.log.warning("modules directory %s does not exist", self.modules_dir)
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
                self.log.warning("skipped module %s: %s", child.name, e)
                continue
            self._catalog[meta["name"]] = meta
            self.log.info("found module '%s' v%s", meta["name"], meta["version"])
        return list(self._catalog)

    def scan(self):
        return self.rescan()

    def _validate(self, data, mod_dir: Path) -> dict:
        if not isinstance(data, dict):
            raise ModuleError("manifest must be a JSON object")
        for k in REQUIRED_FIELDS:
            if k not in data:
                raise ModuleError(f"missing required field '{k}'")
        if data["type"] not in SUPPORTED_TYPES:
            raise ModuleError(f"unknown module type: {data['type']}")
        if not isinstance(data["name"], str) or not _NAME_RE.match(data["name"]):
            raise ModuleError("module name must match ^[a-z0-9_]+$")
        t = data["targets"]
        if not isinstance(t, list) or not t or not all(isinstance(x, str) for x in t):
            raise ModuleError("targets must be a non-empty list of strings")
        rank = data["rank"]
        if not isinstance(rank, int) or not 1 <= rank <= 256:
            raise ModuleError("rank must be an integer 1..256")
        alpha = data.get("alpha", rank)
        if not isinstance(alpha, (int, float)) or alpha <= 0:
            raise ModuleError("alpha must be a positive number")
        w = data.get("weights")
        if w is not None and ".." in str(w):
            raise ModuleError("weights must not leave the module directory")
        compat = data.get("core_compat")
        if compat is not None and not isinstance(compat, dict):
            raise ModuleError("core_compat must be an object")
        return {**data, "alpha": alpha, "dir": mod_dir}

    # ---------- lifecycle ----------

    def connect(self, name: str):
        """Connect a module: stack its LoRA on the target layers and (if
        present) load weights. Several modules may share the same layers;
        their contributions add up."""
        if name not in self._catalog:
            raise ModuleError(f"module '{name}' not found in {self.modules_dir}")
        if name in self._connected:
            raise ModuleError(f"module '{name}' is already connected")
        meta = self._catalog[name]

        compat = meta.get("core_compat", {})
        mism = {
            k: (v, getattr(self.core.config, k, None))
            for k, v in compat.items()
            if getattr(self.core.config, k, None) != v
        }
        if mism:
            self.log.warning(
                "module '%s' was built for a different core: %s", name, mism
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
                    raise ModuleError(
                        f"weights of module '{name}' are incompatible: {e}"
                    ) from e
            else:
                self.log.warning(
                    "module '%s' has no weights (untrained): core behavior "
                    "is unchanged", name,
                )
        self._connected[name] = {"meta": meta, "handles": handles}
        self.log.info(
            "connected module '%s' v%s targets=%s rank=%s",
            name, meta["version"], meta["targets"], meta["rank"],
        )

    def disconnect(self, name: str):
        """Disconnect a module; the remaining stacked modules keep working.

        Implementation: snapshot the adapter weights of the remaining
        modules (in connect order), unwrap the whole chain down to the
        original core layers, then re-inject the remaining modules with
        their weights. The disconnected module leaves no trace.
        """
        if name not in self._connected:
            raise ModuleError(f"module '{name}' is not connected")

        remaining = []
        for other, info in self._connected.items():
            if other == name:
                continue
            remaining.append((other, info["meta"], lora_state_dict(info["handles"])))

        detach_lora(self.core)
        for other, meta, snap in remaining:
            handles = inject_lora(
                self.core, meta["targets"], meta["rank"], meta["alpha"]
            )
            load_lora_state(handles, snap)
            self._connected[other]["handles"] = handles

        del self._connected[name]
        self.log.info("disconnected module '%s' (%d modules remain)", name,
                      len(remaining))

    # ---------- the knowledge of the system ----------

    def connected(self):
        return list(self._connected)

    def describe(self):
        """What the AI knows right now = the list of connected modules."""
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
