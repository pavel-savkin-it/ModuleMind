"""Unified logging setup for ModuleMind (file + console)."""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, log_dir="logs") -> logging.Logger:
    """Logger with a file handler (INFO+) and a console handler (WARNING+).

    Calling again with the same name never duplicates handlers.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_mm_configured", False):
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    d = Path(log_dir)
    d.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        d / "modulemind.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(ch)

    logger._mm_configured = True
    return logger
