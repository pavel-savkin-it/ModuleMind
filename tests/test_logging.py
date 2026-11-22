"""Мини-тесты логирования (чтобы всегда было видно, «в чём дело»)."""
from modulemind.core.log_setup import get_logger


def test_log_file_written(tmp_path):
    lg = get_logger("mm_test_file", log_dir=tmp_path)
    lg.info("hello-test-123")
    for h in lg.handlers:
        h.flush()
    content = (tmp_path / "modulemind.log").read_text(encoding="utf-8")
    assert "hello-test-123" in content


def test_no_duplicate_handlers(tmp_path):
    a = get_logger("mm_test_dup", log_dir=tmp_path)
    n_before = len(a.handlers)
    b = get_logger("mm_test_dup", log_dir=tmp_path)
    assert a is b
    assert len(a.handlers) == n_before
