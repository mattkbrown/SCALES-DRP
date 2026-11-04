import logging
import sys

import pytest
from keckdrpframework.config.framework_config import ConfigClass

from scalesdrp.core import gpu
from scalesdrp.core.scales_pkg_resources import get_resource_path


@pytest.fixture(autouse=True)
def _reset_gpu_state():
    gpu.gpu_status.cache_clear()
    gpu._logged.clear()
    yield
    gpu.gpu_status.cache_clear()
    gpu._logged.clear()


def _fake_status(monkeypatch, available, reason="test"):
    monkeypatch.setattr(gpu, "gpu_status", lambda: (available, reason))


@pytest.mark.parametrize("value, expected", [
    ("auto", "auto"), ("AUTO", "auto"), ("'auto'", "auto"),
    (True, True), ("true", True), ("True", True), ("gpu", True),
    (False, False), ("false", False), ("cpu", False), (None, False),
])
def test_parse_use_gpu(value, expected):
    assert gpu.parse_use_gpu(value) == expected


def test_parse_use_gpu_rejects_unknown():
    with pytest.raises(ValueError):
        gpu.parse_use_gpu("sometimes")


def test_false_uses_cpu_without_checking_gpu(monkeypatch):
    def boom():
        raise AssertionError("GPU should not be probed when use_gpu = False")
    monkeypatch.setattr(gpu, "gpu_status", boom)
    assert gpu.select_backend(False) == gpu.CPU


def test_auto_falls_back_to_cpu(monkeypatch):
    _fake_status(monkeypatch, False)
    assert gpu.select_backend("auto") == gpu.CPU


def test_auto_uses_gpu_when_available(monkeypatch):
    _fake_status(monkeypatch, True)
    assert gpu.select_backend("auto") == gpu.GPU


def test_true_uses_gpu_when_available(monkeypatch):
    _fake_status(monkeypatch, True)
    assert gpu.select_backend(True) == gpu.GPU


def test_true_raises_without_gpu(monkeypatch):
    _fake_status(monkeypatch, False, "no CUDA devices found")
    with pytest.raises(gpu.GPUUnavailableError, match="no CUDA devices found"):
        gpu.select_backend(True)


def test_backend_logged_once_per_run(monkeypatch, caplog):
    _fake_status(monkeypatch, False)
    with caplog.at_level(logging.INFO, logger=gpu.__name__):
        for _ in range(3):
            gpu.select_backend("auto")
    assert sum("CPU backend" in r.getMessage() for r in caplog.records) == 1


def test_gpu_status_without_cupy(monkeypatch):
    # A None entry in sys.modules makes "import cupy" raise ImportError.
    monkeypatch.setitem(sys.modules, "cupy", None)
    available, reason = gpu.gpu_status()
    assert available is False
    assert "cupy could not be imported" in reason
    assert gpu.gpu_available() is False


def test_scales_cfg_use_gpu_defaults_to_auto():
    cfg_path = get_resource_path("scalesdrp", "configs/scales.cfg")
    config = ConfigClass(cfg_path, default_section="SCALES")
    assert gpu.parse_use_gpu(config.use_gpu) == "auto"
