"""GPU detection and compute-backend selection.

cupy is an optional dependency (``pip install "scalesdrp[gpu]"``). Nothing in
this module imports cupy at module level, so the package imports and runs on
machines without cupy or a CUDA device.

The backend is chosen from the ``use_gpu`` option in ``scales.cfg``:

* ``'auto'``  - use the GPU if one is usable, otherwise the CPU (default)
* ``True``    - GPU required; raise ``GPUUnavailableError`` if none is usable
* ``False``   - always use the CPU
"""

import functools
import logging

CPU = "cpu"
GPU = "gpu"

_AUTO = "auto"
_TRUE_STRINGS = {"true", "yes", "on", "1", "gpu"}
_FALSE_STRINGS = {"false", "no", "off", "0", "cpu", "none", ""}

# Backend decisions already logged in this process, so each is logged once
# per run instead of once per frame.
_logged = set()


class GPUUnavailableError(RuntimeError):
    """Raised when ``use_gpu = True`` but no usable GPU is found."""


@functools.lru_cache(maxsize=1)
def gpu_status():
    """Check once per process whether a usable CUDA GPU is present.

    Returns:
        (available, reason): ``available`` is True if cupy imports and at
        least one CUDA device can be used. ``reason`` is a short description
        for log messages.
    """
    try:
        import cupy as cp
    except Exception as e:  # ImportError, or cupy failing to load CUDA libraries
        return False, f"cupy could not be imported ({type(e).__name__}: {e})"

    try:
        n_devices = cp.cuda.runtime.getDeviceCount()
    except Exception as e:  # CUDARuntimeError: no driver, no device, etc.
        return False, f"no usable CUDA device ({type(e).__name__}: {e})"

    if n_devices < 1:
        return False, "no CUDA devices found"

    try:
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
    except Exception:
        name = "unknown device"
    return True, f"{n_devices} CUDA device(s) found; using {name}"


def gpu_available():
    """Return True if cupy imports and at least one CUDA device is usable."""
    return gpu_status()[0]


def parse_use_gpu(value):
    """Normalise a ``use_gpu`` config value to ``'auto'``, True, or False.

    keckdrpframework turns ``True``/``False`` in a .cfg file into bools and
    leaves other words as strings, so both forms are accepted
    (case-insensitive). A missing option (None) means CPU.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().strip("'\"").lower()
    if text == _AUTO:
        return _AUTO
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    raise ValueError(
        f"Invalid use_gpu value {value!r}; expected 'auto', True, or False.")


def select_backend(use_gpu=_AUTO, logger=None):
    """Choose the compute backend for the current run.

    Args:
        use_gpu: the ``use_gpu`` config value ('auto', True, or False).
        logger: logger for the decision; defaults to this module's logger.

    Returns:
        ``CPU`` or ``GPU``.

    Raises:
        GPUUnavailableError: ``use_gpu`` is True and no GPU is usable.
        ValueError: ``use_gpu`` is not a recognised value.
    """
    logger = logger or logging.getLogger(__name__)
    mode = parse_use_gpu(use_gpu)

    if mode is False:
        backend, message = CPU, "use_gpu = False; using the CPU backend."
    else:
        available, reason = gpu_status()
        if available:
            backend, message = GPU, f"Using the GPU backend: {reason}."
        elif mode is True:
            raise GPUUnavailableError(
                f"use_gpu = True but no GPU is usable: {reason}. Install the "
                "GPU extra (pip install \"scalesdrp[gpu]\") on a CUDA machine, "
                "or set use_gpu = 'auto' or False.")
        else:
            backend, message = CPU, f"No GPU available ({reason}); using the CPU backend."

    key = (mode, backend)
    if key not in _logged:
        _logged.add(key)
        logger.info(message)
    return backend
