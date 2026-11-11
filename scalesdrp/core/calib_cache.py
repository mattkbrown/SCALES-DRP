"""
Local cache for calibration files.

Layout under the cache root:

    caldb/index.json            caldb metadata records and when they were fetched
    caldb/files/<id>/<filename> files downloaded from the caldb
    derived/bpmat/<key>.fits    bad-pixel matrices built locally (see bpmat.py)

The root is, in order: an explicit path, $SCALES_CALIB_CACHE, or
~/.cache/scalesdrp. Writes go to a temporary file and are moved into place,
under a per-file lock, so several DRP processes can share one cache.
"""
import contextlib
import fcntl
import hashlib
import os

ENV_CACHE_DIR = "SCALES_CALIB_CACHE"

_md5_memo = {}


def cache_root(path=None):
    root = path or os.environ.get(ENV_CACHE_DIR) or os.path.join(
        os.path.expanduser("~"), ".cache", "scalesdrp")
    os.makedirs(root, exist_ok=True)
    return root


def file_md5(path, chunk=1 << 22):
    """MD5 of a file, remembered per (path, size, mtime) for this process."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _md5_memo:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        _md5_memo[key] = h.hexdigest()
    return _md5_memo[key]


@contextlib.contextmanager
def file_lock(path):
    """Exclusive lock on <path>.lock for the duration of the block."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(f"{path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def tmp_path_for(path):
    """A temporary name next to path, for writing then os.replace."""
    return f"{path}.tmp{os.getpid()}"
