"""
Bad-pixel matrices (bpmats), built locally from a bad pixel mask and, for
IFS frames, the rectmat of the frame's mode, then cached.

A bpmat is a sparse (n_pix, n_pix) matrix applied to a flattened slope image:
good pixels map to themselves, and each bad pixel is replaced by an
inverse-distance weighted average of good pixels in the smallest square box
(3x3, 5x5, ... up to max_box_size) that holds at least min_good_pixels of
them. Bad pixels with no such box are left unchanged.

With a rectmat, a bad pixel inside the rectmat footprint is only filled
from good pixels that belong to the same lenslet (the lenslet with the
largest weight at that pixel), so light from a neighbouring lenslet's trace
is never mixed in. If the box limit is reached without enough such pixels,
it falls back to the rule above. Pixels outside the footprint follow the
rule above.

Without a rectmat (imager frames, or IFS calibration frames taken before a
rectmat exists) the result is the same as bpm_create.bpm_correction.

bpmats are not served by the caldb: get_bpmat builds one on first use and
caches it under derived/bpmat/ in the calibration cache, keyed by the MD5s
of its inputs, the parameters and BPMAT_VERSION.
"""
import hashlib
import logging
import os
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from scipy import sparse

from scalesdrp.core import rectmat_io
from scalesdrp.core.calib_cache import cache_root, file_lock, file_md5, tmp_path_for
from scalesdrp.core.sparse_matrix_utils import pack_matrix_fits, unpack_matrix_fits

# Bump when build_bpmat changes, so cached bpmats are rebuilt
BPMAT_VERSION = 1
MIN_GOOD_PIXELS = 4
MAX_BOX_SIZE = 21

log = logging.getLogger("SCALES")

_memo = OrderedDict()
_MEMO_SIZE = 4


def pixel_owners(rect, det_shape):
    """
    (ny_det, nx_det) lenslet index (y*nx + x) with the largest weight at each
    detector pixel, or -1 outside the rectmat footprint.
    """
    _, ny, nx = rect.cube_shape
    coo = rect.to_opt().tocoo()
    keep = coo.data > 0
    flux, pix, val = coo.row[keep], coo.col[keep], coo.data[keep]
    order = np.lexsort((-val, pix))
    pix, flux = pix[order], flux[order]
    first = np.ones(pix.size, dtype=bool)
    first[1:] = pix[1:] != pix[:-1]
    owners = np.full(int(np.prod(det_shape)), -1, dtype=np.int64)
    owners[pix[first]] = flux[first] % (ny * nx)
    return owners.reshape(det_shape)


def _box_weights(i, j, good, min_good, max_box, owners=None):
    """
    (row, col) indices and weights for bad pixel (i, j) from the good pixels
    around it (only those with the same owner as (i, j), if owners is given),
    or None if no box up to max_box has min_good of them.
    """
    ypix, xpix = good.shape
    sbox = 3
    while sbox <= max_box:
        half = sbox // 2
        ys, ye = max(0, i - half), min(ypix, i + half + 1)
        xs, xe = max(0, j - half), min(xpix, j + half + 1)
        box = good[ys:ye, xs:xe]
        if owners is not None:
            box = box & (owners[ys:ye, xs:xe] == owners[i, j])
        if np.count_nonzero(box) >= min_good:
            yy, xx = np.nonzero(box)
            yy, xx = yy + ys, xx + xs
            dist = np.hypot(yy - i, xx - j)
            keep = dist > 0
            if not np.any(keep):
                return None
            w = 1.0 / dist[keep]
            return yy[keep], xx[keep], w / w.sum()
        sbox += 2
    return None


def build_bpmat(bpmap, rect=None, min_good_pixels=MIN_GOOD_PIXELS, max_box_size=MAX_BOX_SIZE):
    """
    Build a bpmat.

    bpmap : 2-D array, nonzero = bad pixel.
    rect : rectmat_io.Rectmat, optional. Restricts the fill of bad pixels in
        its footprint to pixels of the same lenslet.
    """
    bad = np.asarray(bpmap) != 0
    ypix, xpix = bad.shape
    good = ~bad
    owners = pixel_owners(rect, bad.shape) if rect is not None else None

    rows = [np.flatnonzero(good)]
    cols = [rows[0]]
    vals = [np.ones(rows[0].size)]

    bad_y, bad_x = np.nonzero(bad)
    for i, j in zip(bad_y, bad_x):
        pix = i * xpix + j
        fill = None
        if owners is not None and owners[i, j] >= 0:
            fill = _box_weights(i, j, good, min_good_pixels, max_box_size, owners)
        if fill is None:
            fill = _box_weights(i, j, good, min_good_pixels, max_box_size)
        if fill is None:
            rows.append([pix]); cols.append([pix]); vals.append([1.0])
            continue
        yy, xx, w = fill
        rows.append(np.full(w.size, pix)); cols.append(yy * xpix + xx); vals.append(w)

    n = ypix * xpix
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def _cache_key(bpm_md5, rect_md5, min_good, max_box):
    text = f"{BPMAT_VERSION}|{bpm_md5}|{rect_md5 or 'none'}|{min_good}|{max_box}"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def _expected_header(bpm_path, rect_path, bpm_md5, rect_md5, min_good, max_box):
    hdr = fits.Header()
    hdr["BPMATVER"] = (BPMAT_VERSION, "bpmat algorithm version")
    hdr["BPMFILE"] = (os.path.basename(bpm_path), "bad pixel mask")
    hdr["BPMMD5"] = (bpm_md5, "MD5 of the bad pixel mask")
    hdr["RMATFILE"] = (os.path.basename(rect_path) if rect_path else "NONE", "rectmat used")
    hdr["RMATMD5"] = (rect_md5 or "NONE", "MD5 of the rectmat")
    hdr["MINGOOD"] = (min_good, "min good pixels to fill a bad one")
    hdr["MAXBOX"] = (max_box, "largest box searched [pix]")
    return hdr


def get_bpmat(bpm_path, rectmat_path=None, cache_dir=None, min_good_pixels=MIN_GOOD_PIXELS,
              max_box_size=MAX_BOX_SIZE, lenslet_grid=None, logger=None):
    """
    The bpmat for a bad pixel mask (and rectmat), from the cache or built
    and cached now. Returns a scipy.sparse CSR matrix.
    """
    logger = logger or log
    bpm_md5 = file_md5(bpm_path)
    rect_md5 = file_md5(rectmat_path) if rectmat_path else None
    key = _cache_key(bpm_md5, rect_md5, min_good_pixels, max_box_size)
    if key in _memo:
        _memo.move_to_end(key)
        return _memo[key]

    path = os.path.join(cache_root(cache_dir), "derived", "bpmat", key + ".fits")
    expected = _expected_header(bpm_path, rectmat_path, bpm_md5, rect_md5,
                                min_good_pixels, max_box_size)
    with file_lock(path):
        matrix = _read_cached(path, expected, logger)
        if matrix is None:
            logger.info(f"Building bpmat from {os.path.basename(bpm_path)}"
                        + (f" and {os.path.basename(rectmat_path)}" if rectmat_path else ""))
            rect = (rectmat_io.read_rectmat(rectmat_path, lenslet_grid=lenslet_grid)
                    if rectmat_path else None)
            matrix = build_bpmat(fits.getdata(bpm_path), rect, min_good_pixels, max_box_size)
            tmp = tmp_path_for(path)
            pack_matrix_fits(matrix, header=expected).writeto(tmp, overwrite=True)
            os.replace(tmp, path)
            logger.info(f"Cached bpmat {path}")

    _memo[key] = matrix
    while len(_memo) > _MEMO_SIZE:
        _memo.popitem(last=False)
    return matrix


def _read_cached(path, expected, logger):
    if not os.path.exists(path):
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul[0].header
            if any(hdr.get(k) != expected[k] for k in ("BPMATVER", "BPMMD5", "RMATMD5",
                                                       "MINGOOD", "MAXBOX")):
                logger.warning(f"Cached bpmat {path} does not match its inputs; rebuilding")
                return None
            return unpack_matrix_fits(hdul).tocsr()
    except (OSError, KeyError, ValueError) as e:
        logger.warning(f"Cached bpmat {path} is unreadable ({e}); rebuilding")
        return None
