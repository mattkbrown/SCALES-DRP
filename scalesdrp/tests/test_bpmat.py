import numpy as np
import pytest
from astropy.io import fits
from scipy import sparse

from scalesdrp.core import bpmat, rectmat_io
from scalesdrp.primitives.bpm_create import bpm_correction


def random_bpm(shape=(40, 50), frac=0.08, seed=1):
    rng = np.random.default_rng(seed)
    bpm = (rng.random(shape) < frac).astype(np.uint16)
    bpm[0:6, 0:6] = 1   # a cluster that needs a bigger box
    return bpm


def assert_close(a, b):
    assert a.shape == b.shape
    diff = (a - b).tocoo()
    assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-12


@pytest.mark.parametrize("min_good, max_box", [(4, 21), (8, 5), (1, 3)])
def test_without_rectmat_matches_bpm_correction(min_good, max_box):
    bpm = random_bpm()
    assert_close(bpmat.build_bpmat(bpm, min_good_pixels=min_good, max_box_size=max_box),
                 bpm_correction(bpm, min_good_pixels=min_good, max_box_size=max_box))


def two_lenslet_rect(det_shape):
    """
    A toy OPT rectmat on a small detector: lenslet 0 owns columns 0-4, lenslet
    1 owns columns 5-9 (one wavelength, lenslet grid 1x2).
    """
    ny, nx = det_shape
    rows, cols, vals = [], [], []
    for y in range(ny):
        for x in range(nx):
            rows.append(0 if x < 5 else 1)
            cols.append(y * nx + x)
            vals.append(1.0)
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(2, ny * nx))
    return rectmat_io.Rectmat(m, (1, 1, 2), "OPT")


def test_rectmat_fill_uses_only_the_same_lenslet():
    det = (9, 10)
    bpm = np.zeros(det, np.uint16)
    bpm[4, 4] = 1           # last column of lenslet 0, next to lenslet 1
    m = bpmat.build_bpmat(bpm, two_lenslet_rect(det), min_good_pixels=4)
    row = m.getrow(4 * 10 + 4).tocoo()
    assert np.all(row.col % 10 <= 4), "filled from lenslet 1's pixels"
    assert row.data.sum() == pytest.approx(1.0)

    plain = bpmat.build_bpmat(bpm, None, min_good_pixels=4).getrow(44).tocoo()
    assert np.any(plain.col % 10 == 5), "the plain fill should use both sides"


def test_rectmat_fill_falls_back_when_its_lenslet_has_too_few_good_pixels():
    det = (9, 10)
    bpm = np.zeros(det, np.uint16)
    bpm[:, 0:5] = 1
    bpm[4, 4] = 1
    m = bpmat.build_bpmat(bpm, two_lenslet_rect(det), min_good_pixels=4, max_box_size=5)
    row = m.getrow(44).tocoo()
    assert np.all(row.col % 10 >= 5)
    assert row.data.sum() == pytest.approx(1.0)


def test_pixels_outside_the_footprint_use_the_plain_rule():
    det = (9, 10)
    bpm = random_bpm(det, frac=0.15, seed=3)
    rows, cols = [0], [0]   # footprint is a single pixel
    rect = rectmat_io.Rectmat(sparse.csr_matrix(([1.0], (rows, cols)), shape=(1, 90)), (1, 1, 1), "OPT")
    owners = bpmat.pixel_owners(rect, det)
    assert (owners >= 0).sum() == 1
    with_rect = bpmat.build_bpmat(bpm, rect)
    plain = bpmat.build_bpmat(bpm)
    for pix in np.flatnonzero(bpm.ravel()):
        if pix != 0:
            assert_close(with_rect.getrow(pix), plain.getrow(pix))


def test_get_bpmat_builds_once_and_caches(tmp_path, monkeypatch):
    bpm_path = tmp_path / "bpm.fits"
    fits.writeto(bpm_path, random_bpm())
    bpmat._memo.clear()
    calls = []
    real = bpmat.build_bpmat
    monkeypatch.setattr(bpmat, "build_bpmat", lambda *a, **k: calls.append(1) or real(*a, **k))

    first = bpmat.get_bpmat(str(bpm_path), cache_dir=str(tmp_path / "cache"))
    bpmat._memo.clear()
    second = bpmat.get_bpmat(str(bpm_path), cache_dir=str(tmp_path / "cache"))
    assert len(calls) == 1
    assert_close(first, second)
    cached = list((tmp_path / "cache" / "derived" / "bpmat").glob("*.fits"))
    assert len(cached) == 1
    assert fits.getheader(cached[0])["BPMMD5"] and fits.getheader(cached[0])["RMATMD5"] == "NONE"


def test_a_different_bpm_gets_a_different_bpmat(tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    fits.writeto(a, random_bpm(seed=1))
    fits.writeto(b, random_bpm(seed=2))
    bpmat._memo.clear()
    ma = bpmat.get_bpmat(str(a), cache_dir=str(tmp_path))
    mb = bpmat.get_bpmat(str(b), cache_dir=str(tmp_path))
    assert (ma != mb).nnz > 0
    assert len(list((tmp_path / "derived" / "bpmat").glob("*.fits"))) == 2


def test_cached_file_that_does_not_match_is_rebuilt(tmp_path):
    bpm_path = tmp_path / "bpm.fits"
    fits.writeto(bpm_path, random_bpm())
    bpmat._memo.clear()
    bpmat.get_bpmat(str(bpm_path), cache_dir=str(tmp_path))
    cached, = (tmp_path / "derived" / "bpmat").glob("*.fits")
    with fits.open(cached, mode="update") as hdul:
        hdul[0].header["BPMMD5"] = "tampered"
    bpmat._memo.clear()
    bpmat.get_bpmat(str(bpm_path), cache_dir=str(tmp_path))
    assert fits.getheader(cached)["BPMMD5"] != "tampered"


# ---------- frame-level resolution ----------

from types import SimpleNamespace

from scalesdrp.core import calib_resolve


def ifs_frame_setup(tmp_path, monkeypatch, with_rectmat=True):
    """A full-size BPM with a few bad pixels, and (optionally) a LowRes-K rectmat."""
    calib = tmp_path / "calib"
    calib.mkdir()
    bpm = np.zeros((2048, 2048), np.uint8)
    bpm[100, 100] = bpm[500, 7] = 1
    fits.writeto(calib / "bpm_ifs.fits", bpm)
    if with_rectmat:
        m = sparse.csr_matrix((np.ones(4), ([0, 0, 1, 1], [100 * 2048 + 99, 100 * 2048 + 101,
                                                           100 * 2048 + 100, 7])),
                              shape=(2, rectmat_io.N_DET_PIX))
        rectmat_io.write_rectmat(calib / "LowRes-K_OPT.fits", m, (1, 1, 2), [2.0])
    monkeypatch.setattr(calib_resolve, "get_resource_path", lambda pkg, p: calib)
    ctx = SimpleNamespace(calib_file_path="calib/", bpm_ifs="bpm_ifs.fits", bpm_img="bpm_img.fits",
                          OPT_rmat_LowRes_K="LowRes-K_OPT.fits", lowres_lenslet_grid=(1, 2))
    cfg = SimpleNamespace(instrument=SimpleNamespace(caldb_cache_dir=str(tmp_path / "cache")))
    hdr = fits.Header({"CAMERA": "IFS", "MCLOCK": "5.0 MHz", "MODSLNAM": "LowRes",
                       "DSPRSNAM": "K-Prism", "FILTER": "None"})
    return cfg, ctx, hdr


def cached_headers(tmp_path):
    return [fits.getheader(p) for p in (tmp_path / "cache" / "derived" / "bpmat").glob("*.fits")]


def test_ifs_science_frame_bpmat_uses_the_modes_rectmat(tmp_path, monkeypatch):
    cfg, ctx, hdr = ifs_frame_setup(tmp_path, monkeypatch)
    bpmat._memo.clear()
    bpm_path = calib_resolve.resolve_bpm(cfg, ctx, hdr)
    calib_resolve.resolve_bpmat(cfg, ctx, hdr, bpm_path)
    h, = cached_headers(tmp_path)
    assert h["RMATFILE"] == "LowRes-K_OPT.fits"


def test_calibration_frame_bpmat_does_not_use_a_rectmat(tmp_path, monkeypatch):
    cfg, ctx, hdr = ifs_frame_setup(tmp_path, monkeypatch)
    bpmat._memo.clear()
    bpm_path = calib_resolve.resolve_bpm(cfg, ctx, hdr)
    calib_resolve.resolve_bpmat(cfg, ctx, hdr, bpm_path, use_rectmat=False)
    h, = cached_headers(tmp_path)
    assert h["RMATFILE"] == "NONE"


def test_missing_rectmat_falls_back_to_the_bpm_alone(tmp_path, monkeypatch):
    cfg, ctx, hdr = ifs_frame_setup(tmp_path, monkeypatch, with_rectmat=False)
    bpmat._memo.clear()
    bpm_path = calib_resolve.resolve_bpm(cfg, ctx, hdr)
    m = calib_resolve.resolve_bpmat(cfg, ctx, hdr, bpm_path)
    assert m.shape == (rectmat_io.N_DET_PIX, rectmat_io.N_DET_PIX)
    h, = cached_headers(tmp_path)
    assert h["RMATFILE"] == "NONE"
