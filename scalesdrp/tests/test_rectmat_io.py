import numpy as np
import pytest
from astropy.io import fits
from scipy import sparse

from scalesdrp.core import rectmat_io as rio

CUBE = (5, 3, 4)  # nlam, ny, nx
N_FLUX = int(np.prod(CUBE))


def opt_matrix(seed=0, explicit_zeros=0):
    """Small OPT rectmat: each flux element sees a few detector pixels."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(N_FLUX), 3)
    cols = rng.integers(0, rio.N_DET_PIX, size=rows.size)
    vals = rng.random(rows.size) + 0.1
    if explicit_zeros:
        rows = np.concatenate([rows, np.zeros(explicit_zeros, int)])
        cols = np.concatenate([cols, np.arange(explicit_zeros)])
        vals = np.concatenate([vals, np.zeros(explicit_zeros)])
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(N_FLUX, rio.N_DET_PIX))
    m.sum_duplicates()
    return m


def waves():
    return np.linspace(2.0, 2.4, CUBE[0])


def assert_same(a, b):
    assert a.shape == b.shape
    assert (a != b).nnz == 0


def test_round_trip_opt(tmp_path):
    m = opt_matrix()
    path = rio.write_rectmat(tmp_path / "r.fits", m, CUBE, waves(),
                             header={"IFSMODE": "LowRes-K", "STATEID": ("abc", "state")})
    rect = rio.read_rectmat(path)
    assert_same(rect.matrix, m)
    assert rect.kind == "OPT"
    assert rect.cube_shape == CUBE
    np.testing.assert_array_equal(rect.wavelengths, waves())
    assert rect.header["CALTYPE"] == "rectmat_opt"
    assert rect.header["IFSMODE"] == "LowRes-K"
    assert rect.header["STATEID"] == "abc"
    assert rect.header["MORIENT"] == "FLUX,DET"


def test_round_trip_c2(tmp_path):
    m = opt_matrix().T.tocsr()
    rect = rio.read_rectmat(rio.write_rectmat(tmp_path / "r.fits", m, CUBE, waves()))
    assert rect.kind == "C2"
    assert rect.header["CALTYPE"] == "rectmat_c2"
    assert_same(rect.matrix, m)
    assert_same(rect.to_opt(), opt_matrix())


def test_explicit_zeros_are_dropped(tmp_path):
    m = opt_matrix(explicit_zeros=50)
    stored = m.nnz
    rect = rio.read_rectmat(rio.write_rectmat(tmp_path / "r.fits", m, CUBE, waves()))
    assert rect.matrix.nnz == stored - 50
    assert rect.header["NSTORED"] == stored
    assert m.nnz == stored, "the caller's matrix must not be modified"


def test_wavelength_table(tmp_path):
    w = np.array([2.0, 2.1, 2.25, 2.3, 2.4])
    path = rio.write_rectmat(tmp_path / "r.fits", opt_matrix(), CUBE, w, monowave=w + 0.001)
    with fits.open(path) as hdul:
        tab = hdul[rio.WAVE_EXTNAME]
        np.testing.assert_array_equal(tab.data["CHANNEL"], np.arange(5))
        np.testing.assert_array_equal(tab.data["WAVE"], w)
        np.testing.assert_allclose(tab.data["MONOWAVE"], w + 0.001)
        assert tab.header["UNIFORM"] is False
        assert tab.header["WAVSTART"] == 2.0 and tab.header["WAVEND"] == 2.4


def test_uniform_flag():
    assert rio.is_uniform(np.linspace(1.95, 2.45, 27))
    assert not rio.is_uniform([2.0, 2.1, 2.25])


def test_lensmap_counts_slices_per_lenslet(tmp_path):
    nlam, ny, nx = CUBE
    # lenslet (y=1, x=2) responds in 3 slices, everything else not at all
    flux = [2 + 1 * nx + lam * nx * ny for lam in (0, 2, 4)]
    m = sparse.csr_matrix((np.ones(3), (flux, [10, 20, 30])), shape=(N_FLUX, rio.N_DET_PIX))
    rect = rio.read_rectmat(rio.write_rectmat(tmp_path / "r.fits", m, CUBE, waves()))
    expected = np.zeros((ny, nx), int)
    expected[1, 2] = 3
    np.testing.assert_array_equal(rect.lensmap, expected)
    assert rect.header["NLENSUSE"] == 1


@pytest.mark.parametrize("bad", [
    dict(cube_shape=(4, 3, 4)),
    dict(wavelengths=np.linspace(2, 3, 4)),
    dict(kind="C2"),
])
def test_write_rejects_inconsistent_input(tmp_path, bad):
    args = dict(matrix=opt_matrix(), cube_shape=CUBE, wavelengths=waves())
    args.update(bad)
    with pytest.raises(ValueError):
        rio.write_rectmat(tmp_path / "r.fits", **args)


def test_sources_table(tmp_path):
    src = [{"filename": "a.fits", "monowave": 2.0, "date_obs": "2026-01-01T00:00:00"},
           {"filename": "bb.fits", "monowave": 2.1, "date_obs": "2026-01-01T00:01:00"}]
    path = rio.write_rectmat(tmp_path / "r.fits", opt_matrix(), CUBE, waves(), sources=src)
    with fits.open(path) as hdul:
        assert list(hdul[rio.SOURCES_EXTNAME].data["FILENAME"]) == ["a.fits", "bb.fits"]


def test_legacy_npz_with_cube_shape(tmp_path):
    m = opt_matrix()
    path = tmp_path / "old.npz"
    np.savez_compressed(path, format=np.array(b"csr"), shape=np.array(m.shape), data=m.data,
                        indices=m.indices, indptr=m.indptr, cube_shape=np.array(CUBE))
    rect = rio.read_rectmat(str(path))
    assert rect.cube_shape == CUBE
    assert rect.wavelengths is None


def test_legacy_npz_needs_lenslet_grid(tmp_path):
    path = tmp_path / "old.npz"
    sparse.save_npz(path, opt_matrix())
    with pytest.raises(ValueError):
        rio.read_rectmat(str(path))
    assert rio.read_rectmat(str(path), lenslet_grid=(3, 4)).cube_shape == CUBE


def test_convert_npz(tmp_path):
    npz = tmp_path / "old.npz"
    sparse.save_npz(npz, opt_matrix())
    lams = tmp_path / "old_lams.fits"
    fits.writeto(lams, waves())
    out = rio.convert_npz_rectmat(str(npz), tmp_path / "new.fits", str(lams),
                                  header={"IFSMODE": "LowRes-K"}, lenslet_grid=(3, 4))
    rect = rio.read_rectmat(out)
    assert_same(rect.matrix, opt_matrix())
    np.testing.assert_array_equal(rect.wavelengths, waves())
    assert "Converted from old.npz" in str(rect.header["HISTORY"])


def test_scales_basic_load_rectmat_reads_fits(tmp_path):
    import scalesdrp.primitives.scales_basic as scbasic
    path = rio.write_rectmat(tmp_path / "r.fits", opt_matrix(), CUBE, waves())
    m, shape = scbasic.load_rectmat(path)
    assert shape == CUBE
    assert_same(m, opt_matrix())
