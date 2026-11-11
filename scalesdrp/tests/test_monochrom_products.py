import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from scalesdrp.core import rectmat_io
from scalesdrp.primitives.ProcessMonochrom import ProcessMonochrom


@pytest.fixture
def pm():
    obj = ProcessMonochrom.__new__(ProcessMonochrom)
    obj.logger = logging.getLogger("test")
    obj.config = SimpleNamespace(instrument=SimpleNamespace(apply_destripe=True))
    obj.lmin, obj.lmax = 1.95, 2.45
    obj.mono_rows = pd.DataFrame({
        "filename": ["b.fits", "a.fits"],
        "MONOWAVE": [2000.0, 2100.0],
        "DATE-OBS": ["2026-05-02T10:00:00", "2026-05-01T09:00:00"],
        "MJD": [61162.4, 61161.3],
        "CAMERA": ["IFS", "IFS"],
        "MODSLNAM": ["LowRes", "LowRes"],
        "DSPRSNAM": ["K-Prism", "K-Prism"],
        "STATEID": ["s1", "s1"],
    })
    return obj


def header(pm, interpolated=True):
    return pm._rectmat_header("LowRes-K", interpolated=interpolated, cropsize=8, fluxcut=0.0,
                              oversamp=2, ninterp=27, spot_thresh=90.0, spot_sigma=1.2,
                              match_dist=3)


def test_rectmat_header_records_state_dates_and_parameters(pm):
    hdr = fits.Header()
    rectmat_io.merge_header(hdr, header(pm))
    assert hdr["IFSMODE"] == "LowRes-K"
    assert hdr["STATEID"] == "s1"
    assert hdr["DSPRSNAM"] == "K-Prism"
    assert hdr["DATE-OBS"] == "2026-05-01T09:00:00"
    assert hdr["DATE-END"] == "2026-05-02T10:00:00"
    assert hdr["MJD-OBS"] == pytest.approx(61161.3)
    assert hdr["INTERP"] is True and hdr["NINTERP"] == 27
    assert hdr["NFRAMES"] == 2 and hdr["CROPSIZE"] == 8 and hdr["DESTRIPE"] is True


def test_non_interpolated_header_has_no_ninterp(pm):
    assert "NINTERP" not in header(pm, interpolated=False)


def test_sources_are_monochromator_frames_in_um(pm):
    src = pm._mono_sources()
    assert [s["filename"] for s in src] == ["b.fits", "a.fits"]
    assert [s["monowave"] for s in src] == [2.0, 2.1]


def test_lsf_and_lflat_link_back_to_their_rectmat(pm, tmp_path):
    opt = tmp_path / "LowRes-K_OPT_intp_rectmat.fits"
    opt.write_bytes(b"not really a rectmat")
    waves = np.linspace(2.0, 2.4, 3)
    lsf = np.random.default_rng(0).random((3, 3, 2, 4))
    pm._write_lsf_lflat(str(tmp_path / "LowRes-K_OPT_intp"), lsf, lsf[0, 0],
                        waves, header(pm), str(opt))
    with fits.open(tmp_path / "LowRes-K_OPT_intp_lsf.fits") as hdul:
        np.testing.assert_array_equal(hdul[0].data, lsf)
        h = hdul[0].header
        assert h["CALTYPE"] == "lsf"
        assert (h["NIN"], h["NLAM"], h["NLENSY"], h["NLENSX"]) == (3, 3, 2, 4)
        assert h["RMATFILE"] == opt.name
        assert len(h["RMATMD5"]) == 32
        np.testing.assert_array_equal(hdul[rectmat_io.WAVE_EXTNAME].data["WAVE"], waves)
    with fits.open(tmp_path / "LowRes-K_OPT_intp_lflat.fits") as hdul:
        assert hdul[0].header["RMATFILE"] == opt.name
        assert hdul[0].data.shape == (2, 4)


def test_parse_files_pairs_frames_with_their_wavelengths(pm, tmp_path):
    # rows for another mode come first, so the LowRes-K rows do not start at index 0
    df = pd.DataFrame({
        "filename": ["x.fits", "k2.fits", "k1.fits"],
        "IFSMODE": ["MedRes-L", "LowRes-K", "LowRes-K"],
        "MONOWAVE": [3000.0, 2100.0, 2000.0],
        "MCLOCK": ["5.0 MHz"] * 3,
    })
    for name, value in (("x.fits", 0), ("k2.fits", 2), ("k1.fits", 1)):
        fits.writeto(tmp_path / name, np.full((2, 2), value, dtype=np.float32))
    ims, lams = pm.parse_files(df, "LowRes-K", str(tmp_path))
    np.testing.assert_array_equal(lams, [2.0, 2.1])
    assert [im[0, 0] for im in ims] == [1, 2]
