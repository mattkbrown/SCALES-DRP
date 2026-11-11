import io

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import scalesdrp.primitives.scales_basic as scbasic


def ifs_header(mode="LowRes-K"):
    h = fits.Header()
    h["IFSMODE"] = mode
    h["RA"] = "10:00:00.0"
    h["DEC"] = "+20:00:00.0"
    return h


def cube_header(shape, wavelengths, mode="LowRes-K"):
    wcs, info = scbasic.create_scales_wcs(shape, ifs_header(mode), wavelengths=wavelengths)
    hdr = scbasic.wcs_header_update(np.zeros(shape), ifs_header(mode), wcs, info)
    return hdr, info


def test_linear_grid_from_rectmat_wavelengths():
    w = np.linspace(1.9, 2.5, 7)
    hdr, info = cube_header((7, 4, 4), w)
    assert info["WAVEGRID"] == "LINEAR"
    assert hdr["CTYPE3"] == "WAVE"
    assert hdr["CRVAL3"] == pytest.approx(1.9)
    assert hdr["CDELT3"] == pytest.approx(0.1)
    assert hdr["WAVEGRID"] == "LINEAR"
    assert scbasic.wavelength_extra_hdus(info) == []


def test_rectmat_wavelengths_override_the_per_mode_table():
    # LowRes-K in the built-in table is 1.95 - 2.45 um
    hdr, _ = cube_header((5, 4, 4), np.linspace(2.0, 2.4, 5))
    assert hdr["WAVSTART"] == pytest.approx(2.0)
    assert hdr["WAVEND"] == pytest.approx(2.4)


def test_non_linear_grid_is_a_lookup_table_that_astropy_reads():
    w = np.array([2.0, 2.1, 2.25, 2.3, 2.4])
    shape = (5, 4, 4)
    hdr, info = cube_header(shape, w)
    assert info["WAVEGRID"] == "TABLE"
    assert hdr["CTYPE3"] == "WAVE-TAB"
    extra = scbasic.wavelength_extra_hdus(info)
    assert [h.name for h in extra] == [scbasic.WAVETAB_EXTNAME]

    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(np.zeros(shape, np.float32), header=hdr)] + extra).writeto(buf)
    buf.seek(0)
    with fits.open(buf) as hdul:
        wcs = WCS(hdul[0].header, fobj=hdul)
        world = wcs.pixel_to_world_values(np.zeros(5), np.zeros(5), np.arange(5))
    np.testing.assert_allclose(world[2], w)


def test_wrong_number_of_wavelengths_is_an_error():
    with pytest.raises(ValueError):
        scbasic.create_scales_wcs((5, 4, 4), ifs_header(), wavelengths=np.arange(4.0))


@pytest.mark.parametrize("mode, start", [("LowRes-KLM", 2.0), ("LowRes-Ls", 3.1)])
def test_legacy_table_knows_the_modes_select_ifsmode_returns(mode, start):
    # Without rectmat wavelengths these used to fall back to WAVSTART / 2.0 um
    _, info = scbasic.create_scales_wcs((5, 4, 4), ifs_header(mode))
    assert info["WAVSTART"] == pytest.approx(start)
