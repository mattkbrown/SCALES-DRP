import json

import numpy as np
import pytest
from astropy.io import fits
from scipy import sparse

from scalesdrp.core import rectmat_io
from scalesdrp.scripts import scales_caldb

# Required by the caldb's scales schema (hispec-caldb schemas.py)
REQUIRED = {"id", "filename", "cal_type", "instrument_era", "mjd_start", "datetime_obs",
            "instrument_mode", "target_name", "drp_version", "cal_version", "data_quality",
            "file_md5", "origin", "last_updated", "detector", "read_mode"}
ALLOWED = REQUIRED | {"koa_id", "koa_filepath", "master_cal", "last_processed", "note",
                      "stateid", "interpolated", "file_size"}


def rectmat_file(tmp_path, name="LowRes-K_C2_intp_rectmat.fits"):
    m = sparse.csr_matrix((np.ones(2), ([0, 1], [5, 9])), shape=(2, rectmat_io.N_DET_PIX)).T.tocsr()
    hdr = {"IFSMODE": "LowRes-K", "STATEID": "abc", "INTERP": True, "CAMERA": "IFS",
           "DATE-OBS": "2026-05-01T09:00:00", "MJD-OBS": 61161.375, "DRPVER": "1.2.3"}
    return str(rectmat_io.write_rectmat(tmp_path / name, m, (2, 1, 1), [2.0, 2.1], header=hdr))


def publish_records(capsys, argv):
    scales_caldb.main(["publish", "--dry-run", "--instrument-era", "1"] + argv)
    return json.loads(capsys.readouterr().out)


def test_publish_builds_the_caldb_record_from_the_rectmat_header(tmp_path, capsys):
    path = rectmat_file(tmp_path)
    rec, = publish_records(capsys, [path, "--koa-filepath", "/SCALES/calib"])
    assert REQUIRED <= set(rec) <= ALLOWED
    assert rec["cal_type"] == "rectmat_c2"
    assert rec["instrument_mode"] == "LowRes-K"
    assert rec["read_mode"] == "ANY"
    assert rec["detector"] == "IFS"
    assert rec["stateid"] == "abc"
    assert rec["interpolated"] is True
    assert rec["mjd_start"] == pytest.approx(61161.375)
    assert rec["datetime_obs"].startswith("2026-05-01T09:00:00")
    assert rec["drp_version"] == "1.2.3"
    assert rec["koa_filepath"] == "/SCALES/calib/LowRes-K_C2_intp_rectmat.fits"


def test_publish_id_is_stable_for_the_same_file(tmp_path, capsys):
    path = rectmat_file(tmp_path)
    a, = publish_records(capsys, [path])
    b, = publish_records(capsys, [path])
    assert a["id"] == b["id"]


def test_publish_bpm_uses_its_clock_and_camera(tmp_path, capsys):
    path = tmp_path / "bpm_ifs.fits"
    fits.writeto(path, np.zeros((4, 4), np.uint16),
                 header=fits.Header({"CAMERA": "IFS", "MCLOCK": "5.0 MHz",
                                     "DATE-OBS": "2026-04-01T00:00:00"}))
    rec, = publish_records(capsys, [str(path), "--cal-type", "bpm"])
    assert (rec["cal_type"], rec["detector"], rec["read_mode"], rec["instrument_mode"]) == \
        ("bpm", "IFS", "5.0 MHz", "ANY")
    assert REQUIRED <= set(rec) <= ALLOWED


def test_publish_without_a_cal_type_is_an_error(tmp_path):
    path = tmp_path / "x.fits"
    fits.writeto(path, np.zeros(2), header=fits.Header({"DATE-OBS": "2026-04-01T00:00:00"}))
    with pytest.raises(SystemExit):
        scales_caldb.main(["publish", "--dry-run", "--instrument-era", "1", str(path)])


def test_convert(tmp_path, capsys):
    m = sparse.csr_matrix((np.ones(2), ([0, 1], [5, 9])), shape=(2, rectmat_io.N_DET_PIX))
    npz = tmp_path / "old.npz"
    sparse.save_npz(npz, m)
    lams = tmp_path / "old_lams.fits"
    fits.writeto(lams, np.array([2.0, 2.1]))
    out = tmp_path / "new.fits"
    scales_caldb.main(["convert", str(npz), str(out), "--lams", str(lams), "--ifsmode", "LowRes-K",
                       "--lenslet-grid", "1", "1", "--no-interp", "--stateid", "s"])
    rect = rectmat_io.read_rectmat(str(out))
    assert rect.header["IFSMODE"] == "LowRes-K" and rect.header["INTERP"] is False
    np.testing.assert_array_equal(rect.wavelengths, [2.0, 2.1])
