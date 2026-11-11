import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

import scalesdrp.primitives.scales_basic as scbasic
from scalesdrp.core import calib_resolve
from scalesdrp.core.caldb import CalibStore


def fits_bytes(data, **header):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32), header=fits.Header(header))
    import io
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue()


def record(id_, content, cal_type, detector="IFS", read_mode="5.0 MHz", mode="ANY",
           when="2026-05-01T00:00:00+00:00"):
    return {"id": id_, "filename": f"{id_}.fits", "cal_type": cal_type, "detector": detector,
            "read_mode": read_mode, "instrument_mode": mode, "datetime_obs": when,
            "cal_version": "001", "data_quality": "GOOD", "file_size": len(content),
            "file_md5": hashlib.md5(content).hexdigest()}


class FakeClient:
    def __init__(self, records, files):
        self.records, self.files, self.downloads = records, files, []

    def last_updated(self):
        return "t"

    def query(self, **kw):
        return list(self.records)

    def download(self, cal_id, dest):
        self.downloads.append(cal_id)
        with open(dest, "wb") as f:
            f.write(self.files[cal_id])


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A calib dir, a caldb store with given files, and the config / context for them."""
    calib = tmp_path / "calib"
    calib.mkdir()
    monkeypatch.setattr(calib_resolve, "get_resource_path", lambda pkg, p: calib)
    monkeypatch.chdir(tmp_path)

    def make(entries):
        records = [e[0] for e in entries]
        files = {e[0]["id"]: e[1] for e in entries}
        client = FakeClient(records, files)
        store = CalibStore(client=client, cache_dir=str(tmp_path / "cache"), ttl=0)
        monkeypatch.setattr(calib_resolve, "get_store", lambda c, logger=None: store)
        cfg = SimpleNamespace(instrument=SimpleNamespace(use_caldb=True))
        ctx = SimpleNamespace(calib_file_path="calib/")
        return cfg, ctx, client, calib
    return make


def ifs_header(**extra):
    return fits.Header({"CAMERA": "IFS", "MCLOCK": "5.0 MHz", "DATE-OBS": "2026-06-01T00:00:00",
                        "EXPTIME": 10.0, "IMTYPE": "OBJECT", **extra})


# ---------- readnoise / saturation / linearity ----------

def test_detector_file_comes_from_the_caldb_for_the_frames_clock(env):
    rn5 = fits_bytes(np.ones((2, 2)))
    rn20 = fits_bytes(np.zeros((2, 2)))
    cfg, ctx, client, _ = env([(record("rn5", rn5, "readnoise"), rn5),
                               (record("rn20", rn20, "readnoise", read_mode="20.0 MHz"), rn20)])
    calib_file = calib_resolve.detector_file_resolver(cfg, ctx, ifs_header())
    path = calib_file("readnoise", "readnoise_ifs_fast1.0.fits")
    assert path.endswith("rn5.fits")
    assert client.downloads == ["rn5"]


def test_detector_file_falls_back_to_the_configured_local_file(env):
    cfg, ctx, _, calib = env([])
    (calib / "sat.fits").write_bytes(b"x")
    calib_file = calib_resolve.detector_file_resolver(cfg, ctx, ifs_header())
    assert calib_file("saturation", "sat.fits") == str(calib / "sat.fits")
    with pytest.raises(FileNotFoundError):
        calib_file("readnoise", "missing.fits")


def test_optional_detector_file_returns_its_configured_path(env):
    cfg, ctx, _, calib = env([])
    calib_file = calib_resolve.detector_file_resolver(cfg, ctx, ifs_header())
    assert calib_file("linearity", "lin.fits", required=False) == str(calib / "lin.fits")


def test_imager_frames_select_imager_files(env):
    img = fits_bytes(np.ones((2, 2)))
    ifs = fits_bytes(np.zeros((2, 2)))
    cfg, ctx, _, _ = env([(record("img", img, "linearity", detector="IMG"), img),
                          (record("ifs", ifs, "linearity"), ifs)])
    calib_file = calib_resolve.detector_file_resolver(cfg, ctx, ifs_header(CAMERA="Im"))
    assert calib_file("linearity", "lin.fits").endswith("img.fits")


# ---------- master dark / bias / flat ----------

def test_master_dark_from_caldb_must_match_exptime(env):
    newer = fits_bytes(np.full((2, 2), 5.0), IMTYPE="DARK", CAMERA="IFS", MCLOCK="5.0 MHz", EXPTIME=30.0)
    older = fits_bytes(np.full((2, 2), 7.0), IMTYPE="DARK", CAMERA="IFS", MCLOCK="5.0 MHz", EXPTIME=10.0)
    cfg, ctx, _, _ = env([(record("d30", newer, "dark", when="2026-05-20T00:00:00+00:00"), newer),
                          (record("d10", older, "dark", when="2026-05-01T00:00:00+00:00"), older)])
    keys = {"CAMERA": "IFS", "MCLOCK": "5.0 MHz", "EXPTIME": 10.0}
    data, _ = scbasic.load_single_master_file(keys, "DARK", config=cfg, header=ifs_header())
    assert data[0, 0] == 7.0


def test_tonights_master_in_redux_wins_over_the_caldb(env, tmp_path):
    caldb_bias = fits_bytes(np.full((2, 2), 1.0), IMTYPE="BIAS", CAMERA="IFS", MCLOCK="5.0 MHz")
    cfg, ctx, client, _ = env([(record("b", caldb_bias, "bias"), caldb_bias)])
    (tmp_path / "redux").mkdir()
    (tmp_path / "redux" / "x_mbias.fits").write_bytes(
        fits_bytes(np.full((2, 2), 2.0), IMTYPE="BIAS", CAMERA="IFS", MCLOCK="5.0 MHz"))
    keys = {"CAMERA": "IFS", "MCLOCK": "5.0 MHz", "EXPTIME": 10.0}
    data, _ = scbasic.load_single_master_file(keys, "BIAS", config=cfg, header=ifs_header())
    assert data[0, 0] == 2.0
    assert client.downloads == []


def test_master_flat_from_caldb_then_calib_dir(env):
    flat = fits_bytes(np.full((2, 2), 3.0), IMTYPE="FLATLAMP", CAMERA="IFS", MCLOCK="5.0 MHz")
    cfg, ctx, _, calib = env([(record("f", flat, "detector_flat"), flat)])
    keys = {"CAMERA": "IFS", "MCLOCK": "5.0 MHz", "EXPTIME": 10.0}
    data, _ = scbasic.load_single_master_file(keys, "FLATLAMP", config=cfg, header=ifs_header())
    assert data[0, 0] == 3.0
    # without a config the caldb is skipped, and there is nothing in calib/
    assert scbasic.load_single_master_file(keys, "FLATLAMP") == (None, None)


def test_master_loader_works_without_a_calib_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def missing(pkg, path):
        raise FileNotFoundError(path)
    monkeypatch.setattr(scbasic, "get_resource_path", missing)
    assert scbasic.load_single_master_file({"CAMERA": "IFS"}, "BIAS") == (None, None)


# ---------- lenslet flat ----------

def test_lenslet_flat_cube_from_caldb(env):
    cube = fits_bytes(np.ones((3, 2, 2)) * 2.0, IFSMODE="LowRes-K")
    wrong = fits_bytes(np.ones((3, 2, 2)), IFSMODE="LowRes-L")
    cfg, ctx, _, _ = env([(record("k", cube, "lenslet_flat", mode="LowRes-K"), cube),
                          (record("l", wrong, "lenslet_flat", mode="LowRes-L",
                                  when="2026-05-20T00:00:00+00:00"), wrong)])
    flat, _ = scbasic.load_and_normalize_lenslet_flat("LowRes-K", config=cfg, header=ifs_header())
    assert flat is not None and flat.shape == (3, 2, 2)


def test_no_lenslet_flat_anywhere(env):
    cfg, ctx, _, _ = env([])
    assert scbasic.load_and_normalize_lenslet_flat("LowRes-K", config=cfg,
                                                   header=ifs_header()) == (None, None)


# ---------- HISTORY: exactly which file ----------

def test_describe_names_the_caldb_record_and_md5(env):
    content = fits_bytes(np.ones((2, 2)))
    cfg, ctx, _, _ = env([(record("rn5", content, "readnoise"), content)])
    path = calib_resolve.detector_file_resolver(cfg, ctx, ifs_header())("readnoise", "x.fits")
    text = calib_resolve.describe(path)
    assert text == f"rn5.fits (caldb id rn5; md5 {hashlib.md5(content).hexdigest()})"


def test_describe_names_the_directory_of_a_local_file(env):
    cfg, ctx, _, calib = env([])
    (calib / "sat.fits").write_bytes(b"abc")
    text = calib_resolve.describe(str(calib / "sat.fits"))
    assert text == f"sat.fits ({calib}; md5 {hashlib.md5(b'abc').hexdigest()})"
    assert calib_resolve.describe(str(calib / "gone.fits")).endswith("file missing)")


def test_master_and_lenslet_flat_loaders_report_the_file_used(env):
    bias = fits_bytes(np.full((2, 2), 1.0), IMTYPE="BIAS", CAMERA="IFS", MCLOCK="5.0 MHz")
    cube = fits_bytes(np.ones((3, 2, 2)), IFSMODE="LowRes-K")
    cfg, ctx, _, _ = env([(record("b", bias, "bias"), bias),
                          (record("lf", cube, "lenslet_flat", mode="LowRes-K"), cube)])
    keys = {"CAMERA": "IFS", "MCLOCK": "5.0 MHz"}
    *_, path = scbasic.load_single_master_file(keys, "BIAS", config=cfg, header=ifs_header(),
                                               with_path=True)
    assert path.endswith("b.fits")
    *_, path = scbasic.load_and_normalize_lenslet_flat("LowRes-K", config=cfg, header=ifs_header(),
                                                       with_path=True)
    assert path.endswith("lf.fits")
    assert scbasic.load_single_master_file(keys, "DARK", with_path=True) == (None, None, None)
