import hashlib
import io
import zipfile
from types import SimpleNamespace

import pytest
import requests
from astropy.io import fits

from scalesdrp.core import caldb, calib_resolve
from scalesdrp.core.caldb import CalDBError, CalibStore


def record(id_, cal_type="rectmat_opt", mode="LowRes-K", read_mode="ANY", when="2026-05-01T00:00:00+00:00",
           content=b"file bytes", version="001", quality="GOOD", stateid=None, detector="IFS"):
    return {
        "id": id_, "filename": f"{id_}.fits", "cal_type": cal_type, "instrument_mode": mode,
        "read_mode": read_mode, "detector": detector, "datetime_obs": when, "cal_version": version,
        "data_quality": quality, "stateid": stateid, "file_size": len(content),
        "file_md5": hashlib.md5(content).hexdigest(),
    }


class FakeClient:
    def __init__(self, records, files=None, stamp="t1"):
        self.records = records
        self.files = files or {}
        self.stamp = stamp
        self.calls = []
        self.fail = False

    def last_updated(self):
        self.calls.append("lastUpdated")
        if self.fail:
            raise requests.ConnectionError("down")
        return self.stamp

    def query(self, **kw):
        self.calls.append("query")
        return list(self.records)

    def download(self, cal_id, dest):
        self.calls.append(f"download {cal_id}")
        with open(dest, "wb") as f:
            f.write(self.files[cal_id])
        return dest


@pytest.fixture
def store_for(tmp_path):
    def make(client, **kw):
        return CalibStore(client=client, cache_dir=str(tmp_path), **kw)
    return make


# ---------- metadata refresh ----------

def test_refresh_fetches_records_only_when_last_updated_changes(store_for):
    client = FakeClient([record("a")])
    store = store_for(client, ttl=0)
    assert store.refresh() is True
    assert store.refresh() is False
    assert client.calls == ["lastUpdated", "query", "lastUpdated"]
    client.stamp = "t2"
    assert store.refresh() is True


def test_refresh_respects_ttl(store_for):
    client = FakeClient([record("a")])
    store = store_for(client, ttl=3600)
    store.refresh()
    store.refresh()
    assert client.calls == ["lastUpdated", "query"]


def test_unreachable_caldb_keeps_cached_metadata(store_for):
    client = FakeClient([record("a")])
    store = store_for(client, ttl=0)
    store.refresh()
    client.fail = True
    assert store.refresh() is False
    assert [r["id"] for r in store.records()] == ["a"]


# ---------- selection ----------

def test_select_newest_good_match_not_after_the_frame(store_for):
    client = FakeClient([
        record("old", when="2026-01-01T00:00:00+00:00"),
        record("new", when="2026-06-01T00:00:00+00:00"),
        record("future", when="2027-01-01T00:00:00+00:00"),
        record("bad", when="2026-07-01T00:00:00+00:00", quality="BAD"),
        record("othermode", mode="LowRes-L", when="2026-06-15T00:00:00+00:00"),
        record("c2", cal_type="rectmat_c2", when="2026-06-15T00:00:00+00:00"),
    ])
    store = store_for(client)
    store.refresh()
    pick = store.select("rectmat_opt", instrument_mode="LowRes-K",
                        obs_time="2026-08-01T00:00:00Z")
    assert pick["id"] == "new"


def test_select_accepts_any_and_breaks_ties_on_cal_version(store_for):
    when = "2026-06-01T00:00:00+00:00"
    client = FakeClient([
        record("v1", cal_type="bpm", mode="ANY", read_mode="5.0 MHz", when=when, version="001"),
        record("v2", cal_type="bpm", mode="ANY", read_mode="5.0 MHz", when=when, version="002"),
        record("slow", cal_type="bpm", mode="ANY", read_mode="20.0 MHz", when=when, version="009"),
    ])
    store = store_for(client)
    store.refresh()
    assert store.select("bpm", instrument_mode="LowRes-K", read_mode="5.0 MHz")["id"] == "v2"


def test_select_parses_flask_http_dates(store_for):
    client = FakeClient([record("a", when="Fri, 01 May 2026 00:00:00 GMT")])
    store = store_for(client)
    store.refresh()
    assert store.select("rectmat_opt", obs_time="2026-05-02T00:00:00Z")["id"] == "a"
    assert store.select("rectmat_opt", obs_time="2026-04-30T00:00:00Z") is None


def test_select_by_stateid(store_for):
    client = FakeClient([record("s1", stateid="abc"), record("s2", stateid="xyz",
                                                              when="2026-06-01T00:00:00+00:00")])
    store = store_for(client)
    store.refresh()
    assert store.select("rectmat_opt", stateid="abc")["id"] == "s1"


# ---------- files ----------

def test_fetch_downloads_once_and_verifies(store_for):
    client = FakeClient([record("a")], files={"a": b"file bytes"})
    store = store_for(client)
    store.refresh()
    rec = store.select("rectmat_opt")
    path = store.fetch(rec)
    assert open(path, "rb").read() == b"file bytes"
    store.fetch(rec)
    assert client.calls.count("download a") == 1


def test_fetch_rejects_a_corrupt_download(store_for):
    client = FakeClient([record("a")], files={"a": b"something else"})
    store = store_for(client)
    store.refresh()
    with pytest.raises(CalDBError):
        store.fetch(store.select("rectmat_opt"))
    assert store.cached() == []


def test_offline_store_serves_cached_files_only(store_for):
    client = FakeClient([record("a"), record("b", when="2026-06-01T00:00:00+00:00")],
                        files={"a": b"file bytes"})
    online = store_for(client)
    online.refresh()
    online.fetch(online.select("rectmat_opt", obs_time="2026-05-15T00:00:00Z"))

    offline = store_for(None)
    assert offline.fetch(offline.select("rectmat_opt", obs_time="2026-05-15T00:00:00Z"))
    with pytest.raises(CalDBError):
        offline.fetch(offline.select("rectmat_opt"))


# ---------- HTTP client ----------

class FakeResponse:
    def __init__(self, status=200, body=None, content=b"", ctype="application/json"):
        self.status_code = status
        self._body = body
        self._content = content
        self.headers = {"Content-Type": ctype}
        self.text = str(body)

    def json(self):
        return self._body

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeSession:
    def __init__(self, responses):
        self.responses = responses
        self.cookies = requests.cookies.RequestsCookieJar()
        self.requests = []

    def get(self, url, params=None, timeout=None, stream=False):
        self.requests.append((url, params))
        return self.responses.pop(0)


def test_client_sends_observer_cookie_and_normalises_empty_query():
    session = FakeSession([FakeResponse(body={"message": "No matching calibrations found."})])
    client = caldb.CalDBClient("http://caldb:5000/", obsid="42", session=session)
    assert client.query(cal_type="rectmat_opt") == []
    url, params = session.requests[0]
    assert url == "http://caldb:5000/calibrations/scales/query"
    assert params == {"cal_type": "rectmat_opt"}
    assert session.cookies.get("observer") == "obsid=NDI="


def test_client_download_unzips_the_single_member(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("koa_id_here", b"payload")
    session = FakeSession([FakeResponse(content=buf.getvalue(), ctype="application/zip")])
    client = caldb.CalDBClient("http://caldb", session=session)
    dest = tmp_path / "out.fits"
    client.download("abc", str(dest))
    assert dest.read_bytes() == b"payload"


def test_client_download_of_missing_file_is_an_error(tmp_path):
    session = FakeSession([FakeResponse(status=404, body={"error": "No calibration files available"})])
    client = caldb.CalDBClient("http://caldb", session=session)
    with pytest.raises(CalDBError):
        client.download("abc", str(tmp_path / "out.fits"))


# ---------- resolver ----------

def config(**kw):
    return SimpleNamespace(instrument=SimpleNamespace(**kw))


def test_header_obs_time_variants():
    t = calib_resolve.header_obs_time(fits.Header({"DATE-OBS": "2026-05-01T12:00:00"}))
    assert t.isoformat().startswith("2026-05-01T12:00:00")
    t = calib_resolve.header_obs_time(fits.Header({"DATE-OBS": "2026-05-01", "UTC": "03:00:00"}))
    assert t.hour == 3
    t = calib_resolve.header_obs_time(fits.Header({"MJD": 61161.5}))
    assert t.isoformat().startswith("2026-05-01T12:00")
    assert calib_resolve.header_obs_time(fits.Header()) is None


def test_resolve_without_caldb_uses_the_configured_local_file(tmp_path, monkeypatch):
    (tmp_path / "LowRes-K_OPT.fits").write_bytes(b"x")
    monkeypatch.setattr(calib_resolve, "get_resource_path", lambda pkg, p: tmp_path)
    ctx = SimpleNamespace(calib_file_path="calib/", OPT_rmat_LowRes_K="LowRes-K_OPT.fits")
    path = calib_resolve.resolve_rectmat(config(), ctx, "OPT", "LowRes-K", fits.Header())
    assert path == str(tmp_path / "LowRes-K_OPT.fits")


def test_resolve_prefers_caldb_and_falls_back_to_local(tmp_path, monkeypatch):
    local = tmp_path / "calib"
    local.mkdir()
    (local / "LowRes-K_C2.fits").write_bytes(b"x")
    monkeypatch.setattr(calib_resolve, "get_resource_path", lambda pkg, p: local)
    ctx = SimpleNamespace(calib_file_path="calib/", C2_rmat_LowRes_K="LowRes-K_C2.fits",
                          OPT_rmat_LowRes_K="missing.fits")
    client = FakeClient([record("c2", cal_type="rectmat_c2")], files={"c2": b"file bytes"})
    cfg = config(use_caldb=True, caldb_cache_dir=str(tmp_path / "cache"), caldb_ttl=0)
    store = CalibStore(client=client, cache_dir=str(tmp_path / "cache"), ttl=0)
    monkeypatch.setattr(calib_resolve, "get_store", lambda c, logger=None: store)
    hdr = fits.Header({"DATE-OBS": "2026-06-01T00:00:00"})

    path = calib_resolve.resolve_rectmat(cfg, ctx, "C2", "LowRes-K", hdr)
    assert path == store.path_for(client.records[0])

    # nothing for OPT in the caldb, and the configured local file is missing
    with pytest.raises(FileNotFoundError):
        calib_resolve.resolve_rectmat(cfg, ctx, "OPT", "LowRes-K", hdr)


def test_get_store_is_off_unless_use_caldb():
    assert calib_resolve.get_store(config()) is None
