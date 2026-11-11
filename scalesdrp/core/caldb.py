"""
Client for the calibration database (hispec-caldb) and a local cache of the
files it serves.

The caldb stores one metadata record per calibration file in the `scales`
table and serves the files themselves from /download. CalibStore keeps a
copy of all SCALES records (refreshed when /lastUpdated changes) and of every
file it has downloaded, so a reduction only talks to the caldb when something
changed, and keeps working offline with whatever is cached.

Selection (CalibStore.select): among GOOD records with the requested
cal_type, whose instrument_mode / read_mode match (or are ANY) and whose
datetime_obs is not after the frame being reduced, take the newest, then the
highest cal_version.
"""
import json
import logging
import os
import shutil
import time
import zipfile
from base64 import b64encode
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from scalesdrp.core.calib_cache import cache_root, file_lock, file_md5, tmp_path_for

INSTRUMENT = "scales"
ENV_OBSID = "SCALES_CALDB_OBSID"
DEFAULT_TIMEOUT = (10, 600)  # (connect, read) seconds; rectmats are ~100 MB
ANY = "ANY"

log = logging.getLogger("SCALES")


class CalDBError(RuntimeError):
    pass


def parse_time(value):
    """caldb timestamps (ISO 8601 or HTTP date, as Flask sends datetimes) -> aware UTC."""
    if value is None or isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            dt = parsedate_to_datetime(text)
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class CalDBClient:
    """Thin wrapper around the caldb HTTP API for the SCALES table."""

    def __init__(self, base_url, obsid=None, timeout=DEFAULT_TIMEOUT, session=None):
        self.url = base_url.rstrip("/") + f"/calibrations/{INSTRUMENT}"
        self.timeout = timeout
        self.session = session or requests.Session()
        obsid = obsid or os.environ.get(ENV_OBSID) or "scalesdrp"
        self.session.cookies.set("observer", "obsid=" + b64encode(obsid.encode()).decode())

    def _get(self, endpoint, stream=False, **params):
        params = {k: v for k, v in params.items() if v is not None}
        r = self.session.get(f"{self.url}/{endpoint}", params=params,
                             timeout=self.timeout, stream=stream)
        if r.status_code not in (200, 404):
            raise CalDBError(f"caldb {endpoint} returned {r.status_code}: {r.text[:200]}")
        return r

    def last_updated(self):
        r = self._get("lastUpdated")
        return None if r.status_code == 404 else r.json()["last_updated"]

    def query(self, cal_type=None, cal_id=None, date_time_start=None, date_time_end=None):
        r = self._get("query", cal_type=cal_type, cal_id=cal_id,
                      date_time_start=date_time_start, date_time_end=date_time_end)
        body = r.json()
        # No matches comes back as {"message": ...}, not []
        return body if isinstance(body, list) else []

    def download(self, cal_id, dest):
        """Download one file by cal_id to dest. Returns dest."""
        tmp_zip = tmp_path_for(dest) + ".zip"
        try:
            with self._get("download", stream=True, cal_id=cal_id) as r:
                if r.status_code == 404 or "zip" not in r.headers.get("Content-Type", ""):
                    raise CalDBError(f"caldb has no file for {cal_id}: {r.text[:200]}")
                with open(tmp_zip, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 22):
                        f.write(chunk)
            with zipfile.ZipFile(tmp_zip) as z:
                members = z.namelist()
                if len(members) != 1:
                    raise CalDBError(f"expected one file for {cal_id}, got {members}")
                with z.open(members[0]) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 22)
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        return dest

    def add(self, records):
        r = self.session.post(f"{self.url}/add", json=records, timeout=self.timeout)
        if r.status_code != 200:
            raise CalDBError(f"caldb add returned {r.status_code}: {r.text[:500]}")
        return r.json()


class CalibStore:
    """
    Local mirror of the SCALES caldb: metadata in caldb/index.json, files in
    caldb/files/<id>/<filename> under the calibration cache root.

    client=None or offline=True uses only what is already cached.
    """

    def __init__(self, client=None, cache_dir=None, offline=False, ttl=3600, logger=None):
        self.client = client
        self.offline = offline or client is None
        self.ttl = ttl
        self.logger = logger or log
        self.root = os.path.join(cache_root(cache_dir), "caldb")
        self.index_path = os.path.join(self.root, "index.json")
        os.makedirs(self.root, exist_ok=True)
        self._index_memo = (None, None)  # (mtime_ns, index)

    # ---------- metadata ----------

    def _load_index(self):
        """The index, re-read only when the file changed (it is read per lookup)."""
        try:
            mtime = os.stat(self.index_path).st_mtime_ns
            if self._index_memo[0] == mtime:
                return dict(self._index_memo[1])
            with open(self.index_path) as f:
                index = json.load(f)
            self._index_memo = (mtime, index)
            return dict(index)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"last_updated": None, "checked_at": 0, "records": []}

    def _save_index(self, index):
        tmp = tmp_path_for(self.index_path)
        with open(tmp, "w") as f:
            json.dump(index, f, indent=1, default=str)
        os.replace(tmp, self.index_path)

    def records(self):
        return self._load_index()["records"]

    def refresh(self, force=False):
        """
        Re-fetch the metadata if the caldb changed since the last fetch.
        Checks at most once per ttl seconds unless force. Network errors are
        logged and the cached metadata is kept.
        """
        if self.offline:
            return False
        with file_lock(self.index_path):
            index = self._load_index()
            if not force and time.time() - index["checked_at"] < self.ttl:
                return False
            try:
                stamp = self.client.last_updated()
                changed = force or stamp != index["last_updated"]
                if changed:
                    index["records"] = self.client.query()
                    index["last_updated"] = stamp
                    self.logger.info(f"caldb: fetched {len(index['records'])} SCALES records")
            except (requests.RequestException, CalDBError, ValueError) as e:
                self.logger.warning(f"caldb unreachable, using cached metadata: {e}")
                return False
            index["checked_at"] = time.time()
            self._save_index(index)
            return changed

    def select(self, cal_type, **selectors):
        """The record to use, or None. See the module docstring for the rule."""
        matches = self.candidates(cal_type, **selectors)
        return matches[0] if matches else None

    def candidates(self, cal_type, instrument_mode=None, read_mode=None, detector=None,
                   obs_time=None, stateid=None):
        """All matching records, best first (the order select uses)."""
        obs_time = parse_time(obs_time)

        def ok(r):
            return (r.get("cal_type") == cal_type
                    and r.get("data_quality", "GOOD") == "GOOD"
                    and (instrument_mode is None or r.get("instrument_mode") in (instrument_mode, ANY))
                    and (read_mode is None or r.get("read_mode") in (read_mode, ANY))
                    and (detector is None or r.get("detector") == detector)
                    and (stateid is None or r.get("stateid") == stateid)
                    and (obs_time is None or parse_time(r["datetime_obs"]) <= obs_time))

        return sorted((r for r in self.records() if ok(r)), reverse=True,
                      key=lambda r: (parse_time(r["datetime_obs"]), str(r.get("cal_version", ""))))

    # ---------- files ----------

    def path_for(self, record):
        return os.path.join(self.root, "files", str(record["id"]), record["filename"])

    def _verified(self, record, path):
        if not os.path.exists(path):
            return False
        size = record.get("file_size")
        if size is not None and os.path.getsize(path) != int(size):
            return False
        return file_md5(path) == record["file_md5"]

    def fetch(self, record):
        """Local path of a record's file, downloading and verifying it if needed."""
        path = self.path_for(record)
        if self._verified(record, path):
            return path
        with file_lock(path):
            if self._verified(record, path):
                return path
            if self.offline:
                raise CalDBError(f"{record['filename']} ({record['id']}) is not cached "
                                 f"and the caldb is not available")
            self.logger.info(f"caldb: downloading {record['filename']} ({record['id']})")
            tmp = tmp_path_for(path)
            try:
                self.client.download(record["id"], tmp)
                if not self._verified(record, tmp):
                    raise CalDBError(f"{record['filename']}: downloaded file does not match "
                                     f"its size/MD5 in the caldb")
                os.replace(tmp, path)
            except requests.RequestException as e:
                raise CalDBError(f"download of {record['filename']} failed: {e}") from e
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        return path

    def get(self, cal_type, **selectors):
        """Select and fetch: the local path, or None if nothing matches."""
        record = self.select(cal_type, **selectors)
        return None if record is None else self.fetch(record)

    def cached(self):
        """(record, path) for every record whose file is in the cache."""
        return [(r, self.path_for(r)) for r in self.records()
                if os.path.exists(self.path_for(r))]
