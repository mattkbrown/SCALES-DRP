"""
Find the calibration file a primitive should use.

With use_caldb = True in the config, the file is looked up in the caldb (via
the local CalibStore cache) for the frame being reduced. If the caldb has
nothing suitable, or is off, the file named in the config is used from the
calib directory, as before.

Config keys (all optional; [SCALES] section of scales.cfg):

    use_caldb            False
    caldb_url            base URL of the caldb, e.g. 'http://host:5000'
    caldb_cache_dir      cache root (default $SCALES_CALIB_CACHE or ~/.cache/scalesdrp)
    caldb_offline        True to use only what is already cached
    caldb_ttl            seconds between checks of /lastUpdated (default 3600)
    caldb_match_stateid  True to only accept files with the frame's STATEID

The observer id sent to the caldb is taken from $SCALES_CALDB_OBSID.
"""
import logging
import os
from datetime import timezone

from astropy.time import Time

from scalesdrp.core import rectmat_io
from scalesdrp.core.caldb import CalDBClient, CalDBError, CalibStore
from scalesdrp.core.scales_pkg_resources import get_resource_path

log = logging.getLogger("SCALES")

# CAMERA header value -> caldb detector
DETECTORS = {"IFS": "IFS", "IM": "IMG", "IMG": "IMG"}

_stores = {}


def caldb_settings(config):
    inst = getattr(config, "instrument", config)
    return {
        "use_caldb": bool(getattr(inst, "use_caldb", False)),
        "url": getattr(inst, "caldb_url", "") or "",
        "cache_dir": getattr(inst, "caldb_cache_dir", "") or None,
        "offline": bool(getattr(inst, "caldb_offline", False)),
        "ttl": float(getattr(inst, "caldb_ttl", 3600)),
        "match_stateid": bool(getattr(inst, "caldb_match_stateid", False)),
    }


def get_store(config, logger=None):
    """The CalibStore for this config, or None if the caldb is not in use."""
    s = caldb_settings(config)
    if not s["use_caldb"]:
        return None
    key = (s["url"], s["cache_dir"], s["offline"], s["ttl"])
    if key not in _stores:
        client = CalDBClient(s["url"]) if s["url"] and not s["offline"] else None
        _stores[key] = CalibStore(client=client, cache_dir=s["cache_dir"],
                                  offline=s["offline"], ttl=s["ttl"], logger=logger)
    return _stores[key]


def header_obs_time(header):
    """Observation time of a frame (aware UTC datetime), or None."""
    date = str(header.get("DATE-OBS", "") or "").strip()
    if date:
        if "T" not in date:
            utc = str(header.get("UTC", header.get("UT", header.get("TIME-OBS", ""))) or "").strip()
            date = f"{date}T{utc}" if utc else date
        try:
            return Time(date, scale="utc").to_datetime(timezone=timezone.utc)
        except ValueError:
            pass
    mjd = header.get("MJD-OBS", header.get("MJD"))
    if mjd is not None:
        return Time(float(mjd), format="mjd", scale="utc").to_datetime(timezone=timezone.utc)
    return None


def local_calib_path(context, name):
    """name inside the configured calib directory, or None if it is not there."""
    if not name:
        return None
    try:
        calib_dir = str(get_resource_path("scalesdrp", context.calib_file_path))
    except FileNotFoundError:
        return None
    path = os.path.join(calib_dir, name)
    return path if os.path.exists(path) else None


def resolve(config, context, cal_type, header, local_name=None, instrument_mode=None,
            read_mode=None, detector=None, logger=None):
    """
    Path of the calibration file to use for the frame with this header.

    Raises FileNotFoundError if neither the caldb nor the calib directory
    has one.
    """
    logger = logger or log
    store = get_store(config, logger)
    if store is not None:
        settings = caldb_settings(config)
        store.refresh()
        selectors = dict(instrument_mode=instrument_mode, read_mode=read_mode,
                         detector=detector, obs_time=header_obs_time(header))
        if settings["match_stateid"] and header.get("STATEID"):
            selectors["stateid"] = str(header["STATEID"]).strip()
        try:
            record = store.select(cal_type, **selectors)
            if record is not None:
                path = store.fetch(record)
                logger.info(f"caldb: using {record['filename']} ({record['id']}) for {cal_type}")
                return path
            logger.warning(f"caldb: no {cal_type} for {instrument_mode or ''} "
                           f"{read_mode or ''} {detector or ''}; trying the calib directory")
        except CalDBError as e:
            logger.warning(f"caldb: {e}; trying the calib directory")

    path = local_calib_path(context, local_name)
    if path is None:
        raise FileNotFoundError(
            f"No {cal_type} file for {instrument_mode or ''} {read_mode or ''} "
            f"{detector or ''}: not in the caldb"
            + (f", and {local_name} is not in the calib directory." if local_name
               else ", and none is configured locally."))
    return path


def rectmat_config_name(context, kind, ifsmode):
    """The rectmat filename configured for a mode, e.g. context.OPT_rmat_LowRes_K."""
    return getattr(context, f"{kind}_rmat_{ifsmode.replace('-', '_')}", None)


def resolve_rectmat(config, context, kind, ifsmode, header, logger=None):
    """Path of the OPT or C2 rectmat for an IFS frame."""
    return resolve(config, context, rectmat_io.KINDS[kind][0], header,
                   local_name=rectmat_config_name(context, kind, ifsmode),
                   instrument_mode=ifsmode, detector="IFS", logger=logger)


def resolve_bpm(config, context, header, logger=None):
    """Path of the bad pixel mask for a frame's camera and read mode."""
    camera = str(header.get("CAMERA", "")).strip().upper()
    detector = DETECTORS.get(camera)
    local_name = context.bpm_ifs if detector == "IFS" else context.bpm_img
    return resolve(config, context, "bpm", header, local_name=local_name,
                   instrument_mode=None, read_mode=str(header.get("MCLOCK", "")).strip() or None,
                   detector=detector, logger=logger)
