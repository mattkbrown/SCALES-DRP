"""
scales_caldb: work with the SCALES calibration database and local cache.

    scales_caldb list [--cached]
    scales_caldb sync [--mode M ...] [--date ISO] [--detector-files] [--bpmats]
    scales_caldb publish FILE ... --instrument-era ERA [--koa-filepath P] [--dry-run]
    scales_caldb convert OLD.npz NEW.fits --lams LAMS.fits --ifsmode MODE --kind OPT|C2

The caldb URL comes from --url or $SCALES_CALDB_URL, the observer id from
$SCALES_CALDB_OBSID, and the cache root from --cache-dir, $SCALES_CALIB_CACHE
or ~/.cache/scalesdrp (the same cache the pipeline uses).
"""
import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone

from astropy.io import fits
from astropy.time import Time

from scalesdrp.core import bpmat, rectmat_io
from scalesdrp.core.caldb import CalDBClient, CalDBError, CalibStore
from scalesdrp.core.calib_cache import file_md5

IFS_MODES = ["LowRes-K", "LowRes-L", "LowRes-M", "LowRes-KL", "LowRes-KLM", "LowRes-Ls",
             "MedRes-K", "MedRes-L", "MedRes-M"]
READ_MODES = ["5.0 MHz", "9.0 MHz", "20.0 MHz"]
CAL_TYPES = ["rectmat_opt", "rectmat_c2", "lsf", "lenslet_flat", "bpm", "readnoise",
             "linearity", "saturation", "dark", "bias", "detector_flat"]
# Per detector and read mode, fetched by sync --detector-files
DETECTOR_FILES = ["bpm", "readnoise", "saturation", "linearity"]
# cal_types that do not depend on the detector clock
CLOCK_FREE = {"rectmat_opt", "rectmat_c2", "lsf", "lenslet_flat"}
DETECTORS = {"IFS": "IFS", "IM": "IMG", "IMG": "IMG"}
# Fixed namespace so publishing the same file twice gives the same id
ID_NAMESPACE = uuid.UUID("5ca1e5d2-0000-4000-8000-000000000000")


def make_store(args, need_client=True):
    url = args.url or os.environ.get("SCALES_CALDB_URL")
    client = CalDBClient(url) if url and not args.offline else None
    if need_client and client is None:
        sys.exit("No caldb URL: pass --url or set $SCALES_CALDB_URL (or use --offline).")
    return CalibStore(client=client, cache_dir=args.cache_dir, offline=args.offline, ttl=0)


# ---------------- list ----------------

def cmd_list(args):
    store = make_store(args, need_client=False)
    store.refresh(force=True)
    cached = {r["id"] for r, _ in store.cached()}
    rows = [r for r in store.records() if not args.cached or r["id"] in cached]
    rows.sort(key=lambda r: (r["cal_type"], r.get("instrument_mode", ""), str(r["datetime_obs"])))
    for r in rows:
        print(f"{'*' if r['id'] in cached else ' '} {r['cal_type']:<12} {r.get('instrument_mode', ''):<10} "
              f"{r.get('read_mode', ''):<8} {str(r['datetime_obs']):<30} v{r.get('cal_version', '')} "
              f"{r.get('data_quality', ''):<4} {r['filename']}  {r['id']}")
    print(f"{len(rows)} records ({len(cached)} cached, marked *)")


# ---------------- sync ----------------

def cmd_sync(args):
    store = make_store(args)
    store.refresh(force=True)
    when = args.date or datetime.now(timezone.utc).isoformat()
    modes = args.mode or IFS_MODES
    rectmats = {}
    for mode in modes:
        for cal_type in ("rectmat_opt", "rectmat_c2"):
            rec = store.select(cal_type, instrument_mode=mode, detector="IFS", obs_time=when)
            if rec is None:
                print(f"  no {cal_type} for {mode}")
                continue
            path = store.fetch(rec)
            print(f"  {cal_type:<12} {mode:<10} {path}")
            if cal_type == "rectmat_opt":
                rectmats[mode] = path
    if args.detector_files or args.bpmats:
        for detector in ("IFS", "IMG"):
            for read_mode in READ_MODES:
                for cal_type in DETECTOR_FILES if args.detector_files else ():
                    rec = store.select(cal_type, read_mode=read_mode, detector=detector, obs_time=when)
                    if rec is not None:
                        print(f"  {cal_type:<12} {detector:<4} {read_mode:<9} {store.fetch(rec)}")
                rec = store.select("bpm", read_mode=read_mode, detector=detector, obs_time=when)
                if rec is None:
                    continue
                bpm_path = store.fetch(rec)
                if args.bpmats:
                    for mode, rect_path in (rectmats.items() if detector == "IFS" else []):
                        bpmat.get_bpmat(bpm_path, rect_path, cache_dir=args.cache_dir)
                        print(f"    bpmat for {mode} cached")
                    if detector == "IMG":
                        bpmat.get_bpmat(bpm_path, None, cache_dir=args.cache_dir)
                        print("    bpmat cached")


# ---------------- publish ----------------

def record_from_file(path, args):
    """The caldb /add record for a calibration file, from its primary header."""
    hdr = fits.getheader(path)
    cal_type = args.cal_type or hdr.get("CALTYPE")
    if cal_type not in CAL_TYPES:
        raise ValueError(f"{path}: cal_type {cal_type!r} is not one of {CAL_TYPES}; "
                         f"pass --cal-type")
    date_obs = args.date_obs or hdr.get("DATE-OBS")
    if not date_obs:
        raise ValueError(f"{path}: no DATE-OBS in the header; pass --date-obs")
    t = Time(str(date_obs), scale="utc")
    mjd = hdr.get("MJD-OBS", hdr.get("MJD"))
    camera = str(hdr.get("CAMERA", "IFS" if cal_type in CLOCK_FREE else "")).strip().upper()
    detector = args.detector or DETECTORS.get(camera)
    if detector is None:
        raise ValueError(f"{path}: cannot tell IFS from imager (CAMERA={camera!r}); pass --detector")
    read_mode = ("ANY" if cal_type in CLOCK_FREE
                 else args.read_mode or str(hdr.get("MCLOCK", "ANY")).strip() or "ANY")
    md5 = file_md5(path)
    rec = {
        "id": str(uuid.uuid5(ID_NAMESPACE, md5)),
        "filename": os.path.basename(path),
        "cal_type": cal_type,
        "detector": detector,
        "instrument_era": args.instrument_era,
        "instrument_mode": str(hdr.get("IFSMODE", "ANY")).strip() if cal_type in CLOCK_FREE else "ANY",
        "read_mode": read_mode,
        "mjd_start": float(mjd) if mjd is not None else float(t.mjd),
        "datetime_obs": t.isot + "Z",
        "target_name": args.target_name or ("monochromator" if cal_type in CLOCK_FREE else cal_type),
        "drp_version": str(hdr.get("DRPVER", "unknown")),
        "cal_version": args.cal_version or str(hdr.get("CALVER", "001")),
        "data_quality": args.data_quality,
        "file_md5": md5,
        "file_size": os.path.getsize(path),
        "origin": args.origin,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    if hdr.get("STATEID"):
        rec["stateid"] = str(hdr["STATEID"]).strip()
    if "INTERP" in hdr:
        rec["interpolated"] = bool(hdr["INTERP"])
    if args.koa_filepath:
        rec["koa_filepath"] = os.path.join(args.koa_filepath, rec["filename"])
        rec["koa_id"] = rec["filename"]
    if args.note:
        rec["note"] = args.note
    return rec


def cmd_publish(args):
    records = [record_from_file(p, args) for p in args.files]
    if not args.koa_filepath:
        print("warning: no --koa-filepath, so the caldb will not be able to serve these files",
              file=sys.stderr)
    if args.dry_run:
        print(json.dumps(records, indent=2))
        return
    store = make_store(args)
    print(store.client.add(records)["message"])


# ---------------- convert ----------------

def cmd_convert(args):
    header = {"IFSMODE": args.ifsmode, "CAMERA": "IFS"}
    if args.stateid:
        header["STATEID"] = args.stateid
    if args.date_obs:
        header["DATE-OBS"] = args.date_obs
    if args.interp is not None:
        header["INTERP"] = args.interp
    out = rectmat_io.convert_npz_rectmat(args.npz, args.out, args.lams, header=header,
                                         kind=args.kind, lenslet_grid=args.lenslet_grid)
    rect = rectmat_io.read_rectmat(out)
    print(f"{out}: {rect.kind} {rect.cube_shape}, {rect.matrix.nnz} nonzeros "
          f"(was {rect.header['NSTORED']} stored), {os.path.getsize(out) / 1e6:.0f} MB")


def build_parser():
    p = argparse.ArgumentParser(prog="scales_caldb", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", help="caldb base URL (default $SCALES_CALDB_URL)")
    p.add_argument("--cache-dir", help="cache root (default $SCALES_CALIB_CACHE or ~/.cache/scalesdrp)")
    p.add_argument("--offline", action="store_true", help="use only the local cache")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list", help="list SCALES caldb records")
    s.add_argument("--cached", action="store_true", help="only records whose file is cached")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("sync", help="download the current rectmats (and detector files) into the cache")
    s.add_argument("--mode", action="append", choices=IFS_MODES, help="IFS mode (repeatable; default all)")
    s.add_argument("--date", help="select what applies at this time (ISO; default now)")
    s.add_argument("--detector-files", action="store_true",
                   help="also fetch the bad pixel masks, readnoise, saturation and linearity files")
    s.add_argument("--bpmats", action="store_true", help="also build and cache the bpmats")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("publish", help="register calibration files with the caldb")
    s.add_argument("files", nargs="+")
    s.add_argument("--instrument-era", required=True)
    s.add_argument("--koa-filepath", help="directory the files are served from on the caldb host")
    s.add_argument("--cal-type", choices=CAL_TYPES, help="override the header's CALTYPE")
    s.add_argument("--detector", choices=["IFS", "IMG"])
    s.add_argument("--read-mode", choices=READ_MODES + ["ANY"])
    s.add_argument("--date-obs", help="override DATE-OBS (ISO)")
    s.add_argument("--cal-version")
    s.add_argument("--target-name")
    s.add_argument("--data-quality", default="GOOD", choices=["GOOD", "BAD"])
    s.add_argument("--origin", default="LOCAL", choices=["KECK", "LOCAL"])
    s.add_argument("--note")
    s.add_argument("--dry-run", action="store_true", help="print the records instead of sending them")
    s.set_defaults(func=cmd_publish)

    s = sub.add_parser("convert", help="convert a legacy .npz rectmat to FITS")
    s.add_argument("npz")
    s.add_argument("out")
    s.add_argument("--lams", required=True, help="the matching *_lams.fits or *_intp_lams.fits")
    s.add_argument("--ifsmode", required=True, choices=IFS_MODES)
    s.add_argument("--kind", choices=["OPT", "C2"], help="default: from the matrix orientation")
    s.add_argument("--lenslet-grid", type=int, nargs=2, metavar=("NY", "NX"),
                   help="needed if the .npz has no stored cube_shape")
    s.add_argument("--stateid")
    s.add_argument("--date-obs")
    interp = s.add_mutually_exclusive_group()
    interp.add_argument("--interp", dest="interp", action="store_true", default=None)
    interp.add_argument("--no-interp", dest="interp", action="store_false")
    s.set_defaults(func=cmd_convert)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (CalDBError, ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
