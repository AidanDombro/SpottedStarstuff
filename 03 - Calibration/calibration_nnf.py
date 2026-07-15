##======== AIDAN'S CALIBRATOR-INATOR (w/ NEAREST-NIGHT FALLBACK) ========##

import argparse
import csv
import sys
import gc
import tempfile
import time
import warnings

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import numpy as np
import ccdproc as ccdp

from astropy.io import fits
from astropy.nddata import CCDData
from astropy.stats import mad_std

from calibration_gfm import APPLY_OVERSCAN_CORRECTION

warnings.filterwarnings("ignore", message = "resource_tracker", category = UserWarning)

##======== CONFIGURATION BLOCK & SUCH ========##

RAW_ROOT = "/Volumes/starstuff/Frames/NURO_2011/sorted"
OUTPUT_DIR = "/Volumes/starstuff/Frames/NURO_2011/calibrated_nnf"
MASTERS_DIR = "/Volumes/starstuff/Frames/NURO_2011/master_calibrated_nnf"

BIAS_DIRNAME = "BIAS"
FLAT_DIRNAME = "FLAT"       # ROBO_cam flats are for some reason under 'TWILIGHT FLAT' idk
EXCLUDE_DIRNAMES = {BIAS_DIRNAME, FLAT_DIRNAME, "UNSORTED"}

DRY_RUN = False

RESUME = True

MAX_FALLBACK_DAYS = 7
ALLOW_GLOBAL_FALLBACK = True

UNIT = "adu"
MEM_LIMIT_BYTES = 5e9
SIGMA_CLIP_LOW = 5
SIGMA_CLIP_HIGH = 5

APPLY_OVERSCAN_CORRECTION = True
OVERSCAN_SECTION = "[2099:2138, 1:2052]"
TRIM_SECTION = "[55:2098, 1:2052]"

FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fits.gz", ".fit.gz"}
FILTER_KEYWORDS = ["FILTNME1", "FILTNME2", "FILTER", "FILTER1", "FILTNAME"]

MAX_EXPECTED_FRAMES = 6000


def find_fits_files(root: Path, exclude_dirs: set = None):
    """Recursively finds FITS files, skipping macOS sidecar files and
    any directory names listed in exclude_dirs."""
    exclude_dirs = exclude_dirs or set()
    files = []
    for p in root.rglob("*"):
        if any(excl in p.parts for excl in exclude_dirs):
            continue
        if p.is_file() and not p.name.startswith("._"):
            name_lower = p.name.lower()
            if any(name_lower.endswith(ext) for ext in FITS_EXTENSIONS):
                files.append(p)
    return sorted(files)


def get_date_token(header, filename: str):
    date_obs = header.get("DATE-OBS")
    if date_obs:
        return str(date_obs).split("T")[0].replace("-", "").replace("/", "")
    stem = filename.split(".")[0]
    if stem.isdigit():
        return stem
    return "unknown_date"


def get_filter(header):
    for key in FILTER_KEYWORDS:
        val = header.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return "NONE"


def night_to_int(night: str):
    """Converts a YYYYMMDD string to an integer for date arithmetic.
    Returns None if the string is not a valid date token."""
    try:
        return int(night)
    except (ValueError, TypeError):
        return None


def days_between(night_a: str, night_b: str):
    """
    Approximate day difference between two YYYYMMDD strings.
    Uses a simple integer subtraction which is accurate enough for
    finding the nearest night within a few weeks -- we don't need
    calendar-precise arithmetic here.
    """
    from datetime import datetime
    try:
        a = datetime.strptime(night_a, "%Y%m%d")
        b = datetime.strptime(night_b, "%Y%m%d")
        return abs((a - b).days)
    except (ValueError, TypeError):
        return 9999


def find_nearest_night(target_night: str, available_nights: list,
                       max_days: int = MAX_FALLBACK_DAYS):
    """
    Given a target night (YYYYMMDD string) and a list of available nights,
    returns the (nearest_night, days_apart) tuple for the closest available
    night within max_days, or (None, None) if none exists within that window.

    This is the core of the nearest-night fallback -- instead of building
    a master from ALL nights (global), we find the single closest night
    that has calibration frames and use just that night's data. This is
    scientifically preferred because:
      - Flat field dust patterns are stable over days, not indefinitely
      - Bias structure is stable but using a closer night is more rigorous
      - Errors are bounded by how much the detector changes in max_days
    """
    best_night = None
    best_days = 9999
    for candidate in available_nights:
        if candidate == target_night:
            continue  # skip self -- already handled as night-specific
        d = days_between(target_night, candidate)
        if d < best_days:
            best_days = d
            best_night = candidate
    if best_night and best_days <= max_days:
        return best_night, best_days
    return None, None


def overscan_correct_and_trim(ccd):
    if not APPLY_OVERSCAN_CORRECTION:
        return ccd
    ccd = ccdp.subtract_overscan(ccd, fits_section=OVERSCAN_SECTION, median=True)
    ccd = ccdp.trim_image(ccd, fits_section=TRIM_SECTION)
    return ccd


def scan_dir_with_metadata(dir_path: Path, exclude_dirs: set = None):
    results = []
    if not dir_path.is_dir():
        return results
    for fpath in find_fits_files(dir_path, exclude_dirs=exclude_dirs):
        try:
            header = fits.getheader(fpath)
        except Exception as e:
            print(f"  WARNING: could not read header for {fpath.name}: {e}")
            continue
        results.append({
            "path": fpath,
            "date_token": get_date_token(header, fpath.name),
            "filter": get_filter(header),
        })
    return results


def build_master_from_paths(paths, label, progress):
    """
    Overscan-corrects each frame to a temp file one at a time,
    then combines from disk. Peak RAM = one frame + combined output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = []
        for i, p in enumerate(paths):
            progress.set_postfix_str(f"Building {label[:40]}")
            ccd = overscan_correct_and_trim(CCDData.read(p, unit=UNIT))
            tmp_path = Path(tmpdir) / f"tmp_{i:04d}.fits"
            ccd.write(tmp_path, overwrite=True)
            tmp_paths.append(str(tmp_path))
            del ccd
            gc.collect()
            progress.update(1)

        combined = ccdp.combine(
            tmp_paths,
            method="median",
            sigma_clip=True,
            sigma_clip_low_thresh=SIGMA_CLIP_LOW,
            sigma_clip_high_thresh=SIGMA_CLIP_HIGH,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=MEM_LIMIT_BYTES,
            unit=UNIT,
        )
    combined.meta["combined"] = True
    combined.meta["ncombine"] = len(paths)
    return combined


def build_master_flat_from_paths(paths, label, master_bias, progress):
    """
    Overscan-corrects and bias-subtracts each flat frame to a temp file
    one at a time, then combines from disk.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_paths = []
        for i, fp in enumerate(paths):
            progress.set_postfix_str(f"Building {label[:40]}")
            raw = overscan_correct_and_trim(CCDData.read(fp, unit=UNIT))
            raw = ccdp.subtract_bias(raw, master_bias)
            tmp_path = Path(tmpdir) / f"tmp_{i:04d}.fits"
            raw.write(tmp_path, overwrite=True)
            tmp_paths.append(str(tmp_path))
            del raw
            gc.collect()
            progress.update(1)

        combined = ccdp.combine(
            tmp_paths,
            method="median",
            sigma_clip=True,
            sigma_clip_low_thresh=SIGMA_CLIP_LOW,
            sigma_clip_high_thresh=SIGMA_CLIP_HIGH,
            sigma_clip_func=np.ma.median,
            sigma_clip_dev_func=mad_std,
            mem_limit=MEM_LIMIT_BYTES,
            unit=UNIT,
        )
    combined.meta["combined"] = True
    combined.meta["ncombine"] = len(paths)
    return combined


##======== MAIN ========##

def main():
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=RAW_ROOT)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--masters-dir", default=MASTERS_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dry_run = DRY_RUN and not args.execute

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    masters_dir = Path(args.masters_dir).expanduser().resolve()

    excluded = {
        output_dir.name,
        masters_dir.name,
        "calibrated",
        "master_calibrated",
        "master_calibration",
    }

    if not raw_root.is_dir():
        print(f"ERROR: raw root not found: {raw_root}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 1: Scan bias, flat, and light frames
    # -------------------------------------------------------------------------
    print("Scanning BIAS and FLAT folders...")
    bias_records = scan_dir_with_metadata(raw_root / BIAS_DIRNAME,
                                          exclude_dirs=excluded)
    flat_records = scan_dir_with_metadata(raw_root / FLAT_DIRNAME,
                                          exclude_dirs=excluded)

    if not bias_records:
        print(f"ERROR: no bias frames found under {raw_root / BIAS_DIRNAME}")
        sys.exit(1)
    if not flat_records:
        print(f"ERROR: no flat frames found under {raw_root / FLAT_DIRNAME}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 2: Group calibration frames by night (and filter for flats)
    # -------------------------------------------------------------------------
    bias_by_night = defaultdict(list)
    for r in bias_records:
        bias_by_night[r["date_token"]].append(r["path"])

    flat_by_night_filter = defaultdict(list)
    for r in flat_records:
        flat_by_night_filter[(r["date_token"], r["filter"])].append(r["path"])

    # Pre-compute sorted lists of available nights for nearest-night search
    available_bias_nights = sorted(bias_by_night.keys())
    available_flat_nights = sorted(set(k[0] for k in flat_by_night_filter.keys()))

    # -------------------------------------------------------------------------
    # STEP 3: Scan object (light frame) folders
    # -------------------------------------------------------------------------
    object_dirs = sorted(
        d for d in raw_root.iterdir()
        if d.is_dir()
        and d.name not in EXCLUDE_DIRNAMES
        and d.resolve() != output_dir
        and d.resolve() != masters_dir
    )

    light_records_by_object = {}
    nights_needed = set()
    filters_needed_by_night = defaultdict(set)

    print("Scanning object folders...")
    for d in object_dirs:
        recs = scan_dir_with_metadata(d, exclude_dirs=excluded)
        light_records_by_object[d.name] = recs
        for r in recs:
            nights_needed.add(r["date_token"])
            filters_needed_by_night[r["date_token"]].add(r["filter"])

    # -------------------------------------------------------------------------
    # STEP 4: Coverage summary and fallback preview
    # -------------------------------------------------------------------------
    print("\n===BIAS COVERAGE BY NIGHT===")
    for night in sorted(bias_by_night):
        print(f"  {night}: {len(bias_by_night[night])} frames")

    print("\n===FLAT COVERAGE BY (night, filter)===")
    for (night, filt) in sorted(flat_by_night_filter):
        print(f"  night={night} filter={filt}: "
              f"{len(flat_by_night_filter[(night, filt)])} frames")

    print("\n===OBJECT FOLDERS FOUND===")
    for obj_name, recs in sorted(light_records_by_object.items()):
        print(f"  {obj_name}: {len(recs)} frames")

    # Show exactly which fallback strategy each missing night/filter will use
    print("\n===FALLBACK PREVIEW (nights missing night-specific calibration)===")
    any_issues = False
    for night in sorted(nights_needed):

        # --- Bias fallback preview ---
        if night not in bias_by_night:
            any_issues = True
            nearest, days = find_nearest_night(night, available_bias_nights)
            if nearest:
                print(f"  BIAS  night={night}: no frames -- "
                      f"will use nearest night {nearest} ({days} days away)")
            elif ALLOW_GLOBAL_FALLBACK:
                print(f"  BIAS  night={night}: no frames within {MAX_FALLBACK_DAYS} days -- "
                      f"will use GLOBAL fallback (last resort)")
            else:
                print(f"  BIAS  night={night}: no frames within {MAX_FALLBACK_DAYS} days -- "
                      f"WILL BE SKIPPED (no global fallback)")

        # --- Flat fallback preview ---
        for filt in filters_needed_by_night[night]:
            if (night, filt) not in flat_by_night_filter:
                any_issues = True
                # Find nearest night that has flats for this specific filter
                nights_with_this_filt = [
                    k[0] for k in flat_by_night_filter.keys()
                    if k[1] == filt
                ]
                nearest, days = find_nearest_night(night, nights_with_this_filt)
                if nearest:
                    print(f"  FLAT  night={night} filter={filt}: no frames -- "
                          f"will use nearest night {nearest} ({days} days away)")
                elif ALLOW_GLOBAL_FALLBACK:
                    print(f"  FLAT  night={night} filter={filt}: "
                          f"no frames within {MAX_FALLBACK_DAYS} days -- "
                          f"will use GLOBAL fallback (last resort)")
                else:
                    print(f"  FLAT  night={night} filter={filt}: "
                          f"no frames within {MAX_FALLBACK_DAYS} days -- "
                          f"WILL BE SKIPPED")

    if not any_issues:
        print("  All nights have night-specific calibration coverage. GOOD STUFF!")

    if dry_run:
        print("\n*** DRY RUN: no masters built, no files calibrated. ***")
        print("Review the fallback preview above -- check that nearest nights")
        print("look reasonable before executing. Set DRY_RUN = False to proceed.")
        return

    # -------------------------------------------------------------------------
    # STEP 5: Setup progress bar and resume logic
    # -------------------------------------------------------------------------
    all_recs = [
        (obj_name, rec)
        for obj_name, recs in light_records_by_object.items()
        for rec in recs
    ]

    if len(all_recs) > MAX_EXPECTED_FRAMES:
        print(f"ERROR: {len(all_recs)} frames found -- more than MAX_EXPECTED_FRAMES "
              f"({MAX_EXPECTED_FRAMES}). Check that output dirs aren't being scanned "
              f"as inputs. Aborting.")
        sys.exit(1)

    if RESUME:
        pending_recs = [
            (obj_name, rec) for obj_name, rec in all_recs
            if not (output_dir / obj_name / rec["date_token"] /
                    rec["path"].name).exists()
        ]
        already_done = len(all_recs) - len(pending_recs)
        print(f"\nRESUME: {already_done} frames already done, "
              f"{len(pending_recs)} remaining")
    else:
        pending_recs = all_recs
        already_done = 0

    masters_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_bias_frames = sum(len(v) for v in bias_by_night.values())
    total_flat_frames = sum(len(v) for v in flat_by_night_filter.values())
    TOTAL_WORK = total_bias_frames + total_flat_frames + len(pending_recs)

    overall_progress = tqdm(
        total=TOTAL_WORK,
        desc="Overall Progress",
        unit="frame",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[Elapsed: {elapsed} | ETA: {remaining} | {rate_fmt}] "
            "{percentage:3.0f}%"
        ),
    )

    master_bias_cache = {}
    master_flat_cache = {}

    # -------------------------------------------------------------------------
    # STEP 6: Master bias builder
    # Fallback hierarchy: night-specific → nearest night → global
    # -------------------------------------------------------------------------
    def get_master_bias(science_night):
        if science_night in master_bias_cache:
            return master_bias_cache[science_night]

        # --- Resume: load from disk if already built ---
        expected_path = masters_dir / f"master_bias_{science_night}.fits"
        if RESUME and expected_path.exists():
            tqdm.write(f"  RESUME: loading master bias for night {science_night}")
            mb = CCDData.read(expected_path, unit=UNIT)
            master_bias_cache[science_night] = (mb, "night-specific", 0)
            return mb, "night-specific", 0

        # --- Night-specific ---
        if science_night in bias_by_night:
            paths = bias_by_night[science_night]
            tqdm.write(f"\n  BUILDING MASTER BIAS (night {science_night}, "
                       f"{len(paths)} frames)")
            mb = build_master_from_paths(
                paths, f"MASTER BIAS (night {science_night})", overall_progress
            )
            gc.collect()
            mb.write(masters_dir / f"master_bias_{science_night}.fits",
                     overwrite=True)
            source, days_offset = "night-specific", 0
            master_bias_cache[science_night] = (mb, source, days_offset)
            return mb, source, days_offset

        # --- Nearest-night fallback ---
        nearest, days = find_nearest_night(science_night, available_bias_nights)
        if nearest:
            cache_key = f"NEAREST_BIAS_{nearest}"
            if RESUME:
                nn_path = masters_dir / f"master_bias_{nearest}.fits"
                if nn_path.exists():
                    tqdm.write(f"  RESUME: loading nearest-night master bias "
                               f"(night {nearest}, {days}d away)")
                    mb = CCDData.read(nn_path, unit=UNIT)
                    master_bias_cache[science_night] = (mb, "nearest-night", days)
                    return mb, "nearest-night", days

            if cache_key not in master_bias_cache:
                paths = bias_by_night[nearest]
                tqdm.write(f"\n  BUILDING NEAREST-NIGHT MASTER BIAS "
                           f"(using night {nearest}, {days} days from {science_night}, "
                           f"{len(paths)} frames)")
                mb = build_master_from_paths(
                    paths, f"MASTER BIAS (nearest {nearest})", overall_progress
                )
                gc.collect()
                # Save under the source night's name so it's reusable
                mb.write(masters_dir / f"master_bias_{nearest}.fits", overwrite=True)
                master_bias_cache[cache_key] = (mb, "nearest-night", days)

            mb, source, _ = master_bias_cache[cache_key]
            master_bias_cache[science_night] = (mb, source, days)
            tqdm.write(f"  WARNING: night {science_night} has no bias -- "
                       f"using nearest night {nearest} ({days} days away)")
            return mb, source, days

        # --- Global fallback (last resort) ---
        if ALLOW_GLOBAL_FALLBACK:
            global_key = "GLOBAL_BIAS"
            global_path = masters_dir / "master_bias_GLOBAL.fits"
            if RESUME and global_path.exists():
                tqdm.write(f"  RESUME: loading GLOBAL master bias")
                if global_key not in master_bias_cache:
                    mb = CCDData.read(global_path, unit=UNIT)
                    master_bias_cache[global_key] = (mb, "global", 9999)
                mb, source, days = master_bias_cache[global_key]
                master_bias_cache[science_night] = (mb, source, days)
                return mb, source, days

            if global_key not in master_bias_cache:
                all_paths = [r["path"] for r in bias_records]
                tqdm.write(f"\n  BUILDING GLOBAL MASTER BIAS "
                           f"({len(all_paths)} frames, last resort)")
                mb = build_master_from_paths(
                    all_paths, "MASTER BIAS GLOBAL", overall_progress
                )
                gc.collect()
                mb.write(global_path, overwrite=True)
                master_bias_cache[global_key] = (mb, "global", 9999)

            mb, source, days = master_bias_cache[global_key]
            master_bias_cache[science_night] = (mb, source, days)
            tqdm.write(f"  WARNING: night {science_night} -- no bias within "
                       f"{MAX_FALLBACK_DAYS} days, using GLOBAL fallback")
            return mb, source, days

        return None, None, None

    # -------------------------------------------------------------------------
    # STEP 7: Master flat builder
    # Fallback hierarchy: night-specific → nearest night (same filter) → global
    # -------------------------------------------------------------------------
    def get_master_flat(science_night, filt, bias_for_subtraction):
        key = (science_night, filt)
        if key in master_flat_cache:
            return master_flat_cache[key]

        safe_filt = "".join(c if c.isalnum() else "_" for c in filt)

        # --- Resume: load from disk if already built ---
        expected_path = masters_dir / f"master_flat_{science_night}_{safe_filt}.fits"
        if RESUME and expected_path.exists():
            tqdm.write(f"  RESUME: loading master flat "
                       f"(night {science_night}, filter {filt})")
            mf = CCDData.read(expected_path, unit=UNIT)
            master_flat_cache[key] = (mf, "night-specific", 0)
            return mf, "night-specific", 0

        # --- Night-specific ---
        if key in flat_by_night_filter:
            paths = flat_by_night_filter[key]
            tqdm.write(f"\n  BUILDING MASTER FLAT "
                       f"(night {science_night}, filter {filt}, {len(paths)} frames)")
            combined = build_master_flat_from_paths(
                paths,
                f"MASTER FLAT (night {science_night}, filter {filt})",
                bias_for_subtraction,
                overall_progress
            )
            gc.collect()
            combined = combined.divide(np.nanmedian(combined.data))
            combined.write(expected_path, overwrite=True)
            master_flat_cache[key] = (combined, "night-specific", 0)
            return combined, "night-specific", 0

        # --- Nearest-night fallback ---
        # Only search nights that have flats for this specific filter
        nights_with_filt = [
            k[0] for k in flat_by_night_filter.keys() if k[1] == filt
        ]
        nearest, days = find_nearest_night(science_night, nights_with_filt)

        if nearest:
            nn_key = (nearest, filt)
            nn_path = masters_dir / f"master_flat_{nearest}_{safe_filt}.fits"

            if RESUME and nn_path.exists():
                tqdm.write(f"  RESUME: loading nearest-night master flat "
                           f"(night {nearest}, filter {filt}, {days}d away)")
                if nn_key not in master_flat_cache:
                    mf = CCDData.read(nn_path, unit=UNIT)
                    master_flat_cache[nn_key] = (mf, "nearest-night", days)
                mf, source, _ = master_flat_cache[nn_key]
                master_flat_cache[key] = (mf, source, days)
                return mf, source, days

            if nn_key not in master_flat_cache:
                paths = flat_by_night_filter[nn_key]
                tqdm.write(f"\n  BUILDING NEAREST-NIGHT MASTER FLAT "
                           f"(using night {nearest}, filter {filt}, "
                           f"{days} days from {science_night}, {len(paths)} frames)")
                combined = build_master_flat_from_paths(
                    paths,
                    f"MASTER FLAT (nearest {nearest}, filter {filt})",
                    bias_for_subtraction,
                    overall_progress
                )
                gc.collect()
                combined = combined.divide(np.nanmedian(combined.data))
                combined.write(nn_path, overwrite=True)
                master_flat_cache[nn_key] = (combined, "nearest-night", days)

            mf, source, _ = master_flat_cache[nn_key]
            master_flat_cache[key] = (mf, source, days)
            tqdm.write(f"  WARNING: night {science_night}, filter {filt} -- "
                       f"no flat, using nearest night {nearest} ({days} days away)")
            return mf, source, days

        # --- Global fallback (last resort) ---
        if ALLOW_GLOBAL_FALLBACK:
            global_key = ("GLOBAL", filt)
            global_path = masters_dir / f"master_flat_GLOBAL_{safe_filt}.fits"

            if RESUME and global_path.exists():
                tqdm.write(f"  RESUME: loading GLOBAL master flat (filter {filt})")
                if global_key not in master_flat_cache:
                    mf = CCDData.read(global_path, unit=UNIT)
                    master_flat_cache[global_key] = (mf, "global", 9999)
                mf, source, days = master_flat_cache[global_key]
                master_flat_cache[key] = (mf, source, days)
                return mf, source, days

            if global_key not in master_flat_cache:
                paths = [r["path"] for r in flat_records if r["filter"] == filt]
                if not paths:
                    tqdm.write(f"  ERROR: no flat frames exist for filter {filt} "
                               f"across any night -- cannot calibrate")
                    return None, None, None
                tqdm.write(f"\n  BUILDING GLOBAL MASTER FLAT "
                           f"(filter {filt}, {len(paths)} frames, last resort)")
                combined = build_master_flat_from_paths(
                    paths,
                    f"MASTER FLAT GLOBAL (filter {filt})",
                    bias_for_subtraction,
                    overall_progress
                )
                gc.collect()
                combined = combined.divide(np.nanmedian(combined.data))
                combined.write(global_path, overwrite=True)
                master_flat_cache[global_key] = (combined, "global", 9999)

            mf, source, days = master_flat_cache[global_key]
            master_flat_cache[key] = (mf, source, days)
            tqdm.write(f"  WARNING: night {science_night}, filter {filt} -- "
                       f"no flat within {MAX_FALLBACK_DAYS} days, using GLOBAL")
            return mf, source, days

        return None, None, None

    # -------------------------------------------------------------------------
    # STEP 8: Calibrate all light frames
    # -------------------------------------------------------------------------
    print(f"\n===CALIBRATING {len(pending_recs)} LIGHT FRAMES===")
    log_rows = []
    n_ok, n_skipped = 0, 0

    for obj_name, rec in pending_recs:
        overall_progress.set_postfix_str(f"Calibrating {obj_name}")
        night = rec["date_token"]
        filt = rec["filter"]
        fpath = rec["path"]

        out_dir = output_dir / obj_name / night
        out_path = out_dir / fpath.name

        if RESUME and out_path.exists():
            n_ok += 1
            overall_progress.update(1)
            log_rows.append({
                "object": obj_name, "night": night, "filter": filt,
                "source_file": str(fpath), "output_file": str(out_path),
                "bias_source": "resumed", "bias_days_offset": 0,
                "flat_source": "resumed", "flat_days_offset": 0,
            })
            continue

        master_bias, bias_source, bias_days = get_master_bias(night)
        if master_bias is None:
            tqdm.write(f"  SKIP {fpath.name}: no bias available")
            n_skipped += 1
            overall_progress.update(1)
            continue

        master_flat, flat_source, flat_days = get_master_flat(
            night, filt, master_bias
        )
        if master_flat is None:
            tqdm.write(f"  SKIP {fpath.name}: no flat available for filter {filt}")
            n_skipped += 1
            overall_progress.update(1)
            continue

        try:
            raw = overscan_correct_and_trim(CCDData.read(fpath, unit=UNIT))
            bias_sub = ccdp.subtract_bias(raw, master_bias)
            flat_corrected = ccdp.flat_correct(bias_sub, master_flat)
        except Exception as e:
            tqdm.write(f"  SKIP {fpath.name}: could not process ({e})")
            n_skipped += 1
            overall_progress.update(1)
            continue

        flat_corrected.data[~np.isfinite(flat_corrected.data)] = np.nan

        # Stamp calibration provenance into the header so every frame
        # self-documents exactly how it was calibrated and how far the
        # calibration frames were from the science observation date.
        flat_corrected.header['BIASCORR'] = True
        flat_corrected.header['OVRSCAN'] = True
        flat_corrected.header['FLATCORR'] = True
        flat_corrected.header['BIASSRC'] = bias_source
        flat_corrected.header['BIASDAY'] = bias_days  # days between science and bias night
        flat_corrected.header['FLATSRC'] = flat_source
        flat_corrected.header['FLATDAY'] = flat_days  # days between science and flat night

        out_dir.mkdir(parents=True, exist_ok=True)
        flat_corrected.write(out_path, overwrite=True)

        log_rows.append({
            "object": obj_name,
            "night": night,
            "filter": filt,
            "source_file": str(fpath),
            "output_file": str(out_path),
            "bias_source": bias_source,
            "bias_days_offset": bias_days,
            "flat_source": flat_source,
            "flat_days_offset": flat_days,
        })
        n_ok += 1
        overall_progress.update(1)

    # -------------------------------------------------------------------------
    # STEP 9: Write calibration log
    # -------------------------------------------------------------------------
    log_path = Path("calibration_log.csv").resolve()
    with open(log_path, "w", newline="") as f:
        fieldnames = [
            "object", "night", "filter", "source_file", "output_file",
            "bias_source", "bias_days_offset", "flat_source", "flat_days_offset"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    overall_progress.close()

    # Summary counts
    n_bias_night = sum(1 for r in log_rows if r["bias_source"] == "night-specific")
    n_bias_nearest = sum(1 for r in log_rows if r["bias_source"] == "nearest-night")
    n_bias_global = sum(1 for r in log_rows if r["bias_source"] == "global")
    n_flat_night = sum(1 for r in log_rows if r["flat_source"] == "night-specific")
    n_flat_nearest = sum(1 for r in log_rows if r["flat_source"] == "nearest-night")
    n_flat_global = sum(1 for r in log_rows if r["flat_source"] == "global")

    elapsed = time.perf_counter() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n===DONE===")
    print(f"  {n_ok} frames calibrated, {n_skipped} skipped")
    print(f"\n  Bias sources:")
    print(f"    Night-specific : {n_bias_night}")
    print(f"    Nearest-night  : {n_bias_nearest}")
    print(f"    Global fallback: {n_bias_global}")
    print(f"\n  Flat sources:")
    print(f"    Night-specific : {n_flat_night}")
    print(f"    Nearest-night  : {n_flat_nearest}")
    print(f"    Global fallback: {n_flat_global}")
    print(f"\n  Calibration log : {log_path}")
    print(f"  Calibrated frames: {output_dir}")
    print(f"  Master frames    : {masters_dir}")
    print()
    print("=" * 60)
    print(f"TOTAL RUNTIME : {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()


