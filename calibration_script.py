##======== AIDAN'S CALIBRATOR-INATOR ========##

import argparse
import csv
import sys

from pathlib import Path
from collections import defaultdict

import numpy as np
import ccdproc as ccdp                      # BING, BANG, BOOM

from astropy.io import fits                 # reading the .fits files
from astropy.nddata import CCDData          # ccdproc's data container
from astropy.stats import mad_std

##======== CONFIGURATION BLOCK & SUCH ========##

RAW_ROOT = r"/Users/aidandombrosky/Desktop/ROBO_data"
OUTPUT_DIR = r"/Users/aidandombrosky/Desktop/calibrated"
MASTERS_DIR = r"/Users/aidandombrosky/Desktop/master_calibrated"

BIAS_DIRNAME = "BIAS"
FLAT_DIRNAME = "TWILIGHT FLAT"
EXCLUDE_DIRNAMES = {BIAS_DIRNAME, FLAT_DIRNAME, "UNSORTED"}

DRY_RUN = False
ALLOW_FALLBACK_TO_GLOBAL_MASTER = True

UNIT = "adu"
MEM_LIMIT_BYTES = 2e9
SIGMA_CLIP_LOW = 5
SIGMA_CLIP_HIGH = 5

APPLY_OVERSCAN_CORRECTION = True
OVERSCAN_SECTION = "[2099:2138, 1:2052]"
TRIM_SECTION = "[1:2048, 1:2052]"

FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fits.gz", ".fit.gz"}
FILTER_KEYWORDS = ["FILTNME1", "FILTNME2", "FILTER", "FILTER1", "FILTNAME"]

def find_fits_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file():
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

def overscan_correct_and_trim(ccd):
    if not APPLY_OVERSCAN_CORRECTION:
        return ccd
    ccd = ccdp.subtract_overscan(ccd, fits_section = OVERSCAN_SECTION, median = True)
    ccd = ccdp.trim_image(ccd, fits_section = TRIM_SECTION)
    return ccd

def scan_dir_with_metadata(dir_path: Path):
    results = []
    if not dir_path.is_dir():
        return results
    for fpath in find_fits_files(dir_path):
        try:
            header = fits.getheader(fpath)
        except Exception as e:
            print(f" WARNING: could not read header for {fpath}: {e}")
            continue
        results.append({
            "path": fpath,
            "date_token": get_date_token(header, fpath.name),
            "filter": get_filter(header),
        })
    return results

def combine_paths(filepaths, label):
    print(f" Combining {len(filepaths)} frames for {label}...")
    combined = ccdp.combine(
        [str(p) for p in filepaths],
        method = "median",
        sigma_clip = True,
        sigma_clip_low_thresh = SIGMA_CLIP_LOW,
        sigma_clip_high_thresh = SIGMA_CLIP_HIGH,
        sigma_clip_func = np.ma.median,
        sigma_clip_dev_func = mad_std,
        mem_limit = MEM_LIMIT_BYTES,
        unit = UNIT,
    )
    combined.meta["combined"] = True
    combined.meta["ncombine"] = len(filepaths)
    return combined

def combine_ccddata_list(ccddata_list, label):
    print(f" Combining {len(ccddata_list)} frames for {label}...")
    combined = ccdp.combine(
        ccddata_list,
        method = "median",
        sigma_clip = True,
        sigma_clip_low_thresh = SIGMA_CLIP_LOW,
        sigma_clip_high_thresh = SIGMA_CLIP_HIGH,
        sigma_clip_func = np.ma.median,
        sigma_clip_dev_func = mad_std,
        mem_limit = MEM_LIMIT_BYTES,
    )
    combined.meta["combined"] = True
    combined.meta["ncombine"] = len(ccddata_list)
    return combined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default = RAW_ROOT)
    parser.add_argument("--out-dir", default = OUTPUT_DIR)
    parser.add_argument("--masters-dir", default=MASTERS_DIR)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dry_run = DRY_RUN and not args.execute

    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    masters_dir = Path(args.masters_dir).expanduser().resolve()

    if not raw_root.is_dir():
        print(f"ERROR: raw root not found: {raw_root}")
        sys.exit(1)

    bias_records = scan_dir_with_metadata(raw_root / BIAS_DIRNAME)
    flat_records = scan_dir_with_metadata(raw_root / FLAT_DIRNAME)

    print(f"DEBUG: flat_records has {len(flat_records)} entries")
    for r in flat_records[:5]:
        print(f"  {r['path'].name}  night={r['date_token']}  filter={r['filter']}")

    if not bias_records:
        print(f"ERROR: no bias frames found under {raw_root / BIAS_DIRNAME}")
        sys.exit(1)
    if not flat_records:
        print(f"ERROR: no flat frames found under {raw_root / FLAT_DIRNAME}")
        sys.exit(1)

    bias_by_night = defaultdict(list)
    for r in bias_records:
        bias_by_night[r["date_token"]].append(r["path"])

    flat_by_night_filter = defaultdict(list)
    for r in flat_records:
        flat_by_night_filter[(r["date_token"], r["filter"])].append(r["path"])

    object_dirs = sorted(
        d for d in raw_root.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DIRNAMES
        and d.resolve() != output_dir
        and d.resolve() != masters_dir
    )

    print("===BIAS COVERAGE BY NIGHT===")
    for night in sorted(bias_by_night):
        print(f" {night}: {len(bias_by_night[night])} frames")

    print("\n===FLAT COVERAGE BY (night, filter)===")
    for (night, filt) in sorted(flat_by_night_filter):
        print(f" night = {night} filter = {filt}: {len(flat_by_night_filter [(night, filt)])} frames")

    print(f"\n===OBJECT FOLDERS FOUND===")
    for d in object_dirs:
        print(f" {d.name}")

    light_records_by_object = {}
    nights_needed = set()
    filters_needed_by_night = defaultdict(set)
    for d in object_dirs:
        recs = scan_dir_with_metadata(d)
        light_records_by_object[d.name] = recs
        for r in recs:
            nights_needed.add(r["date_token"])
            filters_needed_by_night[r["date_token"]].add(r["filter"])

    print("\n===CHECKING CALIBRATION COVERAGE AGAINST LIGHT FRAMES===")
    any_fallback_needed = False
    for night in sorted(nights_needed):
        if night not in bias_by_night:
            any_fallback_needed = True
            print(f" WARNING: night {night} has light frames but NO bias frames"
                  f"--will need fallback" if ALLOW_FALLBACK_TO_GLOBAL_MASTER
                else f"ERROR: night {night} has light frames but NO bias frames")
        for filt in filters_needed_by_night[night]:
            if (night, filt) not in flat_by_night_filter:
                any_fallback_needed = True
                msg = f"  WARNING: night {night}, filter {filt} has light frames but NO matching flat"
                print(msg + (
                    " -- will need fallback" if ALLOW_FALLBACK_TO_GLOBAL_MASTER else " (ERROR: will be skipped)"))

    if not any_fallback_needed:
        print("All nights & filters needed by light frames have matching bias/flat coverage. GOOD STUFF!")

    if dry_run:
        print("\n*** DRY RUN: no master frames built, no files calibrated. ***")
        print("Please review the coverage summary above. If it looks a-okay, set DRY_RUN = False")
        print("in the configuration block (or re-run with --execute) to actually calibrate.")
        return

    masters_dir.mkdir(parents = True, exist_ok = True)
    output_dir.mkdir(parents = True, exist_ok = True)

    master_bias_cache = {}
    master_flat_cache = {}

    def get_master_bias(night):
        if night in master_bias_cache:
            return master_bias_cache[night]

        if night in bias_by_night:
            paths = bias_by_night[night]
            print(f" COMBINING {len(paths)} FRAMES for MASTER BIAS (night {night})")
            overscan_corrected = [overscan_correct_and_trim(CCDData.read(p, unit = UNIT)) for p in paths]
            mb = combine_ccddata_list(overscan_corrected, f"MASTER BIAS (night {night})")
            source = "night-specific"
            mb.write(masters_dir / f"master_bias_{night}.fits", overwrite = True)
        elif ALLOW_FALLBACK_TO_GLOBAL_MASTER:
            if "GLOBAL" not in master_bias_cache:
                all_paths = [r["path"] for r in bias_records]
                print(f" COMBINING {len(all_paths)} FRAMES for MASTER BIAS as (GLOBAL FALLBACK)")
                overscan_corrected = [overscan_correct_and_trim(CCDData.read(p, unit = UNIT)) for p in all_paths]
                mb_global = combine_ccddata_list(overscan_corrected, "MASTER BIAS (GLOBAL FALLBACK - ALL NIGHTS")
                mb_global.write(masters_dir / "master_bias_GLOBAL.fits", overwrite = True)
                master_bias_cache["GLOBAL"] = (mb_global, "GLOBAL FALLBACK")
            mb, source = master_bias_cache["GLOBAL"]
            print(f" WARNING: NO BIAS for night {night}; using GLOBAL FALLBACK MASTER BIAS")
        else:
            return None, None

        master_bias_cache[night] = (mb, source)
        return mb, source

    def get_master_flat(night, filt, bias_for_subtraction):
        key = (night, filt)
        if key in master_flat_cache:
            return master_flat_cache[key]

        if key in flat_by_night_filter:
            paths = flat_by_night_filter[key]
            source = "night-specific"
            cache_key = key
        elif ALLOW_FALLBACK_TO_GLOBAL_MASTER:
            global_key = ("GLOBAL", filt)
            if global_key in master_flat_cache:
                return master_flat_cache[global_key]
            paths = [r["path"] for r in flat_records if r["filter"] == filt]

            if not paths:
                return None, None
            source = "GLOBAL FALLBACK"
            cache_key = global_key
            print(f" WARNING: NO FLAT for night {night}, filter {filt}; using GLOBAL FALLBACK MASTER FLAT")
        else:
            return None, None

        bias_subtracted = []
        for fp in paths:
            raw = overscan_correct_and_trim(CCDData.read(fp, unit = UNIT))
            bias_subtracted.append(ccdp.subtract_bias(raw, bias_for_subtraction))

        label = f"MASTER FLAT (night {night}, filter {filt})" if source == "night-specific" \
            else f"MASTER FLAT (GLOBAL FALLBACK, filter {filt})"
        combined_flat = combine_ccddata_list(bias_subtracted, label)

        combined_flat = combined_flat.divide(np.nanmedian(combined_flat.data))

        safe_filt = "".join(c if c.isalnum() else "_" for c in filt)
        if source == "night-specific":
            out_name = f"master_flat_{night}_{safe_filt}.fits"
        else:
            out_name = f"master_flat_GLOBAL_{safe_filt}.fits"
        combined_flat.write(masters_dir / out_name, overwrite = True)

        master_flat_cache[cache_key] = (combined_flat, source)
        return combined_flat, source

    print("\n===CALIBRATING LIGHT FRAMES===")
    log_rows = []
    n_ok, n_skipped = 0, 0

    for obj_name, recs in light_records_by_object.items():
        print(f"\n--OBJECT: {obj_name} ({len(recs)} frames)--")
        for rec in recs:
            night = rec["date_token"]
            filt = rec["filter"]
            fpath = rec["path"]

            master_bias, bias_source = get_master_bias(night)
            if master_bias is None:
                print(f" SKIP {fpath.name}: NO BIAS available for night {night}")
                n_skipped += 1
                continue

            master_flat, flat_source = get_master_flat(night, filt, master_bias)

            if master_flat is None:
                print(f" SKIP {fpath.name}: NO FLAT available for filter {filt}")
                n_skipped += 1
                continue

            try:
                raw = overscan_correct_and_trim(CCDData.read(fpath, unit=UNIT))
                bias_sub = ccdp.subtract_bias(raw, master_bias)
                flat_corrected = ccdp.flat_correct(bias_sub, master_flat)
            except Exception as e:
                print(f"  SKIP {fpath.name}: could not read file ({e})")
                n_skipped += 1
                continue

            flat_corrected.data[~np.isfinite(flat_corrected.data)] = np.nan

            flat_corrected.header['BIASCORR'] = True
            flat_corrected.header['OVRSCAN'] = True
            flat_corrected.header['FLATCORR'] = True

            out_dir = output_dir / obj_name / night
            out_dir.mkdir(parents = True, exist_ok = True)
            out_path = out_dir / fpath.name
            flat_corrected.write(out_path, overwrite = True)

            log_rows.append({
                "object": obj_name,
                "night": night,
                "filter": filt,
                "source_file": str(fpath),
                "output_file": str(out_path),
                "bias_source": bias_source,
                "flat_source": flat_source,
            })
            n_ok += 1

    log_path = Path("calibration_log.csv").resolve()
    with open(log_path, "w", newline = "") as f:
        fieldnames = ["object", "night", "filter", "source_file", "output_file", "bias_source", "flat_source"]
        writer = csv.DictWriter(f, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    n_fallback_bias = sum(1 for r in log_rows if r["bias_source"] != "night-specific")
    n_fallback_flat = sum(1 for r in log_rows if r["flat_source"] != "night-specific")

    print(f"\n DONE: light frames calibrated, {n_skipped} skipped.")
    print(f" {n_fallback_bias} frames used a GLOBAL FALLBACK MASTER BIAS")
    print(f" {n_fallback_flat} frames used a GLOBAL FALLBACK MASTER FLAT")
    print(f"Wrote calibration log: {log_path}")
    print(f"Calibrated frames are in: {output_dir}")
    print(f"Master calibration frames are in: {masters_dir}")

if __name__ == "__main__":
    main()