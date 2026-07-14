import argparse
import sys
import time
import gc

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc as ccdp
from tqdm import tqdm

INPUT_DIR = r"/Volumes/starstuff/Frames/ROBO_data/calibrated"

BACKUP_DIR = r"/Volumes/starstuff/Frames/ROBO_data/calibrated_trimmed"

BACKUP = True
DRY_RUN = False


TRIM_SECTION = "[55:2098, 1:2052]"


WIDTH_NEEDS_TRIM = 2138  # raw width including both overscan strips
WIDTH_PARTIAL_TRIM = 2048  # old trim that removed postscan but kept prescan
WIDTH_ALREADY_DONE = 2044  # correct trimmed science width

FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fits.gz", ".fit.gz"}

def find_fits_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("._"):
            name_lower = p.name.lower()
            if any(name_lower.endswith(ext) for ext in FITS_EXTENSIONS):
                files.append(p)
    return sorted(files)


def needs_trimming(fpath: Path):

    try:
        header = fits.getheader(fpath)
        width = header.get("NAXIS1", 0)
        return width in (WIDTH_NEEDS_TRIM, WIDTH_PARTIAL_TRIM), width
    except Exception as e:
        tqdm.write(f"  WARNING: could not read header for {fpath.name}: {e}")
        return False, 0


def trim_frame(fpath: Path, out_path: Path):

    ccd = CCDData.read(fpath, unit="adu")
    trimmed = ccdp.trim_image(ccd, fits_section=TRIM_SECTION)

    trimmed.header["TRIMFIX"] = True
    trimmed.header["TRIMFROM"] = ccd.data.shape[1]  # original width
    trimmed.header["TRIMTO"] = trimmed.data.shape[1]  # new width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed.write(out_path, overwrite=True)

    del ccd, trimmed
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--backup", default=BACKUP_DIR)
    parser.add_argument("--execute", action="store_true",
                        help="Actually trim files. Without this, runs as dry run.")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite originals instead of writing to backup dir.")
    args = parser.parse_args()

    dry_run = DRY_RUN and not args.execute
    do_backup = BACKUP and not args.inplace

    input_dir = Path(args.input).expanduser().resolve()
    backup_dir = Path(args.backup).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    print(f"Scanning {input_dir} ...")
    all_files = find_fits_files(input_dir)
    print(f"Found {len(all_files)} FITS files\n")


    to_trim = []  # (input_path, output_path, current_width)
    already_ok = []
    unreadable = []

    for fpath in all_files:
        needs, width = needs_trimming(fpath)
        if width == 0:
            unreadable.append(fpath)
            continue
        if needs:
            if do_backup:
                rel = fpath.relative_to(input_dir)
                out_path = backup_dir / rel
            else:
                out_path = fpath  # overwrite in place
            to_trim.append((fpath, out_path, width))
        else:
            already_ok.append(fpath)


    print(f"=== TRIAGE SUMMARY ===")
    print(f"  Need trimming  : {len(to_trim)} frames")
    print(f"  Already correct: {len(already_ok)} frames (NAXIS1={WIDTH_ALREADY_DONE})")
    print(f"  Unreadable     : {len(unreadable)} frames")

    if to_trim:

        from collections import Counter
        width_counts = Counter(w for _, _, w in to_trim)
        print(f"\n  Width breakdown of frames to trim:")
        for w, count in sorted(width_counts.items()):
            label = "raw (both strips)" if w == WIDTH_NEEDS_TRIM else "partial trim (prescan remaining)"
            print(f"    NAXIS1={w} ({label}): {count} frames")

    if do_backup:
        print(f"\n  Trimmed frames will be written to: {backup_dir}")
        print(f"  Originals in {input_dir} will NOT be modified")
    else:
        print(f"\n  Frames will be OVERWRITTEN IN PLACE under {input_dir}")

    if dry_run:
        print(f"\n*** DRY RUN: no files modified. ***")
        print("Set DRY_RUN = False or pass --execute to actually trim.")
        return

    if not to_trim:
        print("\nNothing to do -- all frames already correctly trimmed!")
        return


    print(f"\n=== TRIMMING {len(to_trim)} FRAMES ===")
    start = time.perf_counter()
    n_ok = 0
    n_failed = 0

    for fpath, out_path, width in tqdm(to_trim, desc="Trimming", unit="frame",
                                       dynamic_ncols=True):
        try:
            trim_frame(fpath, out_path)
            n_ok += 1
        except Exception as e:
            tqdm.write(f"  SKIP {fpath.name}: {e}")
            n_failed += 1

    elapsed = time.perf_counter() - start
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"\n=== DONE ===")
    print(f"  {n_ok} frames trimmed successfully")
    print(f"  {n_failed} frames failed")
    print(f"  {len(already_ok)} frames skipped (already correct width)")
    if do_backup:
        print(f"  Trimmed frames written to: {backup_dir}")
    print(f"\n  TOTAL RUNTIME: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")


if __name__ == "__main__":
    main()