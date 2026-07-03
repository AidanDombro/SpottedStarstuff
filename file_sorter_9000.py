##======== AIDAN'S .FITS FILE SORTER 9000, Mk. II (C) TM Patent Pending ========##

import argparse
import csv      # will write the manifest (ship's log, arghh)
import shutil
import sys

from pathlib import Path
from collections import defaultdict

from astropy.io import fits

##======== CONFIGURATION BLOCK & SUCH ========##

# i'm setting up this as a source directory to change for each of the file names & nights in Dr. Milingo's (JM) thumb drive
# NOTE: just need to change the path for each separate file then we're good to go
SOURCE_DIR = r"/Users/aidandombrosky/Desktop/NURO_2011/07 - Mar17_11"

# this will be the destination of the folders after the code searches the .fits files recursively [INSERT RECURSION JOKE HERE]
DEST_DIR = r"/Users/aidandombrosky/Desktop"

MODE = "copy"  # for first test run, i'll keep JM's files untouched then un-comment following line
# MODE = "move"

# same idea for trial run (set to = True)---touch nothing!!
DRY_RUN = False     # set to False after successful trial to actually move files---touch everything!!

SUBFOLDER_BY_DATE = False

FITS_EXTENSIONS = {".fits", ".fit", ".fts", ".fits.gz", ".fit.gz"}  # every possible mutation of .fits--zipped/unzipped

OBJECT_KEYWORDS = ["OBJECT", "OBJNAME"]  # JM's .fits headers have these keywords for object type

# this def locates all of the .fits files that I defined under FITS_EXTENSIONS above just in case they are other extraneous file extensions in data folders
# walks through entire source folder recursively [recursively, recursively, etc.] with Path.rglob("*") to descend into each subfolder automatically
# finds each file, checks whether the filename ends in any of the FITS extensions i defined in line 28 (many a headache here)
# spits out a sorted list of every FITS file path under my source directory
def find_fits_file(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            name_lower = p.name.lower()
            if any(name_lower.endswith(ext) for ext in FITS_EXTENSIONS):
                files.append(p)
    return sorted(files)

# def looks into .fits file metadata inside of headers to search for keywords iterating over the two keywords i provided under OBJECT_KEYWORDS
# reads ONLY header block, not pixel data to keep pixel array untouched
def get_object_name(header):
    for key in OBJECT_KEYWORDS:
        val = header.get(key)
        if val and str(val).strip():
            raw = str(val).strip()
            normalized = " ".join(raw.split()).upper()  # normalizes raw string so any permutation of an object name with differing capitalization all map to the same folder name
            return raw, normalized
    return None, None

#
def get_date_token(header, filename: str):
    date_obs = header.get("DATE-OBS")
    if date_obs:
        return str(date_obs).split("T")[0].replace("-", "")

    stem = filename.split(".")[0]
    if stem.isdigit():
        return stem
    return "unknown_date"


def safe_folder_name(name: str) -> str:
    bad_chars = "<>:\"/\\|?*"
    cleaned = "".join(c if c not in bad_chars else "_" for c in name)
    return cleaned.strip().rstrip(".") or "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE_DIR)
    parser.add_argument("--dest", default=DEST_DIR)
    parser.add_argument("--mode", choices=["copy", "move"], default=MODE)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--by-date", action="store_true", default=SUBFOLDER_BY_DATE)

    args = parser.parse_args()

    dry_run = DRY_RUN and not args.execute
    if args.execute:
        dry_run = False

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()

    if not source.is_dir():
        print(f"ERROR: source folder not found: {source}")
        print("Please edit SOURCE_DIR in the configuration block, or pass --source on the command line.")
        sys.exit(1)

    files = find_fits_file(source)
    print(f"Found {len(files)} FITS files under {source}\n")
    if not files:
        sys.exit(0)

    by_object = defaultdict(list)
    unclassified = []
    manifest_rows = []

    for i, fpath in enumerate(files, 1):
        try:
            header = fits.getheader(fpath)
        except Exception as e:
            unclassified.append((fpath, f"unreadable hearder: {e}"))
            continue

        raw_obj, norm_obj = get_object_name(header)
        if norm_obj is None:
            unclassified.append((fpath, "no OBJECT keyword found"))
            continue

        date_token = get_date_token(header, fpath.name) if args.by_date else None
        by_object[norm_obj].append((fpath, raw_obj, date_token))
        manifest_rows.append({
            "filepath": str(fpath),
            "object_raw": raw_obj,
            "object_normalized": norm_obj,
            "date_token": date_token or "",
        })

        if i % 200 == 0:
            print(f" ... scanned {i}/{len(files)}")

    print("\n=== OBJECTS FOUND ===")
    for obj_norm, entries in sorted(by_object.items()):
        raw_examples = sorted(set(e[1] for e in entries))
        print(f" {obj_norm!r}: {len(entries)} files"
              f" (raw header value(s) seen: {raw_examples})")

    if unclassified:
        print(f"\n*** {len(unclassified)} files had no usable OBJECT keyword ***")
        for fpath, reason in unclassified[:10]:
            print(f" {fpath.name}: {reason}")
        if len(unclassified) > 10:
            print(f" ...& {len(unclassified) - 10} more")
        print("These files will be placed in a folder named 'UNSORTED' if execute.")

    manifest_path = Path("object_sort_manifest.csv").resolve()
    with open(manifest_path, "w", newline="") as f:
        fieldnames = ["filepath", "object_raw", "object_normalized", "date_token"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWROTE MANIFEST: {manifest_path}")

    if dry_run:
        print("\n*** DRY RUN: no files were copied or moved. ***")
        print("Please review the object list above. If it looks right, either:")
        print("  - set DRY_RUN = False in the configuration block & re-run, or;")
        print("  - re-run from the terminal with --execute.")
        return

    print(f"\n=== EXECUTING: {args.mode} files into {dest} ===")
    dest.mkdir(parents=True, exist_ok=True)
    n_done = 0

    def place(fpath, obj_folder_name, date_token):
        nonlocal n_done
        target_dir = dest / safe_folder_name(obj_folder_name)
        if args.by_date and date_token:
            target_dir = target_dir / safe_folder_name(date_token)
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / fpath.name
        if target_path.exists():
            target_path = target_dir / f"{fpath.stem}__dup{n_done}{fpath.suffix}"

        if args.mode == "copy":
            shutil.copy2(fpath, target_path)
        else:
            shutil.move(str(fpath), str(target_path))
        n_done += 1

    for obj_norm, entries in by_object.items():
        for fpath, _raw_obj, date_token in entries:
            place(fpath, obj_norm, date_token)

    for fpath, _reason in unclassified:
        place(fpath, "UNSORTED", None)

    print(f"\nDone. {n_done} files {('copied' if args.mode == 'copy' else 'moved')} into {dest}")
    print("FOLDER LAYOUT:")
    for obj_norm in sorted(by_object.keys()):
        print(f" {dest / safe_folder_name(obj_norm)}")
    if unclassified:
        print(f" {dest / 'UNSORTED'}")


if __name__ == "__main__":
    main()
