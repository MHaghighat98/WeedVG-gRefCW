#!/usr/bin/env python3
"""
build_dataset.py — reconstruct the gRef-CW image set locally.

gRef-CW ships only the ANNOTATIONS (``grefs(unc).json`` and ``instances.json``).
The underlying images are part of the CropAndWeed dataset (AIT) and, under its
non-commercial licence, **may not be redistributed**. This script therefore
obtains the images from the official CropAndWeed source and arranges them to
match the gRef-CW annotations.

Pipeline
--------
  1. (``--auto``) clone the official CropAndWeed repo and run its downloader
     (``python cnw/setup.py``), or point at an existing CropAndWeed checkout
     with ``--cnw-images``.
  2. index the downloaded CropAndWeed images by file name / stem.
  3. for every image referenced by ``instances.json``, link (or copy) it into
     ``<data-dir>/images/``.
  4. verify that every referenced image is present; report anything missing.

Usage
-----
  # Fully automatic: clone CropAndWeed, download images, then build.
  python scripts/build_dataset.py --data-dir data --auto

  # If you already have CropAndWeed images on disk:
  python scripts/build_dataset.py --data-dir data \
      --cnw-images /path/to/cropandweed-dataset/data/images

  # Copy instead of symlink (e.g. across filesystems):
  python scripts/build_dataset.py --data-dir data --cnw-images <dir> --copy

Notes
-----
* gRef-CW file names match the original CropAndWeed image names, e.g.
  ``vwg-0286-0002.jpg`` / ``ave-0159-0023.jpg`` (``ave``/``vwg`` = the Application /
  Experimental sets). Matching is by file name, falling back to the stem, so it is
  tolerant of differing extensions (``.jpg``/``.png``).
* Stdlib only — no third-party dependencies.
* Licence details: see ``DATASHEET.md`` and ``LICENSE``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CNW_REPO = "https://github.com/cropandweed/cropandweed-dataset.git"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def log(msg: str) -> None:
    print(f"[build_dataset] {msg}", flush=True)


def referenced_file_names(instances_json: Path) -> list[str]:
    """Return the list of image file names referenced by instances.json (COCO-style)."""
    with instances_json.open(encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images")
    if not images:
        sys.exit(f"ERROR: no 'images' array found in {instances_json}. "
                 "Confirm this is a COCO-style instances file.")
    names = []
    for img in images:
        name = img.get("file_name") or img.get("filename") or img.get("name")
        if not name:
            sys.exit(f"ERROR: an entry in {instances_json} has no file_name field: {img!r}")
        names.append(name)
    return names


def index_images(images_dir: Path) -> dict[str, Path]:
    """Map both full file name and stem -> path for every image under images_dir."""
    if not images_dir.is_dir():
        sys.exit(f"ERROR: CropAndWeed images directory not found: {images_dir}")
    index: dict[str, Path] = {}
    count = 0
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
            count += 1
    log(f"indexed {count} image files under {images_dir}")
    if count == 0:
        sys.exit("ERROR: no image files found. Did the CropAndWeed download succeed? "
                 "Pass the correct --cnw-images directory.")
    return index


def resolve(name: str, index: dict[str, Path]) -> Path | None:
    """Find a CropAndWeed image for a referenced gRef-CW file name."""
    if name in index:
        return index[name]
    stem = Path(name).stem
    return index.get(stem)


def auto_download(workdir: Path) -> Path:
    """Clone CropAndWeed and run its downloader. Returns the images directory."""
    repo_dir = workdir / "cropandweed-dataset"
    if not repo_dir.exists():
        log(f"cloning CropAndWeed into {repo_dir} ...")
        subprocess.run(["git", "clone", "--depth", "1", CNW_REPO, str(repo_dir)], check=True)
    else:
        log(f"reusing existing CropAndWeed checkout at {repo_dir}")

    setup = repo_dir / "cnw" / "setup.py"
    if not setup.exists():
        sys.exit(f"ERROR: expected CropAndWeed downloader at {setup} but it is missing. "
                 "Check the CropAndWeed repo layout and use --cnw-images instead.")
    log("running CropAndWeed downloader (python cnw/setup.py) — this fetches the images ...")
    subprocess.run([sys.executable, "cnw/setup.py"], cwd=str(repo_dir), check=True)

    # CropAndWeed extracts images under data/images by convention; fall back to a search.
    candidate = repo_dir / "data" / "images"
    if candidate.is_dir() and any(candidate.rglob("*")):
        return candidate
    for sub in repo_dir.rglob("images"):
        if sub.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in sub.rglob("*") if p.is_file()):
            return sub
    sys.exit("ERROR: CropAndWeed download finished but no images directory was found. "
             "Locate it manually and re-run with --cnw-images.")


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(os.path.relpath(src, dst.parent), dst)
        except OSError:
            shutil.copy2(src, dst)  # symlinks unsupported (e.g. some Windows setups)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconstruct the gRef-CW image set from CropAndWeed.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="gRef-CW data dir containing instances.json; images written to <data-dir>/images.")
    ap.add_argument("--instances-json", type=Path, default=None,
                    help="Path to instances.json (default: <data-dir>/instances.json).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--auto", action="store_true",
                     help="Clone CropAndWeed and run its official downloader automatically.")
    src.add_argument("--cnw-images", type=Path,
                     help="Path to an existing CropAndWeed images directory.")
    ap.add_argument("--workdir", type=Path, default=Path(".cropandweed_cache"),
                    help="Where to clone CropAndWeed when using --auto.")
    ap.add_argument("--copy", action="store_true",
                    help="Copy image files instead of symlinking.")
    args = ap.parse_args()

    instances_json = args.instances_json or (args.data_dir / "instances.json")
    if not instances_json.exists():
        sys.exit(f"ERROR: {instances_json} not found. Download the gRef-CW annotations first "
                 "(see README 'Data and Weights').")

    out_dir = args.data_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    cnw_images = auto_download(args.workdir) if args.auto else args.cnw_images
    log(f"CropAndWeed images: {cnw_images}")

    index = index_images(cnw_images)
    names = referenced_file_names(instances_json)
    log(f"{len(names)} images referenced by {instances_json.name}")

    matched, missing = 0, []
    for name in names:
        src_path = resolve(name, index)
        if src_path is None:
            missing.append(name)
            continue
        link_or_copy(src_path, out_dir / Path(name).name, args.copy)
        matched += 1

    log(f"linked/copied {matched}/{len(names)} images into {out_dir}")
    if missing:
        log(f"WARNING: {len(missing)} referenced images were NOT found in CropAndWeed. "
            f"First few: {missing[:10]}")
        log("This usually means the CropAndWeed download is incomplete or the file-name "
            "convention differs — verify the source and re-run.")
        return 1
    log("Done — every referenced image was found. Dataset is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
