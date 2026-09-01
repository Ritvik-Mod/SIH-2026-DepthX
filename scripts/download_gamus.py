#!/usr/bin/env python
"""Fetch GAMUS from HuggingFace (earthflow/GAMUS) into data/GAMUS/.

8,724 tiles across images/ heights/ classes/ x train/val/test.  Check free disk before
starting -- the raw tiles are 1024x1024 uint8 RGB + float32 height + float32 class.
"""
import argparse, shutil, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/GAMUS")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--splits", nargs="*", default=None,
                    help="e.g. --splits val   (default: everything)")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    free = shutil.disk_usage(Path(a.out).parent if Path(a.out).parent.exists() else ".").free
    print(f"free disk: {free/1e9:.1f} GB")

    patterns = None
    if a.splits:
        patterns = [f"{d}/{s}/*" for d in ("images", "heights", "classes") for s in a.splits]
        print("patterns:", patterns)

    p = snapshot_download("earthflow/GAMUS", repo_type="dataset", local_dir=a.out,
                          allow_patterns=patterns, max_workers=a.workers)
    print("downloaded to", p)
    for sub in ("images", "heights", "classes"):
        for split in ("train", "val", "test"):
            d = Path(p) / sub / split
            if d.is_dir():
                print(f"  {sub}/{split}: {len(list(d.glob('*.h5')))} files")


if __name__ == "__main__":
    main()
