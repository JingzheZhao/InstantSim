"""Renumber dataset images sequentially.

Used when a new batch of generated samples is appended to an existing folder,
so that paired A/B files keep matching indices.

Example:
    python scripts/rename_dataset.py dataset/A --start 3600
    python scripts/rename_dataset.py dataset/A --start 3600 --dry-run
"""

import argparse
import os
import sys

EXTENSIONS = (".png", ".jpg", ".jpeg")


def rename(folder: str, start: int, dry_run: bool = False) -> int:
    if not os.path.isdir(folder):
        sys.exit(f"Not a directory: {folder}")

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS))
    if not files:
        sys.exit(f"No images found in {folder}")

    print(f"{folder}: {len(files)} files, renumbering from {start:04d}")

    count = 0
    for i, filename in enumerate(files):
        new_name = f"{start + i:04d}.png"
        if new_name == filename:
            count += 1
            continue
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        if os.path.exists(dst):
            print(f"  skipped {filename}: {new_name} already exists")
            continue
        if dry_run:
            print(f"  would rename {filename} -> {new_name}")
        else:
            os.rename(src, dst)
        count += 1

    print(f"Done: {count} files, last index {start + count - 1:04d}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="folder containing the images to renumber")
    parser.add_argument("--start", type=int, default=0, help="first index (default: 0)")
    parser.add_argument("--dry-run", action="store_true", help="print the renames without applying them")
    args = parser.parse_args()
    rename(args.folder, args.start, args.dry_run)


if __name__ == "__main__":
    main()
