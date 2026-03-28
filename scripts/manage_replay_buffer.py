#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.replay_buffers.storage import (
    rotate,
    scan_snapshots,
    total_size_bytes,
)

_BASE_DIR = "outputs/replay_buffers"
_MAX_GB = 50.0
_KEEP_N = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=_BASE_DIR)
    p.add_argument("--algorithm", default=None)
    p.add_argument("--task", default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--rotate", action="store_true")
    p.add_argument("--keep_n", type=int, default=_KEEP_N)
    p.add_argument("--max_size_gb", type=float, default=_MAX_GB)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    max_bytes = int(args.max_size_gb * 1024 ** 3)
    snapshots = scan_snapshots(args.base_dir, args.algorithm, args.task)

    if args.list or not args.rotate:
        if not snapshots:
            print("No snapshots found.")
            return
        print(f"{'Algorithm':<12} {'Task':<34} {'Step':>10} {'MB':>8}  Saved at")
        print("-" * 82)
        for s in snapshots:
            print(
                f"{s['algorithm']:<12} {s['task']:<34} {s['step']:>10,} "
                f"{s['size_bytes'] / 1024**2:>8.1f}  {s['saved_at'][:19]}"
            )
        print(f"\n{len(snapshots)} snapshot(s)  {total_size_bytes(snapshots) / 1024**3:.2f} GB")
        return

    if args.rotate:
        deleted = rotate(
            args.base_dir,
            algorithm=args.algorithm,
            task=args.task,
            keep_n=args.keep_n,
            max_bytes=max_bytes,
            dry_run=args.dry_run,
        )
        if not deleted:
            print("Nothing to delete.")
        else:
            verb = "Would delete" if args.dry_run else "Deleted"
            for path in deleted:
                print(f"  {verb}: {path}")
            print(f"  {len(deleted)} snapshot(s) removed.")

if __name__ == "__main__":
    main()
