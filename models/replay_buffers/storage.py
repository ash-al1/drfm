import json
import os
import shutil

_DEFAULT_BASE_DIR = "outputs/replay_buffers"
_DEFAULT_MAX_SIZE_BYTES = 50 * 1024 ** 3


def scan_snapshots(base_dir=_DEFAULT_BASE_DIR, algorithm=None, task=None):
    snapshots = []
    for alg in (_subdirs(base_dir) if algorithm is None else [algorithm]):
        alg_path = os.path.join(base_dir, alg)
        for tsk in (_subdirs(alg_path) if task is None else [task]):
            meta_path = os.path.join(alg_path, tsk, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for entry in meta.values():
                if os.path.exists(entry["path"]):
                    snapshots.append({**entry, "algorithm": alg, "task": tsk})
    return sorted(snapshots, key=lambda x: x["step"])


def total_size_bytes(snapshots):
    return sum(s["size_bytes"] for s in snapshots)


def rotate(base_dir=_DEFAULT_BASE_DIR, algorithm=None, task=None,
           keep_n=5, max_bytes=None, dry_run=False):
    snapshots = scan_snapshots(base_dir, algorithm, task)
    to_delete = []

    groups = {}
    for s in snapshots:
        groups.setdefault((s["algorithm"], s["task"]), []).append(s)
    for group in groups.values():
        if len(group) > keep_n:
            to_delete.extend(sorted(group, key=lambda x: x["step"])[: len(group) - keep_n])

    if max_bytes is not None:
        delete_ids = {id(s) for s in to_delete}
        remaining = sorted(
            [s for s in snapshots if id(s) not in delete_ids], key=lambda x: x["step"]
        )
        while total_size_bytes(remaining) > max_bytes and remaining:
            to_delete.append(remaining.pop(0))

    deleted = []
    for snap in to_delete:
        if not dry_run:
            _remove(snap, base_dir)
        deleted.append(snap["path"])
    return deleted


def _remove(snap, base_dir):
    if os.path.isdir(snap["path"]):
        shutil.rmtree(snap["path"])
    elif os.path.exists(snap["path"]):
        os.remove(snap["path"])
    meta_path = os.path.join(base_dir, snap["algorithm"], snap["task"], "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta.pop(f"step_{snap['step']:08d}", None)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def _subdirs(path):
    if not os.path.isdir(path):
        return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
