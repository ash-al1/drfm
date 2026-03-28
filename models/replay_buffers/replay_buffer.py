import json
import os
from datetime import datetime

from skrl.memories.torch import RandomMemory

_DEFAULT_BASE_DIR = "outputs/replay_buffers"


class ReplayBuffer(RandomMemory):
    def __init__(self, capacity, num_envs, device, **kwargs):
        super().__init__(memory_size=capacity, num_envs=num_envs, device=device, **kwargs)

    def snapshot_size_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def save_snapshot(self, step, algorithm, task, base_dir=_DEFAULT_BASE_DIR):
        dir_path = os.path.join(base_dir, algorithm.lower(), task)
        os.makedirs(dir_path, exist_ok=True)
        buf_path = os.path.join(dir_path, f"step_{step:08d}.pt")
        self.save(buf_path)
        size_bytes = sum(
            os.path.getsize(os.path.join(buf_path, f))
            for f in os.listdir(buf_path)
            if os.path.isfile(os.path.join(buf_path, f))
        )
        meta_path = os.path.join(dir_path, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        meta[f"step_{step:08d}"] = {
            "step": step,
            "saved_at": datetime.utcnow().isoformat(),
            "size_bytes": size_bytes,
            "path": buf_path,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return buf_path

    def load_snapshot(self, path):
        self.load(path)
