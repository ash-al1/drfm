#

import torch


def log(env, keys, value):
    if "metrics" not in env.extras:
        env.extras["metrics"] = {}
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise TypeError("keys must be a list of strings.")
    if len(keys) != value.shape[1]:
        raise ValueError(f"Length of keys ({len(keys)}) must match value dim ({value.shape[1]}).")
    for i, key in enumerate(keys):
        env.extras["metrics"][key] = value[:, i]
