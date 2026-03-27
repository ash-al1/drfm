# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import os
from datetime import datetime

import torch

from drfm.utils.plotter import generate_plots


class CSVLogger:
    def __init__(self, folder_path="."):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(folder_path, f"log_{timestamp}.csv")
        self.keys = []
        self.file_initialized = False

    def log(self, data_dict):
        for key, tensor in data_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Value for key '{key}' must be a tensor.")
            if tensor.ndim != 1 or tensor.shape[0] != 1:
                raise ValueError(f"Tensor for key '{key}' must have shape (1,), but got {tensor.shape}.")

        row = {key: tensor.item() for key, tensor in data_dict.items()}

        if not self.file_initialized:
            self.keys = list(row.keys())
            with open(self.file_path, mode="w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.keys).writeheader()
            self.file_initialized = True

        new_keys = [k for k in row if k not in self.keys]
        if new_keys:
            self.keys.extend(new_keys)
            with open(self.file_path) as f:
                rows = list(csv.DictReader(f))
            with open(self.file_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.keys)
                writer.writeheader()
                writer.writerows(rows)

        with open(self.file_path, mode="a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.keys).writerow({k: row.get(k, "") for k in self.keys})

    def save(self):
        if not self.file_initialized:
            raise RuntimeError("No data has been logged yet.")
        generate_plots(self.file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.file_path = os.path.join(os.path.dirname(self.file_path), f"log_{timestamp}.csv")
        self.keys = []
        self.file_initialized = False


def log(env, keys, value):
    if "metrics" not in env.extras:
        env.extras["metrics"] = {}
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise TypeError("keys must be a list of strings.")
    if len(keys) != value.shape[1]:
        raise ValueError(f"Length of keys ({len(keys)}) must match value dim ({value.shape[1]}).")
    for i, key in enumerate(keys):
        env.extras["metrics"][key] = value[:, i]
