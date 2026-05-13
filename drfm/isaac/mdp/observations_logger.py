# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def post_step_log(env: ManagerBasedRLEnv) -> None:
    """Read the concatenated policy observation buffer and log each named component."""
    # The policy group concatenates terms in declaration order:
    #   target_pos_b [3] | waypoints_remaining [1] | attitude [4] |
    #   lin_vel [3] | ang_vel [3] | rwr [30] | drfm_state [8]
    obs = env.observation_manager.compute_group("policy")  # [N, D]
    if obs is None:
        return

    log(env, ["target_x", "target_y", "target_z"], obs[:, 0:3])
    log(env, ["waypoints_remaining"], obs[:, 3:4])
    log(env, ["qw", "qx", "qy", "qz"], obs[:, 4:8])
    log(env, ["vx", "vy", "vz"], obs[:, 8:11])
    log(env, ["wx", "wy", "wz"], obs[:, 11:14])

    # RWR block: 3 radars × 10 dims each (indices 14–43)
    for i, name in enumerate(["sacq", "pd", "mono"]):
        base = 14 + i * 10
        log(env, [f"{name}_rng", f"{name}_az", f"{name}_el",
                  f"{name}_rng_rate", f"{name}_snr",
                  f"{name}_s0", f"{name}_s1", f"{name}_s2", f"{name}_s3",
                  f"{name}_tq"], obs[:, base : base + 10])

    # DRFM state: technique one-hot [4] + por/500 + vpor/200 + coord + power (indices 44–51)
    log(env, ["tech_off", "tech_rgpo", "tech_vgpo", "tech_rvgpo",
              "por_norm", "vpor_norm", "coord", "power"], obs[:, 44:52])
