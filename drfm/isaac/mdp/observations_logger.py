# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def post_step_log(env: ManagerBasedRLEnv, env_ids: object = None) -> None:
    """Read the concatenated policy observation buffer and log each named component."""
    obs = env.observation_manager.compute_group("policy")
    if obs is None:
        return

    o = obs[0]
    print(
        f"[obs] target=({o[0]:.2f},{o[1]:.2f},{o[2]:.2f}) "
        f"wp_rem={o[3]:.0f} "
        f"alt={o[8]:.2f} vvel={o[9]:.2f} "
        f"vel=({o[10]:.2f},{o[11]:.2f},{o[12]:.2f}) "
        f"rwr_rx=({o[17]:.2f},{o[25]:.2f},{o[33]:.2f}) "
        f"drfm_tech={o[48:52].tolist()} pow={o[55]:.2f}",
        flush=True,
    )

    log(env, ["target_x", "target_y", "target_z"], obs[:, 0:3])
    log(env, ["waypoints_remaining"], obs[:, 3:4])
    log(env, ["qw", "qx", "qy", "qz"], obs[:, 4:8])
    log(env, ["altitude"], obs[:, 8:9])
    log(env, ["vertical_vel"], obs[:, 9:10])
    log(env, ["vx", "vy", "vz"], obs[:, 10:13])
    log(env, ["wx", "wy", "wz"], obs[:, 13:16])

    for i, name in enumerate(["sacq", "pd", "mono"]):
        base = 16 + i * 8
        log(env, [f"{name}_bearing", f"{name}_rx", f"{name}_illum", f"{name}_piv",
                  f"{name}_freq0", f"{name}_freq1", f"{name}_freq2", f"{name}_trend"],
            obs[:, base : base + 8])

    log(env, ["tech_off", "tech_rgpo", "tech_vgpo", "tech_rvgpo",
              "por_norm", "vpor_norm", "coord", "power"], obs[:, 48:56])
