#
#

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Drone-Recon-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_recon_env_cfg:DroneReconEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Recon-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_recon_env_cfg:DroneReconEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

# ---------------------------------------------------------------------------
# Phase 2: Navigation + Radar + DRFM jamming
# Action: 11D (4 flight + 7 DRFM)   Obs: 62D (14 nav + 40 RWR + 8 DRFM)
# ---------------------------------------------------------------------------
gym.register(
    id="Isaac-Drone-Recon-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_recon_env_cfg_p2:DroneReconEnvCfgP2",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Drone-Recon-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.drone_recon_env_cfg_p2:DroneReconEnvCfgP2_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_cfg.yaml",
    },
)
