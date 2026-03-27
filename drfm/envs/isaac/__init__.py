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
