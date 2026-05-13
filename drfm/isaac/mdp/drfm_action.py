from __future__ import annotations
from typing import TYPE_CHECKING

import logging

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .radar import LOCK, RADAR_NAMES, TECHNIQUE_NAMES, RadarManager
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Action slice: 7D  (receives [:, 4:11] of the full 11D tensor)
#   [0:4]  technique logits → argmax → {0=OFF, 1=RGPO, 2=VGPO, 3=RVGPO}
#   [4]    pull_off_rate       norm [-1,1] → [50, 500] m/s
#   [5]    vel_pull_off_rate   norm [-1,1] → [10, 200] m/s²
#   [6]    coordination_ratio  norm [-1,1] → [0, 1]

logger = logging.getLogger(__name__)

POR_MIN  =  50.0;  POR_MAX  = 500.0
VPR_MIN  =  10.0;  VPR_MAX  = 200.0

POWER_COST = [0.0, 0.001, 0.001, 0.003]  # per control step; RVGPO depletes ~step 333


def _denorm(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (x.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


class DrfmAction(ActionTerm):
    cfg: "DrfmActionCfg"

    def __init__(self, cfg: "DrfmActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        self.radar_manager = RadarManager(self.num_envs, self.device, cfg.radar_positions, cfg.obstacle_geom)
        self._technique      = torch.zeros(self.num_envs, dtype=torch.long,    device=self.device)
        self._prev_technique = torch.zeros(self.num_envs, dtype=torch.long,    device=self.device)
        self._pull_off_rate      = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._vel_pull_off_rate  = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._coord          = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._power          = torch.ones (self.num_envs, dtype=torch.float32, device=self.device)
        self._power_cost     = torch.tensor(POWER_COST, device=self.device)
        self._step           = 0
        self._prev_esm       = torch.zeros(3, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    def process_actions(self, actions: torch.Tensor) -> None:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        self._prev_technique[:] = self._technique

        technique = actions[:, :4].argmax(dim=1).long()
        technique = torch.where(self._power <= 0.0, torch.zeros_like(technique), technique)
        self._technique[:] = technique

        self._pull_off_rate     = _denorm(actions[:, 4], POR_MIN,  POR_MAX)
        self._vel_pull_off_rate = _denorm(actions[:, 5], VPR_MIN,  VPR_MAX)
        self._coord = _denorm(actions[:, 6], 0.0, 1.0)

        switched  = (self._technique != self._prev_technique)
        robot     = self._env.scene["robot"]
        pos_local = robot.data.root_pos_w - self._env.scene.env_origins
        vel_w     = robot.data.root_lin_vel_w

        self.radar_manager.update(
            drone_pos          = pos_local,
            drone_vel          = vel_w,
            technique          = self._technique,
            pull_off_rate      = self._pull_off_rate,
            vel_pull_off_rate  = self._vel_pull_off_rate,
            coordination_ratio = self._coord,
            technique_switched = switched,
            dt                 = float(self._env.step_dt),
        )

        self._power = (self._power - self._power_cost[self._technique]).clamp(min=0.0)
        self._step += 1
        self._debug_print(switched)
        self._log_metrics()

    def apply_actions(self) -> None:
        pass  # DRFM is pure computation; nothing to write to physics

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._technique[env_ids]      = 0
        self._prev_technique[env_ids] = 0
        self._pull_off_rate[env_ids]     = 0.0
        self._vel_pull_off_rate[env_ids] = 0.0
        self._coord[env_ids]          = 0.0
        self._power[env_ids]          = 1.0
        if (env_ids == 0).any():
            self._prev_esm[:] = False
        self.radar_manager.reset(env_ids)

    def get_state_obs(self) -> torch.Tensor:
        """[N, 8]: technique_onehot(4), pull_off_rate/500, vel_pull_off_rate/200, coord, power"""
        N   = self.num_envs
        obs = torch.zeros(N, 8, device=self.device)
        tech_oh = torch.zeros(N, 4, device=self.device)
        tech_oh.scatter_(1, self._technique.unsqueeze(1), 1.0)
        obs[:, :4] = tech_oh
        obs[:, 4]  = self._pull_off_rate     / POR_MAX
        obs[:, 5]  = self._vel_pull_off_rate / VPR_MAX
        obs[:, 6]  = self._coord
        obs[:, 7]  = self._power
        return obs

    def _debug_print(self, switched: torch.Tensor) -> None:
        env0_tech = int(self._technique[0].item())
        rm        = self.radar_manager
        pwr       = self._power[0].item()
        tech_name = TECHNIQUE_NAMES[env0_tech]

        # Technique switch (event-driven)
        if switched[0].item():
            old = TECHNIQUE_NAMES[int(self._prev_technique[0].item())]
            if old != tech_name:
                logger.debug("switch  s=%5d  %s -> %s  pwr=%.0f%%", self._step, old, tech_name, pwr * 100)

        # ESM cue (event-driven)
        esm_now = rm.esm_triggered[0]
        new_esm = esm_now & ~self._prev_esm
        if new_esm.any():
            cued = [RADAR_NAMES[i] for i in range(3) if new_esm[i].item()]
            logger.debug("esm     s=%5d  cued: %s", self._step, ", ".join(cued))
        self._prev_esm = esm_now.clone()

        # Radar lock (event-driven)
        if rm.any_locked[0].item():
            for i in range(3):
                if int(rm.state[0, i].item()) == LOCK:
                    logger.debug(
                        "lock    s=%5d  %s  track_quality=%.2f  technique=%s",
                        self._step, RADAR_NAMES[i], rm.track_quality[0, i].item(), tech_name,
                    )

        # Periodic radar/drfm state is tracked via TensorBoard through _log_metrics().

    def _log_metrics(self) -> None:
        rm = self.radar_manager
        log(self._env, ["radar_0_state"], rm.state[:, 0:1].float())
        log(self._env, ["radar_1_state"], rm.state[:, 1:2].float())
        log(self._env, ["radar_2_state"], rm.state[:, 2:3].float())
        log(self._env, ["radar_0_tq"],    rm.track_quality[:, 0:1])
        log(self._env, ["radar_1_tq"],    rm.track_quality[:, 1:2])
        log(self._env, ["radar_2_tq"],    rm.track_quality[:, 2:3])
        log(self._env, ["any_lock"],           rm.any_locked.float().unsqueeze(1))
        log(self._env, ["drfm_technique"],     self._technique.float().unsqueeze(1))
        log(self._env, ["drfm_power"],         self._power.unsqueeze(1))
        log(self._env, ["illumination_penalty"],rm.track_quality.max(dim=1).values.unsqueeze(1))


@configclass
class DrfmActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DrfmAction
    asset_name: str = "robot"
    radar_positions: tuple = (
        ( 7.0,  1.0, 0.0),
        (25.0, -9.0, 0.0),
        (26.0,  9.0, 0.0),
    )
    obstacle_geom: tuple = ()   # (cx, cy, hw, hh) per obstacle; set from env config
