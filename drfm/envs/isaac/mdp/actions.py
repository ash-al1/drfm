# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import Allocation, Motor
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    """Applies thrust and torque wrench to the drone body from [-1,1] action commands."""

    cfg: ControlActionCfg

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._allocation = Allocation(
            num_envs=self.num_envs,
            arm_length=self.cfg.arm_length,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    def process_actions(self, actions: torch.Tensor):

        self._raw_actions[:] = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        clamped = self._raw_actions.clamp_(-1.0, 1.0)
        mapped = (clamped + 1.0) / 2.0
        omega_ref = self.cfg.omega_max * mapped
        omega_real = self._motor.compute(omega_ref)
        self._processed_actions = self._allocation.compute(omega_real)

        log(self._env, ["a1", "a2", "a3", "a4"], self._raw_actions)
        log(self._env, ["w1", "w2", "w3", "w4"], omega_real)

    def apply_actions(self):
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        self._robot.permanent_wrench_composer.set_forces_and_torques(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._motor.reset(env_ids)
        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = ControlAction

    asset_name: str = "robot"
    arm_length: float = 0.035
    drag_coef: float = 1.5e-9
    thrust_coef: float = 2.25e-7
    omega_max: float = 5145.0
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    init: list[float] = (2572.5, 2572.5, 2572.5, 2572.5)
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    use_motor_model: bool = False
