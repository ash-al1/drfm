# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import cv2
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import TiledCamera
from isaaclab.utils import configclass

from .events import reset_after_prev_gate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GateTargetingCommand(CommandTerm):
    cfg: GateTargetingCommandCfg

    def __init__(self, cfg: GateTargetingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.cfg = cfg

        if self.cfg.record_fpv:
            self.video_id = 0
            self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera")
            self.sensor: TiledCamera = self._env.scene.sensors[self.sensor_cfg.name]

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.gate_size = cfg.gate_size
        self.num_gates = self.track.num_objects

        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_robot_pos_w = self.robot.data.root_pos_w
        self._gate_missed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.next_gate_w = torch.zeros(self.num_envs, 7, device=self.device)

    def __str__(self) -> str:
        msg = "GateTargetingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7): [x, y, z, qw, qx, qy, qz]."""
        return self.next_gate_w

    @property
    def gate_missed(self) -> torch.Tensor:
        return self._gate_missed

    @property
    def gate_passed(self) -> torch.Tensor:
        return self._gate_passed

    @property
    def previous_pos(self) -> torch.Tensor:
        return self.prev_robot_pos_w

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        if hasattr(self, "out") and self.cfg.record_fpv:
            self.out.release()
            print(f"FPV video saved as fpv_{self.video_id}.mp4")
            self.video_id += 1

        if self.cfg.record_fpv:
            self.out = cv2.VideoWriter(f"fpv_{self.video_id}.mp4", self.fourcc, 100, (1000, 1000))

        if self.cfg.randomise_start is None:
            self.next_gate_idx[env_ids] = 0

        else:
            if self.cfg.randomise_start:
                self.next_gate_idx[env_ids] = torch.randint(
                    low=0, high=self.num_gates, size=(len(env_ids),), device=self.device, dtype=torch.int32
                )
            else:
                self.next_gate_idx[env_ids] = 1

            gate_indices = self.next_gate_idx - 1
            gate_positions = self.track.data.object_com_pos_w[self.env_ids, gate_indices]
            gate_orientations = self.track.data.object_quat_w[self.env_ids, gate_indices]
            gate_w = torch.cat([gate_positions, gate_orientations], dim=1)

            reset_after_prev_gate(
                env=self._env,
                env_ids=env_ids,
                gate_pose=gate_w,
                pose_range={
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-torch.pi / 4, torch.pi / 4),
                    "pitch": (-torch.pi / 4, torch.pi / 4),
                    "yaw": (-torch.pi / 4, torch.pi / 4),
                },
                velocity_range={
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                asset_cfg_name=self.cfg.asset_name,
            )

    def _update_command(self):
        if self.cfg.record_fpv:
            image = self.sensor.data.output["rgb"][0].cpu().numpy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.out.write(image)

        next_gate_positions = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
        next_gate_orientations = self.track.data.object_quat_w[self.env_ids, self.next_gate_idx]
        self.next_gate_w = torch.cat([next_gate_positions, next_gate_orientations], dim=1)

        (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(self.next_gate_w[:, 3:7])
        normal = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        pos_old_projected = (self.prev_robot_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.prev_robot_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (self.robot.data.root_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.robot.data.root_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)

        self._gate_passed = passed_gate_plane & (
            torch.all(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) < (self.gate_size / 2), dim=1)
        )

        self._gate_missed = passed_gate_plane & (
            torch.any(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) > (self.gate_size / 2), dim=1)
        )

        self.next_gate_idx[self._gate_passed] += 1
        self.next_gate_idx = self.next_gate_idx % self.num_gates

        self.prev_robot_pos_w = self.robot.data.root_pos_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # update the markers
        self.target_visualizer.visualize(self.next_gate_w[:, :3], self.next_gate_w[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)


class WaypointCommand(CommandTerm):
    cfg: "WaypointCommandCfg"

    def __init__(self, cfg: "WaypointCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 7, device=self.device)
        self._command[:, 3] = 1.0
        self._previous_pos = self.robot.data.root_pos_w.clone()

    def __str__(self) -> str:
        msg = "WaypointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def previous_pos(self) -> torch.Tensor:
        return self._previous_pos

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        rx = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_x_range)
        ry = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_y_range)
        rz = torch.empty(n, device=self.device).uniform_(*self.cfg.goal_z_range)
        self._command[env_ids, :3] = torch.stack([rx, ry, rz], dim=1) + self._env.scene.env_origins[env_ids]

    def _update_command(self):
        self._previous_pos = self.robot.data.root_pos_w.clone()

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.target_visualizer.visualize(self._command[:, :3], self._command[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)


@configclass
class WaypointCommandCfg(CommandTermCfg):
    class_type: type = WaypointCommand

    asset_name: str = MISSING
    goal_x_range: tuple = (15.0, 25.0)
    goal_y_range: tuple = (-10.0, 10.0)
    goal_z_range: tuple = (1.0, 5.0)

    target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ReconCommand/goal_ring",
        markers={
            "ring": sim_utils.CylinderCfg(
                radius=1.5,
                height=0.25,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.8, 0.0),
                    emissive_color=(0.6, 0.45, 0.0),
                    opacity=0.85,
                ),
            )
        },
    )
    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/ReconCommand/body_pose"
    )
    drone_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)


@configclass
class GateTargetingCommandCfg(CommandTermCfg):
    class_type: type = GateTargetingCommand

    asset_name: str = MISSING
    track_name: str = MISSING
    randomise_start: bool | None = None
    record_fpv: bool = False
    gate_size: float = 1.5

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")

    target_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
    drone_visualizer_cfg.markers["frame"].scale = (0.0001, 0.0001, 0.0001)
