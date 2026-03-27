#
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
                radius=0.375,
                height=0.0625,
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
