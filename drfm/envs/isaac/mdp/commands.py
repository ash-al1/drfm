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

_OBSTACLES = [
    (torch.tensor([6.0, 8.0, 2.5]), torch.tensor([1.0, 1.0, 2.5])),
    (torch.tensor([13.5, -8.0, 2.0]), torch.tensor([1.0, 0.5, 2.0])),
    (torch.tensor([22.0, 8.0, 3.5]), torch.tensor([1.5, 1.5, 3.5])),
    (torch.tensor([27.0, -5.0, 1.5]), torch.tensor([0.5, 2.0, 1.5])),
]


class WaypointCommand(CommandTerm):
    cfg: WaypointCommandCfg

    def __init__(self, cfg: WaypointCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 7, device=self.device)
        self._command[:, 3] = 1.0
        self._previous_pos = self.robot.data.root_pos_w.clone()
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoints_visited = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._waypoints = torch.zeros(
            self.num_envs, cfg.waypoints_per_episode, 3, device=self.device
        )
        self._all_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        msg = "WaypointCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tWaypoints per episode: {self.cfg.waypoints_per_episode}\n"
        msg += f"\tArrival threshold: {self.cfg.arrival_threshold}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def previous_pos(self) -> torch.Tensor:
        return self._previous_pos

    @property
    def waypoints_remaining(self) -> torch.Tensor:
        return (self.cfg.waypoints_per_episode - self._waypoints_visited).clamp(min=0).float()

    @property
    def all_done(self) -> torch.Tensor:
        return self._all_done

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        total = self.cfg.waypoints_per_episode
        wp = torch.zeros(n, total, 3, device=self.device)
        for i in range(total):
            wp[:, i, :] = self._sample_valid_positions(n)
            wp[:, i, :] += self._env.scene.env_origins[env_ids]
        self._waypoints[env_ids] = wp
        self._waypoint_idx[env_ids] = 0
        self._waypoints_visited[env_ids] = 0
        self._all_done[env_ids] = False
        self._command[env_ids, :3] = wp[:, 0, :]
        self._command[env_ids, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

    def _sample_valid_positions(self, n: int) -> torch.Tensor:
        ranges = [self.cfg.goal_x_range, self.cfg.goal_y_range, self.cfg.goal_z_range]
        pos = torch.stack(
            [torch.empty(n, device=self.device).uniform_(*r) for r in ranges], dim=1
        )
        valid = self._is_clear(pos)
        for _ in range(10):
            if valid.all():
                break
            count = (~valid).sum().item()
            new_pos = torch.stack(
                [torch.empty(count, device=self.device).uniform_(*r) for r in ranges], dim=1
            )
            pos[~valid] = new_pos
            valid = self._is_clear(pos)
        return pos

    def _is_clear(self, pos: torch.Tensor) -> torch.Tensor:
        margin = self.cfg.obstacle_margin
        clear = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        for center, half in _OBSTACLES:
            c = center.to(pos.device)
            h = (half + margin).to(pos.device)
            clear &= ~(torch.abs(pos - c) < h).all(dim=1)
        return clear

    def _update_command(self):
        if self.cfg.waypoints_per_episode <= 0:
            self._previous_pos = self.robot.data.root_pos_w.clone()
            return

        idx = torch.arange(self.num_envs, device=self.device)
        current_goals = self._waypoints[idx, self._waypoint_idx]
        dist = torch.norm(self.robot.data.root_pos_w - current_goals, dim=1)
        arrived_mask = (dist < self.cfg.arrival_threshold) & ~self._all_done

        if arrived_mask.any():
            self._waypoint_idx[arrived_mask] += 1
            self._waypoints_visited[arrived_mask] += 1

            newly_done = self._waypoint_idx >= self.cfg.waypoints_per_episode
            self._all_done |= newly_done

            advanced = arrived_mask & ~newly_done
            if advanced.any():
                next_goals = self._waypoints[idx, self._waypoint_idx]
                self._command[advanced, :3] = next_goals[advanced]
                self._command[advanced, 3:] = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0], device=self.device
                )

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
    waypoints_per_episode: int = 5
    arrival_threshold: float = 2.5
    obstacle_margin: float = 2.0

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
