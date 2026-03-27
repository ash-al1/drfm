#
#

"""
File:   drone_recon_env_cfg.py
Use:    Phase 1 navigation with terrain, boundary walls, and altitude cap.
        No radar, no DRFM. Pure navigation with PPO.
Update: Thu, 26 Mar 2026
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

from assets.five_in_drone import FIVE_IN_DRONE  # isort:skip

_OBSTACLE_NAMES = [
    "obstacle_box1", "obstacle_box2", "obstacle_box3", "obstacle_box4",
    "wall_north", "wall_south", "wall_east", "wall_west",
]


def _reset_contact_sensor(env, env_ids):
    contact_sensor = env.scene.sensors["collision_sensor"]
    contact_sensor.data.net_forces_w[env_ids] = 0.0


def _box(prim: str, size, pos, color) -> RigidObjectCfg:
    """Helper: kinematic cuboid with collision."""
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


@configclass
class DroneReconSceneCfg(InteractiveSceneCfg):
    """Flat terrain, 4 box obstacles, 4 boundary walls, directional sun."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.2, 0.35, 0.2)),
    )

    robot: ArticulationCfg = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    collision_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=True
    )

    # --- Interior obstacles ---
    obstacle_box1: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleBox1", (2.0, 2.0, 5.0), (6.0,   8.0, 2.5), (0.6, 0.4, 0.2)
    )
    obstacle_box2: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleBox2", (2.0, 1.0, 4.0), (13.5, -8.0, 2.0), (0.5, 0.5, 0.65)
    )
    obstacle_box3: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleBox3", (3.0, 3.0, 7.0), (22.0,  8.0, 3.5), (0.55, 0.5, 0.35)
    )
    obstacle_box4: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleBox4", (1.0, 4.0, 3.0), (27.0, -5.0, 1.5), (0.4, 0.55, 0.4)
    )

    # --- Boundary walls (grey) ---
    wall_north: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallNorth",
        (1.0, 26.0, 10.0),
        (30.5, 0.0, 5.0),
        (0.45, 0.45, 0.45),
    )
    wall_south: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallSouth",
        (1.0, 26.0, 10.0),
        (-3.5, 0.0, 5.0),
        (0.45, 0.45, 0.45),
    )
    wall_east: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallEast",
        (35.0, 1.0, 10.0),
        (13.5, 13.5, 5.0),
        (0.45, 0.45, 0.45),
    )
    wall_west: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallWest",
        (35.0, 1.0, 10.0),
        (13.5, -13.5, 5.0),
        (0.45, 0.45, 0.45),
    )

    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.85), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.906, 0.423, 0.0, 0.0)),
    )


@configclass
class ActionsCfg:
    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)


@configclass
class ObservationsCfg:
    """14D: goal in body frame + remaining waypoints + quaternion + linear vel + angular vel."""

    @configclass
    class PolicyCfg(ObsGroup):
        target_pos_b      = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        waypoints_remaining = ObsTerm(func=mdp.waypoints_remaining, params={"command_name": "target"})
        attitude           = ObsTerm(func=mdp.root_quat_w)
        lin_vel            = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel            = ObsTerm(func=mdp.root_ang_vel_b)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    target = mdp.WaypointCommandCfg(
        asset_name="robot",
        goal_x_range=(5.0, 28.0),
        goal_y_range=(-11.0, 11.0),
        goal_z_range=(1.0, 3.5),
        waypoints_per_episode=5,
        arrival_threshold=2.5,
        obstacle_margin=2.0,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.5, 0.5),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_contact = EventTerm(func=_reset_contact_sensor, mode="reset")


@configclass
class RewardsCfg:
    progress         = RewTerm(func=mdp.progress,          weight=40.0,   params={"command_name": "target"})
    heading          = RewTerm(func=mdp.heading_to_goal,    weight=2.0,    params={"command_name": "target"})
    arrived          = RewTerm(func=mdp.arrived,            weight=500.0,  params={"command_name": "target", "threshold": 2.5})
    completion_bonus = RewTerm(func=mdp.completion_bonus,   weight=2000.0, params={"command_name": "target"})
    distance_penalty = RewTerm(func=mdp.distance_to_goal,   weight=-0.3,   params={"command_name": "target"})
    step_penalty     = RewTerm(func=mdp.step_penalty,       weight=-0.05)
    ang_vel_l2       = RewTerm(func=mdp.ang_vel_l2,         weight=-0.001)
    proximity        = RewTerm(
        func=mdp.proximity_penalty,
        weight=-3.0,
        params={"obstacle_names": _OBSTACLE_NAMES, "safe_dist": 2.5, "max_dist": 6.0},
    )
    terminating      = RewTerm(func=mdp.is_terminated,      weight=-500.0)


@configclass
class TerminationsCfg:
    time_out         = DoneTerm(func=mdp.time_out,          time_out=True)
    all_waypoints    = DoneTerm(func=mdp.all_waypoints_done, params={"command_name": "target"})
    collision        = DoneTerm(func=mdp.illegal_contact,    params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01})
    flyaway          = DoneTerm(func=mdp.flyaway,            params={"command_name": "target", "distance": 50.0})
    too_high         = DoneTerm(func=mdp.too_high,           params={"max_z": 4.0})


@configclass
class DroneReconEnvCfg(ManagerBasedRLEnvCfg):
    scene: DroneReconSceneCfg = DroneReconSceneCfg(num_envs=4096, env_spacing=60.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30        # 5 waypoints × ~6s each
        self.viewer.eye = (-4.0, 0.0, 1.0)
        self.viewer.lookat = (10.0, 0.0, 2.0)
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneReconEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    scene: DroneReconSceneCfg = DroneReconSceneCfg(num_envs=1, env_spacing=60.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30
        self.sim.dt = 1 / 400
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        # Chase-cam: 3 m behind (world -x), 2 m above; look 1 m ahead of the drone.
        # eye/lookat are offsets from the drone's world position.
        self.viewer.eye = (-3.0, 0.0, 2.0)
        self.viewer.lookat = (1.0, 0.0, 0.0)
        self.sim.render_interval = self.decimation
