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
    "obstacle_center",
    "obstacle_n1", "obstacle_n2",
    "obstacle_s1", "obstacle_s2",
    "wall_north", "wall_south", "wall_east", "wall_west",
]

# Region: 50 m x 30 m, centered at (25, 0).
# Drones spawn on the left (x=2..6), objective zone on the right (x=44..48).
# Two corridors (north/south) around a center wall create route-choice pressure.
_RADAR_POSITIONS = (
    (15.0,  0.0, 0.0),   # SAcq  -- center, first detection line
    (35.0, -8.0, 0.0),   # PD   -- south, deep coverage
    (38.0,  8.0, 0.0),   # Mono -- north, deep coverage (cued by SAcq)
)

_RADAR_EXCLUSION_RADIUS = 8.0
_RADAR_EXCLUSION_ZONES = tuple(
    (p[0], p[1], _RADAR_EXCLUSION_RADIUS) for p in _RADAR_POSITIONS
)

# 2D obstacle footprints (cx, cy, half_width, half_height).
# half_width  = size_x / 2,  half_height = size_y / 2.
_OBSTACLE_GEOM = (
    (25.0,   0.0, 4.0, 1.5),   # center wall: size=(8,3,6)
    (20.0,   9.0, 2.0, 1.5),   # n1: size=(4,3,5)
    (33.0,  10.0, 1.5, 2.0),   # n2: size=(3,4,5)
    (18.0,  -8.0, 1.5, 2.0),   # s1: size=(3,4,5)
    (32.0,  -6.0, 2.0, 1.5),   # s2: size=(4,3,5)
)


_NUM_ROBOTS = 2   # increment here when adding more drones to the scene


def _reset_contact_sensor(env, env_ids):
    suffixes = [""] + [f"_{i}" for i in range(1, _NUM_ROBOTS)]
    for sfx in suffixes:
        env.scene.sensors[f"collision_sensor{sfx}"].data.net_forces_w[env_ids] = 0.0


def _box(prim: str, size, pos, color) -> RigidObjectCfg:
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


def _drone_slot(index: int) -> tuple[ArticulationCfg, ContactSensorCfg]:
    """Return (ArticulationCfg, ContactSensorCfg) for drone `index`.

    Index 0 is the policy-controlled agent; higher indices are inert wingmen.
    Spawn positions are offset in Y by 3 m per slot so drones never overlap.
    """
    suffix = "" if index == 0 else str(index)
    prim   = f"{{ENV_REGEX_NS}}/Robot{suffix}"
    y_off  = float(index) * 3.0
    articulation = FIVE_IN_DRONE.replace(
        prim_path=prim,
        init_state=FIVE_IN_DRONE.init_state.__class__(pos=(0.0, y_off, 0.5)),
    )
    sensor = ContactSensorCfg(prim_path=f"{prim}/.*", debug_vis=(index == 0))
    return articulation, sensor


def _radar_cone(prim: str, xy, radius: float, height: float, diffuse, emissive) -> RigidObjectCfg:
    x, y = xy
    return RigidObjectCfg(
        prim_path=prim,
        spawn=sim_utils.ConeCfg(
            radius=radius,
            height=height,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse,
                emissive_color=emissive,
                roughness=0.15,
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, height / 2.0)),
    )


@configclass
class DroneReconSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(color=(0.2, 0.35, 0.2)),
    )

    robot,   collision_sensor   = _drone_slot(0)
    robot_1, collision_sensor_1 = _drone_slot(1)

    obstacle_center: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleCenter", (8.0, 3.0, 6.0), (25.0,  0.0, 3.0), (0.50, 0.45, 0.40)
    )
    obstacle_n1: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleN1",     (4.0, 3.0, 5.0), (20.0,  9.0, 2.5), (0.45, 0.50, 0.55)
    )
    obstacle_n2: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleN2",     (3.0, 4.0, 5.0), (33.0, 10.0, 2.5), (0.45, 0.50, 0.55)
    )
    obstacle_s1: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleS1",     (3.0, 4.0, 5.0), (18.0, -8.0, 2.5), (0.55, 0.50, 0.45)
    )
    obstacle_s2: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/ObstacleS2",     (4.0, 3.0, 5.0), (32.0, -6.0, 2.5), (0.55, 0.50, 0.45)
    )

    wall_west:  RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallWest",  ( 1.0, 32.0, 10.0), ( 0.0,   0.0, 5.0), (0.45, 0.45, 0.45),
    )
    wall_east:  RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallEast",  ( 1.0, 32.0, 10.0), (50.0,   0.0, 5.0), (0.45, 0.45, 0.45),
    )
    wall_north: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallNorth", (52.0,  1.0, 10.0), (25.0,  16.0, 5.0), (0.45, 0.45, 0.45),
    )
    wall_south: RigidObjectCfg = _box(
        "{ENV_REGEX_NS}/WallSouth", (52.0,  1.0, 10.0), (25.0, -16.0, 5.0), (0.45, 0.45, 0.45),
    )

    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(color=(1.0, 0.95, 0.85), intensity=2500.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.906, 0.423, 0.0, 0.0)),
    )


@configclass
class DroneReconSceneCfgDRFM(DroneReconSceneCfg):
    radar_marker_search: RigidObjectCfg = _radar_cone(
        "{ENV_REGEX_NS}/RadarMarkerSearch",
        (_RADAR_POSITIONS[0][0], _RADAR_POSITIONS[0][1]),
        radius=1.5, height=0.5,
        diffuse=(0.05, 0.05, 0.9),
        emissive=(0.1, 0.1, 3.0),
    )

    radar_marker_pd: RigidObjectCfg = _radar_cone(
        "{ENV_REGEX_NS}/RadarMarkerPD",
        (_RADAR_POSITIONS[1][0], _RADAR_POSITIONS[1][1]),
        radius=1.5, height=0.5,
        diffuse=(0.8, 0.02, 0.02),
        emissive=(3.0, 0.05, 0.05),
    )

    radar_marker_mono: RigidObjectCfg = _radar_cone(
        "{ENV_REGEX_NS}/RadarMarkerMono",
        (_RADAR_POSITIONS[2][0], _RADAR_POSITIONS[2][1]),
        radius=1.5, height=0.5,
        diffuse=(1.0, 0.45, 0.0),
        emissive=(3.0, 1.2, 0.0),
    )


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (2.0, 6.0), "y": (-10.0, 10.0), "z": (1.5, 1.5),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_contact = EventTerm(func=_reset_contact_sensor, mode="reset")
    # Uncomment to log observation components each step (adds overhead; use for debugging only):
    # log_observations = EventTerm(func=mdp.post_step_log, mode="post_step")


@configclass
class CommandsCfg:
    target = mdp.WaypointCommandCfg(
        asset_name="robot",
        goal_x_range=(44.0, 48.0),
        goal_y_range=(-10.0, 10.0),
        goal_z_range=(1.0, 3.0),
        waypoints_per_episode=3,
        arrival_threshold=1.0,
        obstacle_margin=2.0,
        exclusion_zones=_RADAR_EXCLUSION_ZONES,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)
    drfm_action:    mdp.DrfmActionCfg    = mdp.DrfmActionCfg(
        radar_positions=_RADAR_POSITIONS,
        obstacle_geom=_OBSTACLE_GEOM,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        target_pos_b        = ObsTerm(func=mdp.target_pos_b,        params={"command_name": "target"})
        waypoints_remaining = ObsTerm(func=mdp.waypoints_remaining,  params={"command_name": "target"})
        attitude            = ObsTerm(func=mdp.root_quat_w)
        lin_vel             = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel             = ObsTerm(func=mdp.root_ang_vel_b)
        rwr        = ObsTerm(func=mdp.rwr_observations)
        drfm_state = ObsTerm(func=mdp.drfm_state_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    progress         = RewTerm(func=mdp.progress,         weight=20.0,   params={"command_name": "target"})
    forward_speed    = RewTerm(func=mdp.forward_speed,     weight=8.0,    params={"command_name": "target", "target_speed": 4.0})
    heading          = RewTerm(func=mdp.heading_to_goal,   weight=4.0,    params={"command_name": "target"})
    arrived          = RewTerm(func=mdp.arrived,           weight=50.0,  params={"command_name": "target", "threshold": 1.0})
    completion_bonus = RewTerm(func=mdp.completion_bonus,  weight=100.0, params={"command_name": "target"})
    distance_penalty = RewTerm(func=mdp.distance_to_goal,  weight=-0.5,   params={"command_name": "target"})
    terminating      = RewTerm(func=mdp.is_terminated,     weight=-500.0)
    step_penalty     = RewTerm(func=mdp.step_penalty,      weight=-0.01)
    ang_vel_l2       = RewTerm(func=mdp.ang_vel_l2,        weight=-0.01)
    proximity        = RewTerm(func=mdp.proximity_penalty, weight=-10.0,
                               params={"obstacle_names": _OBSTACLE_NAMES, "safe_dist": 2.5, "max_dist": 6.0})
    illumination_low  = RewTerm(func=mdp.illumination_penalty, weight=-5.0)
    power_conserve    = RewTerm(func=mdp.power_conserve,       weight=0.5)


@configclass
class TerminationsCfg:
    time_out      = DoneTerm(func=mdp.time_out,           time_out=True)
    all_waypoints = DoneTerm(func=mdp.all_waypoints_done, params={"command_name": "target"})
    collision     = DoneTerm(func=mdp.illegal_contact,    params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01})
    flyaway       = DoneTerm(func=mdp.flyaway,            params={"command_name": "target", "distance": 50.0})
    too_high      = DoneTerm(func=mdp.too_high,           params={"max_z": 4.0})
    radar_lock    = DoneTerm(func=mdp.radar_lock)


@configclass
class DroneReconEnvCfg(ManagerBasedRLEnvCfg):
    scene:        DroneReconSceneCfgDRFM = DroneReconSceneCfgDRFM(num_envs=4096, env_spacing=70.0)
    observations: ObservationsCfg        = ObservationsCfg()
    actions:      ActionsCfg             = ActionsCfg()
    commands:     CommandsCfg            = CommandsCfg()
    events:       EventCfg               = EventCfg()
    rewards:      RewardsCfg             = RewardsCfg()
    terminations: TerminationsCfg        = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation       = 8
        self.episode_length_s = 25.0
        self.viewer.eye       = (-5.0, 0.0, 20.0)
        self.viewer.lookat    = (25.0, 0.0,  1.0)
        self.sim.dt              = 1 / 400
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1


@configclass
class DroneReconEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    scene:        DroneReconSceneCfgDRFM = DroneReconSceneCfgDRFM(num_envs=1, env_spacing=70.0)
    observations: ObservationsCfg        = ObservationsCfg()
    actions:      ActionsCfg             = ActionsCfg()
    commands:     CommandsCfg            = CommandsCfg()
    events:       EventCfg               = EventCfg()
    rewards:      RewardsCfg             = RewardsCfg()
    terminations: TerminationsCfg        = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation          = 8
        self.episode_length_s    = 25.0
        self.sim.dt              = 1 / 400
        self.viewer.origin_type  = "world"
        self.viewer.eye          = (-5.0, -30.0, 25.0)
        self.viewer.lookat       = (25.0,   0.0,  1.0)
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_velocity_iteration_count = 1
