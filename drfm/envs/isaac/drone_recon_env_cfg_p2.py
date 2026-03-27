import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .drone_recon_env_cfg import _OBSTACLE_NAMES, DroneReconSceneCfg, EventCfg

# Radar positions (env-local, metres).
# Navigable region: x∈[5,28]  y∈[-11,11]
# Tune here without touching anything else.
_RADAR_POSITIONS = (
    ( 7.0,  1.0, 0.0),   # Search/Acq    - middle-left
    (25.0, -9.0, 0.0),   # Pulse-Doppler - bottom-right
    (26.0,  9.0, 0.0),   # Monopulse     - top-right 
)

# Waypoints must not spawn within this radius (m) of any radar - prevents the
# drone from having to push through a radar's immediate footprint to reach a goal.
_RADAR_EXCLUSION_RADIUS = 8.0
_RADAR_EXCLUSION_ZONES = tuple(
    (p[0], p[1], _RADAR_EXCLUSION_RADIUS) for p in _RADAR_POSITIONS
)


def _radar_cone(prim: str, xy, radius: float, height: float, diffuse, emissive) -> RigidObjectCfg:
    """Kinematic glowing cone marker at a radar site. Tip points up."""
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
class DroneReconSceneCfgP2(DroneReconSceneCfg):
    """Phase 2 scene: inherits all obstacles/walls, adds glowing cone radar markers."""

    # Search/Acq - dark blue narrow cone
    radar_marker_search: RigidObjectCfg = _radar_cone(
        "{ENV_REGEX_NS}/RadarMarkerSearch",
        (_RADAR_POSITIONS[0][0], _RADAR_POSITIONS[0][1]),
        radius=0.8, height=6.0,
        diffuse=(0.05, 0.05, 0.9),
        emissive=(0.1, 0.1, 3.0),
    )

    # Pulse-Doppler - dark red narrow cone
    radar_marker_pd: RigidObjectCfg = _radar_cone(
        "{ENV_REGEX_NS}/RadarMarkerPD",
        (_RADAR_POSITIONS[1][0], _RADAR_POSITIONS[1][1]),
        radius=0.8, height=6.0,
        diffuse=(0.8, 0.02, 0.02),
        emissive=(3.0, 0.05, 0.05),
    )

    # Monopulse - amber wide block cone
    radar_marker_mono: RigidObjectCfg = _radar_cone(
        "{ENV_REGEX_NS}/RadarMarkerMono",
        (_RADAR_POSITIONS[2][0], _RADAR_POSITIONS[2][1]),
        radius=2.0, height=3.5,
        diffuse=(1.0, 0.45, 0.0),
        emissive=(3.0, 1.2, 0.0),
    )


@configclass
class CommandsCfgP2:
    target = mdp.WaypointCommandCfg(
        asset_name="robot",
        goal_x_range=(5.0, 28.0),
        goal_y_range=(-11.0, 11.0),
        goal_z_range=(1.0, 3.5),
        waypoints_per_episode=3,
        arrival_threshold=1.0,
        obstacle_margin=2.0,
        exclusion_zones=_RADAR_EXCLUSION_ZONES,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class ActionsCfgP2:
    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)
    drfm_action:    mdp.DrfmActionCfg    = mdp.DrfmActionCfg(radar_positions=_RADAR_POSITIONS)


@configclass
class ObservationsCfgP2:
    @configclass
    class PolicyCfg(ObsGroup):
        # 14D navigation (same as Phase 1)
        target_pos_b        = ObsTerm(func=mdp.target_pos_b,        params={"command_name": "target"})
        waypoints_remaining = ObsTerm(func=mdp.waypoints_remaining,  params={"command_name": "target"})
        attitude            = ObsTerm(func=mdp.root_quat_w)
        lin_vel             = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel             = ObsTerm(func=mdp.root_ang_vel_b)
        # 40D RWR + 8D DRFM state (new)
        rwr        = ObsTerm(func=mdp.rwr_observations)
        drfm_state = ObsTerm(func=mdp.drfm_state_obs)

        def __post_init__(self) -> None:
            self.enable_corruption  = False
            self.concatenate_terms  = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfgP2:
    progress         = RewTerm(func=mdp.progress,          weight=20.0,    params={"command_name": "target"})
    forward_speed    = RewTerm(func=mdp.forward_speed,      weight=8.0,     params={"command_name": "target", "target_speed": 4.0})
    heading          = RewTerm(func=mdp.heading_to_goal,    weight=2.0,     params={"command_name": "target"})
    arrived          = RewTerm(func=mdp.arrived,            weight=500.0,   params={"command_name": "target", "threshold": 1.0})
    completion_bonus = RewTerm(func=mdp.completion_bonus,   weight=1000.0,  params={"command_name": "target"})
    terminating      = RewTerm(func=mdp.is_terminated,      weight=-1000.0)
    step_penalty     = RewTerm(func=mdp.step_penalty,       weight=-0.05)
    proximity        = RewTerm(func=mdp.proximity_penalty,  weight=-1.0,
                               params={"obstacle_names": _OBSTACLE_NAMES, "safe_dist": 2.5, "max_dist": 6.0})


@configclass
class TerminationsCfgP2:
    time_out      = DoneTerm(func=mdp.time_out,          time_out=True)
    all_waypoints = DoneTerm(func=mdp.all_waypoints_done, params={"command_name": "target"})
    collision     = DoneTerm(func=mdp.illegal_contact,    params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01})
    flyaway       = DoneTerm(func=mdp.flyaway,            params={"command_name": "target", "distance": 50.0})
    too_high      = DoneTerm(func=mdp.too_high,           params={"max_z": 4.0})
    radar_lock    = DoneTerm(func=mdp.radar_lock)


@configclass
class DroneReconEnvCfgP2(ManagerBasedRLEnvCfg):
    scene:        DroneReconSceneCfgP2 = DroneReconSceneCfgP2(num_envs=4096, env_spacing=60.0)
    observations: ObservationsCfgP2   = ObservationsCfgP2()
    actions:      ActionsCfgP2        = ActionsCfgP2()
    commands:     CommandsCfgP2        = CommandsCfgP2()
    events:       EventCfg            = EventCfg()
    rewards:      RewardsCfgP2        = RewardsCfgP2()
    terminations: TerminationsCfgP2   = TerminationsCfgP2()

    def __post_init__(self) -> None:
        self.decimation       = 8       # 400 Hz physics / 8 = 50 Hz control = 0.02 s steps
        self.episode_length_s = 20.0    # 1000 control steps
        self.viewer.eye    = (-4.0, 0.0, 1.0)
        self.viewer.lookat = (10.0, 0.0, 2.0)
        self.sim.dt              = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneReconEnvCfgP2_PLAY(ManagerBasedRLEnvCfg):
    scene:        DroneReconSceneCfgP2 = DroneReconSceneCfgP2(num_envs=1, env_spacing=60.0)
    observations: ObservationsCfgP2   = ObservationsCfgP2()
    actions:      ActionsCfgP2        = ActionsCfgP2()
    commands:     CommandsCfgP2        = CommandsCfgP2()
    events:       EventCfg            = EventCfg()
    rewards:      RewardsCfgP2        = RewardsCfgP2()
    terminations: TerminationsCfgP2   = TerminationsCfgP2()

    def __post_init__(self) -> None:
        self.decimation       = 8
        self.episode_length_s = 20.0
        self.sim.dt           = 1 / 400
        self.viewer.origin_type  = "asset_root"
        self.viewer.asset_name   = "robot"
        self.viewer.env_index    = 0
        self.viewer.eye          = (-4.0, 0.0, 1.0)
        self.viewer.lookat       = (1.0,  0.0, 0.0)
        self.sim.render_interval = self.decimation
