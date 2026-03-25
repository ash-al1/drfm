import argparse

from isaaclab.app import AppLauncher

"""
# Default spawn prim method:

# Config per prim for relationships, material, shading
# Every prim gets a config
cfg = MyPrimeCfg()
prim_path = "/home/ic3/ic3/drfm/assets/iris.usd"

cfg.func(prim_path, cfg, translation=[0,0,0], orientation=[1,0,0,0],
         scale=[1,1,1])

"""

parser = argparse.ArgumentParser(description="Empty example")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def design_scene():
    # Ground plan
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # Light 
    cfg_light_distance = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distance.func("/World/lightDistance", cfg_light_distance,
                            translation=(1,0,10))

    # Transform prims group other prims
    # Or define transformation
    sim_utils.create_prim("/World/Objects", "Xform")

    # Spawn objects, by default material properties are disabled
    cfg_cone = sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.0,0.0)),
    )
    cfg_cone.func("/World/Objects/Cone1", cfg_cone, translation=(-1.0,1.0,1.0))
    cfg_cone.func("/World/Objects/Cone2", cfg_cone, translation=(-1.0,-1.0,1.0))

    # Third cone with rigid physics
    cfg_cone_rigid=sim_utils.ConeCfg(
            radius=0.15,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0)),
    )
    cfg_cone_rigid.func(
            "/World/Objects/ConeRigid", cfg_cone_rigid, translation=(-0.1,0.0,2.0),
            orientation=(0.5,0.0,0.5,0.0)
    )

    # Cuboid with deformable physics
    # Deformable means motion w.r.t vertices. like cloth, rubber, jello
    cfg_cuboid_deformable = sim_utils.MeshCuboidCfg(
            size=(0.2, 0.5, 0.2),
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(),
    )
    cfg_cuboid_deformable.func("/World/Objects/CuboidDeformable",
                               cfg_cuboid_deformable, translation=(0.15,0.0,2.0))


def main():
    # Start environment
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set camera
    sim.set_camera_view([2.5, 0.0, 2.5], [-0.5,0.0,0.5])
    
    # Run through scene
    design_scene()
    sim.reset()
    print("Setup complete")

    # Step through scene, just like RL
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
