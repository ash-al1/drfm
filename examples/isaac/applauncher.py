import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Empty example")
parser.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="Size-length of cuboid"
)
parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of viewport",
)
parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of viewport",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils

def design_scene():
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    cfg_light_distant=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.75,0.75,0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant,
                           translation=(1,0,10))

    cfg_cuboid = sim_utils.CuboidCfg(
            size=[args_cli.size]*3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,1.0,1.0)),
    )
    cfg_cuboid.func("/World/Object", cfg_cuboid,
                    translation=(0.0,0.0,args_cli.size/2))

def main():
    # Simulation start
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    # Camera
    sim.set_camera_view([2.5, 0.0, 2.5], [-0.5, 0.0, 0.5])

    # Scene
    design_scene()
    sim.reset()
    
    # Step through simulation
    print("[INF)] Setup complete... ")
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
