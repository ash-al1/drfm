import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Empty example")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext

sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

sim.reset()
print("Setup complete")
while simulation_app.is_running():
    sim.step()

simulation_app.close()
