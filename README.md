# Reinforcement Learning & DRFM

<table align="center">
    <tr>
        <td>
            <img src="media/bae.png" height="200px" />
        </td>
        <td>
            <img src="media/draw.png" height="200px" />
        </td>
    </tr>
</table>

This project aims to create a realistic Digital Radio Frequency Memory module
embedded on a drone that operates using Reinforcement Learning algorithms, the
drone itself also maneuvers using an algorithm tasked with surviving Radar
tracking. Maneuverability is based on [9] which is based off the research paper
[4]. Drone is used to survive some electromagnetic environment with
deterministic radars that attempt to gain a lock, the DRFM module is trained to
survive using realistic jamming techniques: transponder and repeater false
targeting, combination of RGPO and VGPO, and SAR active decoy.

## Foundations

We highly recommend readers to go through `docs/` directory for quickly catching up to speed with how foundational things are implemented in this project or building a mental mindmap on where inspiration came from.

<table align="center">
    <tr>
        <td>
            <img src="media/rgpo.gif" height="200px" /><br>
            RGPO
        </td>
        <td>
            <img src="media/vgpo.gif" height="200px" /><br>
            VGPO
        </td>
        <td>
            <img src="media/rgpo+vgpo.gif" height="200px" /><br>
            Coordinated
        </td>
    </tr>
</table>

## Setup

Make sure Isaac Sim and Lab are [installed](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#verifying-the-isaac-lab-installation) & environment is setup properly:
```
conda env create -f environment.yaml -n [name]
conda activate [name]
export ISAACSIM_PATH="${HOME}/isaacsim/_build/linux-x86_64/release"
export ISAACSIM_PYTHON_EXE="${HOME}/isaacsim/_build/linux-x86_64/release/python.sh"
ln -s ${ISAACSIM_PATH} _isaac_sim
```

Modify robot path in `drfm/robots/five_in_drone.py`

## Usage

<table align="center">
    <tr>
        <td>
            <img src="media/phase2.gif" height="200px" /><br>
        </td>
    </tr>
</table>

Environment is split into two phases: (1) navigation, (2) DRFM. This allows us
to test different agents, architectures on invidiual problems. Later the agent
will be packaged without any regards for which phase to use.

Full task (navigation + DRFM):
```sh
python3 scripts/train.py --task singleDRFM --headless --num_envs 4096 --algorithm PPO_GRU --log-level INFO
python3 scripts/play.py --task singleDRFM --num_envs 1 --algorithm PPO_GRU --debug
```

Scaffolding
```sh
python3 scripts/train.py --task singleDRFM_stage1 --headless --num_envs 8192 --algorithm PPO_GRU --log-level INFO
python3 scripts/train.py --task singleDRFM_stage2 --headless --num_envs 8192 --algorithm PPO_GRU --log-level INFO --checkpoint path/to/stage1/best_agent.pt

python3 scripts/play.py --task singleDRFM_stage1 --num_envs 1 --algorithm PPO_GRU --debug
python3 scripts/play.py --task singleDRFM_stage2 --num_envs 1 --algorithm PPO_GRU --debug
```

## Environment

**Observations**

| Term | Description |
|---|---|
| `target_pos_b` | Next waypoint in body frame |
| `waypoints_remaining` | Count of remaining waypoints |
| `attitude` | Quaternion orientation |
| `altitude` | Altitude error from target z=3 m |
| `vertical_vel` | Vertical velocity |
| `lin_vel` | Linear velocity in body frame |
| `ang_vel` | Angular velocity in body frame |
| `rwr` | Radar warning receiver per radar |
| `drfm_state` | DRFM jammer state |

**Actions** (11D)

| Term | Dim | Description |
|---|---|---|
| `control_action` | 4 | Per-motor thrust [-1, 1] |
| `drfm_technique` | 4 | Logits → OFF / RGPO / VGPO / RVGPO |
| `drfm_params` | 3 | Pull-off rate, velocity pull-off, coordination ratio |

**Rewards**

| Term | Weight | Description |
|---|---|---|
| `waypoint_reached` | +50 | Per waypoint bonus |
| `completion_bonus` | +100 | All waypoints done |
| `progress` | +5 | Forward progress toward waypoint |
| `forward_speed` | +2 | Speed toward goal (target 5 m/s) |
| `heading` | +2 | Aligned heading to goal |
| `drfm_effective` | +2 | Jamming an active radar |
| `smart_jam` | +1 | Jamming the right radar for the threat |
| `power_conserve` | +0.5 | Low DRFM power when not needed |
| `upright` | +1 | Upright orientation |
| `terminating` | -200 | Bad termination (collision / radar lock) |
| `altitude_band` | -5 | Deviation from z=3 m ±1 m |
| `illumination_low` | -2 | Radar illumination on drone |
| `proximity` | -3 | Within 2.5–6 m of obstacle |
| `ang_vel_l2` | -0.02 | Angular velocity magnitude |
| `action_smooth` | -0.01 | Action jitter between steps |
| `step_penalty` | -0.01 | Time alive penalty |

## Algorithms

We used Proximal Policy Optimization (PPO) as the backbone throughout the
project with Soft Actor-Critic (SAC) added later for ablation & replay buffer
comparison. Both agents support hybrid discrete-continuous actions which is
critical for the DRFM module - technique selection is discrete (OFF, RGPO,
VGPO, RVGPO) while each technique's parameters (pull-off rate, velocity
pull-off rate, coordination ratio) are continuous. PPO and SAC cover decent
variance since one is on-policy and the other is off-policy.

We also implemented PPO_GRU (PPO with a GRU recurrent encoder) specifically
to handle partial observability in the radar environment. The drone receives
Radar Warning Receiver (RWR) observations including: bearing, power,
illumination rate, pulse interval variance which are noisy single-timestamp
snapshots. A memoryless MLP policy cannot distinguish whether a radar is ramping up
toward lock or cooling down from a failed track. The GRU encodes the temporal
RWR stream (32D) into a hidden state while passing static observations
(attitude, velocity, DRFM state) through directly. Theoretically, this
split-stream design lets the agent build a mental model of radar over
time without forcing navigation state through recurrence.

All other agents mentioned, DQN, REINFORCE, vanilla Actor-Critic, DDPG, TD3 and
TRPO cannot be used for any of these reasons: discrete only, continuous only,
higher variance. Also PPO is pretty popular compared to all the others ...

## Project Organization

```
  ├── LICENSE / NOTICE
  ├── README.md
  ├── environment.yaml
  ├── docs/               # Research notes & technical challenges
  │   ├── drfm.md
  │   ├── radar.md
  │   ├── meta.md
  │   ├── references.md
  │   └── technical-challenges.md
  │
  ├── media/
  ├── scripts/
  │   ├── train.py
  │   └── play.py
  │
  ├── outputs/
  ├── drfm/
  │   ├── assets/
  │   │   └── configuration/
  │   ├── robots/
  │   ├── dynamics/
  │   ├── algorithms/
  │   ├── agents/
  │   ├── utils/
  │   └── isaac/
  │       ├── drfm_env.py
  │       ├── agents/
  │       └── mdp/
  │
  └── models/
      ├── architectures/
      ├── checkpoints/
      └── replay_buffers/
```

## Note on AI Use and Assets

Claude helped in refactoring old SKRL version 1.4.3 to 2.1.0, although couldn't
really tell whether it was working or not since agents broke as a result. The
initial baseline foundations of this project was based on isaac drone racer,
and we largely stuck with the some allocation and motor usage, largely nothing
has changed here besides some testing we did with how thrust initializes. The
MDP structure for actions, observations rewards we took from isaac drone racer
too and added on top including waypoints, and a lot of reward structure
primarily so that the drone is limited in altitude (minz and maxz) and urged
towards the waypoint while preventing itself from colliding with objects.

The radar for both deterministic and probabilistic was designed based on the
Phillip E. Pace book and I got claude to validate some of the mathematics for
calculating DRFM and drone interactions. The equations were pulled directly from
the book or online (SPJ). I did not create any of the assets used (drone mesh,
USD, URDF and so on), most media was created by me besides the AI generated
image of BAE systems decoy as header in this file.

## Future 

```
1. Properly get MAPPO & PPO GRU working.
    - I need an easier way to validate whats going on, visualize inconsistencies
      and debug easier.
2. Change reward structure so its not unbearably fragile.
3. Change environment to be more realistic.
4. Add IQ waveforms using USRP recorded signals instead of janky radar
   interactions we current have.
```


## References

1. Schulman, John, et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
1. Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." International conference on machine learning. Pmlr, 2018.
1. Wang, Chao, et al. *"Autonomous navigation of UAV in large-scale unknown complex environment with deep reinforcement learning."* GlobalSIP 2017
1. Kaufmann, E., et al. *"Champion-level drone racing using deep reinforcement learning."* Nature, 2023
1. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction.* MIT Press, 2018.
1. Merrick, R. *Getting Started with FPGAs: Digital Circuit Design, Verilog, and VHDL for Beginners.* No Starch Press, 2023.
1. Pace, P. E. *Developing Digital RF Memories and Transceiver Technologies for Electromagnetic Warfare.* Artech House, 2022.
1. Salimpour, Sahar, et al. "Sim-to-real transfer for mobile robots with reinforcement learning: from nvidia isaac sim to gazebo and real ros 2 robots." arXiv preprint arXiv:2501.02902 (2025).
1. [PPO SKRL](https://skrl.readthedocs.io/en/latest/api/agents/ppo.html)
1. [Isaac Drone Racer](https://github.com/kousheekc/isaac_drone_racer)
1. [Isaac Sim: Foundation Model](https://github.com/isaac-sim/IsaacSim)
1. [Isaac Lab: RL Environments](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)
1. [Isaac Lab: Actuators](https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/transferring-robot-learning-policies-from-simulation-to-reality/04-bridging-the-gap-real-world-data-integration/01-actuator-modeling.html)
1. [Radar Equations - MIT Lincoln Lab](https://www.ll.mit.edu/sites/default/files/outreach/doc/2018-07/lecture%202.pdf)
1. [Radar Jamming and Deception - Wikipedia](https://en.wikipedia.org/wiki/Radar_jamming_and_deception)
1. [DRFM: History, Circuit & Testing - Rohde & Schwarz](https://www.rohde-schwarz.taipei/data/activity/file/1644474835378405224.pdf)
1. [TD Learning - Stanford CME241](https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/11-12-TD.pdf)
1. [Bellman Equation - Wikipedia](https://en.wikipedia.org/wiki/Bellman_equation#The_Bellman_equation)
1. [Bellman's Principle of Optimality - Wikipedia](https://en.wikipedia.org/wiki/Bellman_equation#Bellman's_principle_of_optimality)
1. [MDP Algorithms: Value & Policy Iteration - Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process#Algorithms)
1. [AN/ALE-55 Fiber-Optic Towed Decoy (FOTD) Image - BAE SYSTEMS](https://www.baesystems.com/en-us/product/anale55-fiberoptic-towed-decoy)
1. [Radar Tutorials: Self Protection Jammer](https://www.radartutorial.eu/16.eccm/ja11.en.html)
1. Claude (Anthropic)
