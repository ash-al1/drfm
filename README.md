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
tracking. Maneuverability is based on [4] which is based off the research paper
[1]. Drone is used to survive some electromagnetic environment with
deterministic radars that attempt to gain a lock, the DRFM module is trained to
survive using realistic jamming techniques: transponder and repeater false
targeting, combination of RGPO and VGPO, and SAR active decoy.

## DRFM Action Space

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

## Phases

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

Navigation only: simple case of drone that navigates to three separate waypoints
that are randomly placed within boundary walls.
```sh
HYDRA_FULL_ERROR=1 python3 scripts/train.py --task Isaac-Drone-Recon-v0 --headless --num_envs 4096 --phase 1 --algorithm SAC --save_buffer_interval 10000 --buffer_keep_n 3

python3 scripts/play.py --task Isaac-Drone-Recon-Play-v0 --num_envs 1 --algorithm SAC --checkpoint models/checkpoints/[CHECKPOINT]/agent_best.pt
```

Navigation and DRFM: three separate radars are present one for each action space
above sparsely located to not interfere with each other. DRFM has no direct
connection to rewards, this needs to be thought through better.
```sh
python3 scripts/train.py --task Isaac-Drone-Recon-v1 --headless --num_envs 2048 --phase 2 --algorithm PPO

python3 scripts/play.py --task Isaac-Drone-Recon-Play-v1 --num_envs 1 --algorithm PPO --debug --checkpoint models/checkpoints/[CHECKPOINT]/agent_best.pt
```

## Justification

We used Proximal Policy Optimazation (PPO) throughout the project and only added
Soft Actor-Critic (SAC) later for ablation & replay buffer. Both these agents
support hybrid discrete-continuous actions which is critical for the DRFM
module, which supports 3 discrete actions: RGPO, VGPO, RVGPO - when one of these
is selected each has continuous set based on its position w.r.t radar. PPO and
SAC covers decent variance since one is on-policy and the other is off-policy.

All other agents mentioned, DQN, REINFORCE, vanilla Actor-Critic, DDPG, TD3 and
TRPO cannot be used for any of these reasons: discrete only, continuous only,
higher variance. Also PPO is pretty popular compared to all the others ...

For thought process in building the environment, how we structure separate
navigation and DRFM, and jointly incorporate them take a look in
./docs/thoughts.md. This file also includes some major issues we ran into.

## Setup

Make sure Isaac Sim and Lab are [installed](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#verifying-the-isaac-lab-installation) & environment is setup properly:
```
conda env create -f environment.yaml -n [name]
conda activate [name]
export ISAACSIM_PATH="${HOME}/isaacsim/_build/linux-x86_64/release"
export ISAACSIM_PYTHON_EXE="${HOME}/isaacsim/_build/linux-x86_64/release/python.sh"
ln -s ${ISAACSIM_PATH} _isaac_sim
```

## Project Organization

```
├── LICENSE
├── README.md
├── pyproject.toml
│
├── configs/
├── docs/               # Thought process, task list, updates, research
├── media/ 
├── examples/           # Old tutorial for bookkeeping
├── scripts/            # Main run files
│   ├── train.py
│   └── play.py
│
├── drfm/
│   ├── __init__.py
│   ├── assets/         # Iris USD
│   ├── robots/         # Drone file
│   ├── envs/
│   │   ├── isaac/      # Drone recon cfg
│   │   │   └── mdp/    # Actions, Observations, etc
│   │   └── gym/
│   ├── agents/         # Skrl hyperparameters
│   ├── algorithms/     # Classic RL algos
│   ├── dynamics/       # Drone dynamics (Isaac drone racer)
│   └── utils/
```

## TODOS

[ ] Rewards that take into account radar and DRFM module
[ ] More agents for wider study
[ ] Incorporate GNU Radio for direct RF signals
[ ] Pull FPGA statistics, compute for DRFM usage
[ ] Change drone to fixed wing or non-racing drone
[ ] Change environment to be more realistic

## References

1. Wang, Chao, et al. *"Autonomous navigation of UAV in large-scale unknown complex environment with deep reinforcement learning."* GlobalSIP 2017
1. Kaufmann, E., et al. *"Champion-level drone racing using deep reinforcement learning."* Nature, 2023
2. Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction.* MIT Press, 2018.
3. Merrick, R. *Getting Started with FPGAs: Digital Circuit Design, Verilog, and VHDL for Beginners.* No Starch Press, 2023.
4. Pace, P. E. *Developing Digital RF Memories and Transceiver Technologies for Electromagnetic Warfare.* Artech House, 2022.
5. [Isaac Drone Racer](https://github.com/kousheekc/isaac_drone_racer)
6. [Isaac Sim: Foundation Model](https://github.com/isaac-sim/IsaacSim)
7. [Isaac Lab: RL Environments](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html)
8. [Radar Equations - MIT Lincoln Lab](https://www.ll.mit.edu/sites/default/files/outreach/doc/2018-07/lecture%202.pdf)
9. [Radar Jamming and Deception - Wikipedia](https://en.wikipedia.org/wiki/Radar_jamming_and_deception)
10. [DRFM: History, Circuit & Testing - Rohde & Schwarz](https://www.rohde-schwarz.taipei/data/activity/file/1644474835378405224.pdf)
11. [TD Learning - Stanford CME241](https://web.stanford.edu/class/cme241/lecture_slides/rich_sutton_slides/11-12-TD.pdf)
12. [Bellman Equation - Wikipedia](https://en.wikipedia.org/wiki/Bellman_equation#The_Bellman_equation)
13. [Bellman's Principle of Optimality - Wikipedia](https://en.wikipedia.org/wiki/Bellman_equation#Bellman's_principle_of_optimality)
14. [MDP Algorithms: Value & Policy Iteration - Wikipedia](https://en.wikipedia.org/wiki/Markov_decision_process#Algorithms)
16. [AN/ALE-55 Fiber-Optic Towed Decoy (FOTD) Image - BAE SYSTEMS](https://www.baesystems.com/en-us/product/anale55-fiberoptic-towed-decoy)
15. Claude (Anthropic)
