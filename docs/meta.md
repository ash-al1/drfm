# Tasks:

+ Add SKRL multi agent
+ 
+ POMDP - RWR and radar have info when they should not
    - fix rwr\_observations
    - agent infers radar state from signal behavior instead
    - give timestep history of RWR, remove tq and radar state
    - replace MLP with GRU or LSTM
+ Fix --debug
+ Need outputs!!! Plots, visualizations!!! Saliency? or similar?
+ Replay, memory need to be properly managed
+ Submission stuff to keep in check:
    - Change readme
    - Choose better images, and include results

---

# Ideas

+ Create difficult environments that can train UAV navigation and DRFM module
  separately, or together that are unrealistic yet help train an agent.
+ Use Isaac sim to export drone pos/vel. emitter location, LoS or not. Feed
  data into RF model, compute J/S and detection. Echo, Jamming power,J/S ratio,
  etc. Agent receives J/S ratio and detection status as observation vector, then
  calculates reward based off agent perforamnce. +1 reward every timestep alive,
  maybe bonus if radar loses track, penalties for being detected and locked on.
+ Use the nature racing drone concept of implementing controller and firmware
  for input peculiarities to the drone input, we can do the same for DRFM module
  Mesarcik's DRFM implementation ... plug and play
+ Pegasus Simulator?
+ GNU Radio connector for RF backend?
+ FPGA usb 3.0 connected, isolated, latency w.r.t. simulator: handles DRFM
  compute isolated from simulator RF backend & drone robotics?
+ Mesarcik uses: Frequency + doppler, time, amplitude shifts - what should we
  use? Check the patent and other files I have on USB

---

# Qs:

+ In what situations does the DRFM module do: X - i.e. when to time/freq etc?
+ Should I focus on theoretical guarantees? This devolves the project down into
  reading theoretical books on digital signal processing, wireless
  communications and other books (radar) for the purpose of extracting
  mathematical theorems and algorithms beneficial specifically for those signals
  and appending them into layers at feature extraction points or other
  mathematically attractive positions. -> All this to say, is this really good
  enough for the project?

---

# Goal:

+ Closed loop environment, continuous state space, actions
+ More complicated: multi-target tracking
[ ] RF environment model
    + Radar echo, jamming-to-signal ratio, burn-through-range
    + Parallel physics approach
[ ] Isaac Sim model
    + Basic terrain map with drone and single structure/radar/block
    + Start pulling entity positions and velocities
    + Terrain LOS queries
[ ] DRFM
    + Capture and store 'signals'
    + Shift/delay, etc and retransmit
[ ] Agent
    + Select shifts,delays, etc of DRFM module to keep it alive
    + What other tasks can it get? Maybe RL agent that flies the drone?

---

# Versions

## V3

+ Calculate distance from drone to objects properly (to surface, not center)
+ Better outputs, accumulated terminations, and other metrics printed on train
+ Terminate when too close to ground
+ Better rewards in general
    - remove distance penalty, robot learned to die faster to get less negatives
+ Remove CSVLogger() which does plots(?) idk
+ Add --log-level in train, remove debug interval
+ Remove redundant debugs/printing in train and switch to tensorboard
+ Add physx
+ Refactor train::main() to be small
+ Change printing to logging
+ Proper licensing with notice
+ Add one-liner docstrings
+ Cleanup train script, move odd functions to drfm/agents
+ Cleanup fmtTime function
+ Removed drone racer and phase 0 stuff only kept singleDRFM environment
+ Add paper references for SAC,PPO - add more later for MARL
+ Move sac and PPO building to agents/builder.py
+ Remove useless __init__ files

## V2
+ DRFM
    + Still work in progress, power consumption, delays need tuning
    + Each radar is weak against some DRFM and not others
+ Radar
    + Search/Acquisition radar/Pulse Doppler/Monopulse simulations
    + View mod_radar_types and thoughts.md
+ Drone navigation within bounded region, with sparse objects
    + Waypoints/goals are gold disks drone has to go to
    + Rewards and penalties to urge fast completion, prevent hover, dying and
      collision with objects - still needs additional work
+ Defined DRFM action space as [Off, RGPO, VGPO, RVGPO]
    - Discrete action space, continuous parameter selection per technique
    - PPO with discrete head, continuous head per technique for parameters
+ Pull Isaac drone racer project

## V1

+ Tabular RL Q-learning model implementation to test a simple agent can learn to
  survive. Agents actions could be to select DRFM module parameters. Drone
  maneuvarability should be continuous algorithm. (`drfm_grid_env.py`)
+ Implement core RF (basic) modeling on a standalone python file. -> (`basic_model.py`)
    + Radar echo, Jamming power, J/S ratio, burn-through range
+ Monte Carlo
+ TD(0), TD(1), TD(Lambda) - compare if front or backwards view
+ SARSA
+ Q-Learning
+ Agent wrapper
+ Expose Q-value in libmdp
+ Map continuous to map discrete values
+ Port libmdp and libsparse over from mini projects

---

# Poor:

Poor ideas we had that got dropped later at some point.

+ Can not just use isaac drone racer out of the box, algorithm and MDP dynamics
  are incomplete for our task
+ [DRFM - Mesarcik](https://github.com/mesarcik/DRFM) is not useful DRFM action
  space, not built on real world techniques. It does amplitude, frequency,
  phase, etc changes - which is fundamentally accurate yet mathematically
  nebulous. It simplifies a DRFM module too much.
+ Using an FPGA is problematic because of time constraints, can't wait for data
  in/out for each iteration on many agents in an environment.
