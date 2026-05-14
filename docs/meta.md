## Goal:

+ Closed loop environment, continuous state space, actions
+ More complicated: multi-target tracking, multi-agent
+ Multi agent drone enviroment that can fly from source to destination while
  dodging obstacles and preventing lock-ons from radar.
    - Chooses best DRFM technique against radar
    - Manages its power usage over time (can't use best technique on all radars)
    - Proper collision avoidance against static objects AND other drones
+ RF environment model
    - Throughout the semester I tried to think about how to incorporate GNU
      Radio which is an application that can generate (simulate or real world)
      IQ signals and incorporate them with the RADAR implementation directly.
    - Could not figure out a method that would realistically work given the time
      I had (and didn't have ...). One method of my best thoughts here is to
      save IQ waveforms and use offline training similar to what we learned in
      class, this was the motivation after those lectures. Isaac also does
      raytracing which I can swap our 'light' for RF but there is no easy way
      after having a quick ~10 minute dive into the IsaacSim github page.
    - Settled on probabilistic model (check commits), using Bernoulli on SNR and
      applied noise directly on drone.
+ Isaac Sim enviroment
    - I had many ideas including recording Gaussian Splat of an environment and
      loading into Isaac directly, or
      [Cesium](https://cesium.com/platform/cesium-ion/) which is an EXTREMELY
      cool thing to have done, but given how laggy and GPU hungry Isaac already
      is I decided not to. I would've picked a cool spot on earth and loaded it
      into isaac instead of having to build an environment.
    - Settled on a simple obstacle corridor example, where drone has to avoid
      obstacles to pass through from start to finish (non-trivial obstacles).
      Which include (N)LOS characteristics via radar interaction.
+ DRFM
    - Start of this project, I wanted to incorporate an FPGA to this project.
      Later learned how unbelievably slow this would turn out to be. I
      specifically chose this drone project because the ETH drone racing paper
      VS. professional drone racers had the same idea, use real flight control
      data instead of simulating everything.
    - But the FPGA I had was a MUCH less capable device than what deemed worthy
      to use. I had an older Lattice iCE40 HX1k chip I could use but it has
      1,280 4-input LUTs and no dedicated DSP blocks.
    - I settled on simulating realistic techniques found in Phillip E. Pace
      book.
+ Agent
    - Algorithm-wise I just wanted something to work and didn't bother looking
      into any SOTA models. As long as one could work and theoretically made
      sense, given we learned about PPO and saw how popular this algorithm was
      decided to sit on it.
    - I picked SKRL as the library because Isaac drone racer github repository
      did the same, later researched a few other libraries and realized how
      unbelievably fragmented the RL world is compared to torch, cuda and so on.
      I attempted to use RayLib but found it had no QMIX anyways so chose SKRL,
      which had PPO PPO(GRU on the 2.1.0 version) and MAPPO.
    - I initially built on 1.4.3 but had to refactor later.
    - PPO and PPO+GRU literally did not work at all. Agent would either fly
      straight, hover and die. Or fly to the ground/sky where it would
      terminate. MAPPO worked surprisingly.
    - I read a few papers which I didn't really like, the direction many
      researchers have taken w.r.t drones, Signal Processing and Radio Frequency
      in RL seems to be too much in simulations. I immediately didn't bother
      with digging deeper in that scope, but picked up the partial observability
      content from this domain, which many authors have worked out already.

---

## Versions

This section contains problems and solutions I encountered in each version of
this project. For V3, PPO and PPO GRU methods did NOT work at all. Reward
shaping was a pain, I could not get the agents to operate at all.

Every solution trapped us in another problem causing model collapse. I wish I
knew how to visualize or debug these things easier, but one method I used to
save me some time was remembering that about 15-20% of the run would show more
or less the result of our modification. This was specifically for
hyperparameter tuning, but I used the same concept here for critical
modifications. So 10K timesteps for lots of training, around 2K for quick
checks. 25K timesteps for final (45~ minutes).

### V3

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

### V2
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

### V1

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

## Poor:

Poor ideas we had that got dropped later at some point.

+ Can not just use isaac drone racer out of the box, algorithm and MDP dynamics
  are incomplete for our task
+ [DRFM - Mesarcik](https://github.com/mesarcik/DRFM) is not useful DRFM action
  space, not built on real world techniques. It does amplitude, frequency,
  phase, etc changes - which is fundamentally accurate yet mathematically
  nebulous. It simplifies a DRFM module too much.
+ Using an FPGA is problematic because of time constraints, can't wait for data
  in/out for each iteration on many agents in an environment.

---

## Ideas

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

## Qs:

+ In what situations does the DRFM module do: X - i.e. when to time/freq etc?
+ Should I focus on theoretical guarantees? This devolves the project down into
  reading theoretical books on digital signal processing, wireless
  communications and other books (radar) for the purpose of extracting
  mathematical theorems and algorithms beneficial specifically for those signals
  and appending them into layers at feature extraction points or other
  mathematically attractive positions. -> All this to say, is this really good
  enough for the project?

