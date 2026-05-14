# Problem, solutions, thought processing

File that contains issues, and their solutions plus thought processing. Even
though ./docs/ contains general information on how things work we need to modify
to fit our project ~ non-trivially.

## V3 Problems and solutions

1. MAPPO actually works a lot better than PPO and PPO_GRU i dont know why. All
   drones share a single policy by sharing their parameters and the global gets
   flattened to a single dimension (global state, i.e. centralized critic). 
    - MAPPO differing from the others actually passes observation inputs not
      state inputs.
    - Current fault is sharing centralized critic and cooperative gaming.
      Surprisingly this worked better than I expected, given Mini4 content this
      should not have worked that well. My assumption is because of how easy the
      environment is, this worked fine w.r.t static environment.
    - MAPPO has no scaffolding it trains both navigation and DRFM modules at the
      same time (PPO+GRU contains scaffolding and still fails).
1. PPO_GRU fails to hover and navigate completely, high collision rate ; agent
   just gradually falls to the ground
    - Action mapping (motor, allocation) is not good. Added piecewise-linear
      over bias with action=0 hover thrust.
1. Struggling to fix the agent just flying upward/downward and insta-terminating
    - Added new observations for altitude & vertical velocity
    - Also the initial bias the drone has is to climb up caused by the control
      action configuration
1. Fixed proximity penalty given we calculate distance to objects properly
    - Remove outer boundary walls from proximity penalty, there already is a
      termination on them ; penalty applied on drones spawning next to walls
    - Upper bound clamp added
1. Migrate SKRL 1.4.3 -> 2.1.0 for PPO RNN implementation
    - Replace custom messed up GRU hidden state with SKRLs built in PPO RNN,
      this has get_specification() and proper auto hidden state management
    - Change builder/train/play to proper calls now, PPO_CFG, keyword inits and
      critically 2-tuple compute/act returns, observations input key,
      ExperimentCfg and SAC_CFG
    - Add new PPO_GRU option to train and play instead of bolting to old PPO
1. Altitude is an issue, drone flies into ground or upper limit to terminate
   because distance penalty causing agent to learn to die faster to accumulate
   less rewards
   - Completely remove distance penalty
   - Added ground limit similar to max_z, min_z terminates agent 0.3m from
     ground
   - This change is more non-trivial than I thought, adding altitude band and
     penalizing doesn't solve, adding only ground penalty doesn't solve.
   - This also kicks up our need to train for longer ... not ideal.
1. Interaction of agent to objects in the environment feels like im making a
   video game
   - Slab intersection calculating for LoS between radar and drone
   - Calculate distance between center of obstacle to surface, and distance to
     drone
1. Thrust is a quadratic value in mdp/actions.py problem is our actions are
   linear
   - PPO uses gaussian policy, we see differences in training and inference when
     running train.py or play.py
   - I think whats going on is the noise in training pushes drone above ground
     enough for it to continue forward, but during inference that noise does not
     exist. Change thrust to be linear in action
1. I hate reward shaping so much ; small changes in values completely changes
   agent outcome ; very fragile and clunky to work with.
   - Adding more metrics, debugging during training helps shed some light in
   black box a lot

## V1/V2 Solutions

1. Debug_viz helped solve problem:
    + https://github.com/isaac-sim/IsaacLab/discussions/2516
    + Probably spent ~5 hours trying to fix this s...
    + Still don't know what problem is but its disappeared for now
2. Manager based RL env hardcodes action space to (-inf, inf)
    + Set bounded action space before passing to skrl
3. skrl doesn't have proper good support for verbosity during training
    + Copy tracking_data directly after each update
4. Training bootstrap was getting injected during runtime
    + Hardcode to False, prevents GAE causing NaNs - i don't know how/why
5. Many training issues, catastrophic forgetting
    + Start on easy tasks and ramp up

## Thought process

Question is how do we structure environment and radar + drone interaction?
+ What does drone see in rf rxr?
+ Power consumption?
+ What is drones observation space?
    + Position, velocity, orientation, distance/bearing to current waypoint,
      radar signal, power budget
+ Rewards/penalties:
    + Reward for getting closer to checkpoint/finish
    + Penalty constant to prevent loitering, urge to finish
    + Penalty for getting too close to objects
    + Reward for pointing in the right direction
+ Mission data files
    + File of what the mission entails: start, checkpoint and final positions
    + Include intermediate tasks, i.e. hover over a location for X time

---

0. DRFM and Radar glue
    + Threat library matches fingerprint against mission data files
    + RWR measures signal parameters (PRF, Fc, Pulse width, scan pattern,
      polarization) then matches to threat library.
    + RWR can distinguish between illumination beam and high-PRF pencil beam

1. Training
    + How? ( Using Alex's comments )
        - Train both flying and DRFM together
        - Use scaffolding: idea of starting off eacy (training wheels) and
          gradually adding complexity helping the model learn. Time decaying.
        - Avoid catastrophic forgetting but not re-training
        - Avoid gradual learning because moving loss function prevents finding
          local extrema

2. Define what type of radars are in use
    + How do they operate deterministic or stochastic?
        - Deterministisc
    + How do they exactly operate?
        - Search -> Detect -> Track -> Lock -> Kill
    + Do all radars have different parameters?
        - They have to, to allow drone to trigger different movement & DRFM
    + What kind of radars?
        - Pulse Doppler, Search/acq., monopulse

3. Define DRFM
    + How?
        - Discrete techniques, continuous variable selection
    + What is its action space?
        - Off, RGPO, VGPO, RVGPO, add more later
    + When does it trigger what action?
        - Based on radar type, a radar may be vulnerable to only RGPO, or VGPO,
          or has capability to counter either or but not both.
    + Model? No idea - discrete-action
    + Penalties and rewards?
        - Reward for choosing correct technique vs. specific radar, penalize for
          getting locked on/tracked, Penalize for incorrect parameters given
          individual techniques (develop further).
    + Sensors?
        - RF wideband receiver, RF transmitter, FPGA for memory & fast DSP
    + Limitations?
        - Power consumption simulation to limit RVGPO usage

4. Drone Maneuverability:
    + How does drone know distance from itself to objects in multiple directions
        - Camera
    + Orientation angles
        - IMU sensor simulation (X,Y,Z?)
    + Distance and angle to 'finish' line
        - Needs GPS coordinates
    + Model? Actor-critic is best used here
        - POMDP because partial observation & no path planning
    + Penalties and rewards:
        - urge to finish quick, urge to take best orientation/angles, penalize
          for risky decisions (too close to objects)

## Radar & DRFM Interaction

### Search/Acquisition Radar

Simple, non-coherent, no Doppler processing. Measures range:
```
R_measured = c · t_echo / 2 # Range, t_echo is delay of strongest return

# Effect of RGPO -> DRFM return is stronger than skin return, walk range gate
Δt(step) = Δt(step-1) + (2 · pull_off_rate · dt) / c
R_apparent = R_true + c · Δt / 2

# When RGPO active and J/S > 1:
# α is effeciveness rate
gate_separation = |R_apparent - R_true| / gate_width
capture_factor = clamp(gate_separation, 0, 1)
tq -= α_rgpo · capture_factor · dt
```

VGPO does not affect this radar, no velocity gate, no Doppler processing.

Sweep model: mechanical rotation, antenna beamwidth, scan period. Beam is on
target for certain dwell time. Detection only on dwell timer, non otherwise.
```
tq -= bleed_rate · dt # bleed_rate ~ 0.02/s
```

### Pulse-Doppler Radar

Measures range and radial velocity independently. Validates one against another.
Range: `R_measured = c · t_echo / 2`, velocity from Doppler:
```
f_d = 2 · fc · v_radial / c
v_measured = f_d · c / (2 · fc)
```

Why RGPO fails: `v_from_range = ΔR_measured / Δt_between_pulses`, then measure
`consistency_error = |v_from_range - v_measured|`. Check consistency:
```
if consistency_error > consistency_threshold:
    reject range measurement, hold last valid gate position
    tq += α_detect · (1 - tq) · dt # tracking continues normally
```

What VGPO does - inject delta f, gate follows v_apparent consistency check now
helps with deception.
```
Δf(step) = Δf(step-1) + (2 · velocity_pull_off_rate · dt) / λ
v_apparent = v_true + Δf · λ / 2
```

What RVGPO does - more effective than VGPO, finds physically plausible
trajectory both gates are pulled together
```
range_rate_from_delay = c · d(Δt)/dt / 2     # implied velocity from range pull
velocity_from_shift = Δf · λ / 2             # implied velocity from Doppler pull
coordination_error = |range_rate_from_delay - velocity_from_shift|
pqf = 1 - clamp(coordination_error / max_allowed_error, 0, 1)
```

Sweep model: Dwell fraction typically high, no mechanical sweep, electronically
steered beam dwells on target most of the time.

### Monopulse Radar

Measures angle error, range and velocity - 4 beams. `θ_error = Re(Δ / Σ)`.

No gate to walk off, ratio is compute on every pulse. RVGPO degrades range and
velocity enough that combined destabilizes.
```
# RVGPO effect on monopulse: indirect, through track filter destabilization
range_vel_degradation = α_rvgpo_mono · pqf · combined_capture_factor · dt
# But the angle channel provides a stabilizing correction:
angle_stabilization = α_angle · (1 - θ_error_normalized) · dt
# Net effect:
tq -= (range_vel_degradation - angle_stabilization)
```

RGPO and VGPO alone has minimal if any effect.

Sweep mode: Cued tracker, no sweep. Continuous dwell on target. Dwell = 1.0.
