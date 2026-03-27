


Multiple ML models under different directories
Only best trained agent not all
Pass type of model to train (done), PPO etc
Pass type of neural network (not done) - all MLP right now

Issues:
    + PPO training has catastrophic forgetting R=+2 -> R=-5 permanently
    + Whether to use training bootstrap hyperparameter or not
        + This either caused poor training performance (not sure why), NaN
          values (figured out action space was (-inf, inf) or was getting
          overwritten in runtime - hardcoded this now


Debug_viz helped solve problem:
    + https://github.com/isaac-sim/IsaacLab/discussions/2516
    + Probably spent ~5 hours trying to fix this s...
    + Still don't know what problem is but its disappeared for now



Problems:
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


0. DRFM and Radar glue
    + Threat library matches fingerprint against mission data files
    + RWR measures signal parameters (PRF, Fc, Pulse width, scan pattern,
      polarization) then matches to threat library.
    + RWR can distinguish between illumination beam and high-PRF pencil beam


0. Training
    + How? ( Using Alex's comments )
        - Train both flying and DRFM together
        - Use scaffolding: idea of starting off eacy (training wheels) and
          gradually adding complexity helping the model learn. Time decaying.
        - Avoid catastrophic forgetting but not re-training
        - Avoid gradual learning because moving loss function prevents finding
          local extrema


1. Define what type of radars are in use
    + How do they operate deterministic or stochastic?
        - Deterministisc
    + How do they exactly operate?
        - Search -> Detect -> Track -> Lock -> Kill
    + Do all radars have different parameters?
        - They have to, to allow drone to trigger different movement & DRFM
    + What kind of radars?
        - Pulse Doppler, Search/acq., monopulse


2. Define DRFM
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


3. Drone Maneuverability:
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
