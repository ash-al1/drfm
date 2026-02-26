---                                                                                 

# Tasks:                                                                            
+ Reverse engineer drfm github [repository](https://github.com/mesarcik/DRFM)
+ Create simple grid, drone and radar objects in Isaac sim
    + Script a flight path, try and pull entity poses can be exported per
      timestep
+ Build a simple gymnasium wrapper environment mockup
    + Define observation space (pos, J/s ratio) and discrete action space
+ Use Isaac sim to export drone pos/vel. emitter location, LoS or not. Feed
  data into RF model, compute J/S and detection. Echo, Jamming power,J/S ratio,
  etc. Agent receives J/S ratio and detection status as observation vector, then
  calculates reward based off agent perforamnce. +1 reward every timestep alive,
  maybe bonus if radar loses track, penalties for being detected and locked on.
+ Tabular RL Q-learning model implementation to test a simple agent can learn to
  survive. Agents actions could be to select DRFM module parameters.

---

# Read
+ [Drone Racing](https://www.nature.com/articles/s41586-023-06419-4.pdf)

---

# Ideas
+ Pegasus Simulator?
+ GNU Radio connector for RF backend?
+ FPGA usb 3.0 connected, isolated, latency w.r.t. simulator: handles DRFM
  compute isolated from simulator RF backend & drone robotics?

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

# Poor:

---

# Done:
+ Implement core RF (basic) modeling on a standalone python file
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

# People:

