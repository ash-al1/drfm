---                                                                                 

# Tasks:                                                                            
+ Implement basic GNU Radio DRFM with minimal constraints
+ Reverse engineer drfm github [repository](https://github.com/mesarcik/DRFM)
+ Add resume flag to resume training given a model
+ Pass policy/value iter as argument
+ Pass any subtask as argument instead of hardcoding to CartPole
+ Soon:
    + Monte carlo
    + TD Methods TD(n) TD(λ)
    + SARSA
    + Q-Learn
    + Exploration

---

# Read
+ [Drone Racing](https://www.nature.com/articles/s41586-023-06419-4.pdf)

---

# Qs:
+ Should I focus on theoretical guarantees? This devolves the project down into
  reading theoretical books on digital signal processing, wireless
  communications and other books (radar) for the purpose of extracting
  mathematical theorems and algorithms beneficial specifically for those signals
  and appending them into layers at feature extraction points or other
  mathematically attractive positions. -> All this to say, is this really good
  enough for the project?
+ 

---

# Goal:
+ DRFM + RL with robotics; flesh out main idea soon


---

# Poor:

---

# Done:
+ Agent wrapper
+ Expose Q-value in libmdp
+ Map continuous to map discrete values
+ Port libmdp and libsparse over from mini projects

---

# People:

