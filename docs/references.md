# References

This file contains my summarization on research papers I read. This is not a
comprehensive references file as some of my notes are on hard paper, or littered
throughout many file ... or in my head.

---

- Poisson: Modeling rare or independent random events over fixed time/space
- Bernoulli: Modeling coin flips, does this pixel contain real target or no
- Markovian: Whats the future state given only current state (memoryless)
- Monte Carlo: Random simulations then approximate probability functions
- Bayesian: Start with initial belief (uncertainty) then update using new evidence

---

Pinto, Juliano, et al. "Next generation multitarget trackers: Random finite set
methods vs transformer-based deep learning." 2021 IEEE 24th International
Conference on Information Fusion (FUSION). IEEE, 2021.

MTT differs from normal CV because it has memory and attempts to ID targets
over time-axis because: systems that require MTT have high clutter/noise and
high pruning requirements think RADAR chaff, clutter, etc. Bayesian techniques
like PMBM and δ-GLMB (Poisson and Bernoulli). MTT has high dimensional
parameters, long term time tracking.

Environment is "standard multitarget transition and observation models for point
objects without target spawning". New processes arrive following some Poisson
point process and a λ birthing intensity others depart with iid markovian and
survival probability is p\_s(x). Clutter and object measurements independant of
all other objects. MT-estimation using moving window i.e. given τ time-steps.

Model: DETR style transformer encoder/decoder with deformable attention. Two 
output boxes: state and object classifiers. Selection mechanism architecture
with 2 FFNs output Top-K selection to better help decoder. Iterative refinement
(taken from original DETR) to help decoder output across multiple decoder
layers. Loss functions (with ablations) based on localization error and
probabilities of missed and false targets given best match between GT and Pred.
Generalization effort using constrastive aux learning creates a constrastive
loss L, maximize invariance.

---

Zhang, Chudi, et al. "Radar jamming decision-making in cognitive electronic
warfare: A review." IEEE Sensors Journal 23.11 (2023): 11383-11403.


---

Hiles, Alex, et al. "Jammer Versus Radar in a Cognitive Electronic Warfare
Environment." Authorea Preprints (2025).

---

Mesarcik, Michael B., Daniel W. O'Hagan, and Stephen Paine. "Low Cost FPGA based
Implementation of a DRFM System." 2019 IEEE Radar Conference (RadarConf). IEEE,
2019.

---

[SPJ](https://www.radartutorial.eu/16.eccm/ja11.en.html)

---

You, Shixun, Ming Diao, and Lipeng Gao. "Deep reinforcement learning for target
searching in cognitive electronic warfare." IEEE Access 7 (2019): 37432-37447.

---

Sequential Monte Carlo Cardinalized probability hypothesized density filter
based on Track-Before-Detect for fluctuating targets in heavy-tailed clutter

Baysian filtering in heavy-tailed clutter environment. Heavy mathematics,
frankly didn't read much besides a quick pass this is way over my head -
although fundamentally explains the kind of environment and method of problem
solving traditional and existing methods use. Heavy-tailed clutter
(k-distribution and G0-distribution instead of Rayeligh), and fluctuating
targets (swerling types 1 and 3 models). Track-before-detect operates on raw
signal data before threshold, building confidence over multiple frames before
declaring a target exists. Papers often do this kind of thing, where confidence
is the measure they use to build belief whether a target exists or not, because
RDR is extremely noise heavy.

---

A robust multiple target tracking framework with Transformer-based data
association and dynamic trajectory management in challenging environments

Cross transformer with data association within traditional tracking (BLUE and
trajectory management). Attention gives features from position only measurement
as opposed to angle of arrival or rough position estimates, not full state
vectors. Some mechanism in place that handles targets appearing and
disappearing, losing sight etc. More realistic that what the project aims to be
in its infancy.
