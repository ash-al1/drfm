# Radar Types

We chose to use deterministic radars, there are many kinds and have different
pros/cons. We cannot use a single type of radar across the environment because
we have to urge the drone to use different techniques. Different techniques
handle different use cases.

Basic assumption:
1. No radar redundancy
2. Each DRFM Technique works against some, not all radars
3. In the case of a strong technique (RVGPO) limit using power consumption sim

Thus:
- RGPO vs. search/acq.
- VGPO vs. pulse doppler
- RVGPO vs. monopulse


## Pulse Doppler

Use Doppler effect to find target objects velocity, distance based of echo'd
radio signal. Doppler frequency: (2 x fc x radial velocity) / C. As reflector
move, phase shift occurs from pulse to pulse consequently Doppler modulation. So
1. Range -> from time delay
2. Radial velocity -> from doppler shifts

Radar has a range and velocity gate, cross-checks for consistency so RGPO
doesn't work. VGPO does work.

## Search/Acquisition

Non-coherent. No Doppler. Fixed PRF. Transmit and listens to pulses, detection
based on power exceeding a threshold. RGPO works here. So basic radar equation is used here:
$$P_r = \frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 R^4}$$

## Monopulse

Uses sum and difference antenna pattern on every pulse. Calculates angle error
based on real values of delta/sum. RVGPO partially breaks, degrades range and
velocity gates enough that it cannot track. 

# References

1. [Electronic Counter-countermeasure](https://en.wikipedia.org/wiki/Electronic_counter-countermeasure)
1. [Pulse-Doppler Radar](https://en.wikipedia.org/wiki/Pulse-Doppler_radar)
1. [List of radar types](https://en.wikipedia.org/wiki/List_of_radar_types)
1. [Monopulse radar](https://en.wikipedia.org/wiki/Monopulse_radar)
