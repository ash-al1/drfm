
# DRFM:

This markdown file contains information I researched while going through
"Developing Digital RF Memories and Transceiver Technologies for Electromagnetic
Warfare", by Phillip E. Pace. It contains high level information on how DRFM
modules work, the "action space", what the module practically does.

---

## Observation:
+ Many players on the battlefield, every part is a piece of the puzzle
	+ joint sensor fusion provides full picture
+ High third-order-intercept (TOI) of +15 dBm
	+ resistance against parasitic RF signals and EA
+ Low noise figure (< 3 dB)
	+ evaluate weak RF signals
	+ determines range of observation
+ sensitivity:
	+ boltzman constant, ambient temperature, input bandwidth to rx, noise
	factor, input required signal to noise ratio of intercepting rxr
	+ linear power amp. often needed, but log-amplifiers used for video input
	+ Sen. = k x T_{0} x B_{I} x (NF) x SNR_{Ii}

---

## Orientation:

+ state of spectrum (SOS)
+ automatic detection of available channels for wireless transmission
+ track frequency contacts -> maintain a codebook/track files
	+ how does each track file change over time
	+ attach real world targets to states
+ kinematic states -> without care for cause, changing motion of mechanics
	+ stochastic/probabilistic/deterministic signal, and actual geolocation too

---

##  Action space:

### Transponder False Targeting

DRFM transponder mode for false target generation.
Surface-to-air EA for deception against tactical fire control airborne radar:
	+ Real Beam Ground Map (RBGM)
	+ Radar Navigation Ground Map (RNGM)
	+ Doppler Beam Sharpening (DBS)
	+ SAR, ISAR

How: trigger threshold
+ Rx -> magnitude detection -> save in memory -> Tx if above mag. threshold
+ Tx'd are amplified, identical pulses
+ Considered a "constant power false target generator"
+ Gate off our rxr while tx'ing to not overwhelm


Skin return power:
$$P_{rT} = \frac{P_r G_T^2 \sigma_T \lambda^2}{(4\pi)^3 R^4} \quad \text{(10.38)}$$

DRFM transponder power:
$$P_{rj} = \frac{P_j G_j G_r \lambda^2}{(4\pi R)^2} \quad \text{(10.39)}$$

First-order approximation to JSR:
$$\frac{J}{S} = \frac{P_j G_j 4\pi R^2}{P_r G_r \sigma_T} \quad \text{(10.40)}$$


RGBM signal processing:
+ ADC -> AGC -> Pulse compression -> Ripple suppression -> Mag. Detect -> Postdetection Integration -> Display Processing -> Display

RGBM: convert raw I/Q to magnitude, increase SNR with minimal change to
characteristic of the signal. Display is a ground backscatter, after RX we use
an AGC for even the display appearance across the map.
1. Find peak value over a set of pulses, used to set sensitivity of map
2. Pulse compression & ripple suppression used to increase SNR and minimize
   sidelobe structure
3. Complex signal -> magnitude detection
4. PDI to increase SNR before display


### Repeat False Targeting

Change received signal of radar or comm signal by modulating in time, frequency,
amplitude, etc of the Tx signal. DRFM repeater modeled as constant gain system.



### EA: Range Gate Pull-Off (RGPO)

### EA: Velocity Gate Pull-Off (VGPO)

### Coordinated RGPA-VGPO

### SAR Active Decoy EA Technique
