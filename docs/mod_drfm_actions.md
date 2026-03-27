
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
	+ Sen. = $$ k \, T_{0} \, B_{I} \, (\text{NF}) \, \text{SNR}_{Ii} $$

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
$$P_{rT} = \frac{P_r G_T^2 \sigma_T \lambda^2}{(4\pi)^3 R^4}$$

DRFM transponder power:
$$P_{rj} = \frac{P_j G_j G_r \lambda^2}{(4\pi R)^2}$$

First-order approximation to JSR:
$$\frac{J}{S} = \frac{P_j G_j 4\pi R^2}{P_r G_r \sigma_T}$$


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

- Power at radar from the DRFM jammer is (Poynting vector x antenna effective
aperture x EA jammer signal): P_rj = $$\left( \frac{P_r G_r}{4\pi R^2} \right) \left( \frac{G_r \lambda^2}{4\pi} \right) \left( \frac{G_{jrcv} G G_{jxmt} \lambda^2}{4\pi R^2} \right)$$

- Power at radar from RCS: P_rT = $$ \left( \frac{P_r G_r}{4\pi R^2} \right) \left( \frac{G_r \lambda^2}{4\pi} \right) \left( \frac{\sigma_T}{4\pi R^2} \right) $$

- JSR ratio is: P_rj / P_rT = J/S = $$ \left( \frac{\lambda^2}{4\pi} \right) \frac{G_{jrcv} G G_{jxmt}}{\sigma_T^2} $$

- Critically, range is not part of the equation anymore.

Information of an intercepted radar signal is carried in phase, not amp.

EA signal in quantized signal (Fourier transform domain): \hat{X}(f) = $$ \sum_{m=-\infty}^{+\infty} \operatorname{sinc}\left(m + \frac{1}{N}\right) \delta[f - (N m + 1)f_0] $$


### EA: Range Gate Pull-Off (RGPO)

RGPO implements a smooth time delay to walk a radar off its skin return.
JSR goes to infinity as a result.

Radar split gates centers using search information, balances signal energy
between early and late range gates.

Delay time smoothly (or in stair-step realistically) increases, the phase change
over time is: $$\Delta\phi = -2\pi \left( f_0 \beta - [2f_0\beta] + [f_0\beta] \right)$$

Its important to take into consideration rate of pull-off of the DRFM, if its
too quick then radar will clip back to skin return; typically either linear or
parabolic time delay generation.

Initially there is always a time delay at the start until DRFM starts to pull
off target, determined by "how fast a signal can be intercepted, detected,
classified and response called up to send back out the DRFM."

RGPO typically used with other techniques: RGPO-to-chaff, RGPO-to-decoy,
RGPO -> Cross polarization, RPGPO -> Swept Square Wave.

Hook pulse added to the end of RGPO to prevent radar from immediately locking
back when radar starts to initiate a search.


### EA: Velocity Gate Pull-Off (VGPO)

Modulates frequency or velocity (pulse Doppler, vel. is related to freq as range
is related to time). Freq. modulation is good vs narrowband filtering (Doppler
radar, LPI FSK).

Even betetr to pull radars tracking gate (in frequency) into clutter, also since
its freq. it can be pulled up/down.

### Coordinated RGPO-VGPO

Combination of RGPO and VGPO acts as a true target in range-Doppler space.

Can move skin return in range and filter bins. Combining them is useful against
counter-RGPO/VGPO; radar can compare both and if they dont align then it knows
its being deceived.

Sampling rate of ADC on DRFM needs to be higher than sampling rate of radar.

### SAR Active Decoy EA Technique

SAR use scorrelation processing, noise EA is not effective.

Range deception modulator can produce decoy target at delta x: $$H_n(\omega) = \sigma_n \exp(j 2 k \Delta x_n$$

Azimuth deception can produce a decoy target at delta y, a deception modulator
in azimuth can be represented as: $$H_n(\omega, u) = \sigma_n \exp\left( jk \frac{ (\Delta y)^2 - 2u\Delta y }{R} \right)$$

EA needs to know radar location at every moment for azimuth deception. The
azimuth deception modulator can be simplified as: $$H_n(\omega, \theta) = \sigma_n \exp(-j 2 k \Delta y \sin \theta)$$

## References

1. Pace, P. E. *Developing Digital RF Memories and Transceiver Technologies for Electromagnetic Warfare.* Artech House, 2022.
