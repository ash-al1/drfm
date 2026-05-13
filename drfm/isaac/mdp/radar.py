from __future__ import annotations
import math
import torch
from torch import Tensor

C = 3e8

# Per-radar constants  (index: 0=Search/Acq  1=Pulse-Doppler  2=Monopulse)
DET_RANGE  = [1500.0,  800.0,  600.0]   # detection range (m)
FC         = [3e9,    10e9,   10e9]
LAMBDA     = [C / f for f in FC]         # wavelength (m)
GATE_WIDTH = [200.0,  100.0,  100.0]    # range gate (m)
VEL_GATE   = [  0.0,   15.0,   10.0]   # velocity gate (m/s); 0 = no gate

# Dwell fraction [radar, state]  states: 0=Search 1=Detect 2=Track 3=Lock
# Once a radar detects something it commits more beam time to that bearing.
# Monopulse has no scan in Search/Detect - only tracks when cued by SAcq.
#
#              Search  Detect  Track  Lock
DWELL_FRAC = [
    [0.03,   0.30,   0.50,  1.00],  # Search/Acq
    [0.10,   0.40,   0.80,  1.00],  # Pulse-Doppler
    [0.00,   0.00,   1.00,  1.00],  # Monopulse
]

# track_quality accumulation alpha [radar, state]
ALPHA_ACC = [
    [0.00, 0.15, 0.25, 0.25],
    [0.00, 0.20, 0.35, 0.35],
    [0.00, 0.00, 0.40, 0.40],
]

BLEED_RATE = [0.02, 0.03, 0.01]  # passive track_quality loss/s when beam is off

# DRFM degradation alpha [technique, radar]
# technique: 0=OFF  1=RGPO  2=VGPO  3=RVGPO
# PD-RGPO weak (0.03): consistency check baked in
# Mono-RVGPO (0.25): raw value; angle stabilisation (0.12) applied live
DRFM_ALPHA = [
    [0.00, 0.00, 0.00],
    [0.40, 0.03, 0.04],
    [0.02, 0.35, 0.04],
    [0.15, 0.50, 0.25],
]

RGPO_PLAUS_MAX = 300.0;  RGPO_FALLOFF = 50.0   # m/s
VGPO_PLAUS_MAX = 150.0;  VGPO_FALLOFF = 30.0   # m/s²
ALPHA_ANGLE_MONO = 0.12                          # monopulse angle-channel stabilisation /s
ESM_RANGE = 2000.0                               # passive ESM detection range (m)

# State machine thresholds
TQ_DETECT_THRESH  = 0.40
TQ_LOCK_THRESH    = 0.85
TQ_BREAK_THRESH   = 0.20
TQ_LOST_THRESH    = 0.05
TQ_MONO_CUE_INIT  = 0.20
TQ_MONO_LOSE      = 0.15
CONSEC_MISS_THRESH = 5

SEARCH = 0;  DETECT = 1;  TRACK = 2;  LOCK = 3

RADAR_NAMES     = ["SAcq", "PD", "Mono"]
STATE_NAMES     = ["Search", "Detect", "Track", "LOCK"]
TECHNIQUE_NAMES = ["OFF", "RGPO", "VGPO", "RVGPO"]

# Beam model constants
BEAM_WIDTH  = [0.3, 0.15, 0.08]    # half-power beamwidth (rad) per radar type
SWEEP_SPEED = [1.5, 1.0,  0.5]     # base sweep rate (rad/s) scaled by dwell_frac

# SNR-based detection parameters
SNR_THRESHOLD   = 0.5
SNR_TEMPERATURE = 0.2

# RWR sliding window length (steps)
WINDOW = 10


class RadarManager:
    """Three radar state machines batched over N environments. All tensors [N, 3]."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        radar_positions: list,
        obstacles: list[tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.device   = device
        self.positions = torch.tensor([list(p) for p in radar_positions],
                                       dtype=torch.float32, device=device)  # [3, 3]

        self._det_range  = torch.tensor(DET_RANGE,  device=device)
        self._lambda     = torch.tensor(LAMBDA,     device=device)
        self._gate_width = torch.tensor(GATE_WIDTH, device=device)
        self._vel_gate   = torch.tensor(VEL_GATE,   device=device)
        self._dwell_frac = torch.tensor(DWELL_FRAC, device=device)  # [3, 4]
        self._alpha_acc  = torch.tensor(ALPHA_ACC,  device=device)  # [3, 4]
        self._bleed_rate = torch.tensor(BLEED_RATE, device=device)
        self._drfm_alpha = torch.tensor(DRFM_ALPHA, device=device)  # [4, 3]
        self._ridx       = torch.arange(3, device=device).unsqueeze(0)  # [1, 3]
        self._beam_width  = torch.tensor(BEAM_WIDTH,  dtype=torch.float32, device=device)
        self._sweep_speed = torch.tensor(SWEEP_SPEED, dtype=torch.float32, device=device)

        self.state            = torch.zeros(num_envs, 3, dtype=torch.long,    device=device)
        self.track_quality    = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.delta_t          = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.delta_f          = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.scan_det_count   = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.scan_total_count = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.consec_misses    = torch.zeros(num_envs, 3, dtype=torch.long,    device=device)
        self.esm_triggered    = torch.zeros(num_envs, 3, dtype=torch.bool,    device=device)

        # Beam state for realistic emission model
        self._beam_azimuth = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self._beam_history = torch.zeros(num_envs, 3, WINDOW, dtype=torch.bool, device=device)
        self._beam_ptr     = 0   # ring buffer write index

        # 2D axis-aligned obstacle rectangles for LOS checks: [M, 4] cx, cy, hw, hh
        if obstacles:
            self._obstacles = torch.tensor(obstacles, dtype=torch.float32, device=device)
        else:
            self._obstacles = torch.zeros(0, 4, dtype=torch.float32, device=device)

    @property
    def any_locked(self) -> Tensor:
        return (self.state == LOCK).any(dim=1)

    def reset(self, env_ids: Tensor) -> None:
        self.state[env_ids]            = SEARCH
        self.track_quality[env_ids]    = 0.0
        self.delta_t[env_ids]          = 0.0
        self.delta_f[env_ids]          = 0.0
        self.scan_det_count[env_ids]   = 0.0
        self.scan_total_count[env_ids] = 0.0
        self.consec_misses[env_ids]    = 0
        self.esm_triggered[env_ids]    = False
        self._beam_azimuth[env_ids]    = 0.0
        self._beam_history[env_ids]    = False

    def _check_los(self, drone_pos: Tensor) -> Tensor:
        """Return los_blocked [N, 3] bool. True when an obstacle occludes the drone-to-radar segment.

        Uses slab intersection in 2D (x, y only).  Loops over M obstacles; M is small (≤8).
        """
        N = self.num_envs
        M = self._obstacles.shape[0]
        if M == 0:
            return torch.zeros(N, 3, dtype=torch.bool, device=self.device)

        d_xy  = drone_pos[:, :2]                           # [N, 2]
        r_xy  = self.positions[:, :2]                      # [3, 2]

        # Segment: P(t) = d_xy + t * (r_xy - d_xy),  t in [0, 1]
        # We check for each (env, radar) pair against each obstacle slab.
        seg   = r_xy.unsqueeze(0) - d_xy.unsqueeze(1)     # [N, 3, 2]  direction

        # Obstacle slabs: [M, 2] for min/max in x and y
        obs_min = self._obstacles[:, :2] - self._obstacles[:, 2:]   # [M, 2]  cx-hw, cy-hh
        obs_max = self._obstacles[:, :2] + self._obstacles[:, 2:]   # [M, 2]  cx+hw, cy+hh

        # Expand for broadcasting: [N, 3, M, 2]
        o_min = obs_min.unsqueeze(0).unsqueeze(0)   # [1, 1, M, 2]
        o_max = obs_max.unsqueeze(0).unsqueeze(0)   # [1, 1, M, 2]

        d  = d_xy.unsqueeze(1).unsqueeze(2)         # [N, 1, 1, 2]
        s  = seg.unsqueeze(2)                        # [N, 3, 1, 2]

        # Parametric slab intersection along each axis
        # Avoid division by zero for axis-aligned segments
        inv_s  = torch.where(s.abs() > 1e-9, 1.0 / s, torch.full_like(s, 1e9))
        t_min  = (o_min - d) * inv_s                # [N, 3, M, 2]
        t_max  = (o_max - d) * inv_s                # [N, 3, M, 2]

        t_enter = torch.min(t_min, t_max).amax(dim=-1)   # [N, 3, M]  max of per-axis enters
        t_exit  = torch.max(t_min, t_max).amin(dim=-1)   # [N, 3, M]  min of per-axis exits

        # Intersection if t_enter < t_exit and the interval overlaps [0, 1]
        hit = (t_enter < t_exit) & (t_exit > 0.0) & (t_enter < 1.0)   # [N, 3, M]
        los_blocked = hit.any(dim=-1)                                    # [N, 3]
        return los_blocked

    def update(
        self,
        drone_pos:          Tensor,   # [N, 3] env-local XYZ
        drone_vel:          Tensor,   # [N, 3] m/s
        technique:          Tensor,   # [N] int
        pull_off_rate:      Tensor,   # [N] m/s
        vel_pull_off_rate:  Tensor,   # [N] m/s²
        coordination_ratio: Tensor,   # [N] [0, 1]
        technique_switched: Tensor,   # [N] bool
        dt:                 float,
    ) -> None:
        N = self.num_envs
        radar_indices = self._ridx.expand(N, -1)  # [N, 3]

        # Reset Δt/Δf on technique switch
        technique_switched_expanded = technique_switched.unsqueeze(1).expand(-1, 3)
        self.delta_t = torch.where(technique_switched_expanded, torch.zeros_like(self.delta_t), self.delta_t)
        self.delta_f = torch.where(technique_switched_expanded, torch.zeros_like(self.delta_f), self.delta_f)

        # Geometry
        relative_pos = drone_pos.unsqueeze(1) - self.positions.unsqueeze(0)   # [N, 3, 3]
        ranges       = relative_pos.norm(dim=2).clamp(min=0.1)                # [N, 3]

        # ESM: any jamming active → cue Search-state radars to Detect (once per episode)
        jamming_on = (technique != 0).unsqueeze(1).expand(-1, 3)
        can_esm = jamming_on & (self.state == SEARCH) & ~self.esm_triggered & (ranges < ESM_RANGE)
        self.state            = torch.where(can_esm, torch.full_like(self.state, DETECT), self.state)
        self.track_quality    = torch.where(can_esm, torch.full_like(self.track_quality, 0.1), self.track_quality)
        self.esm_triggered    = self.esm_triggered | can_esm

        # Beam model: advance azimuth and compute gain at target angle
        dwell_frac = self._dwell_frac[radar_indices, self.state.clamp(0, 3)]   # [N, 3]
        sweep      = self._sweep_speed.unsqueeze(0) * dwell_frac * dt           # [N, 3]
        self._beam_azimuth = self._beam_azimuth + sweep
        self._beam_azimuth = torch.atan2(
            torch.sin(self._beam_azimuth), torch.cos(self._beam_azimuth)
        )

        target_az  = torch.atan2(relative_pos[:, :, 1], relative_pos[:, :, 0])  # [N, 3]
        angle_diff = target_az - self._beam_azimuth
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))   # wrap
        sinc_arg   = angle_diff * (math.pi / self._beam_width.unsqueeze(0))      # [N, 3]
        # sinc²: handle near-zero case to avoid 0/0
        beam_gain  = torch.where(
            sinc_arg.abs() < 1e-4,
            torch.ones_like(sinc_arg),
            (torch.sin(sinc_arg) / sinc_arg) ** 2,
        )

        # LOS check: blocked segments cannot detect
        los_blocked = self._check_los(drone_pos)   # [N, 3]

        # Probabilistic detection: P(detect) = sigmoid((SNR - threshold) / temperature)
        # SNR normalised so SNR=1 at det_range with beam_gain=1
        snr      = ((self._det_range.unsqueeze(0) / ranges) ** 2 * beam_gain).clamp(0.0, 5.0)
        snr_noise = torch.randn(N, 3, device=self.device) * 0.1
        p_detect  = torch.sigmoid((snr + snr_noise - SNR_THRESHOLD) / SNR_TEMPERATURE)
        p_detect  = p_detect * (~los_blocked).float()   # zero detection probability when blocked
        beam_on   = torch.bernoulli(p_detect).bool()

        # Detection event (beam on AND in range implied by SNR model; keep in_range for ESM/mono)
        in_range = ranges < self._det_range.unsqueeze(0)
        detected = beam_on & in_range

        # 3-in-5 dwell rule (Search state only) → transition to Detect
        in_search = (self.state == SEARCH)
        self.scan_total_count += beam_on.float() * in_search.float()
        self.scan_det_count   += detected.float() * in_search.float()
        window_done    = (self.scan_total_count >= 5) & in_search
        search_to_det  = window_done & (self.scan_det_count >= 3)
        self.scan_total_count = torch.where(window_done, torch.zeros_like(self.scan_total_count), self.scan_total_count)
        self.scan_det_count   = torch.where(window_done, torch.zeros_like(self.scan_det_count),   self.scan_det_count)
        self.state            = torch.where(search_to_det, torch.full_like(self.state, DETECT), self.state)
        self.track_quality    = torch.where(search_to_det, torch.full_like(self.track_quality, 0.1), self.track_quality)

        # Consecutive miss tracking (for Detect→Search fallback)
        in_dt = (self.state == DETECT) | (self.state == TRACK)
        self.consec_misses = torch.where(beam_on & in_dt, torch.zeros_like(self.consec_misses), self.consec_misses)
        self.consec_misses = self.consec_misses + (~beam_on & in_dt).long()

        # track_quality accumulation and passive bleed (with Gaussian noise σ=0.05)
        alpha_acc = self._alpha_acc[radar_indices, self.state.clamp(0, 3)]
        tq_noise  = torch.randn(N, 3, device=self.device) * 0.05
        self.track_quality = torch.where(
            beam_on & in_dt,
            (self.track_quality + alpha_acc * (1.0 - self.track_quality) * dt + tq_noise * dt).clamp(0.0, 1.0),
            self.track_quality,
        )
        self.track_quality = torch.where(
            ~beam_on & in_dt,
            (self.track_quality - self._bleed_rate.unsqueeze(0) * dt).clamp(min=0.0),
            self.track_quality,
        )

        # Δt/Δf accumulation
        technique_expanded       = technique.unsqueeze(1).expand(-1, 3)
        pull_off_rate_expanded   = pull_off_rate.unsqueeze(1).expand(-1, 3)
        vel_pull_off_rate_expanded = vel_pull_off_rate.unsqueeze(1).expand(-1, 3)

        do_range = (technique_expanded == 1) | (technique_expanded == 3)
        do_vel   = (technique_expanded == 2) | (technique_expanded == 3)
        is_off   = (technique_expanded == 0)

        self.delta_t = torch.where(do_range, self.delta_t + (2.0 * pull_off_rate_expanded * dt) / C, self.delta_t)
        self.delta_f = torch.where(do_vel,   self.delta_f + (2.0 * vel_pull_off_rate_expanded * dt) / self._lambda.unsqueeze(0), self.delta_f)
        self.delta_t = torch.where(is_off, torch.zeros_like(self.delta_t), self.delta_t)
        self.delta_f = torch.where(is_off, torch.zeros_like(self.delta_f), self.delta_f)

        # DRFM tq degradation
        drfm_active = beam_on & in_dt & ~is_off

        range_diff    = C * self.delta_t / 2.0
        range_capture = (range_diff / self._gate_width.unsqueeze(0)).clamp(0.0, 1.0)

        vel_diff    = self.delta_f * self._lambda.unsqueeze(0) / 2.0
        vel_capture = (vel_diff / self._vel_gate.unsqueeze(0).clamp(min=1.0)).clamp(0.0, 1.0)
        vel_capture = torch.where((self._vel_gate == 0.0).unsqueeze(0), torch.zeros_like(vel_capture), vel_capture)

        # Plausibility clamp: exponential penalty above the believable max pull-off rate
        range_capture = range_capture * torch.exp(-((pull_off_rate_expanded     - RGPO_PLAUS_MAX).clamp(min=0.0)) / RGPO_FALLOFF)
        vel_capture   = vel_capture   * torch.exp(-((vel_pull_off_rate_expanded - VGPO_PLAUS_MAX).clamp(min=0.0)) / VGPO_FALLOFF)

        capture = torch.where(technique_expanded == 1, range_capture,                        torch.zeros_like(range_capture))
        capture = torch.where(technique_expanded == 2, vel_capture,                           capture)
        capture = torch.where(technique_expanded == 3, (range_capture + vel_capture) / 2.0,   capture)

        # Scales DRFM effectiveness by coordination quality; only meaningful for RVGPO (technique==3)
        param_quality_factor = torch.where(technique == 3, coordination_ratio.clamp(0.0, 1.0), torch.ones(N, device=self.device))

        drfm_alpha  = self._drfm_alpha[technique.long()]              # [N, 3]
        tq_loss     = drfm_alpha * capture * param_quality_factor.unsqueeze(1).expand(-1, 3) * dt

        # Monopulse angle stabilisation partially resists RVGPO
        rvgpo_env = (technique == 3)
        tq_loss[:, 2] = torch.where(rvgpo_env,
                                     (tq_loss[:, 2] - ALPHA_ANGLE_MONO * dt).clamp(min=0.0),
                                     tq_loss[:, 2])

        self.track_quality = torch.where(drfm_active, (self.track_quality - tq_loss).clamp(min=0.0), self.track_quality)
        self.track_quality = self.track_quality.clamp(0.0, 1.0)

        # Monopulse cueing: SAcq enters Detect + drone in Mono range → Mono Track
        sacq_in_detect   = (self.state[:, 0] == DETECT)
        mono_in_range    = ranges[:, 2] < self._det_range[2]
        mono_untracked   = (self.state[:, 2] == SEARCH)
        should_cue       = sacq_in_detect & mono_in_range & mono_untracked
        self.state[:, 2] = torch.where(should_cue, torch.full((N,), TRACK,            dtype=torch.long, device=self.device), self.state[:, 2])
        self.track_quality[:, 2] = torch.where(should_cue, torch.full((N,), TQ_MONO_CUE_INIT, device=self.device),           self.track_quality[:, 2])

        # Monopulse falls back if track_quality drops too low
        mono_lose        = (self.state[:, 2] == TRACK) & (self.track_quality[:, 2] < TQ_MONO_LOSE)
        self.state[:, 2] = torch.where(mono_lose, torch.full((N,), SEARCH, dtype=torch.long, device=self.device), self.state[:, 2])

        # State machine transitions with per-env threshold noise (σ=0.05)
        thresh_noise = torch.randn(N, 3, device=self.device) * 0.05
        tq_detect_thresh = TQ_DETECT_THRESH + thresh_noise
        tq_lock_thresh   = TQ_LOCK_THRESH   + thresh_noise
        tq_break_thresh  = TQ_BREAK_THRESH  + thresh_noise

        in_detect = (self.state == DETECT)
        in_track  = (self.state == TRACK)
        self.state = torch.where(in_detect & (self.track_quality > tq_detect_thresh), torch.full_like(self.state, TRACK),  self.state)
        self.state = torch.where(in_track  & (self.track_quality > tq_lock_thresh),   torch.full_like(self.state, LOCK),   self.state)
        self.state = torch.where(in_track  & (self.track_quality < tq_break_thresh),  torch.full_like(self.state, DETECT), self.state)

        lost = in_detect & ((self.track_quality < TQ_LOST_THRESH) | (self.consec_misses >= CONSEC_MISS_THRESH))
        self.state            = torch.where(lost, torch.full_like(self.state, SEARCH),             self.state)
        self.track_quality    = torch.where(lost, torch.zeros_like(self.track_quality),            self.track_quality)
        self.scan_det_count   = torch.where(lost, torch.zeros_like(self.scan_det_count),           self.scan_det_count)
        self.scan_total_count = torch.where(lost, torch.zeros_like(self.scan_total_count),         self.scan_total_count)
        self.consec_misses    = torch.where(lost, torch.zeros_like(self.consec_misses),            self.consec_misses)

        # Update beam history ring buffer
        self._beam_history[:, :, self._beam_ptr] = beam_on
        self._beam_ptr = (self._beam_ptr + 1) % WINDOW

    def get_rwr_observations(self, drone_pos: Tensor, drone_quat: Tensor) -> Tensor:
        """Realistic RWR observations, shape [N, 32] (4 emitter slots × 8 dims).
        Slots 0-2 correspond to the three radars; slot 3 is zero-padded."""
        N   = self.num_envs
        obs = torch.zeros(N, 32, device=self.device)

        relative_pos = drone_pos.unsqueeze(1) - self.positions.unsqueeze(0)   # [N, 3, 3]
        ranges       = relative_pos.norm(dim=2).clamp(min=0.1)               # [N, 3]
        los_blocked  = self._check_los(drone_pos)                             # [N, 3]

        qw, qx, qy, qz = drone_quat[:, 0], drone_quat[:, 1], drone_quat[:, 2], drone_quat[:, 3]
        drone_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Build true-type logits once: identity matrix scaled so correct type wins
        # type_logits[i] is a [3] vector with logit 3.0 at index i, 0.0 elsewhere
        type_logits = torch.eye(3, device=self.device) * 3.0  # [3, 3]

        for i in range(3):
            b = i * 8

            # Bearing with 5 deg Gaussian noise (sigma=0.087 rad), normalised to [-1, 1]
            bearing = torch.atan2(relative_pos[:, i, 1], relative_pos[:, i, 0]) - drone_yaw
            bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
            bearing_noise = torch.randn(N, device=self.device) * 0.087
            obs[:, b] = (bearing + bearing_noise) / math.pi

            # Received power: one-way link ∝ 1/R², normalised to [0, 1].
            # Value is 1.0 at R = det_range/2, 0.25 at R = det_range.
            # Attenuate by 0.1 when LOS is blocked (signal leaks around edges).
            rx_norm  = ((self._det_range[i] / ranges[:, i]) ** 2).clamp(0.0, 4.0) / 4.0
            los_atten = torch.where(los_blocked[:, i], torch.full_like(rx_norm, 0.1), torch.ones_like(rx_norm))
            rx_noise = torch.randn(N, device=self.device) * 0.05
            obs[:, b + 1] = (rx_norm * los_atten + rx_noise).clamp(0.0, 1.0)

            # Illumination rate: fraction of last WINDOW steps with beam_on  [0, 1]
            illum_rate = self._beam_history[:, i, :].float().mean(dim=-1)   # [N]
            obs[:, b + 2] = illum_rate

            # Pulse-interval variance: empirical variance of beam_on over the window,
            # normalised by maximum Bernoulli variance (0.25)
            bh  = self._beam_history[:, i, :].float()                        # [N, WINDOW]
            piv = bh.var(dim=-1) / 0.25                                       # [N]
            obs[:, b + 3] = piv.clamp(0.0, 1.0)

            # Frequency-band soft classification: softmax over true logits + noise
            logits_i = type_logits[i].unsqueeze(0).expand(N, -1)             # [N, 3]
            freq_noise = torch.randn(N, 3, device=self.device) * 0.5
            obs[:, b + 4 : b + 7] = torch.softmax(logits_i + freq_noise, dim=-1)

            # Trend: delta(illumination_rate) between first and second half of window
            half = WINDOW // 2
            illum_old = self._beam_history[:, i, :half].float().mean(dim=-1)
            illum_new = self._beam_history[:, i, half:].float().mean(dim=-1)
            obs[:, b + 7] = (illum_new - illum_old).clamp(-1.0, 1.0)

        # Slot 3 (indices 24-31) stays zero-padded (future emitter slot)
        return obs

    def debug_state(self, env_id: int, radar_idx: int) -> str:
        """Single radar: 'Track 0.72'"""
        state_idx    = int(self.state[env_id, radar_idx].item())
        track_quality = self.track_quality[env_id, radar_idx].item()
        return f"{STATE_NAMES[state_idx]:<6s} {track_quality:.2f}"

    def debug_string(self, env_id: int = 0) -> str:
        return "  ".join(
            f"{RADAR_NAMES[i]}:{self.debug_state(env_id, i)}" for i in range(3)
        )
