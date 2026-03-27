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
# Monopulse has no scan in Search/Detect — only tracks when cued by SAcq.
#
#              Search  Detect  Track  Lock
DWELL_FRAC = [
    [0.03,   0.30,   0.50,  1.00],  # Search/Acq
    [0.10,   0.40,   0.80,  1.00],  # Pulse-Doppler
    [0.00,   0.00,   1.00,  1.00],  # Monopulse
]

# tq accumulation alpha [radar, state]
ALPHA_ACC = [
    [0.00, 0.15, 0.25, 0.25],
    [0.00, 0.20, 0.35, 0.35],
    [0.00, 0.00, 0.40, 0.40],
]

BLEED_RATE = [0.02, 0.03, 0.01]  # passive tq loss/s when beam is off

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


class RadarManager:
    """Three radar state machines batched over N environments. All tensors [N, 3]."""

    def __init__(self, num_envs: int, device: str, radar_positions: list) -> None:
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

        self.state            = torch.zeros(num_envs, 3, dtype=torch.long,    device=device)
        self.tq               = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.delta_t          = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.delta_f          = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.scan_det_count   = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.scan_total_count = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.consec_misses    = torch.zeros(num_envs, 3, dtype=torch.long,    device=device)
        self.esm_triggered    = torch.zeros(num_envs, 3, dtype=torch.bool,    device=device)

    @property
    def any_locked(self) -> Tensor:
        return (self.state == LOCK).any(dim=1)

    def reset(self, env_ids: Tensor) -> None:
        self.state[env_ids]            = SEARCH
        self.tq[env_ids]               = 0.0
        self.delta_t[env_ids]          = 0.0
        self.delta_f[env_ids]          = 0.0
        self.scan_det_count[env_ids]   = 0.0
        self.scan_total_count[env_ids] = 0.0
        self.consec_misses[env_ids]    = 0
        self.esm_triggered[env_ids]    = False

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
        ridx = self._ridx.expand(N, -1)  # [N, 3]

        # Reset Δt/Δf on technique switch
        sw3 = technique_switched.unsqueeze(1).expand(-1, 3)
        self.delta_t = torch.where(sw3, torch.zeros_like(self.delta_t), self.delta_t)
        self.delta_f = torch.where(sw3, torch.zeros_like(self.delta_f), self.delta_f)

        # Geometry
        rel = drone_pos.unsqueeze(1) - self.positions.unsqueeze(0)   # [N, 3, 3]
        R   = rel.norm(dim=2).clamp(min=0.1)                         # [N, 3]

        # ESM: any jamming active → cue Search-state radars to Detect (once per episode)
        jamming_on = (technique != 0).unsqueeze(1).expand(-1, 3)
        can_esm = jamming_on & (self.state == SEARCH) & ~self.esm_triggered & (R < ESM_RANGE)
        self.state         = torch.where(can_esm, torch.full_like(self.state, DETECT), self.state)
        self.tq            = torch.where(can_esm, torch.full_like(self.tq, 0.1), self.tq)
        self.esm_triggered = self.esm_triggered | can_esm

        # Dwell roll
        dwell_frac = self._dwell_frac[ridx, self.state.clamp(0, 3)]   # [N, 3]
        beam_on    = torch.rand(N, 3, device=self.device) < dwell_frac

        # Detection event
        in_range = R < self._det_range.unsqueeze(0)
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
        self.tq               = torch.where(search_to_det, torch.full_like(self.tq, 0.1), self.tq)

        # Consecutive miss tracking (for Detect→Search fallback)
        in_dt = (self.state == DETECT) | (self.state == TRACK)
        self.consec_misses = torch.where(beam_on & in_dt, torch.zeros_like(self.consec_misses), self.consec_misses)
        self.consec_misses = self.consec_misses + (~beam_on & in_dt).long()

        # tq accumulation (unimpeded when beam on) and passive bleed
        alpha_acc = self._alpha_acc[ridx, self.state.clamp(0, 3)]
        self.tq = torch.where(beam_on & in_dt, self.tq + alpha_acc * (1.0 - self.tq) * dt, self.tq)
        self.tq = torch.where(~beam_on & in_dt, (self.tq - self._bleed_rate.unsqueeze(0) * dt).clamp(min=0.0), self.tq)

        # Δt/Δf accumulation
        tech3 = technique.unsqueeze(1).expand(-1, 3)
        por3  = pull_off_rate.unsqueeze(1).expand(-1, 3)
        vpor3 = vel_pull_off_rate.unsqueeze(1).expand(-1, 3)

        do_range = (tech3 == 1) | (tech3 == 3)
        do_vel   = (tech3 == 2) | (tech3 == 3)
        is_off   = (tech3 == 0)

        self.delta_t = torch.where(do_range, self.delta_t + (2.0 * por3 * dt) / C, self.delta_t)
        self.delta_f = torch.where(do_vel,   self.delta_f + (2.0 * vpor3 * dt) / self._lambda.unsqueeze(0), self.delta_f)
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
        range_capture = range_capture * torch.exp(-((por3  - RGPO_PLAUS_MAX).clamp(min=0.0)) / RGPO_FALLOFF)
        vel_capture   = vel_capture   * torch.exp(-((vpor3 - VGPO_PLAUS_MAX).clamp(min=0.0)) / VGPO_FALLOFF)

        capture = torch.where(tech3 == 1, range_capture,                        torch.zeros_like(range_capture))
        capture = torch.where(tech3 == 2, vel_capture,                           capture)
        capture = torch.where(tech3 == 3, (range_capture + vel_capture) / 2.0,   capture)

        pqf = torch.where(technique == 3, coordination_ratio.clamp(0.0, 1.0), torch.ones(N, device=self.device))

        drfm_alpha  = self._drfm_alpha[technique.long()]              # [N, 3]
        tq_loss     = drfm_alpha * capture * pqf.unsqueeze(1).expand(-1, 3) * dt

        # Monopulse angle stabilisation partially resists RVGPO
        rvgpo_env = (technique == 3)
        tq_loss[:, 2] = torch.where(rvgpo_env,
                                     (tq_loss[:, 2] - ALPHA_ANGLE_MONO * dt).clamp(min=0.0),
                                     tq_loss[:, 2])

        self.tq = torch.where(drfm_active, (self.tq - tq_loss).clamp(min=0.0), self.tq)
        self.tq = self.tq.clamp(0.0, 1.0)

        # Monopulse cueing: SAcq enters Detect + drone in Mono range → Mono Track
        sacq_in_detect   = (self.state[:, 0] == DETECT)
        mono_in_range    = R[:, 2] < self._det_range[2]
        mono_untracked   = (self.state[:, 2] == SEARCH)
        should_cue       = sacq_in_detect & mono_in_range & mono_untracked
        self.state[:, 2] = torch.where(should_cue, torch.full((N,), TRACK,         dtype=torch.long, device=self.device), self.state[:, 2])
        self.tq[:, 2]    = torch.where(should_cue, torch.full((N,), TQ_MONO_CUE_INIT, device=self.device),                self.tq[:, 2])

        # Monopulse falls back if tq drops too low
        mono_lose        = (self.state[:, 2] == TRACK) & (self.tq[:, 2] < TQ_MONO_LOSE)
        self.state[:, 2] = torch.where(mono_lose, torch.full((N,), SEARCH, dtype=torch.long, device=self.device), self.state[:, 2])

        # State machine transitions
        in_detect = (self.state == DETECT)
        in_track  = (self.state == TRACK)
        self.state = torch.where(in_detect & (self.tq > TQ_DETECT_THRESH), torch.full_like(self.state, TRACK),  self.state)
        self.state = torch.where(in_track  & (self.tq > TQ_LOCK_THRESH),   torch.full_like(self.state, LOCK),   self.state)
        self.state = torch.where(in_track  & (self.tq < TQ_BREAK_THRESH),  torch.full_like(self.state, DETECT), self.state)

        lost = in_detect & ((self.tq < TQ_LOST_THRESH) | (self.consec_misses >= CONSEC_MISS_THRESH))
        self.state            = torch.where(lost, torch.full_like(self.state, SEARCH), self.state)
        self.tq               = torch.where(lost, torch.zeros_like(self.tq),           self.tq)
        self.scan_det_count   = torch.where(lost, torch.zeros_like(self.scan_det_count),   self.scan_det_count)
        self.scan_total_count = torch.where(lost, torch.zeros_like(self.scan_total_count), self.scan_total_count)
        self.consec_misses    = torch.where(lost, torch.zeros_like(self.consec_misses),    self.consec_misses)

    def get_observations(self, drone_pos: Tensor, drone_quat: Tensor) -> Tensor:
        """[N, 40] RWR: 4 slots × 10 dims. Slots 0-2 = radars, slot 3 = zero-padded.
        Per slot: [bearing_norm, signal_strength, type_onehot(3), state_onehot(4), tq]"""
        N   = self.num_envs
        obs = torch.zeros(N, 40, device=self.device)
        rel = drone_pos.unsqueeze(1) - self.positions.unsqueeze(0)   # [N, 3, 3]
        R   = rel.norm(dim=2).clamp(min=0.1)

        qw, qx, qy, qz = drone_quat[:, 0], drone_quat[:, 1], drone_quat[:, 2], drone_quat[:, 3]
        drone_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        for i in range(3):
            b = i * 10
            bearing = torch.atan2(rel[:, i, 1], rel[:, i, 0]) - drone_yaw
            bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))
            obs[:, b]     = bearing / math.pi
            obs[:, b + 1] = (1.0 - R[:, i] / self._det_range[i]).clamp(0.0, 1.0)
            obs[:, b + 2 + i] = 1.0   # radar type one-hot
            state_oh = torch.zeros(N, 4, device=self.device)
            state_oh.scatter_(1, self.state[:, i].clamp(0, 3).unsqueeze(1), 1.0)
            obs[:, b + 5 : b + 9] = state_oh
            obs[:, b + 9] = self.tq[:, i]

        return obs

    def debug_state(self, env_id: int, radar_idx: int) -> str:
        """Single radar: 'Track 0.72'"""
        s  = int(self.state[env_id, radar_idx].item())
        tq = self.tq[env_id, radar_idx].item()
        return f"{STATE_NAMES[s]:<6s} {tq:.2f}"

    def debug_string(self, env_id: int = 0) -> str:
        return "  ".join(
            f"{RADAR_NAMES[i]}:{self.debug_state(env_id, i)}" for i in range(3)
        )
