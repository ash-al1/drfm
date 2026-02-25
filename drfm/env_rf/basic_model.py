"""

File: basic_model
Use: Define the RF environment in 4 basic radar equations
Update: Wed, 25 Feb 2026 22:14:43

1. Radar echo power   : S_echo = (Pt * Gt * Gr * λ² * σ) / ((4π)³ * R⁴ * L)
2. DRFM jamming power : S_jam  = (Pj * Gj * Gr * λ²)     / ((4π)² * R² * L)
3. J/S ratio          : J/S    = (Pj * Gj * 4π * R²)     / (Pt * Gt * σ)
4. Burn-through range : R_BT   = sqrt((Pt * Gt * σ)      / (Pj * Gj * 4π))
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rf_helpers import db_to_linear, linear_to_db, freq_to_lambda

def r_echo(
    Pt: float,       # R_tx pow [W]
    Gt: float,       # R_tx antenna gain
    Gr: float,       # R_rx antenna gain
    lam: float,      # Wavelength
    sigma: float,    # Drone radar cross section
    R: float,        # meters
    L: float = 1.0,  # Loss, 1 because I can
) -> float:
    """Radar echo power at R"""
    return (Pt * Gt * Gr * lam**2 * sigma) / ((4 * math.pi)**3 * R**4 * L)


def d_to_r_jp(
    Pj: float,       # D_tx power
    Gj: float,       # D antenna gain
    Gr: float,       # R_rx antenna gain
    lam: float,      # Wavelength
    R: float,        # Range
    L: float = 1.0,  # System losses
) -> float:
    """J power at R_rx"""
    return (Pj * Gj * Gr * lam**2) / ((4 * math.pi)**2 * R**2 * L)


def js_ratio(
    Pj: float,       # D_tx power
    Gj: float,       # D antenna gain
    Pt: float,       # R transmit power
    Gt: float,       # R_tx antenna gain
    sigma: float,    # target cross section
    R: float,        # meters
) -> float:
    """
    J/S > 1  →  jamming wins
    J/S < 1  →  radar signal wins
    """
    return (Pj * Gj * 4 * math.pi * R**2) / (Pt * Gt * sigma)


def burn_through(
    Pt: float,   # Radar transmit power [W]
    Gt: float,   # Radar transmit antenna gain [linear]
    sigma: float,# Target RCS [m²]
    Pj: float,   # Jammer transmit power [W]
    Gj: float,   # Jammer antenna gain [linear]
) -> float:
    """Range at which radar can always get adequate target skin

    Below R_BT, J/S < 1 and the radar detects the target.

    Ref: https://en.wikipedia.org/wiki/Radar_jamming_and_deception#Radar_burn-through 
    """
    return math.sqrt((Pt * Gt * sigma) / (Pj * Gj * 4 * math.pi))


def run_scenario(
    *,
    freq_hz: float,
    Pt_dBW: float,
    Gt_dBi: float,
    Gr_dBi: float,
    sigma_dBsm: float,
    Pj_dBW: float,
    Gj_dBi: float,
    R_m: float,
    L_dB: float = 0.0,
) -> dict:

    lam    = freq_to_lambda(freq_hz);       Pt     = db_to_linear(Pt_dBW)
    Gt     = db_to_linear(Gt_dBi);          Gr     = db_to_linear(Gr_dBi)
    sigma  = db_to_linear(sigma_dBsm);      Pj     = db_to_linear(Pj_dBW)
    Gj     = db_to_linear(Gj_dBi);          L      = db_to_linear(L_dB)

    S_echo = r_echo(Pt, Gt, Gr, lam, sigma, R_m, L)
    S_jam  = d_to_r_jp(Pj, Gj, Gr, lam, R_m, L)
    js     = js_ratio(Pj, Gj, Pt, Gt, sigma, R_m)
    r_bt   = burn_through(Pt, Gt, sigma, Pj, Gj)

    return {
        "freq_hz":      freq_hz,
        "wavelength_m": lam,
        "range_m":      R_m,
        # linear
        "S_echo_W":     S_echo,
        "S_jam_W":      S_jam,
        "JS_linear":    js,
        "R_BT_m":       r_bt,
        # dB
        "S_echo_dBW":   linear_to_db(S_echo),
        "S_jam_dBW":    linear_to_db(S_jam),
        "JS_dB":        linear_to_db(js),
        # status
        "jamming_wins": js > 1.0,
        "radar_wins":   R_m < r_bt,
    }


if __name__ == "__main__":
    
    # Basic bogus numbers as first time scenario
    scenario = dict(
        freq_hz    = 10e9,
        Pt_dBW     = 40.0,
        Gt_dBi     = 30.0,
        Gr_dBi     = 30.0,
        sigma_dBsm = -20.0,
        Pj_dBW     = 10.0,
        Gj_dBi     = 5.0,
        R_m        = 5000.0,
        L_dB       = 3.0,
    )

    res = run_scenario(**scenario)

    print()
    print(f"  Frequency      : {res['freq_hz']/1e9:.1f} GHz")
    print(f"  Wavelength     : {res['wavelength_m']*100:.2f} cm")
    print(f"  Range          : {res['range_m']/1e3:.1f} km")
    print(f"  ---")
    print(f"  Radar echo     : {res['S_echo_dBW']:.1f} dBW")
    print(f"  Jamming power  : {res['S_jam_dBW']:.1f} dBW")
    print(f"  J/S ratio      : {res['JS_dB']:.1f} dB  ({'JAMMER wins' if res['jamming_wins'] else 'RADAR wins'})")
    print(f"  Burn-through   : {res['R_BT_m']:.1f} m")
    print(f"  Radar detects? : {'No' if res['jamming_wins'] else 'Yes'}")
