"""
File: RF Helpers
Use: Bunch of helpers for RF environment modeling, conversions
Update: Wed, 25 Feb 2026
"""

import numpy as np


def db_to_linear(db: float) -> float:
    return 10 ** (db / 10)

def linear_to_db(linear: float) -> float:
    return 10 * np.log10(linear)

def freq_to_lambda(freq_hz: float) -> float:
    return 3e8 / freq_hz
