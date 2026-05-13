# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mohammad Ali
#
# Copyright (c) 2025, Kousheek Chakraborty
# Original work licensed under the BSD-3-Clause License.
# Built on the IsaacLab framework (https://github.com/isaac-sim/IsaacLab).

import torch


class Allocation:
    """Maps rotor speeds to body thrust and torques for an X-config quadrotor."""

    def __init__(self, num_envs, arm_length, thrust_coeff, drag_coeff, device="cpu", dtype=torch.float32):
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        L = arm_length * sqrt2_inv
        A = torch.tensor(
            [
                [1.0,  1.0,  1.0,  1.0],
                [ L,   -L,   -L,    L ],
                [-L,   -L,    L,    L ],
                [drag_coeff, -drag_coeff, drag_coeff, -drag_coeff],
            ],
            dtype=dtype,
            device=device,
        )
        self._A = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._kt = thrust_coeff

    def compute(self, omega: torch.Tensor) -> torch.Tensor:
        """Returns [total_thrust, roll_torque, pitch_torque, yaw_torque]."""
        return torch.bmm(self._A, (self._kt * omega**2).unsqueeze(-1)).squeeze(-1)
