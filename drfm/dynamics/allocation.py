# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""
Modified From Isaac Drone Racer GitHub by Kousheek Chakraborty

File:   Allocation.py
Use:    Thrust allocation model for X-configuration quadrotor
Update: Fri, 20 Mar 2026

Quadroter assigned 4 separate rotors ararnged in X pattern. Rotors produce
upward thrust vector proportional to speed^2. Reaction torque around yaw
from aerodynamic drag.

The allocation matrix A maps rotor forces [f1, f2, f3, f4] to body wrench
[F_z, tau_x, tau_y, tau_z]:

    [F_z  ]   [  1      1      1      1   ] [f1]
    [tau_x] = [  L/√2  -L/√2  -L/√2   L/√2] [f2]
    [tau_y]   [ -L/√2  -L/√2   L/√2   L/√2] [f3]
    [tau_z]   [  kd    -kd     kd    -kd  ] [f4]

Where L = arm length, kd = drag coefficient.
"""

import torch

class Allocation:
    """Map rotors, thurst, torque"""
    def __init__(
        self,
        num_envs: int,
        arm_length: float,
        thrust_coeff: float,
        drag_coeff: float,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Args:
            num_envs:     N parallel simulations
            arm_length:   Distance (m) from center to rotor
            thrust_coeff: Thrust force constant kt, (f = kt * omega^2 (N·s^2/rad^2))
            drag_coeff:   Aerodynamic drag torque constant kd for yaw (N·m·s^2s/rad^2)
            device:       Torch device
            dtype:        Torch datatype
        """
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

        # Allocation matrix rows: [total thrust, roll torque, pitch torque, yaw torque]
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coeff, -drag_coeff, drag_coeff, -drag_coeff],
            ],
            dtype=dtype,
            device=device,
        )

        # Replicate for all environments
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coeff

    def compute(self, omega: torch.Tensor) -> torch.Tensor:
        """
        Convert rotor speeds to body thrust and torques

        Returns:
            Wrench [total_thrust, roll_torque, pitch_torque, yaw_torque]
        """
        rotor_thrusts = self._thrust_coeff * omega**2
        thrust_torque = torch.bmm(self._allocation_matrix, rotor_thrusts.unsqueeze(-1)).squeeze(-1)
        return thrust_torque
