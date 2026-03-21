# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

"""
Modified From Isaac Drone Racer GitHub by Kousheek Chakraborty

File:   Motor.py
Use:    First-order motor dynamics model for quadrotor simulation
Update: Fri, 20 Mar 2026

Simulates motor spin up/down using rotor speed (omega_ref), acceleration
determined by constant tau bounded by min and max rate.

The integration formula applied each timestep:
    omega_rate = (1 / tau) * (omega_ref - omega)
    omega_rate = clamp(omega_rate, min_rate, max_rate)
    omega += dt * omega_rate

Changes:
+ Comments
+ Documentation
"""

import torch


class Motor:
    """
    First-order motor dynamics model.

    Models the spin-up and spin-down lag of brushless motors. Given a target
    rotor speed (omega_ref), returns the actual rotor speed (omega) after
    simulating the delay imposed by motor inertia and electrical characteristics.

    Supports batched operation across all parallel simulation environments.
    """

    def __init__(
        self,
        num_envs: int,
        taus: list,
        init: list,
        max_rate: list,
        min_rate: list,
        dt: float,
        use: bool,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Args:
            num_envs:  N parallel simulations
            taus:      Time constants per motor; smaller = faster (s)
            init:      Init. rotor speed (rad/s)
            max_rate:  Max spin-up rate per motor   (rad/s²)
            min_rate:  Min spin-down rate per motor (rad/s²)
            dt:        Timestep for Euler integration (s)
            use:       If False, motor lag is bypassed and omega_ref is returned directly
            device:    Torch device
            dtype:     Torch datatype
        """
        self.num_envs = num_envs
        self.num_motors = len(taus)
        self.dt = dt
        self.use = use
        self.init = init
        self.device = device
        self.dtype = dtype

        # Current rotor speeds, initialized to resting state: (num_envs, num_motors)
        self.omega = torch.tensor(init, device=device).expand(num_envs, -1).clone()

        # Motor parameters, broadcast across all environments: (num_envs, num_motors)
        self.tau = torch.tensor(taus, device=device).expand(num_envs, -1)
        self.max_rate = torch.tensor(max_rate, device=device).expand(num_envs, -1)
        self.min_rate = torch.tensor(min_rate, device=device).expand(num_envs, -1)

    def compute(self, omega_ref: torch.Tensor) -> torch.Tensor:
        """
        Advance toward target, one timestep

        Args:
            omega_ref: Target motor speed (rad/s)

        Returns:
            Actual rotor speeds after motor lag (rad/s)
        """
        if not self.use:
            self.omega = omega_ref
            return self.omega

        # First-order lag: rate of change is proportional to the speed error
        omega_rate = (1.0 / self.tau) * (omega_ref - self.omega)

        # Clamp, PS. can introduce overload noise
        omega_rate = omega_rate.clamp(self.min_rate, self.max_rate)

        # Integrate
        self.omega += self.dt * omega_rate
        return self.omega

    def reset(self, env_ids: torch.Tensor) -> None:
        self.omega[env_ids] = torch.tensor(self.init, device=self.device, dtype=self.dtype).expand(len(env_ids), -1)
