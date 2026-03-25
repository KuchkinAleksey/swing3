"""RK4 integrator for triple pendulum dynamics."""

import numpy as np
from dynamics import TriplePendulumDynamics, PendulumParams


class Simulator:

    def __init__(self, params: PendulumParams | None = None):
        self.params = params or PendulumParams()
        self.dynamics = TriplePendulumDynamics(self.params)
        self.dt = self.params.dt

    def step_rk4(self, state: np.ndarray, a: float) -> np.ndarray:
        """Single RK4 integration step."""
        f = self.dynamics.forward_dynamics
        dt = self.dt

        k1 = f(state, a)
        k2 = f(state + 0.5 * dt * k1, a)
        k3 = f(state + 0.5 * dt * k2, a)
        k4 = f(state + dt * k3, a)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Wrap angles to [-π, π]
        new_state[1:4] = (new_state[1:4] + np.pi) % (2 * np.pi) - np.pi

        # Clamp velocities to prevent numerical blowup
        vel_max = 50.0
        new_state[4:] = np.clip(new_state[4:], -vel_max, vel_max)

        return new_state

    def rollout(
        self,
        initial_state: np.ndarray,
        controller,
        duration: float,
    ) -> dict:
        """
        Simulate the system with a given controller.

        Args:
            initial_state: [x, θ₁, θ₂, θ₃, ẋ, θ̇₁, θ̇₂, θ̇₃]
            controller: callable(state, t) -> cart acceleration [m/s²]
            duration: simulation time [s]

        Returns:
            dict with 'times', 'states', 'controls', 'energy'
        """
        n_steps = int(duration / self.dt)
        states = np.zeros((n_steps + 1, 8))
        controls = np.zeros(n_steps)
        times = np.arange(n_steps + 1) * self.dt
        energy = np.zeros((n_steps + 1, 2))

        states[0] = initial_state
        energy[0] = self.dynamics.energy(initial_state)

        for i in range(n_steps):
            a = controller(states[i], times[i])
            a = np.clip(a, -self.params.a_max, self.params.a_max)
            controls[i] = a
            states[i + 1] = self.step_rk4(states[i], a)
            energy[i + 1] = self.dynamics.energy(states[i + 1])

        return {
            "times": times,
            "states": states,
            "controls": controls,
            "energy": energy,
        }
