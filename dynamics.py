"""
Triple inverted pendulum on a cart — physical parameters and equations of motion.

Cart is rigidly on a rail: we control cart acceleration a = ẍ directly.
The pendulum dynamics depend on cart acceleration but NOT vice versa.

Generalized coordinates: q = [x, θ₁, θ₂, θ₃]
  - x:  cart horizontal position (prescribed motion, ẍ = a)
  - θ₁: angle of link 1 from vertical (0 = up, absolute)
  - θ₂: angle of link 2 from vertical (0 = up, absolute)
  - θ₃: angle of link 3 from vertical (0 = up, absolute)

Pendulum equations (3 DOF):
  M_pend(θ) · θ̈ + C(θ,θ̇) + G(θ) + D·θ̇ = -F_coupling(θ) · a
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PendulumParams:
    """Physical parameters for triple inverted pendulum on a cart."""

    # Cart
    m0: float = 1.0        # cart mass [kg]

    # Link 1 (closest to cart)
    # Light plastic rod (~50g) + heavy metal hinge at tip (~250g)
    m1: float = 0.3        # link 1 total mass [kg]
    l1: float = 0.5        # link 1 length [m]
    lc1: float = 0.458     # link 1 center of mass [m] (92% — mass at tip)
    I1: float = 0.0036     # link 1 inertia about CM [kg*m^2]

    # Link 2
    m2: float = 0.3
    l2: float = 0.5
    lc2: float = 0.458
    I2: float = 0.0036

    # Link 3 (top)
    m3: float = 0.3
    l3: float = 0.5
    lc3: float = 0.458
    I3: float = 0.0036

    g: float = 9.81        # gravity [m/s^2]

    # Viscous friction / damping coefficients
    c1: float = 0.05       # joint 1 pivot friction [N*m*s/rad]
    c2: float = 0.05       # joint 2 pivot friction [N*m*s/rad]
    c3: float = 0.05       # joint 3 pivot friction [N*m*s/rad]

    # Cart track limits
    x_min: float = -3.0
    x_max: float = 3.0

    # Acceleration limits
    a_max: float = 250.0    # max cart acceleration [m/s²]

    # Simulation
    dt: float = 0.005      # integration step [s]
    T: float = 5.0         # trajectory optimization time [s]


class TriplePendulumDynamics:

    def __init__(self, params: PendulumParams | None = None):
        self.p = params or PendulumParams()

    def pendulum_mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        """Compute 3×3 pendulum mass matrix M_pend(θ)."""
        p = self.p
        th1, th2, th3 = theta

        c12 = np.cos(th1 - th2)
        c13 = np.cos(th1 - th3)
        c23 = np.cos(th2 - th3)

        b12 = (p.m2 * p.lc2 + p.m3 * p.l2) * p.l1
        b13 = p.m3 * p.lc3 * p.l1
        b23 = p.m3 * p.lc3 * p.l2

        M11 = p.I1 + p.m1 * p.lc1**2 + (p.m2 + p.m3) * p.l1**2
        M22 = p.I2 + p.m2 * p.lc2**2 + p.m3 * p.l2**2
        M33 = p.I3 + p.m3 * p.lc3**2

        M = np.array([
            [M11,        b12 * c12,  b13 * c13],
            [b12 * c12,  M22,        b23 * c23],
            [b13 * c13,  b23 * c23,  M33],
        ])
        return M

    def coupling_vector(self, theta: np.ndarray) -> np.ndarray:
        """Cart-to-pendulum coupling: F_coupling(θ) · a enters the RHS."""
        p = self.p
        th1, th2, th3 = theta
        a1 = p.m1 * p.lc1 + (p.m2 + p.m3) * p.l1
        a2 = p.m2 * p.lc2 + p.m3 * p.l2
        a3 = p.m3 * p.lc3
        return np.array([a1 * np.cos(th1), a2 * np.cos(th2), a3 * np.cos(th3)])

    def coriolis_gravity(self, theta: np.ndarray, dtheta: np.ndarray) -> np.ndarray:
        """Compute C(θ,θ̇)·θ̇ + G(θ) for the 3 pendulum DOFs."""
        p = self.p
        th1, th2, th3 = theta
        dth1, dth2, dth3 = dtheta

        s1 = np.sin(th1)
        s2 = np.sin(th2)
        s3 = np.sin(th3)
        s12 = np.sin(th1 - th2)
        s13 = np.sin(th1 - th3)
        s23 = np.sin(th2 - th3)

        b12 = (p.m2 * p.lc2 + p.m3 * p.l2) * p.l1
        b13 = p.m3 * p.lc3 * p.l1
        b23 = p.m3 * p.lc3 * p.l2

        a1 = p.m1 * p.lc1 + (p.m2 + p.m3) * p.l1
        a2 = p.m2 * p.lc2 + p.m3 * p.l2
        a3 = p.m3 * p.lc3

        # Coriolis
        h = np.array([
            b12 * s12 * dth2**2 + b13 * s13 * dth3**2,
            -b12 * s12 * dth1**2 + b23 * s23 * dth3**2,
            -b13 * s13 * dth1**2 - b23 * s23 * dth2**2,
        ])

        # Gravity
        g_vec = np.array([
            -a1 * p.g * s1,
            -a2 * p.g * s2,
            -a3 * p.g * s3,
        ])

        return h + g_vec

    def forward_dynamics(self, state: np.ndarray, a: float) -> np.ndarray:
        """
        Compute state derivative: ṡ = f(s, a).

        Args:
            state: [x, θ₁, θ₂, θ₃, ẋ, θ̇₁, θ̇₂, θ̇₃]
            a: cart acceleration [m/s²]

        Returns:
            state_dot: [ẋ, θ̇₁, θ̇₂, θ̇₃, ẍ, θ̈₁, θ̈₂, θ̈₃]
        """
        theta = state[1:4]
        dtheta = state[5:8]
        xdot = state[4]

        M = self.pendulum_mass_matrix(theta)
        cg = self.coriolis_gravity(theta, dtheta)
        coupling = self.coupling_vector(theta)
        D = np.array([self.p.c1, self.p.c2, self.p.c3])

        rhs = -cg - D * dtheta - coupling * a
        theta_ddot = np.linalg.solve(M, rhs)

        return np.array([
            xdot, dtheta[0], dtheta[1], dtheta[2],
            a,
            theta_ddot[0], theta_ddot[1], theta_ddot[2],
        ])

    def energy(self, state: np.ndarray) -> tuple[float, float]:
        """Compute kinetic and potential energy."""
        p = self.p
        theta = state[1:4]
        dtheta = state[5:8]
        xdot = state[4]

        th1, th2, th3 = theta

        M = self.pendulum_mass_matrix(theta)
        T_pend = 0.5 * dtheta @ M @ dtheta

        a1 = p.m1 * p.lc1 + (p.m2 + p.m3) * p.l1
        a2 = p.m2 * p.lc2 + p.m3 * p.l2
        a3 = p.m3 * p.lc3
        T_coupling = xdot * (a1 * np.cos(th1) * dtheta[0]
                             + a2 * np.cos(th2) * dtheta[1]
                             + a3 * np.cos(th3) * dtheta[2])
        m_total = p.m1 + p.m2 + p.m3
        T_cart = 0.5 * m_total * xdot**2
        T = T_pend + T_coupling + T_cart

        V = (p.m1 * p.g * p.lc1 * np.cos(th1)
             + p.m2 * p.g * (p.l1 * np.cos(th1) + p.lc2 * np.cos(th2))
             + p.m3 * p.g * (p.l1 * np.cos(th1) + p.l2 * np.cos(th2) + p.lc3 * np.cos(th3)))

        return T, V
