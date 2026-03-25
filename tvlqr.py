"""
Trajectory optimization + Time-Varying LQR tracking controller.

Uses direct collocation (CasADi/IPOPT) to find optimal trajectory,
TV-LQR (backward Riccati) for closed-loop tracking,
and infinite-horizon LQR (CARE) for post-trajectory stabilization.
"""

import numpy as np
import casadi as ca
from scipy.linalg import solve_continuous_are
from dynamics import PendulumParams, TriplePendulumDynamics


# ---------------------------------------------------------------------------
# CasADi symbolic dynamics (for trajectory optimization)
# ---------------------------------------------------------------------------

def _build_casadi_dynamics(p: PendulumParams):
    x = ca.SX.sym("x")
    th1, th2, th3 = ca.SX.sym("th1"), ca.SX.sym("th2"), ca.SX.sym("th3")
    dx = ca.SX.sym("dx")
    dth1, dth2, dth3 = ca.SX.sym("dth1"), ca.SX.sym("dth2"), ca.SX.sym("dth3")
    a = ca.SX.sym("a")

    state = ca.vertcat(x, th1, th2, th3, dx, dth1, dth2, dth3)
    dtheta = ca.vertcat(dth1, dth2, dth3)

    s1, c1 = ca.sin(th1), ca.cos(th1)
    s2, c2 = ca.sin(th2), ca.cos(th2)
    s3, c3 = ca.sin(th3), ca.cos(th3)
    c12, c13, c23 = ca.cos(th1-th2), ca.cos(th1-th3), ca.cos(th2-th3)
    s12, s13, s23 = ca.sin(th1-th2), ca.sin(th1-th3), ca.sin(th2-th3)

    a1 = p.m1*p.lc1 + (p.m2+p.m3)*p.l1
    a2 = p.m2*p.lc2 + p.m3*p.l2
    a3 = p.m3*p.lc3
    b12 = (p.m2*p.lc2 + p.m3*p.l2)*p.l1
    b13 = p.m3*p.lc3*p.l1
    b23 = p.m3*p.lc3*p.l2

    M = ca.SX(3, 3)
    M[0,0] = p.I1 + p.m1*p.lc1**2 + (p.m2+p.m3)*p.l1**2
    M[1,1] = p.I2 + p.m2*p.lc2**2 + p.m3*p.l2**2
    M[2,2] = p.I3 + p.m3*p.lc3**2
    M[0,1] = M[1,0] = b12*c12
    M[0,2] = M[2,0] = b13*c13
    M[1,2] = M[2,1] = b23*c23

    h = ca.vertcat(
        b12*s12*dth2**2 + b13*s13*dth3**2,
        -b12*s12*dth1**2 + b23*s23*dth3**2,
        -b13*s13*dth1**2 - b23*s23*dth2**2,
    )
    g_vec = ca.vertcat(-a1*p.g*s1, -a2*p.g*s2, -a3*p.g*s3)
    coupling = ca.vertcat(a1*c1, a2*c2, a3*c3)
    D = ca.vertcat(p.c1, p.c2, p.c3)

    theta_ddot = ca.solve(M, -h - g_vec - D*dtheta - coupling*a)
    state_dot = ca.vertcat(dx, dth1, dth2, dth3, a, theta_ddot)
    return ca.Function("f", [state, a], [state_dot])


def _rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u)
    k4 = f(x + dt*k3, u)
    return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# LQR gain computation (used internally for post-trajectory stabilization)
# ---------------------------------------------------------------------------

def _compute_lqr_gain(params: PendulumParams, target_angles: np.ndarray) -> np.ndarray:
    """Compute infinite-horizon LQR gain K around target equilibrium."""
    dynamics = TriplePendulumDynamics(params)
    target_state = np.array([0, *target_angles, 0, 0, 0, 0])
    eps = 1e-6

    A = np.zeros((8, 8))
    for i in range(8):
        xp = target_state.copy(); xp[i] += eps
        xm = target_state.copy(); xm[i] -= eps
        A[:, i] = (dynamics.forward_dynamics(xp, 0.0)
                    - dynamics.forward_dynamics(xm, 0.0)) / (2 * eps)

    B = np.zeros((8, 1))
    B[:, 0] = (dynamics.forward_dynamics(target_state, eps)
                - dynamics.forward_dynamics(target_state, -eps)) / (2 * eps)

    Q = np.diag([10.0, 200.0, 200.0, 200.0, 1.0, 10.0, 10.0, 10.0])
    R = np.array([[1.0]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


# ---------------------------------------------------------------------------
# Trajectory optimization via direct collocation
# ---------------------------------------------------------------------------

def solve_trajectory(
    start_angles: np.ndarray,
    end_angles: np.ndarray,
    params: PendulumParams | None = None,
    N: int = 200,
    T: float = 3.0,
    verbose: bool = True,
):
    """Find optimal trajectory between two configurations."""
    params = params or PendulumParams()
    f = _build_casadi_dynamics(params)
    dt = T / N
    nx, nu = 8, 1

    x_start = np.array([0, *start_angles, 0, 0, 0, 0])
    x_end = np.array([0, *end_angles, 0, 0, 0, 0])

    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)

    cost = 0
    constraints = []
    lbg, ubg = [], []

    constraints.append(X[:, 0] - x_start)
    lbg += [0] * nx
    ubg += [0] * nx

    for k in range(N):
        # Ramp up penalties toward end to ensure smooth landing
        w = 1.0 + 4.0 * (k / N)**2
        cost += dt * (U[:, k].T @ U[:, k]) * 0.01
        cost += dt * X[0, k]**2 * 0.1
        cost += dt * (X[5, k]**2 + X[6, k]**2 + X[7, k]**2) * 0.005 * w
        if k > 0:
            du = U[:, k] - U[:, k-1]
            cost += (du.T @ du) * 0.1

        x_next = _rk4_step(f, X[:, k], U[:, k], dt)
        constraints.append(X[:, k+1] - x_next)
        lbg += [0] * nx
        ubg += [0] * nx

    terminal = X[:, N]
    constraints.append(terminal[1:4] - x_end[1:4])
    lbg += [0, 0, 0]
    ubg += [0, 0, 0]
    constraints.append(terminal[4:])
    lbg += [0, 0, 0, 0]
    ubg += [0, 0, 0, 0]
    cost += 100 * terminal[0]**2

    a_max = params.a_max
    x_max = params.x_max
    ang_lb = min(0, *start_angles, *end_angles) - 2 * np.pi
    ang_ub = max(np.pi, *start_angles, *end_angles) + 2 * np.pi

    lbx_x = np.tile([-x_max, ang_lb, ang_lb, ang_lb, -15, -30, -30, -30], N+1)
    ubx_x = np.tile([x_max, ang_ub, ang_ub, ang_ub, 15, 30, 30, 30], N+1)
    lbx_u = np.tile([-a_max], N)
    ubx_u = np.tile([a_max], N)

    opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    nlp = {"x": opt_vars, "f": cost, "g": ca.vertcat(*constraints)}
    opts = {
        "ipopt.print_level": 5 if verbose else 0,
        "ipopt.max_iter": 5000,
        "ipopt.tol": 1e-10,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.warm_start_init_point": "yes",
        "print_time": verbose,
    }

    lbx = np.concatenate([lbx_x, lbx_u])
    ubx = np.concatenate([ubx_x, ubx_u])

    x_init = np.zeros(nx * (N+1))
    for k in range(N+1):
        alpha = k / N
        x_init[k*nx:(k+1)*nx] = (1 - alpha) * x_start + alpha * x_end
    x0_guess = np.concatenate([x_init, np.zeros(N)])

    if verbose:
        print(f"  N={N}, T={T}s, dt={dt:.4f}s, a_max={a_max} m/s²")

    solver = ca.nlpsol("trajectory", "ipopt", nlp, opts)
    sol = solver(x0=x0_guess, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    best_sol = sol

    opt = np.array(best_sol["x"]).flatten()
    states = opt[:nx*(N+1)].reshape(N+1, nx)
    controls = opt[nx*(N+1):]

    if verbose:
        print(f"Final cost: {float(best_sol['f']):.4f}")
        print(f"Final angles (deg): {np.round(np.degrees(states[-1, 1:4]), 2)}")
        print(f"Max accel: {np.max(np.abs(controls)):.1f} m/s²")

    return {"times": np.linspace(0, T, N+1), "states": states, "controls": controls}


# ---------------------------------------------------------------------------
# TV-LQR Controller
# ---------------------------------------------------------------------------

class TVLQRController:
    """Trajectory optimization + TV-LQR tracking + LQR stabilization."""

    def __init__(self, traj: dict, params: PendulumParams,
                 target_angles: np.ndarray):
        self.states_ref = traj["states"]
        self.controls_ref = traj["controls"]
        self.params = params
        self.T_final = traj["times"][-1]
        self.dt_ref = traj["times"][1] - traj["times"][0]
        self.target_state = np.array([0, *target_angles, 0, 0, 0, 0])
        self._prev_u = 0.0

        # LQR for post-trajectory stabilization
        self.lqr_K = _compute_lqr_gain(params, target_angles)

        # TV-LQR gains along trajectory
        dynamics = TriplePendulumDynamics(params)
        self.gains = self._compute_tvlqr(dynamics)

    def _compute_tvlqr(self, dynamics):
        """Backward Riccati pass along reference trajectory."""
        N = len(self.controls_ref)
        nx, nu = 8, 1
        dt = self.dt_ref
        eps = 1e-5

        Q = np.diag([2.0, 50.0, 50.0, 50.0, 0.5, 2.0, 2.0, 2.0])
        R = np.array([[1.0]])
        Qf = np.diag([10.0, 200.0, 200.0, 200.0, 1.0, 10.0, 10.0, 10.0])

        A_list, B_list = [], []
        for k in range(N):
            x_ref = self.states_ref[k]
            u_ref = self.controls_ref[k]

            def rk4_step(x, u):
                k1 = dynamics.forward_dynamics(x, u)
                k2 = dynamics.forward_dynamics(x + dt/2*k1, u)
                k3 = dynamics.forward_dynamics(x + dt/2*k2, u)
                k4 = dynamics.forward_dynamics(x + dt*k3, u)
                return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

            A = np.zeros((nx, nx))
            for i in range(nx):
                xp = x_ref.copy(); xp[i] += eps
                xm = x_ref.copy(); xm[i] -= eps
                A[:, i] = (rk4_step(xp, u_ref) - rk4_step(xm, u_ref)) / (2*eps)

            B = np.zeros((nx, nu))
            B[:, 0] = (rk4_step(x_ref, u_ref+eps) - rk4_step(x_ref, u_ref-eps)) / (2*eps)

            A_list.append(A)
            B_list.append(B)

        P = Qf.copy()
        gains = [None] * N
        for k in range(N-1, -1, -1):
            A, B = A_list[k], B_list[k]
            K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
            gains[k] = K
            P = Q + A.T @ P @ (A - B @ K)

        return gains

    def __call__(self, state: np.ndarray, t: float) -> float:
        if t >= self.T_final:
            x_err = state - self.target_state
            x_err[1:4] = (x_err[1:4] + np.pi) % (2*np.pi) - np.pi
            u_raw = float((-self.lqr_K @ x_err)[0])
        else:
            idx = min(int(t / self.dt_ref), len(self.controls_ref) - 1)
            u_ff = self.controls_ref[idx]
            x_err = state - self.states_ref[idx]
            x_err[1:4] = (x_err[1:4] + np.pi) % (2*np.pi) - np.pi
            u_raw = u_ff + float((-self.gains[idx] @ x_err)[0])

        alpha = 0.3
        u_filt = (1 - alpha) * u_raw + alpha * self._prev_u
        u_out = float(np.clip(u_filt, -self.params.a_max, self.params.a_max))
        self._prev_u = u_out
        return u_out
