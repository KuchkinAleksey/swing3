"""
Transition between arbitrary pendulum configurations using TV-LQR.

Usage:
    main("ddd->uuu")
    main("udd->udu")

Output: results/ddd_uuu.gif
"""

import os
import time

import numpy as np
from dynamics import PendulumParams
from simulator import Simulator
from animator import PendulumAnimator
from tvlqr import solve_trajectory, TVLQRController


ANGLE_MAP = {"u": 0.0, "d": np.pi}


def parse_transition(transition: str):
    """Parse 'uud->duu' into start/end angle arrays."""
    parts = transition.strip().split("->")
    if len(parts) != 2 or len(parts[0]) != 3 or len(parts[1]) != 3:
        raise ValueError(f"Expected format like 'uud->duu', got: '{transition}'")
    start = np.array([ANGLE_MAP[c] for c in parts[0]])
    end = np.array([ANGLE_MAP[c] for c in parts[1]])
    return start, end


def label(angles: np.ndarray) -> str:
    """Short label: [0, π, π] -> 'udd'."""
    return "".join("u" if abs((a + np.pi) % (2 * np.pi) - np.pi) < 0.1
                   else "d" if abs(abs((a + np.pi) % (2 * np.pi) - np.pi) - np.pi) < 0.1
                   else "x" for a in angles)


def main(transition: str = "ddd->uuu"):
    t0 = time.time()

    start_angles, end_angles = parse_transition(transition)
    params = PendulumParams()

    sl, el = label(start_angles), label(end_angles)
    print(f"Transition: {sl} -> {el}")

    # Solve trajectory + build TV-LQR controller
    N_opt = int(params.T / 0.015)
    traj = solve_trajectory(start_angles, end_angles, params, N=N_opt, T=params.T)
    controller = TVLQRController(traj, params, target_angles=end_angles)

    # Simulate
    print(f"\n[{time.time()-t0:.1f}s] Simulating...")
    sim = Simulator(params)
    x0 = np.array([0, *start_angles, 0, 0, 0, 0])
    result = sim.rollout(x0, controller, params.T + 3.0)

    final = result["states"][-1]
    angle_err = (final[1:4] - end_angles + np.pi) % (2 * np.pi) - np.pi
    status = "SUCCESS" if np.all(np.abs(angle_err) < np.radians(10)) else "FAILED"
    print(f"  {status} — angles: {np.degrees(final[1:4]).round(2)} deg")

    # Save GIF
    os.makedirs("results", exist_ok=True)
    gif_name = f"{sl}_{el}.gif"
    PendulumAnimator(params).animate(result, save_path=f"results/{gif_name}", fps=30)

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main("dud->uuu")
