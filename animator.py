"""Matplotlib animation for triple pendulum on a cart."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dynamics import PendulumParams


class PendulumAnimator:

    def __init__(self, params: PendulumParams | None = None):
        self.p = params or PendulumParams()

    def _cart_and_joints(self, state: np.ndarray):
        """Compute cart position and joint coordinates from state."""
        p = self.p
        x, th1, th2, th3 = state[:4]

        cart = np.array([x, 0.0])

        j1 = cart + p.l1 * np.array([np.sin(th1), np.cos(th1)])
        j2 = j1 + p.l2 * np.array([np.sin(th2), np.cos(th2)])
        j3 = j2 + p.l3 * np.array([np.sin(th3), np.cos(th3)])

        return cart, j1, j2, j3

    def animate(
        self,
        result: dict,
        save_path: str | None = None,
        fps: int = 50,
    ):
        """Create real-time animation (1s GIF = 1s sim time)."""
        sim_dt = result["times"][1] - result["times"][0]
        skip = max(1, round(1.0 / (fps * sim_dt)))

        states = result["states"][::skip]
        times = result["times"][::skip]
        controls = result["controls"]

        fig, (ax_anim, ax_ctrl) = plt.subplots(
            2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]}
        )

        total_len = self.p.l1 + self.p.l2 + self.p.l3
        margin = 1.0
        ax_anim.set_xlim(self.p.x_min - margin, self.p.x_max + margin)
        ax_anim.set_ylim(-total_len - 0.5, total_len + 0.5)
        ax_anim.set_aspect("equal")
        ax_anim.grid(True, alpha=0.3)
        ax_anim.axhline(y=0, color="k", linewidth=2)

        cart_w, cart_h = 0.4, 0.2
        cart_patch = patches.Rectangle(
            (-cart_w / 2, -cart_h / 2), cart_w, cart_h,
            facecolor="#2196F3", edgecolor="black", linewidth=1.5,
        )
        ax_anim.add_patch(cart_patch)

        (link_line,) = ax_anim.plot([], [], "o-", color="#333", linewidth=2.5, markersize=6)
        time_text = ax_anim.text(0.02, 0.95, "", transform=ax_anim.transAxes, fontsize=11)

        ctrl_times = result["times"][:-1]
        ax_ctrl.plot(ctrl_times, controls, color="#E91E63", linewidth=0.5, alpha=0.7)
        (ctrl_marker,) = ax_ctrl.plot([], [], "o", color="#E91E63", markersize=5)
        ax_ctrl.set_xlabel("Time [s]")
        ax_ctrl.set_ylabel("Accel [m/s²]")
        ax_ctrl.grid(True, alpha=0.3)
        ax_ctrl.set_xlim(ctrl_times[0], ctrl_times[-1])

        fig.tight_layout()

        def init():
            link_line.set_data([], [])
            cart_patch.set_x(-cart_w / 2)
            cart_patch.set_y(-cart_h / 2)
            time_text.set_text("")
            ctrl_marker.set_data([], [])
            return link_line, cart_patch, time_text, ctrl_marker

        def update(frame):
            state = states[frame]
            t = times[frame]
            cart, j1, j2, j3 = self._cart_and_joints(state)

            cart_patch.set_x(cart[0] - cart_w / 2)
            cart_patch.set_y(-cart_h / 2)

            xs = [cart[0], j1[0], j2[0], j3[0]]
            ys = [cart[1], j1[1], j2[1], j3[1]]
            link_line.set_data(xs, ys)

            time_text.set_text(f"t = {t:.2f} s")

            ctrl_idx = min(int(frame * skip), len(controls) - 1)
            ctrl_marker.set_data([ctrl_times[ctrl_idx]], [controls[ctrl_idx]])

            return link_line, cart_patch, time_text, ctrl_marker

        anim = FuncAnimation(
            fig, update, frames=len(states),
            init_func=init, blit=True, interval=1000 / fps,
        )

        if save_path:
            if save_path.endswith(".gif"):
                anim.save(save_path, writer="pillow", fps=fps)
            else:
                anim.save(save_path, writer="ffmpeg", fps=fps)
            print(f"Saved animation to {save_path}")
        else:
            plt.show()

        return anim
