"""
pendulum.py — Single N-link chain pendulum simulation.

Lagrangian formalism applied to a chain of N point masses connected by
massless rigid rods. The equations of motion are derived analytically and
written in matrix form so numpy can build them in a vectorized way:

    M(α) · α̈ = b(α, α̇)

where (with α_i the angle of link i from the downward vertical):

    M[i,j]   = l² · M̃[i,j] · cos(α_i − α_j)
    b[i]     = −l² Σ_j M̃[i,j] sin(α_i − α_j) α̇_j²
               − g·l·M̂[i]·sin(α_i)
               − d·α̇_i
    M̃[i,j]  = Σ_{k ≥ max(i,j)} m_k     (symmetric "tail mass" matrix)
    M̂[i]    = Σ_{k ≥ i} m_k

A heavy mass on the last link makes the chain behave like a wrecking ball.
Solved with scipy.integrate.solve_ivp; animated with matplotlib.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def simulate_chain(
    N: int = 50,
    l: float = 0.5,
    masses: np.ndarray | None = None,
    g: float = 9.8,
    d: float = 0.05,
    alpha0: float = 0.5,
    omega0: float = 0.0,
    t_max: float = 10.0,
    n_frames: int = 300,
    method: str = "RK45",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-8,
):
    """Integrate an N-link chain pendulum.

    Parameters
    ----------
    N : number of links.
    l : length of each link.
    masses : per-link masses, shape (N,). Defaults to 49×0.25 + 70 (heavy tip).
    g, d : gravity and linear damping coefficient.
    alpha0, omega0 : initial angle and angular velocity (same for every link).
    t_max, n_frames : simulation horizon and number of output frames.
    method : scipy ODE method ("RK45", "Radau", "LSODA", ...).
    rtol, atol : integrator tolerances. With ``d = 0`` energy is conserved
        to ~rtol per integration step; tighten if you need stricter conservation.

    Returns
    -------
    t : (n_frames,)  time grid.
    angles : (N, n_frames)  angle of every link at every frame.
    """
    if masses is None:
        masses = np.full(N, 0.25)
        masses[-1] = 70.0
    masses = np.asarray(masses, dtype=float)
    assert masses.shape == (N,)

    # M̂[i] = Σ_{k≥i} m_k     M̃[i,j] = M̂[max(i,j)]
    M_hat = np.cumsum(masses[::-1])[::-1]
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    M_tilde = M_hat[np.maximum(i_idx, j_idx)]

    def deriv(t, state):
        alpha = state[:N]
        omega = state[N:]
        diff = alpha[:, None] - alpha[None, :]
        M_mat = l * l * M_tilde * np.cos(diff)
        rhs = (
            -l * l * (M_tilde * np.sin(diff)) @ (omega ** 2)
            - g * l * M_hat * np.sin(alpha)
            - d * omega
        )
        accel = np.linalg.solve(M_mat, rhs)
        return np.concatenate([omega, accel])

    state0 = np.concatenate([np.full(N, alpha0), np.full(N, omega0)])
    t_eval = np.linspace(0.0, t_max, n_frames)
    sol = solve_ivp(
        deriv, [0.0, t_max], state0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    return sol.t, sol.y[:N]


def link_positions(angles: np.ndarray, l: float, anchor=(0.0, 0.0)):
    """Convert angle history (N, T) to point coordinates including the pivot.

    Returns x, y of shape (N+1, T) — first row is the pivot, last row is the tip.
    """
    sin_a, cos_a = np.sin(angles), np.cos(angles)
    x = anchor[0] + np.cumsum(l * sin_a, axis=0)
    y = anchor[1] - np.cumsum(l * cos_a, axis=0)
    T = angles.shape[1]
    x = np.vstack([np.full(T, anchor[0]), x])
    y = np.vstack([np.full(T, anchor[1]), y])
    return x, y


def animate_chain(t, angles, l, save_path: str | None = None, fps: int = 30):
    x, y = link_positions(angles, l)
    N = angles.shape[0]
    extent = N * l * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent * 0.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Chain pendulum, N = {N}")

    (line,) = ax.plot([], [], "-", color="royalblue", lw=1.5)
    tip = ax.scatter([], [], s=140, color="crimson", zorder=5)

    def update(frame):
        line.set_data(x[:, frame], y[:, frame])
        tip.set_offsets([[x[-1, frame], y[-1, frame]]])
        return line, tip

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000 / fps, blit=True)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
    return fig, anim


if __name__ == "__main__":
    t, angles = simulate_chain(
        N=50, l=0.5, alpha0=0.5, omega0=0.0,
        d=0.05, t_max=10.0, n_frames=300,
    )
    fig, anim = animate_chain(t, angles, l=0.5, save_path="pendulum.gif")
    print("Saved pendulum.gif")
