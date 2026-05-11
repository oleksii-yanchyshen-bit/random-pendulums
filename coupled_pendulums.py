"""
coupled_pendulums.py — Two N-link chain pendulums coupled by a spring.

Each chain is governed by the same Lagrangian equations as in pendulum.py:

    M(α) · α̈ = b(α, α̇)

The coupling adds a spring potential between two configurable joints:

    U_spring = κ · |r_A^{pA} − r_B^{pB}|²

where pA and pB are the indices of the joints on each chain where the
spring is attached. The generalized force on angle α_k of chain A from
this potential is:

    Q_k^A = −2κ · l_A · [Δ_x · cos(α_k) + Δ_y · sin(α_k)]   if k ≤ pA
    Q_k^A = 0                                                if k >  pA

Each chain has its own link length (l_A, l_B), so chains can be of
different sizes. The system becomes stiff for large κ — we use Radau.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def _chain_matrices(N: int, masses: np.ndarray):
    M_hat = np.cumsum(masses[::-1])[::-1]
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    M_tilde = M_hat[np.maximum(i_idx, j_idx)]
    return M_hat, M_tilde


def _link_positions(angles: np.ndarray, l: float, anchor):
    """Convert angle history (N, T) to coordinates (N+1, T) including pivot."""
    sin_a, cos_a = np.sin(angles), np.cos(angles)
    x = anchor[0] + np.cumsum(l * sin_a, axis=0)
    y = anchor[1] - np.cumsum(l * cos_a, axis=0)
    T = angles.shape[1]
    x = np.vstack([np.full(T, anchor[0]), x])
    y = np.vstack([np.full(T, anchor[1]), y])
    return x, y


def reach_config(pivot, target, n_links: int):
    """Geometry: with `n_links` rigid links of equal length pointing as a
    straight bar from `pivot` to `target`, return (link_length, angle).

    The angle convention matches the simulation:
        x_tip = pivot_x + n·l·sin(α)
        y_tip = pivot_y - n·l·cos(α)
    """
    dx = target[0] - pivot[0]
    dy = target[1] - pivot[1]
    D = float(np.hypot(dx, dy))
    if D == 0.0:
        raise ValueError("target coincides with pivot")
    l = D / n_links
    # x: dx = n·l·sin(α)  →  sin(α) = dx/D
    # y: dy = -n·l·cos(α) →  cos(α) = -dy/D
    alpha = float(np.arctan2(dx, -dy))
    return l, alpha


def simulate_two_chains(
    N: int = 50,
    pivot_A=(-25.0, 0.0),
    pivot_B=(25.0, 0.0),
    spring_anchor: tuple[float, float] | None = None,
    l_A: float | None = None,
    l_B: float | None = None,
    masses_A: np.ndarray | None = None,
    masses_B: np.ndarray | None = None,
    g: float = 9.8,
    d: float = 5.0,
    kappa: float = 1.0e5,
    alpha0: float | np.ndarray | None = None,
    beta0: float | np.ndarray | None = None,
    attach_A: int | None = None,
    attach_B: int | None = None,
    t_max: float = 10.0,
    n_frames: int = 300,
    method: str = "Radau",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
):
    """Simulate two N-link chains coupled by a spring.

    Geometry
    --------
    Each chain hangs from its own pivot. The spring connects joint
    ``attach_A`` on chain A to joint ``attach_B`` on chain B (default:
    tip of each chain).

    The most convenient way to set up an experiment is to pass
    ``spring_anchor=(x, y)`` — the point in space where the two
    attachment joints should coincide at t=0. The function then computes
    ``l_A``, ``l_B``, ``alpha0``, ``beta0`` automatically so that:

      * each chain's first ``attach+1`` links form a straight bar from
        its pivot pointing exactly at ``(x, y)`` (link length scales to
        match the distance — chain "stretches" to reach);
      * remaining links (``attach+1`` … N−1) hang straight down (angle 0).

    Any of ``l_A``, ``l_B``, ``alpha0``, ``beta0`` you pass explicitly
    overrides the auto-computed value, so you can mix-and-match.

    If ``spring_anchor`` is omitted, you must supply ``l_A`` and ``l_B``
    yourself (and probably also ``alpha0``, ``beta0``).

    Notes on energy conservation
    ----------------------------
    A stiff spring (large κ) makes the ODE stiff. With ``d = 0`` and the
    default ``rtol = 1e-8`` / ``atol = 1e-10``, energy is conserved to
    about 1e-10 over a few seconds. Tighten if you need stricter
    conservation (and expect noticeably longer runtime).

    Returns
    -------
    dict with keys:
        t, angles_A, angles_B,
        pivot_A, pivot_B,
        l_A, l_B,
        attach_A, attach_B,
        spring_anchor (None if not given).
    """
    pivot_A = np.asarray(pivot_A, dtype=float)
    pivot_B = np.asarray(pivot_B, dtype=float)

    if attach_A is None:
        attach_A = N - 1
    if attach_B is None:
        attach_B = N - 1
    if not (0 <= attach_A < N):
        raise ValueError(f"attach_A must be in [0, {N-1}], got {attach_A}")
    if not (0 <= attach_B < N):
        raise ValueError(f"attach_B must be in [0, {N-1}], got {attach_B}")

    # --- geometric setup from spring_anchor (if given) ---
    if spring_anchor is not None:
        anchor = np.asarray(spring_anchor, dtype=float)
        l_A_geom, alpha_geom = reach_config(pivot_A, anchor, attach_A + 1)
        l_B_geom, beta_geom = reach_config(pivot_B, anchor, attach_B + 1)

        if l_A is None:
            l_A = l_A_geom
        if l_B is None:
            l_B = l_B_geom
        if alpha0 is None:
            alpha0 = np.zeros(N)
            alpha0[: attach_A + 1] = alpha_geom
            # the rest hang straight down (angle 0)
        if beta0 is None:
            beta0 = np.zeros(N)
            beta0[: attach_B + 1] = beta_geom
    else:
        if l_A is None or l_B is None:
            raise ValueError(
                "Either pass spring_anchor=(x, y) or specify both l_A and l_B."
            )
        if alpha0 is None:
            alpha0 = 0.0
        if beta0 is None:
            beta0 = 0.0

    # --- masses ---
    if masses_A is None:
        masses_A = np.full(N, 0.25)
    if masses_B is None:
        masses_B = np.full(N, 0.25)
    masses_A = np.asarray(masses_A, dtype=float)
    masses_B = np.asarray(masses_B, dtype=float)
    M_hat_A, M_tilde_A = _chain_matrices(N, masses_A)
    M_hat_B, M_tilde_B = _chain_matrices(N, masses_B)

    mask_A = (np.arange(N) <= attach_A).astype(float)
    mask_B = (np.arange(N) <= attach_B).astype(float)

    def chain_rhs(angles, omegas, l, M_hat, M_tilde):
        diff = angles[:, None] - angles[None, :]
        M_mat = l * l * M_tilde * np.cos(diff)
        rhs = (
            -l * l * (M_tilde * np.sin(diff)) @ (omegas ** 2)
            - g * l * M_hat * np.sin(angles)
            - d * omegas
        )
        return M_mat, rhs

    def attach_pos(angles, anchor, attach_idx, l):
        sub = angles[: attach_idx + 1]
        return np.array([
            anchor[0] + l * np.sin(sub).sum(),
            anchor[1] - l * np.cos(sub).sum(),
        ])

    def deriv(t, state):
        a = state[0:N]
        b = state[N:2 * N]
        oa = state[2 * N:3 * N]
        ob = state[3 * N:4 * N]

        Ma, rhs_a = chain_rhs(a, oa, l_A, M_hat_A, M_tilde_A)
        Mb, rhs_b = chain_rhs(b, ob, l_B, M_hat_B, M_tilde_B)

        rA = attach_pos(a, pivot_A, attach_A, l_A)
        rB = attach_pos(b, pivot_B, attach_B, l_B)
        delta = rA - rB
        tau_A = (-2.0 * kappa * l_A) * (
            delta[0] * np.cos(a) + delta[1] * np.sin(a)
        ) * mask_A
        tau_B = (+2.0 * kappa * l_B) * (
            delta[0] * np.cos(b) + delta[1] * np.sin(b)
        ) * mask_B

        accel_a = np.linalg.solve(Ma, rhs_a + tau_A)
        accel_b = np.linalg.solve(Mb, rhs_b + tau_B)

        return np.concatenate([oa, ob, accel_a, accel_b])

    alpha0 = np.broadcast_to(np.asarray(alpha0, dtype=float), (N,)).copy()
    beta0 = np.broadcast_to(np.asarray(beta0, dtype=float), (N,)).copy()
    state0 = np.concatenate([alpha0, beta0, np.zeros(N), np.zeros(N)])

    t_eval = np.linspace(0.0, t_max, n_frames)
    sol = solve_ivp(
        deriv, [0.0, t_max], state0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    return {
        "t": sol.t,
        "angles_A": sol.y[:N],
        "angles_B": sol.y[N:2 * N],
        "pivot_A": pivot_A,
        "pivot_B": pivot_B,
        "l_A": l_A,
        "l_B": l_B,
        "attach_A": attach_A,
        "attach_B": attach_B,
        "spring_anchor": (None if spring_anchor is None else np.asarray(spring_anchor)),
    }


def animate_two_chains(result, save_path: str | None = None,
                       fps: int = 30, margin: float = 0.05,
                       dpi: int = 200, figsize: float = 10.0):
    """Animate the result of ``simulate_two_chains``.

    The plot box is a square sized to fit both chains' natural reach and
    any actual trajectory excursion — robust regardless of geometry.

    Output resolution is ``figsize × dpi`` pixels per side. With the
    defaults (figsize=10, dpi=200) you get 2000×2000 GIF frames. Bump
    further if you really need it (file size grows roughly as resolution²;
    Telegram limit for bot uploads is 50 MB — at 2000² with 240 frames
    you're typically at 5-15 MB, well under). For very long animations
    consider saving as ``.mp4`` instead of ``.gif`` — same visual,
    much smaller files.
    """
    angles_A = result["angles_A"]
    angles_B = result["angles_B"]
    pivot_A = result["pivot_A"]
    pivot_B = result["pivot_B"]
    l_A = result["l_A"]
    l_B = result["l_B"]
    attach_A = result["attach_A"]
    attach_B = result["attach_B"]
    spring_anchor = result.get("spring_anchor", None)

    xA, yA = _link_positions(angles_A, l_A, anchor=tuple(pivot_A))
    xB, yB = _link_positions(angles_B, l_B, anchor=tuple(pivot_B))

    N = angles_A.shape[0]
    aiA = attach_A + 1   # in *_link_positions* coords (which prepend pivot)
    aiB = attach_B + 1

    # Plot box: square that fits everything
    cx = 0.5 * (pivot_A[0] + pivot_B[0])
    cy = 0.5 * (pivot_A[1] + pivot_B[1])
    xs = np.concatenate([xA.ravel(), xB.ravel()])
    ys = np.concatenate([yA.ravel(), yB.ravel()])
    half = max(
        N * l_A, N * l_B,
        np.abs(xs - cx).max(), np.abs(ys - cy).max(),
        0.5 * np.hypot(pivot_B[0] - pivot_A[0], pivot_B[1] - pivot_A[1])
        + max(N * l_A, N * l_B) * 0.2,
    ) * (1.0 + margin)

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=18)
    title = f"Two coupled chains  (N={N}, attach=({attach_A},{attach_B}))"
    if spring_anchor is not None:
        title += (
            f"\nspring anchor at t=0:  ({spring_anchor[0]:.2f}, "
            f"{spring_anchor[1]:.2f})"
        )
    ax.set_title(title, fontsize=22)

    (line_A,) = ax.plot([], [], "-", color="royalblue", lw=2.5)
    (line_B,) = ax.plot([], [], "-", color="seagreen", lw=2.5)
    (spring,) = ax.plot([], [], "--", color="gray", lw=1.8, alpha=0.7)
    tips = ax.scatter([], [], s=140, color="crimson", zorder=5)
    attach_dots = ax.scatter([], [], s=100, color="orange",
                             edgecolors="black", lw=1.0, zorder=6)
    ax.scatter([pivot_A[0], pivot_B[0]], [pivot_A[1], pivot_B[1]],
               s=160, color="black", marker="^", zorder=4)

    def update(frame):
        line_A.set_data(xA[:, frame], yA[:, frame])
        line_B.set_data(xB[:, frame], yB[:, frame])
        spring.set_data(
            [xA[aiA, frame], xB[aiB, frame]],
            [yA[aiA, frame], yB[aiB, frame]],
        )
        tips.set_offsets([
            [xA[-1, frame], yA[-1, frame]],
            [xB[-1, frame], yB[-1, frame]],
        ])
        attach_offsets = []
        if attach_A != N - 1:
            attach_offsets.append([xA[aiA, frame], yA[aiA, frame]])
        if attach_B != N - 1:
            attach_offsets.append([xB[aiB, frame], yB[aiB, frame]])
        attach_dots.set_offsets(np.array(attach_offsets) if attach_offsets
                                else np.empty((0, 2)))
        return line_A, line_B, spring, tips, attach_dots

    anim = FuncAnimation(fig, update, frames=angles_A.shape[1],
                         interval=1000 / fps, blit=True)
    if save_path:
        ext = str(save_path).lower().rsplit(".", 1)[-1]
        if ext == "mp4":
            writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
    return fig, anim


if __name__ == "__main__":
    # Example 1: original Mathematica-style setup, but expressed via spring_anchor.
    # Pivots are 50 apart on the x-axis, spring sits at the origin.
    # Chains will stretch toward (0, 0) — matches old Dist=49.9 setup.
    res = simulate_two_chains(
        N=50,
        pivot_A=(-25.0, 0.0), pivot_B=(25.0, 0.0),
        spring_anchor=(0.0, 0.0),
        kappa=1e5, d=5.0,
        t_max=10.0, n_frames=300,
    )
    print(f"Example 1: l_A={res['l_A']:.3f}, l_B={res['l_B']:.3f} (auto)")
    animate_two_chains(res, save_path="coupled_pendulums.gif")
    print("  saved coupled_pendulums.gif")

    # Example 2: spring anchored ABOVE and to the LEFT of the centre — chains
    # have different lengths and angles to reach it.
    res = simulate_two_chains(
        N=50,
        pivot_A=(-15.0, 0.0), pivot_B=(15.0, 0.0),
        spring_anchor=(-3.0, 8.0),
        kappa=1e4, d=2.0,
        t_max=8.0, n_frames=240,
    )
    print(f"Example 2: l_A={res['l_A']:.3f}, l_B={res['l_B']:.3f}")
    animate_two_chains(res, save_path="coupled_offset_anchor.gif")
    print("  saved coupled_offset_anchor.gif")
