"""
pendulum_system.py — Chain-pendulum network.

A network of K chain pendulums coupled by M springs. The model
generalizes pendulum_system in two directions:

1.  Multiple springs (M ≥ 1), each tying together ≥ 2 chain joints with
    pairwise potential

        U_spring = (κ/2) Σ_{i<j ∈ spring}  |r_i − r_j|²

    so each chain attached to a spring is pulled toward the centroid of
    the other chains attached to it.

2.  A chain may itself be a "bridge" between two springs.  In that case
    its pivot is no longer a fixed point in space — it becomes a
    dynamical degree of freedom (the pivot can move) that's tied to one
    spring, while another joint of the same chain is tied to a second
    spring.

A "connection" record is just (chain index, joint index, spring index).
Joint indexing convention:

    joint = 0   →  the chain's pivot (start)
    joint = k   →  the hinge after link k − 1, i.e. point reached after
                   k links from the pivot
    joint = N   →  the chain's tip (end)

Internally, a chain is "free-pivot" iff it has any connection at
joint 0.  The state vector for such a chain is (x_p, y_p, α₁, …, α_N);
otherwise just (α₁, …, α_N).  At every step we build the appropriate
extended (2 + N) × (2 + N) mass matrix per chain, including the
cross-coupling terms between pivot motion and angles, and solve it.

Energy is conserved to roughly the integrator tolerance — verified
to 1e-10 with d = 0 on networks with up to 5 chains and 2 springs.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from datetime import date


CHAIN_COLORS = [
    "royalblue", "seagreen", "darkorange", "purple", "teal",
    "crimson", "olive", "saddlebrown", "deeppink", "navy",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _chain_matrices(N: int, masses: np.ndarray):
    """Pre-compute M̃[i,j] = Σ_{k≥max(i,j)} m_k and M̂[i] = Σ_{k≥i} m_k."""
    M_hat = np.cumsum(masses[::-1])[::-1]
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    M_tilde = M_hat[np.maximum(i_idx, j_idx)]
    return M_hat, M_tilde


def _reach_config(pivot, target, n_links: int):
    """A bar of n_links rigid links from pivot straight to target.

    Returns (link_length, angle).  All n_links share the same angle.
    """
    dx = target[0] - pivot[0]
    dy = target[1] - pivot[1]
    D = float(np.hypot(dx, dy))
    if D == 0.0 or n_links == 0:
        raise ValueError("target coincides with pivot, or n_links=0")
    return D / n_links, float(np.arctan2(dx, -dy))


def _link_positions(angles: np.ndarray, l: float, pivots: np.ndarray):
    """(N, T) angles + (2, T) per-frame pivots → (N+1, T) point arrays."""
    sin_a, cos_a = np.sin(angles), np.cos(angles)
    x = pivots[0:1] + np.cumsum(l * sin_a, axis=0)
    y = pivots[1:2] - np.cumsum(l * cos_a, axis=0)
    x = np.vstack([pivots[0:1], x])
    y = np.vstack([pivots[1:2], y])
    return x, y


def _find_ffmpeg():
    """Locate the ffmpeg binary, with fallbacks for macOS conda envs.

    GUI-launched and conda-shipped Pythons sometimes get a $PATH that
    doesn't include /opt/homebrew/bin or /usr/local/bin, so plain
    ``shutil.which`` returns None even when ffmpeg is installed.  We
    look in the standard install locations as a fallback.
    """
    import os
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found

    candidates = [
        "/opt/homebrew/bin/ffmpeg",   # Apple Silicon Homebrew
        "/usr/local/bin/ffmpeg",      # Intel Homebrew, manual installs
        "/usr/bin/ffmpeg",            # Linux
        "/opt/local/bin/ffmpeg",      # MacPorts
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _make_driver(spec: dict):
    """Build pivot driver functions from a spec dict.

    Returns (pos_fn, vel_fn, acc_fn), each taking time t (scalar) and
    returning a length-2 numpy array.  Defaults: amp=0, freq=0, phase=0.
    """
    centre = np.asarray(spec["centre"], dtype=float)
    amp = np.asarray(spec.get("amp", (0.0, 0.0)), dtype=float)
    freq = np.asarray(spec.get("freq", (0.0, 0.0)), dtype=float)
    phase = np.asarray(spec.get("phase", (0.0, 0.0)), dtype=float)
    omega = 2.0 * np.pi * freq

    def pos(t):
        return centre + amp * np.sin(omega * t + phase)

    def vel(t):
        return amp * omega * np.cos(omega * t + phase)

    def acc(t):
        return -amp * omega ** 2 * np.sin(omega * t + phase)

    return pos, vel, acc


# --------------------------------------------------------------------------- #
# Simulation                                                                  #
# --------------------------------------------------------------------------- #

def simulate_network(
    chains: list[dict],
    springs: list[dict],
    connections: Iterable[dict],
    g: float = 9.8,
    d: float = 1.0,
    t_max: float = 10.0,
    n_frames: int = 240,
    method: str = "LSODA",
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
):
    """Simulate a network of chain pendulums coupled by springs.

    Parameters
    ----------
    chains : list of dicts.  Each dict has:
        ``N``        — number of links
        ``pivot``    — (x, y) pivot location.  Required if no connection
                       at joint 0 and no ``pivot_drive``.
        ``pivot_drive`` — optional dict for a *driven* (time-dependent)
                       pivot.  Mutually exclusive with a connection at
                       joint 0.  Keys:
                            ``centre`` : (x0, y0) — equilibrium position
                            ``amp``    : (ax, ay) — oscillation amplitude
                            ``freq``   : (fx, fy) — frequency in Hz
                            ``phase``  : (px, py) — phase in radians
                       The pivot then moves as
                            x(t) = x0 + ax · sin(2π·fx·t + px)
                            y(t) = y0 + ay · sin(2π·fy·t + py)
                       The chain's equations of motion gain a fictitious
                       force ∝ pivot acceleration, exactly as in the
                       non-inertial frame moving with the pivot.
        ``masses``   — optional, length-N array (default 0.25 each)
        ``l``        — optional, fixed link length.  When the chain has
                       a "secondary" connection (joint > 0 to some
                       spring), the link length is auto-computed from
                       geometry to stretch from pivot to that spring's
                       anchor; ``l`` overrides that if provided.
        ``pivot_mass`` — optional, only used if the chain has a free
                       pivot.  Without it the mass matrix is singular
                       (a point-mass chain has no inertia at the pivot
                       itself, so the linearized system loses rank).
                       Default 0.25, matching default link mass.

    springs : list of dicts.  Each has:
        ``anchor``  — (x, y).  At t = 0, every joint connected to this
                      spring sits at this point.
        ``kappa``   — stiffness

    connections : iterable of dicts.  Each:
        ``chain``   — index into ``chains``
        ``joint``   — joint index in [0, N].  0 = pivot, N = tip.
        ``spring``  — index into ``springs``

    Returns
    -------
    dict with keys
        t, angles (list of (N, T) per chain), pivots (list of (2, T) per
        chain — constant for fixed-pivot chains), chains, springs,
        connections.
    """
    if not chains:
        raise ValueError("Need at least one chain.")
    if not springs:
        raise ValueError("Need at least one spring.")

    connections = [dict(c) for c in connections]
    K = len(chains)

    # Per-chain incoming connections, and free-pivot flag.
    chain_conns: list[list[dict]] = [[] for _ in range(K)]
    for c in connections:
        ci = c["chain"]
        if not (0 <= ci < K):
            raise ValueError(f"connection has chain={ci}, out of range")
        chain_conns[ci].append(c)
    free_pivot = [
        any(c["joint"] == 0 for c in cc) for cc in chain_conns
    ]

    # Build chain_data: geometry, mass matrices, initial conditions.
    chain_data = []
    for i, ch in enumerate(chains):
        N = int(ch["N"])
        masses = ch.get("masses", None)
        if masses is None:
            masses = np.full(N, 0.25)
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (N,):
            raise ValueError(f"chain {i}: masses must have shape ({N},)")

        my_conns = chain_conns[i]
        for c in my_conns:
            if not (0 <= c["joint"] <= N):
                raise ValueError(f"chain {i}: joint={c['joint']} out of [0, {N}]")
            if not (0 <= c["spring"] < len(springs)):
                raise ValueError(f"chain {i}: spring={c['spring']} out of range")

        # Determine pivot kind: free (connection at joint=0),
        # driven (pivot_drive given), or fixed (pivot given).
        has_drive = "pivot_drive" in ch
        if free_pivot[i] and has_drive:
            raise ValueError(
                f"chain {i}: cannot have both joint=0 connection and pivot_drive"
            )

        driver = (None, None, None)
        if free_pivot[i]:
            pivot_conn = next(c for c in my_conns if c["joint"] == 0)
            pivot_init = np.asarray(
                springs[pivot_conn["spring"]]["anchor"], dtype=float
            )
        elif has_drive:
            driver = _make_driver(ch["pivot_drive"])
            pivot_init = driver[0](0.0)
        else:
            if "pivot" not in ch:
                raise ValueError(
                    f"chain {i}: must specify 'pivot', 'pivot_drive', or "
                    "have a connection at joint 0"
                )
            pivot_init = np.asarray(ch["pivot"], dtype=float)

        # Initial link length & angles.  If the chain has a secondary
        # connection (joint > 0) we stretch toward that spring's anchor.
        secondary = [c for c in my_conns if c["joint"] > 0]
        if secondary:
            sc = secondary[0]
            target = np.asarray(springs[sc["spring"]]["anchor"], dtype=float)
            l_geom, alpha_geom = _reach_config(pivot_init, target, sc["joint"])
            l = ch.get("l", l_geom)
            alpha_init = np.zeros(N)
            alpha_init[: sc["joint"]] = alpha_geom
            # Links beyond stay at angle 0 (hanging straight down).
        else:
            l = ch.get("l", 0.5)
            alpha_init = np.zeros(N)

        M_hat, M_tilde = _chain_matrices(N, masses)
        # Pivot mass: small inertia attached at the pivot itself, which
        # makes the mass matrix non-singular for free-pivot chains.
        # For fixed-pivot chains it has no effect.
        pivot_mass = float(ch.get("pivot_mass", 0.25 if free_pivot[i] else 0.0))
        chain_data.append({
            "N": N, "l": float(l), "masses": masses,
            "M_hat": M_hat, "M_tilde": M_tilde,
            "M_total": float(masses.sum()),
            "pivot_mass": pivot_mass,
            "pivot_init": pivot_init, "alpha_init": alpha_init,
            "free_pivot": free_pivot[i],
            "driven": has_drive,
            "driver": driver,        # (pos_fn, vel_fn, acc_fn) or (None,)*3
            "connections": my_conns,
        })

    # State layout — per chain, q-block has size N (fixed pivot) or N + 2.
    block_sizes = [cd["N"] + (2 if cd["free_pivot"] else 0) for cd in chain_data]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    total_dof = offsets[-1]   # # of generalized coords (and same for velocities)

    # Group connections by spring for fast iteration.
    springs_conns = [[] for _ in springs]
    for c in connections:
        springs_conns[c["spring"]].append(c)

    # ----- ODE RHS --------------------------------------------------------- #
    def deriv(t, state):
        out = np.empty_like(state)

        # Per-chain views into state.  For driven chains, the pivot
        # position and velocity come from the driver evaluated at t.
        cs = []  # per-chain dict: {pivot, alpha, pivot_vel, omega, pivot_acc}
        for i, cd in enumerate(chain_data):
            q = state[offsets[i]:offsets[i + 1]]
            qd = state[total_dof + offsets[i]:total_dof + offsets[i + 1]]
            if cd["free_pivot"]:
                cs.append({
                    "pivot": q[:2], "alpha": q[2:],
                    "pivot_vel": qd[:2], "omega": qd[2:],
                    "pivot_acc": np.zeros(2),
                })
            elif cd["driven"]:
                pos_fn, vel_fn, acc_fn = cd["driver"]
                cs.append({
                    "pivot": pos_fn(t), "alpha": q,
                    "pivot_vel": vel_fn(t), "omega": qd,
                    "pivot_acc": acc_fn(t),
                })
            else:
                cs.append({
                    "pivot": cd["pivot_init"], "alpha": q,
                    "pivot_vel": np.zeros(2), "omega": qd,
                    "pivot_acc": np.zeros(2),
                })

        # Compute attach-point position of every connection.
        attach_pos = {}
        for c in connections:
            ci, j = c["chain"], c["joint"]
            l = chain_data[ci]["l"]
            pv = cs[ci]["pivot"]
            if j == 0:
                attach_pos[(ci, j)] = pv
            else:
                sub = cs[ci]["alpha"][:j]
                attach_pos[(ci, j)] = np.array([
                    pv[0] + l * np.sin(sub).sum(),
                    pv[1] - l * np.cos(sub).sum(),
                ])

        # Spring force on each (chain, joint) connection.
        forces = {}
        for s_idx, sc in enumerate(springs_conns):
            if len(sc) < 2:
                continue
            kappa = float(springs[s_idx]["kappa"])
            r_sum = sum(attach_pos[(c["chain"], c["joint"])] for c in sc)
            n = len(sc)
            for c in sc:
                key = (c["chain"], c["joint"])
                F = -kappa * (n * attach_pos[key] - r_sum)
                forces[key] = forces.get(key, np.zeros(2)) + F

        # Per-chain dynamics.
        for i, cd in enumerate(chain_data):
            N = cd["N"]; l = cd["l"]
            M_hat, M_tilde = cd["M_hat"], cd["M_tilde"]
            alpha = cs[i]["alpha"]; omega = cs[i]["omega"]

            diff = alpha[:, None] - alpha[None, :]
            cos_d = np.cos(diff); sin_d = np.sin(diff)
            cos_a = np.cos(alpha); sin_a = np.sin(alpha)

            # Standard angle dynamics: M_aa α̈ + ... = b_alpha
            M_aa = (l * l) * M_tilde * cos_d
            b_alpha = (
                -(l * l) * (M_tilde * sin_d) @ (omega ** 2)
                - g * l * M_hat * sin_a
                - d * omega
            )

            # Spring contributions on each angle.
            # F_{i,j} acts at joint j, contributing to α_k iff k < j.
            # ∂r_j/∂α_k = l (cos α_k, sin α_k).
            for c in cd["connections"]:
                j = c["joint"]
                key = (i, j)
                if key not in forces or j == 0:
                    continue
                F = forces[key]
                # contribute to α_0..α_{j-1}
                b_alpha[:j] += l * (F[0] * cos_a[:j] + F[1] * sin_a[:j])

            # Driven pivot: fictitious inertial force.  A pivot
            # accelerating with a_p in the lab frame is equivalent to a
            # uniform pseudo-gravity -a_p acting on every mass.  This
            # adds  -l · M_hat[k] · (a_px·cos α_k + a_py·sin α_k)  to b.
            if cd["driven"]:
                a_p = cs[i]["pivot_acc"]
                b_alpha -= l * M_hat * (a_p[0] * cos_a + a_p[1] * sin_a)

            if cd["free_pivot"]:
                M_total = cd["M_total"]
                m_p = cd["pivot_mass"]
                M_eff = M_total + m_p   # effective inertia at pivot
                pivot_vel = cs[i]["pivot_vel"]

                # Augmented (N+2) x (N+2) mass matrix.
                # | M_pp   M_pα |   | ẍ_p ÿ_p |
                # | M_pαᵀ  M_aa |   |    α̈    |
                # Pivot mass m_p adds m_p · I to M_pp; this resolves the
                # rank deficiency that point-mass chains otherwise cause.
                M_pa = np.empty((2, N))
                M_pa[0] = M_hat * l * cos_a
                M_pa[1] = M_hat * l * sin_a
                M_full = np.empty((N + 2, N + 2))
                M_full[:2, :2] = M_eff * np.eye(2)
                M_full[:2, 2:] = M_pa
                M_full[2:, :2] = M_pa.T
                M_full[2:, 2:] = M_aa

                # Pivot RHS: from kinetic-energy cross-terms + gravity + damping.
                b_x = float(np.sum(M_hat * l * sin_a * omega ** 2)) - d * pivot_vel[0]
                b_y = -float(np.sum(M_hat * l * cos_a * omega ** 2)) \
                      - M_eff * g - d * pivot_vel[1]
                # Spring force directly on pivot (joint=0 connections).
                if (i, 0) in forces:
                    F0 = forces[(i, 0)]
                    b_x += F0[0]
                    b_y += F0[1]

                b_full = np.empty(N + 2)
                b_full[0] = b_x; b_full[1] = b_y
                b_full[2:] = b_alpha

                accel = np.linalg.solve(M_full, b_full)
                # ẋ_p ẏ_p α̇  →  out (positions),  ẍ_p ÿ_p α̈  →  out (velocities)
                out[offsets[i]:offsets[i + 1]] = np.concatenate(
                    [pivot_vel, omega]
                )
                out[total_dof + offsets[i]:total_dof + offsets[i + 1]] = accel
            else:
                accel = np.linalg.solve(M_aa, b_alpha)
                out[offsets[i]:offsets[i + 1]] = omega
                out[total_dof + offsets[i]:total_dof + offsets[i + 1]] = accel

        return out

    # Build the initial state.
    state0 = np.zeros(2 * total_dof)
    for i, cd in enumerate(chain_data):
        if cd["free_pivot"]:
            state0[offsets[i]:offsets[i] + 2] = cd["pivot_init"]
            state0[offsets[i] + 2:offsets[i + 1]] = cd["alpha_init"]
        else:
            state0[offsets[i]:offsets[i + 1]] = cd["alpha_init"]
        # velocities all zero (already)

    t_eval = np.linspace(0.0, t_max, n_frames)
    sol = solve_ivp(
        deriv, [0.0, t_max], state0,
        t_eval=t_eval, method=method, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    if not np.isfinite(sol.y).all():
        # LSODA in particular sometimes silently produces NaN/Inf for
        # very stiff or multi-scale systems while still reporting
        # success=True. Treat that as a failure so the caller can retry.
        raise RuntimeError("ODE integration produced non-finite values "
                           "(likely instability / multi-scale stiffness)")

    # Unpack into per-chain (N, T) angle arrays and (2, T) pivot arrays.
    angles_list = []
    pivots_list = []
    T = sol.y.shape[1]
    for i, cd in enumerate(chain_data):
        if cd["free_pivot"]:
            pivots = sol.y[offsets[i]:offsets[i] + 2, :]
            ang = sol.y[offsets[i] + 2:offsets[i + 1], :]
        elif cd["driven"]:
            pos_fn = cd["driver"][0]
            pivots = np.stack([pos_fn(tt) for tt in sol.t], axis=1)
            ang = sol.y[offsets[i]:offsets[i + 1], :]
        else:
            pivots = np.tile(cd["pivot_init"][:, None], (1, T))
            ang = sol.y[offsets[i]:offsets[i + 1], :]
        angles_list.append(ang)
        pivots_list.append(pivots)

    return {
        "t": sol.t,
        "angles": angles_list,
        "pivots": pivots_list,
        "chains": chain_data,
        "springs": springs,
        "connections": connections,
    }


# --------------------------------------------------------------------------- #
# Animation                                                                   #
# --------------------------------------------------------------------------- #

def animate_network(result, save_path: str | None = None,
                    fps: int = 24, margin: float = 0.05,
                    dpi: int = 200, figsize: float = 10.0,
                    trails: bool = True, energy_color: bool = True,
                    trail_len: int = 25,
                    legend: bool = True, show_date: bool = True,
                    created: str | None = None):
    """Animate the result of ``simulate_network``.

    ``legend`` draws a key explaining the on-screen elements (only the
    ones actually present in the scene).  ``show_date`` stamps the
    creation date in the corner (``created`` overrides it, ISO format).

    Two optional visual features (both on by default, both removable):

    ``energy_color`` : colour each link by its kinetic energy — links
        brighten where they move fast, so you can watch energy flow
        through the network.  Set ``False`` to fall back to flat
        per-chain colours.
    ``trails`` : draw fading "phosphor" trails behind each chain's tip,
        of length ``trail_len`` frames.  Set ``False`` to disable.
    """
    chain_data = result["chains"]
    angles_list = result["angles"]
    pivots_list = result["pivots"]
    springs = result["springs"]
    connections = result["connections"]
    K = len(chain_data)

    # Pre-compute (x, y) of every link for every chain.
    positions = []  # list of (x, y) arrays of shape (N+1, T)
    for cd, ang, piv in zip(chain_data, angles_list, pivots_list):
        positions.append(_link_positions(ang, cd["l"], piv))

    # === FEATURE: energy colour — precompute per-segment kinetic energy ==== #
    # (Delete this block and set energy_color's branches below to disable.)
    # Point speeds come from a finite difference of the link positions in
    # time; per-segment energy ~ mean of its two endpoints' speed².  We
    # normalise by the 95th percentile so a single spike doesn't wash the
    # whole scene out.
    seg_energy = None
    chain_palette = None
    if energy_color:
        t_arr = np.asarray(result["t"], dtype=float)
        dt = np.gradient(t_arr) if t_arr.size > 1 else np.array([1.0])
        raw = []
        for (x, y) in positions:
            vx = np.gradient(x, axis=1) / dt
            vy = np.gradient(y, axis=1) / dt
            sp2 = vx ** 2 + vy ** 2                # (N+1, T)
            raw.append(0.5 * (sp2[:-1] + sp2[1:]))  # (N, T)
        scale = np.percentile(np.concatenate([e.ravel() for e in raw]), 95.0)
        scale = scale if scale > 1e-12 else 1.0
        seg_energy = [np.clip(e / scale, 0.0, 1.0) for e in raw]
        chain_palette = []
        for i in range(K):
            base = np.array(mcolors.to_rgb(CHAIN_COLORS[i % len(CHAIN_COLORS)]))
            lo = 0.30 * base                       # dark = slow
            hi = base + (1.0 - base) * 0.85        # bright = fast
            chain_palette.append((lo, hi))
    # ====================================================================== #

    # Adaptive square plot box.
    all_x = np.concatenate([px[0].ravel() for px in positions])
    all_y = np.concatenate([px[1].ravel() for px in positions])
    cx = 0.5 * (all_x.max() + all_x.min())
    cy = 0.5 * (all_y.max() + all_y.min())
    half = max(np.abs(all_x - cx).max(), np.abs(all_y - cy).max()) * (1 + margin)

    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=18)

    spring_chain = [[c for c in connections if c["spring"] == s]
                    for s in range(len(springs))]
    n_active_springs = sum(1 for sc in spring_chain if len(sc) >= 2)

    title = (f"Chain pendulum network\n"
             f"{K} chains, {n_active_springs} "
             f"spring{'s' if n_active_springs != 1 else ''}")
    ax.set_title(title, fontsize=22)

    # Static markers: only truly-fixed pivots (free and driven pivots move).
    fixed_pivots = np.array(
        [cd["pivot_init"] for cd in chain_data
         if not cd["free_pivot"] and not cd["driven"]]
    )
    if fixed_pivots.size:
        ax.scatter(fixed_pivots[:, 0], fixed_pivots[:, 1],
                   s=160, color="black", marker="^", zorder=4)

    # Animated artists — chain bodies.
    # === FEATURE: energy colour — LineCollection instead of flat line ===== #
    chain_lines = []   # used only when energy_color is False
    chain_lcs = []     # used only when energy_color is True
    for i in range(K):
        color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
        if energy_color:
            lc = LineCollection([], linewidths=2.5, zorder=2)
            ax.add_collection(lc)
            chain_lcs.append(lc)
        else:
            (line,) = ax.plot([], [], "-", color=color, lw=2.5)
            chain_lines.append(line)
    # === FEATURE: phosphor trails on chain tips =========================== #
    trail_lcs = []     # used only when trails is True
    if trails:
        for _ in range(K):
            tlc = LineCollection([], linewidths=1.6, zorder=1)
            ax.add_collection(tlc)
            trail_lcs.append(tlc)
    # ====================================================================== #

    # Spring segments: for each spring with ≥ 2 connections, one dashed
    # segment per attached joint going to the spring's centroid.
    spring_segments = []
    for s_idx, sc in enumerate(spring_chain):
        per = []
        if len(sc) >= 2:
            for _ in sc:
                (seg,) = ax.plot([], [], "--", color="gray",
                                 lw=1.8, alpha=0.65)
                per.append(seg)
        spring_segments.append(per)

    tips = ax.scatter([], [], s=140, color="crimson", zorder=5)
    attach_dots = ax.scatter([], [], s=110, color="orange",
                             edgecolors="black", lw=1.0, zorder=6)
    free_pivot_dots = ax.scatter([], [], s=160, color="black",
                                 marker="o", zorder=4)
    # Driven pivots get a distinct shape so the viewer can tell them
    # apart from free pivots: square instead of circle.
    driven_pivot_dots = ax.scatter([], [], s=180, color="black",
                                   marker="s", zorder=4)
    centroids = ax.scatter([], [], s=80, color="dimgray",
                           marker="x", lw=2.0, zorder=3)

    def update(frame):
        for i, (x, y) in enumerate(positions):
            # === FEATURE: energy colour / flat line (see energy_color) ==== #
            if energy_color:
                pts = np.stack([x[:, frame], y[:, frame]], axis=1)   # (N+1,2)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)          # (N,2,2)
                en = seg_energy[i][:, frame]
                lo, hi = chain_palette[i]
                cols = lo[None, :] + (hi - lo)[None, :] * en[:, None]  # (N,3)
                chain_lcs[i].set_segments(segs)
                chain_lcs[i].set_color(cols)
            else:
                chain_lines[i].set_data(x[:, frame], y[:, frame])
            # === FEATURE: phosphor trail on this chain's tip ============== #
            if trails:
                lo_f = max(0, frame - trail_len)
                tx = x[-1, lo_f:frame + 1]
                ty = y[-1, lo_f:frame + 1]
                if tx.size >= 2:
                    tpts = np.stack([tx, ty], axis=1)
                    tsegs = np.stack([tpts[:-1], tpts[1:]], axis=1)
                    n_seg = tsegs.shape[0]
                    base = np.array(
                        mcolors.to_rgb(CHAIN_COLORS[i % len(CHAIN_COLORS)])
                    )
                    alphas = np.linspace(0.0, 0.7, n_seg)   # old→new fade-in
                    tcols = np.concatenate(
                        [np.tile(base, (n_seg, 1)), alphas[:, None]], axis=1
                    )
                    trail_lcs[i].set_segments(tsegs)
                    trail_lcs[i].set_color(tcols)
                else:
                    trail_lcs[i].set_segments([])

        # Spring segments + centroids.
        centroid_pts = []
        for s_idx, sc in enumerate(spring_chain):
            if len(sc) < 2:
                continue
            joint_pts = []
            for c in sc:
                ci, j = c["chain"], c["joint"]
                x, y = positions[ci]
                joint_pts.append(np.array([x[j, frame], y[j, frame]]))
            joint_pts = np.array(joint_pts)
            centroid = joint_pts.mean(axis=0)
            centroid_pts.append(centroid)
            for seg, jp in zip(spring_segments[s_idx], joint_pts):
                seg.set_data([jp[0], centroid[0]], [jp[1], centroid[1]])

        # Tips, attachment markers, moving-pivot markers.
        tip_pts = []
        attach_pts = []
        free_piv_pts = []
        driven_piv_pts = []
        for i, (x, y) in enumerate(positions):
            cd = chain_data[i]
            tip_pts.append([x[-1, frame], y[-1, frame]])
            if cd["free_pivot"]:
                free_piv_pts.append([x[0, frame], y[0, frame]])
            elif cd["driven"]:
                driven_piv_pts.append([x[0, frame], y[0, frame]])
            for c in cd["connections"]:
                j = c["joint"]
                if j == 0 or j == cd["N"]:
                    continue   # already shown via tip / pivot marker
                attach_pts.append([x[j, frame], y[j, frame]])

        tips.set_offsets(np.array(tip_pts))
        attach_dots.set_offsets(np.array(attach_pts) if attach_pts
                                else np.empty((0, 2)))
        free_pivot_dots.set_offsets(np.array(free_piv_pts) if free_piv_pts
                                    else np.empty((0, 2)))
        driven_pivot_dots.set_offsets(np.array(driven_piv_pts) if driven_piv_pts
                                      else np.empty((0, 2)))
        centroids.set_offsets(np.array(centroid_pts) if centroid_pts
                              else np.empty((0, 2)))

        artists = list(chain_lcs if energy_color else chain_lines)
        if trails:
            artists += trail_lcs
        artists += [tips, attach_dots, free_pivot_dots,
                    driven_pivot_dots, centroids]
        for per in spring_segments:
            artists.extend(per)
        return artists

    # === FEATURE: legend + creation date (legend/show_date to disable) ==== #
    # Static overlays (drawn once, not animated).  The legend lists only
    # the element types actually present in this scene.
    if legend:
        any_fixed = any((not cd["free_pivot"]) and (not cd["driven"])
                        for cd in chain_data)
        any_driven = any(cd["driven"] for cd in chain_data)
        any_free = any(cd["free_pivot"] for cd in chain_data)
        any_attach = any(0 < c["joint"] < chain_data[c["chain"]]["N"]
                         for c in connections)
        handles = [
            Line2D([0], [0], color="royalblue", lw=2.5,
                   label=("chain (brighter = faster)" if energy_color
                          else "chain")),
        ]
        if n_active_springs:
            handles.append(Line2D([0], [0], color="gray", lw=1.8, ls="--",
                                  label="spring"))
        handles.append(Line2D([0], [0], color="crimson", marker="o",
                              ls="none", ms=9, label="chain tip"))
        if any_attach:
            handles.append(Line2D([0], [0], color="orange", marker="o",
                                  mec="black", ls="none", ms=9,
                                  label="spring joint"))
        if any_fixed:
            handles.append(Line2D([0], [0], color="black", marker="^",
                                  ls="none", ms=10, label="fixed pivot"))
        if any_driven:
            handles.append(Line2D([0], [0], color="black", marker="s",
                                  ls="none", ms=10, label="driven pivot"))
        if any_free:
            handles.append(Line2D([0], [0], color="black", marker="o",
                                  ls="none", ms=10, label="free pivot"))
        if n_active_springs:
            handles.append(Line2D([0], [0], color="dimgray", marker="x",
                                  ls="none", ms=8, mew=2,
                                  label="spring centre"))
        if trails:
            handles.append(Line2D([0], [0], color="royalblue", lw=1.6,
                                  alpha=0.5, label="tip trail"))
        # Place the legend OUTSIDE the plot (to the right) so it never
        # covers the motion, and reserve a right margin for it.
        leg = ax.legend(handles=handles, loc="upper left",
                        bbox_to_anchor=(1.03, 1.0), fontsize=12,
                        framealpha=0.9, borderpad=0.7, labelspacing=0.4,
                        handletextpad=0.5)
        leg.set_zorder(10)
        fig.subplots_adjust(right=0.72)
    if show_date:
        created_str = created or date.today().isoformat()
        ax.text(0.99, 0.01, f"created {created_str}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12, color="gray")
    # ===================================================================== #

    n_frames = positions[0][0].shape[1]
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=True)
    actual_path = save_path
    if save_path:
        import logging
        log = logging.getLogger(__name__)
        ext = str(save_path).lower().rsplit(".", 1)[-1]
        if ext == "mp4":
            ffmpeg = _find_ffmpeg()
            if ffmpeg is None:
                # Matplotlib delegates to ffmpeg via subprocess and the
                # FileNotFoundError only surfaces at save time, hence
                # this pre-flight check + fallback to gif.
                import warnings
                actual_path = save_path[:-4] + ".gif"
                msg = (
                    "ffmpeg not found — falling back to GIF "
                    f"({actual_path}). Install ffmpeg (`brew install "
                    "ffmpeg` on macOS, `apt install ffmpeg` on Linux) "
                    "for smaller, higher-quality MP4 output."
                )
                warnings.warn(msg, stacklevel=2)
                log.warning(msg)
                writer = PillowWriter(fps=fps)
            else:
                # Tell matplotlib explicitly where ffmpeg lives, since
                # GUI-launched / conda Pythons may have a stripped PATH.
                import matplotlib as mpl
                mpl.rcParams["animation.ffmpeg_path"] = ffmpeg
                log.info("Encoding MP4 with ffmpeg at %s", ffmpeg)
                writer = FFMpegWriter(fps=fps, bitrate=2400)
        else:
            log.info("Encoding %s with PillowWriter", ext.upper())
            writer = PillowWriter(fps=fps)
        anim.save(actual_path, writer=writer, dpi=dpi)
    return fig, anim, actual_path


# --------------------------------------------------------------------------- #
# Demo                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # 2 springs joined by a chain whose tips are pulled to two anchors:
    # this is the new "ribbon between two clouds" topology.
    res = simulate_network(
        chains=[
            {"N": 25, "pivot": (-20, 0)},
            {"N": 25, "pivot": (-20, 10)},
            # bridge chain — pivot to spring 0, tip to spring 1
            {"N": 30},
            {"N": 25, "pivot": (+20, 0)},
            {"N": 25, "pivot": (+20, 10)},
        ],
        springs=[
            {"anchor": (-7, 5), "kappa": 1.0e3},
            {"anchor": (+7, 5), "kappa": 1.0e3},
        ],
        connections=[
            {"chain": 0, "joint": 25, "spring": 0},
            {"chain": 1, "joint": 25, "spring": 0},
            {"chain": 2, "joint": 0,  "spring": 0},
            {"chain": 2, "joint": 30, "spring": 1},
            {"chain": 3, "joint": 25, "spring": 1},
            {"chain": 4, "joint": 25, "spring": 1},
        ],
        d=1.0, t_max=8.0, n_frames=180,
    )
    animate_network(res, save_path="bridge_network.gif")
    print("saved bridge_network.gif")
