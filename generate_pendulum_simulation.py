#!/usr/bin/env python3
# """
# generate_pendulum_simulation.py
# ================================
# Generate a chain-pendulum-network animation with random initial
# conditions and (optionally) post it to a Telegram channel.

# Inspired by https://github.com/robolamp/3_body_problem_bot

# Random configurations span 1-3 springs with 2-4 chains attached to each.
# When there are ≥ 2 springs, with some probability we also generate
# "bridge" chains connecting two springs together — both ends of the
# bridge are tied to springs, so its pivot is itself a free, dynamical
# point in space.

# Without ``-T`` and ``-N`` arguments, the script just writes a GIF to disk.
# With them, it also publishes the GIF to the given Telegram channel.

# The first attempt that runs to completion is accepted, no quality
# filtering — with very low damping the system rarely produces a boring
# result, and visual judgment is subjective.

# Per-attempt timeout: extremely stiff springs (κ ≳ 10⁶) make the ODE
# arbitrarily hard, with oscillation periods below μs scale.  We wrap
# each attempt in a ``signal.alarm``-based timeout (Unix only) and just
# move on if a sim takes too long.
# """
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Optional

import numpy as np

from pendulum_system import (
    simulate_network,
    animate_network,
)


# --------------------------------------------------------------------------- #
# Feature toggles                                                             #
# --------------------------------------------------------------------------- #
# Each new visual/sampling feature is gated here.  Flip any of these to
# False (or use the matching --no-* CLI flag) to remove that feature and
# fall back to the original behaviour.  Deleting a feature entirely is a
# matter of removing its gated block; nothing else depends on it.
ENABLE_STAR_TOPOLOGY = True      # occasional 7–10 chain hubs on one spring
ENABLE_CONTRAST_GEOMETRY = True  # mix long-thin and short-stubby chains
ENABLE_TRAILS = True             # phosphor motion trails on chain tips
ENABLE_ENERGY_COLOR = False       # colour links by kinetic energy
ENABLE_DRIVEN_RESONANCE = True   # sometimes tune a driven pivot to the
                                 # chain's own frequency (resonant build-up)
ENABLE_ADAPTIVE_DAMPING = True   # raise damping a bit for fast pivot drives
                                 # (tames chaos + integrator cost)
ENABLE_LEGEND = True             # draw a legend explaining plot elements
ENABLE_CREATION_DATE = True      # stamp the creation date on the animation


def _natural_freq(L: float, g: float = 9.8) -> float:
    """Fundamental frequency (Hz) of a uniform chain of total length ``L``
    hanging from a fixed top pivot.

    For a continuous uniform hanging chain the lowest mode is
        omega_1 = (j0 / 2) * sqrt(g / L),   j0 = 2.4048  (first zero of J0),
    so f1 = omega_1 / (2*pi).  Good enough to place a resonant drive near
    the chain's natural swing rate.
    """
    L = max(float(L), 1e-6)
    return (2.4048 / 2.0) * np.sqrt(g / L) / (2.0 * np.pi)


def _sample_drive(rng: np.random.Generator, pivot, L: float,
                  resonance: bool = True, force: bool = False) -> dict:
    """Sample a sinusoidal pivot-drive spec.

    Amplitudes and frequencies are larger than the original (which capped
    amp at 2.5 and freq at 0.4 Hz).  When ``resonance`` is on, a fraction
    of drives (always, if ``force``) are tuned to the chain's own
    fundamental frequency so it visibly builds up amplitude.  ``force``
    also guarantees a genuinely moving pivot (used for the "at least one
    moving pivot per scene" guarantee).
    """
    f_nat = _natural_freq(L)
    if resonance and (force or rng.random() < 0.40):
        # Resonant drive: match the chain's frequency, moderate amplitude
        # (energy accumulates on its own, so a huge push can go unstable).
        f = float(f_nat * rng.uniform(0.9, 5.1))
        freq_x = freq_y = f
        amp_main = float(rng.uniform(0.5, 3.2))
        if rng.random() < 0.5:
            amp_x, amp_y = amp_main, float(rng.uniform(0.0, 1.0))
        else:
            amp_x, amp_y = float(rng.uniform(0.0, 1.0)), amp_main
    else:
        # General drive: bigger and faster than before.
        amp_x = float(rng.uniform(0, 4.5)) if rng.random() < 0.75 else 0.0
        amp_y = float(rng.uniform(0, 4.5)) if rng.random() < 0.55 else 0.0
        if force and amp_x < 1.0 and amp_y < 1.0:
            amp_x = float(rng.uniform(1.5, 4.5))
        freq_x = float(rng.uniform(0.15, 0.55))
        freq_y = float(rng.uniform(0.15, 0.55))
    return {
        "centre": (float(pivot[0]), float(pivot[1])),
        "amp": (amp_x, amp_y),
        "freq": (freq_x, freq_y),
        "phase": (float(rng.uniform(0, 2 * np.pi)),
                  float(rng.uniform(0, 2 * np.pi))),
    }


# --------------------------------------------------------------------------- #
# Random parameter sampling                                                   #
# --------------------------------------------------------------------------- #

def sample_parameters(rng: np.random.Generator,
                      star_topology: bool | None = None,
                      contrast_geometry: bool | None = None,
                      driven_resonance: bool | None = None,
                      adaptive_damping: bool | None = None) -> dict:
    """Sample a random-but-reasonable network configuration.

    Returns a dict with ``chains``, ``springs``, ``connections`` ready
    to pass straight into ``simulate_network``, plus ``d`` for damping.

    ``star_topology`` / ``contrast_geometry`` / ``driven_resonance``
    default to the module-level ``ENABLE_*`` toggles; pass ``False`` to
    disable any of them.

    At least one pivot in every scene is always driven (moving).
    """
    if star_topology is None:
        star_topology = ENABLE_STAR_TOPOLOGY
    if contrast_geometry is None:
        contrast_geometry = ENABLE_CONTRAST_GEOMETRY
    if driven_resonance is None:
        driven_resonance = ENABLE_DRIVEN_RESONANCE
    if adaptive_damping is None:
        adaptive_damping = ENABLE_ADAPTIVE_DAMPING
    n_springs = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))

    # Spring anchor positions.
    if n_springs == 1:
        spring_centres = np.array([[0.0, 0.0]])
    else:
        radius = rng.uniform(6.0, 10.0)
        angles = (np.linspace(0, 2 * np.pi, n_springs, endpoint=False)
                  + rng.uniform(0, 2 * np.pi))
        spring_centres = np.stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
        ], axis=1)

    springs = []
    # Sample a single "base" log-stiffness per scene, then give each
    # spring a small offset around it. This avoids extreme spreads in
    # one scene which make the ODE multi-scale stiff.  Range 1–400 →
    # very soft to moderately stiff; weak springs let chains move more
    # independently and produce a more chaotic look.
    log10_kappa_base = rng.uniform(0.0, 2.6)
    for s_idx, centre in enumerate(spring_centres):
        anchor = (
            float(centre[0] + rng.normal(0, 1.0)),
            float(centre[1] + rng.normal(0, 1.0)),
        )
        kappa = float(10 ** (log10_kappa_base + rng.uniform(-0.3, 0.3)))
        springs.append({"anchor": anchor, "kappa": kappa})

    chains: list[dict] = []
    connections: list[dict] = []
    pendant_records: list[tuple] = []   # (chain_idx, pivot, L_eff) per pendant

    # ----- Pendant chains hanging off each spring ------------------------- #
    for s_idx, spr in enumerate(springs):
        anchor = spr["anchor"]
        # Bias toward more chains per spring — bigger Y/X/star junctions
        # produce richer collective dynamics than simple pairs.
        # === FEATURE: star/hub topology (star_topology=False to disable) === #
        # Occasionally make this spring a dense hub of 7–10 chains, whose
        # collective dynamics are much richer than simple pairs.
        if star_topology and rng.random() < 0.22:
            n_chains = int(rng.integers(7, 11))   # 7..10
        else:
            n_chains = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.35, 0.25, 0.15]))
        # =================================================================== #
        base_angle = rng.uniform(0, 2 * np.pi)
        for c in range(n_chains):
            theta = (base_angle + 2 * np.pi * c / n_chains
                     + rng.normal(0, 0.25))
            # === FEATURE: contrast geometry (contrast_geometry=False off) == #
            # Some chains are long & thin, others short & stubby.  Link
            # length follows from distance / #links, so this also produces
            # strongly asymmetric link lengths within a single scene.
            if contrast_geometry and rng.random() < 0.40:
                if rng.random() < 0.5:
                    N = int(rng.integers(6, 14))      # short & stubby
                    dist = rng.uniform(6.0, 11.0)
                else:
                    N = int(rng.integers(44, 59))     # long & thin
                    dist = rng.uniform(16.0, 24.0)
            else:
                N = int(rng.integers(20, 41))
                dist = rng.uniform(12.0, 20.0)
            # =============================================================== #
            pivot = (
                anchor[0] + dist * np.cos(theta),
                anchor[1] + dist * np.sin(theta),
            )
            ci = len(chains)

            # 35% attach somewhere in the chain's lower half rather than
            # at the tip → trailing tail hangs free below the spring,
            # adding energy and asymmetry to the motion.  Decide it first
            # so we can size a resonant drive to the chain's real length.
            if rng.random() < 0.35:
                joint = int(rng.integers(N // 2, N))
            else:
                joint = N
            # Effective length of the (full, N-link) hanging chain.  The
            # first ``joint`` links span pivot→anchor (distance ``dist``),
            # so each link is dist/joint long.
            l_link = dist / max(joint, 1)
            L_eff = N * l_link

            # 35% chance the pivot is *driven* — sinusoidally oscillating.
            # Amplitude/frequency come from _sample_drive (bigger & faster
            # than before, sometimes tuned to resonance).
            chain_dict: dict = {"N": N}
            if rng.random() < 0.35:
                chain_dict["pivot_drive"] = _sample_drive(
                    rng, pivot, L_eff, resonance=driven_resonance
                )
            else:
                chain_dict["pivot"] = pivot
            chains.append(chain_dict)
            pendant_records.append((ci, pivot, L_eff))

            connections.append({"chain": ci, "joint": joint, "spring": s_idx})

    # Guarantee: at least one pivot in every scene is driven (moving).
    # If sampling produced none, promote a random pendant chain to driven
    # with a genuinely moving (force=True) drive.
    if pendant_records and not any("pivot_drive" in ch for ch in chains):
        ci, piv, L_eff = pendant_records[int(rng.integers(len(pendant_records)))]
        chains[ci].pop("pivot", None)
        chains[ci]["pivot_drive"] = _sample_drive(
            rng, piv, L_eff, resonance=driven_resonance, force=True
        )

    # ----- Bridge chains between pairs of springs ------------------------- #
    # Pick 0..2 inter-spring bridges (only meaningful when n_springs ≥ 2).
    if n_springs >= 2:
        # All ordered pairs of distinct springs.
        pairs = [(i, j) for i in range(n_springs) for j in range(n_springs) if i < j]
        rng.shuffle(pairs)
        # 60% chance per pair of having a bridge, capped at 2 bridges total.
        n_bridges = 0
        for (i, j) in pairs:
            if n_bridges >= 2:
                break
            if rng.random() < 0.60:
                N = int(rng.integers(25, 41))
                ci = len(chains)
                chains.append({"N": N})
                connections.append({"chain": ci, "joint": 0, "spring": i})
                connections.append({"chain": ci, "joint": N, "spring": j})
                n_bridges += 1

    # Damping is normally tiny (0.001): motion lasts the full simulation
    # window and chains keep transferring energy back and forth instead of
    # settling down.  Below ~1e-4 the system is effectively conservative
    # and visually identical, so 1e-3 is the sweet spot — still measurable
    # but never noticeably dissipative.
    d = 0.001

    # === FEATURE: adaptive damping (adaptive_damping=False to disable) === #
    # Fast pivot drives pump a lot of energy in, which looks over-chaotic
    # and makes the ODE expensive (the integrator is forced into tiny
    # steps).  Bump damping up gently in proportion to the *fastest* drive
    # actually in play.  Only frequencies above a knee are penalised, so
    # low-frequency resonant drives keep their clean build-up undamped.
    if adaptive_damping:
        driven_freqs = []
        for ch in chains:
            drv = ch.get("pivot_drive")
            if drv is None:
                continue
            fs = [f for a, f in zip(drv["amp"], drv["freq"]) if a > 1e-6]
            if fs:
                driven_freqs.append(max(fs))
        if driven_freqs:
            f_max = max(driven_freqs)
            f_knee = 0.30                       # keep light default below this
            d += 0.12 * max(0.0, f_max - f_knee)  # ~+0.03 at the fastest (0.55)
    d = float(d)
    # ==================================================================== #

    return {
        "chains": chains,
        "springs": springs,
        "connections": connections,
        "d": d,
    }


# --------------------------------------------------------------------------- #
# Per-attempt timeout (Unix only)                                             #
# --------------------------------------------------------------------------- #

class _SimTimeout(Exception):
    """Raised when a simulation attempt exceeds its time budget."""


def _alarm_handler(signum, frame):
    raise _SimTimeout()


class attempt_timeout:
    """Context manager: SIGALRM-based timeout for an integration attempt.

    Falls back to a no-op on platforms without SIGALRM (Windows).
    """
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.installed = False

    def __enter__(self):
        if hasattr(signal, "SIGALRM") and self.seconds > 0:
            self._prev = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(self.seconds)
            self.installed = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.installed:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._prev)
        return False


# --------------------------------------------------------------------------- #
# Telegram                                                                    #
# --------------------------------------------------------------------------- #

def post_to_telegram(token: str, channel: str, media_path: str,
                     caption: str = "") -> None:
    """Send the animation file (gif or mp4) to a Telegram channel.

    The default python-telegram-bot timeouts (5 s connect / 5 s read /
    5 s write) are too short for multi-MB animations on slow uplinks.
    We bump them to 30 / 120 / 120 s.

    ``send_animation`` accepts both gif and mp4; Telegram converts gifs
    to mp4 internally anyway, so passing mp4 directly avoids double
    encoding and makes uploads dramatically smaller and faster.
    """
    import asyncio
    import os
    from telegram import Bot, InputFile
    from telegram.request import HTTPXRequest

    async def _send():
        request = HTTPXRequest(
            connection_pool_size=1,
            connect_timeout=30.0,
            read_timeout=120.0,
            write_timeout=120.0,
            pool_timeout=10.0,
        )
        bot = Bot(token=token, request=request)
        with open(media_path, "rb") as fh:
            await bot.send_animation(
                chat_id=channel,
                animation=InputFile(fh, filename=os.path.basename(media_path)),
                caption=caption,
                read_timeout=120.0, write_timeout=120.0,
                connect_timeout=30.0,
            )

    asyncio.run(_send())


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-T", "--token", default=None, help="Telegram bot token")
    p.add_argument("-N", "--channel", default=None,
                   help="Telegram channel name (e.g. @my_channel)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")

    p.add_argument("--max-attempts", type=int, default=20,
                   help="How many random configs to try before giving up")
    p.add_argument("--max-links", type=int, default=0,
                   help="Skip sampled scenes whose total link count exceeds "
                        "this (keeps per-attempt sim time bounded). 0 = no cap")
    p.add_argument("--attempt-timeout", type=int, default=90,
                   help="Per-attempt simulation timeout in seconds (Unix only)")

    p.add_argument("--t-max", type=float, default=25.0,
                   help="Simulation horizon, seconds")
    p.add_argument("--n-frames", type=int, default=360,
                   help="Number of animation frames (at 24 fps, 360 frames = 15 s)")
    p.add_argument("--fps", type=int, default=24, help="Animation FPS")
    p.add_argument("--dpi", type=int, default=200,
                   help="Output image DPI (figsize × dpi → pixels per side)")
    p.add_argument("--figsize", type=float, default=10.0,
                   help="Output figure size in inches (square)")

    p.add_argument("--rtol", type=float, default=1.0e-7)
    p.add_argument("--atol", type=float, default=1.0e-9)

    p.add_argument("-o", "--output", default=None,
                   help="Output animation path (default: pendulum_<timestamp>.mp4)")
    p.add_argument("-V", "--verbose", action="store_true")

    # --- feature switches (all features on by default) ------------------- #
    p.add_argument("--no-star", action="store_true",
                   help="Disable occasional 7–10 chain hub topologies")
    p.add_argument("--no-contrast", action="store_true",
                   help="Disable long-thin vs short-stubby geometry mixing")
    p.add_argument("--no-resonance", action="store_true",
                   help="Disable tuning driven pivots to chain resonance")
    p.add_argument("--no-adaptive-damping", action="store_true",
                   help="Disable raising damping for fast pivot drives")
    p.add_argument("--no-trails", action="store_true",
                   help="Disable phosphor motion trails on chain tips")
    p.add_argument("--no-energy-color", action="store_true",
                   help="Disable kinetic-energy colour coding of links")
    p.add_argument("--no-legend", action="store_true",
                   help="Disable the element legend on the animation")
    p.add_argument("--no-date", action="store_true",
                   help="Disable the creation-date stamp on the animation")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for noisy in ("matplotlib", "PIL", "asyncio", "httpx", "telegram"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("pendulum-bot")

    rng = np.random.default_rng(args.seed)
    if args.seed is None:
        seed_used = int(rng.integers(0, 2 ** 31 - 1))
        rng = np.random.default_rng(seed_used)
    else:
        seed_used = args.seed
    log.info("RNG seed: %d", seed_used)

    chosen: Optional[dict] = None
    chosen_params: Optional[dict] = None

    for attempt in range(1, args.max_attempts + 1):
        params = sample_parameters(
            rng,
            star_topology=ENABLE_STAR_TOPOLOGY and not args.no_star,
            contrast_geometry=ENABLE_CONTRAST_GEOMETRY and not args.no_contrast,
            driven_resonance=ENABLE_DRIVEN_RESONANCE and not args.no_resonance,
            adaptive_damping=(ENABLE_ADAPTIVE_DAMPING
                              and not args.no_adaptive_damping),
        )
        # Skip scenes that are too big to simulate within the per-attempt
        # budget.  Sampling is cheap, so burning an attempt here is far
        # better than a guaranteed timeout mid-integration.
        total_links = sum(c["N"] for c in params["chains"])
        if args.max_links > 0 and total_links > args.max_links:
            log.debug("attempt %d: %d links > --max-links %d — skipping",
                      attempt, total_links, args.max_links)
            continue
        n_springs = len(params["springs"])
        n_chains = len(params["chains"])
        n_bridges = sum(
            1 for c in params["chains"]
            if "pivot" not in c and "pivot_drive" not in c
        )
        n_driven = sum(1 for c in params["chains"] if "pivot_drive" in c)
        kappas = [s["kappa"] for s in params["springs"]]
        log.debug(
            "attempt %d: %d spring(s), %d chain(s) (%d bridge, %d driven), "
            "d=%g, κ=[%s]",
            attempt, n_springs, n_chains, n_bridges, n_driven, params["d"],
            ", ".join(f"{k:.2e}" for k in kappas),
        )
        try:
            t0 = time.time()
            with attempt_timeout(args.attempt_timeout):
                result = simulate_network(
                    chains=params["chains"],
                    springs=params["springs"],
                    connections=params["connections"],
                    d=params["d"],
                    t_max=args.t_max,
                    n_frames=args.n_frames,
                    rtol=args.rtol,
                    atol=args.atol,
                )
            sim_dt = time.time() - t0
        except _SimTimeout:
            log.warning("attempt %d: TIMEOUT after %ds — skipping",
                        attempt, args.attempt_timeout)
            continue
        except Exception as exc:
            log.warning("attempt %d simulation failed: %s", attempt, exc)
            continue

        log.info(
            "attempt %d: %d spring(s), %d chain(s) (%d bridge, %d driven) "
            "[%.1fs] — accepted",
            attempt, n_springs, n_chains, n_bridges, n_driven, sim_dt,
        )
        chosen = result
        chosen_params = params
        break

    if chosen is None:
        log.error("All %d attempts timed out or failed.", args.max_attempts)
        return 1

    out_path = args.output or f"pendulum_{int(time.time())}.mp4"
    log.info("Rendering animation to %s", out_path)
    t0 = time.time()
    fig, _anim, actual_path = animate_network(
        chosen, save_path=out_path,
        fps=args.fps, dpi=args.dpi, figsize=args.figsize,
        trails=ENABLE_TRAILS and not args.no_trails,
        energy_color=ENABLE_ENERGY_COLOR and not args.no_energy_color,
        legend=ENABLE_LEGEND and not args.no_legend,
        show_date=ENABLE_CREATION_DATE and not args.no_date,
    )
    if actual_path != out_path:
        log.warning("Wrote %s instead of %s (ffmpeg unavailable?)",
                    actual_path, out_path)
        out_path = actual_path
    log.info("Render took %.1fs", time.time() - t0)
    import matplotlib.pyplot as plt
    plt.close(fig)

    if args.token and args.channel:
        n_springs = len(chosen_params["springs"])
        n_chains = len(chosen_params["chains"])
        n_bridges = sum(
            1 for c in chosen_params["chains"]
            if "pivot" not in c and "pivot_drive" not in c
        )
        n_driven = sum(1 for c in chosen_params["chains"] if "pivot_drive" in c)
        kappas = [s["kappa"] for s in chosen_params["springs"]]
        extras = []
        if n_bridges:
            extras.append(f"{n_bridges} bridge{'s' if n_bridges != 1 else ''}")
        if n_driven:
            extras.append(f"{n_driven} driven")
        extras_str = f" ({', '.join(extras)})" if extras else ""
        caption = (
            f"Chain pendulum network\n"
            f"{n_chains} chain{'s' if n_chains != 1 else ''}{extras_str}, "
            f"{n_springs} spring{'s' if n_springs != 1 else ''}\n"
            f"d={chosen_params['d']:g}, "
            f"κ=[{', '.join(f'{k:.0f}' for k in kappas)}]\n"
            f"seed={seed_used}"
        )
        log.info("Posting to Telegram channel %s", args.channel)
        try:
            post_to_telegram(args.token, args.channel, out_path, caption)
            log.info("Posted successfully.")
        except Exception as exc:
            log.error("Telegram post failed: %s", exc)
            return 2
    else:
        log.info("No --token / --channel given — skipping Telegram post.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
