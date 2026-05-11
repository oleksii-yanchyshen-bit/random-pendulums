#!/usr/bin/env python3
"""
generate_pendulum_simulation.py
================================
Generate a chain-pendulum-network animation with random initial
conditions and (optionally) post it to a Telegram channel.

Inspired by https://github.com/robolamp/3_body_problem_bot

Random configurations span 1-3 springs with 2-4 chains attached to each.
When there are ≥ 2 springs, with some probability we also generate
"bridge" chains connecting two springs together — both ends of the
bridge are tied to springs, so its pivot is itself a free, dynamical
point in space.

Without ``-T`` and ``-N`` arguments, the script just writes a GIF to disk.
With them, it also publishes the GIF to the given Telegram channel.

The first attempt that runs to completion is accepted, no quality
filtering — with very low damping the system rarely produces a boring
result, and visual judgment is subjective.

Per-attempt timeout: extremely stiff springs (κ ≳ 10⁶) make the ODE
arbitrarily hard, with oscillation periods below μs scale.  We wrap
each attempt in a ``signal.alarm``-based timeout (Unix only) and just
move on if a sim takes too long.
"""
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
# Random parameter sampling                                                   #
# --------------------------------------------------------------------------- #

def sample_parameters(rng: np.random.Generator) -> dict:
    """Sample a random-but-reasonable network configuration.

    Returns a dict with ``chains``, ``springs``, ``connections`` ready
    to pass straight into ``simulate_network``, plus ``d`` for damping.
    """
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

    # ----- Pendant chains hanging off each spring ------------------------- #
    for s_idx, spr in enumerate(springs):
        anchor = spr["anchor"]
        # Bias toward more chains per spring — bigger Y/X/star junctions
        # produce richer collective dynamics than simple pairs.
        n_chains = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.35, 0.25, 0.15]))
        base_angle = rng.uniform(0, 2 * np.pi)
        for c in range(n_chains):
            theta = (base_angle + 2 * np.pi * c / n_chains
                     + rng.normal(0, 0.25))
            dist = rng.uniform(12.0, 20.0)
            pivot = (
                anchor[0] + dist * np.cos(theta),
                anchor[1] + dist * np.sin(theta),
            )
            N = int(rng.integers(20, 41))
            ci = len(chains)

            # 35% chance the pivot is *driven* — sinusoidally oscillating.
            # Driving makes the integrator work harder (the system is no
            # longer autonomous), so we keep frequencies modest.  Drive
            # period stays ≥ 2.5 s — comfortable for LSODA.
            chain_dict: dict = {"N": N}
            if rng.random() < 0.35:
                amp_x = float(rng.uniform(0, 2.5)) if rng.random() < 0.7 else 0.0
                amp_y = float(rng.uniform(0, 2.5)) if rng.random() < 0.5 else 0.0
                freq_x = float(rng.uniform(0.10, 0.40))
                freq_y = float(rng.uniform(0.10, 0.40))
                phase_x = float(rng.uniform(0, 2 * np.pi))
                phase_y = float(rng.uniform(0, 2 * np.pi))
                chain_dict["pivot_drive"] = {
                    "centre": pivot,
                    "amp": (amp_x, amp_y),
                    "freq": (freq_x, freq_y),
                    "phase": (phase_x, phase_y),
                }
            else:
                chain_dict["pivot"] = pivot
            chains.append(chain_dict)

            # 35% attach somewhere in the chain's lower half rather than
            # at the tip → trailing tail hangs free below the spring,
            # adding energy and asymmetry to the motion.
            if rng.random() < 0.35:
                joint = int(rng.integers(N // 2, N))
            else:
                joint = N
            connections.append({"chain": ci, "joint": joint, "spring": s_idx})

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

    # Damping fixed at a very small value: motion lasts the full
    # simulation window and chains keep transferring energy back and
    # forth instead of settling down.  Below ~1e-4 the system is
    # effectively conservative and visually identical, so 1e-3 is the
    # sweet spot — still measurable but never noticeably dissipative.
    d = 0.001

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
    p.add_argument("--attempt-timeout", type=int, default=60,
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
        params = sample_parameters(rng)
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
