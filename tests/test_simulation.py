import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

import pendulum_system as ps
from pendulum import simulate_chain


class SimulationTests(unittest.TestCase):
    def test_fixed_chain_matches_independent_reference(self):
        n, length, angle = 6, 0.5, 0.6
        target = (n * length * np.sin(angle), -n * length * np.cos(angle))
        result = ps.simulate_network(
            chains=[{"N": n, "pivot": (0, 0)}],
            springs=[{"anchor": target, "kappa": 3}],
            connections=[{"chain": 0, "joint": n, "spring": 0}],
            d=0.01, t_max=2, n_frames=80, rtol=1e-7, atol=1e-9,
        )
        _, expected = simulate_chain(
            N=n, l=length, masses=np.full(n, 0.25), alpha0=angle,
            d=0.01, t_max=2, n_frames=80, method="Radau",
            rtol=1e-11, atol=1e-13,
        )
        np.testing.assert_allclose(result["angles"][0], expected, atol=2e-6, rtol=0)

    def test_driven_heavy_chains_match_high_precision_radau(self):
        params = dict(
            chains=[
                {"N": 4, "masses": [0.25, 4, 0.25, 7],
                 "pivot_drive": {"centre": (-2, 1), "amp": (0.6, 0.4),
                                 "freq": (0.4, 0.3), "phase": (0.2, 0.8)}},
                {"N": 3, "pivot": (2, 0)},
            ],
            springs=[{"anchor": (0, -1), "kappa": 20}],
            connections=[{"chain": 0, "joint": 3, "spring": 0},
                         {"chain": 1, "joint": 3, "spring": 0}],
            d=0.01, t_max=1, n_frames=40,
        )
        result = ps.simulate_network(**params, rtol=1e-7, atol=1e-9)
        reference = ps.simulate_network(**params, method="Radau", rtol=1e-11, atol=1e-13)
        for actual, expected in zip(result["angles"], reference["angles"]):
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)
        for actual, expected in zip(result["pivots"], reference["pivots"]):
            np.testing.assert_array_equal(actual, expected)

    def test_bridge_conserves_total_energy(self):
        # Capture velocities from the solver, rather than estimating them
        # from rendered frames; finite differences hide integration errors.
        solutions = []

        def capture(*args, **kwargs):
            sol = solve_ivp(*args, **kwargs)
            solutions.append(sol)
            return sol

        with patch.object(ps, "solve_ivp", side_effect=capture):
            result = ps.simulate_network(
                chains=[{"N": 3, "pivot": (-3, 2)}, {"N": 4},
                        {"N": 3, "pivot": (3, 2)}],
                springs=[{"anchor": (-1, 0), "kappa": 15},
                         {"anchor": (1, 0), "kappa": 20}],
                connections=[{"chain": 0, "joint": 3, "spring": 0},
                             {"chain": 1, "joint": 0, "spring": 0},
                             {"chain": 1, "joint": 4, "spring": 1},
                             {"chain": 2, "joint": 3, "spring": 1}],
                d=0, t_max=3, n_frames=180, rtol=1e-9, atol=1e-11,
            )
        velocities = solutions[0].y[solutions[0].y.shape[0] // 2:]
        energy = np.zeros(len(result["t"]))
        positions = []
        offset = 0
        for cd, angles, pivots in zip(result["chains"], result["angles"], result["pivots"]):
            n, length = cd["N"], cd["l"]
            pivot_v = np.zeros_like(pivots)
            if cd["free_pivot"]:
                pivot_v = velocities[offset:offset + 2]
                offset += 2
                energy += cd["pivot_mass"] * (
                    0.5 * np.sum(pivot_v ** 2, axis=0) + 9.8 * pivots[1])
            omega = velocities[offset:offset + n]
            offset += n
            x, y = ps._link_positions(angles, length, pivots)
            positions.append((x, y))
            vx = pivot_v[0] + np.cumsum(length * np.cos(angles) * omega, axis=0)
            vy = pivot_v[1] + np.cumsum(length * np.sin(angles) * omega, axis=0)
            energy += np.sum(cd["masses"][:, None] *
                             (0.5 * (vx ** 2 + vy ** 2) + 9.8 * y[1:]), axis=0)
        for index, spring in enumerate(result["springs"]):
            joints = [c for c in result["connections"] if c["spring"] == index]
            for i, a in enumerate(joints):
                for b in joints[i + 1:]:
                    xa, ya = positions[a["chain"]]
                    xb, yb = positions[b["chain"]]
                    energy += 0.5 * spring["kappa"] * (
                        (xa[a["joint"]] - xb[b["joint"]]) ** 2
                        + (ya[a["joint"]] - yb[b["joint"]]) ** 2)
        relative_drift = np.max(np.abs(energy - energy[0])) / max(1, abs(energy[0]))
        self.assertLess(relative_drift, 1e-7)


if __name__ == "__main__":
    unittest.main()
