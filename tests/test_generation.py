import contextlib
import io
import os
import unittest
from unittest.mock import patch

import generate_pendulum_simulation as gen


def config(n):
    return {"chains": [{"N": n, "pivot": (0, 0)}],
            "springs": [{"anchor": (1, -1), "kappa": 1}],
            "connections": [{"chain": 0, "joint": n, "spring": 0}], "d": 0.001}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"TG_TOKEN": "", "TG_CHANNEL": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_main(self, samples, outcomes, args):
        with patch.object(gen, "sample_parameters", side_effect=samples) as sample, \
             patch.object(gen, "simulate_network", side_effect=outcomes) as simulate, \
             patch.object(gen, "animate_network", return_value=(None, None, "test.mp4")) as render:
            code = gen.main(["--seed", "0", "-o", "test.mp4", *args])
        return code, sample, simulate, render

    def test_size_rejections_do_not_spend_simulation_attempts(self):
        code, sample, simulate, render = self.run_main(
            [config(500), config(600), config(10)], [{}],
            ["--max-attempts", "1", "--max-links", "280"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(sample.call_count, 3)
        self.assertEqual(simulate.call_count, 1)
        self.assertEqual(simulate.call_args.kwargs["n_frames"], 360)
        self.assertEqual(simulate.call_args.kwargs["rtol"], 1e-7)
        self.assertEqual(simulate.call_args.kwargs["atol"], 1e-9)
        self.assertEqual(render.call_args.kwargs["dpi"], 200)
        self.assertEqual(render.call_args.kwargs["figsize"], 10)
        self.assertEqual(render.call_args.kwargs["fps"], 24)
        self.assertTrue(render.call_args.kwargs["trails"])

    def test_timeout_and_numerical_failure_retry_then_succeed(self):
        code, _, simulate, render = self.run_main(
            [config(10)] * 3, [gen._SimTimeout(), RuntimeError("ODE failed"), {}],
            ["--max-attempts", "3"],
        )
        self.assertEqual(code, 0)
        self.assertEqual(simulate.call_count, 3)
        render.assert_called_once()

    def test_failed_simulations_stop_at_budget(self):
        code, _, simulate, render = self.run_main(
            [config(10)] * 2, [gen._SimTimeout(), gen._SimTimeout()],
            ["--max-attempts", "2"],
        )
        self.assertEqual(code, 1)
        self.assertEqual(simulate.call_count, 2)
        render.assert_not_called()

    def test_impossible_size_limit_terminates(self):
        code, sample, simulate, render = self.run_main(
            [config(10)] * 100, [], ["--max-attempts", "1", "--max-links", "1"],
        )
        self.assertEqual(code, 1)
        self.assertEqual(sample.call_count, 100)
        simulate.assert_not_called()
        render.assert_not_called()

    def test_bad_arguments_fail_before_simulation(self):
        for args in (["--fps", "0"], ["--n-frames", "1"], ["--t-max", "nan"],
                     ["--max-attempts", "0"], ["--token", "only-token"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    gen.parse_args(args)
                self.assertEqual(exc.exception.code, 2)

    def test_environment_credentials_and_cli_override(self):
        with patch.dict(os.environ, {"TG_TOKEN": "env-token", "TG_CHANNEL": "@env"}):
            args = gen.parse_args([])
            self.assertEqual((args.token, args.channel), ("env-token", "@env"))
            args = gen.parse_args(["-T", "cli-token", "-N", "@cli"])
            self.assertEqual((args.token, args.channel), ("cli-token", "@cli"))


if __name__ == "__main__":
    unittest.main()
