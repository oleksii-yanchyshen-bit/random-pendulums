import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut

from generate_pendulum_simulation import post_to_telegram


class TelegramTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "animation.mp4"
        self.payload = b"video payload for upload retry test"
        self.path.write_bytes(self.payload)

    def send(self, outcomes):
        uploads = []
        bots = []

        def make_bot(**kwargs):
            bot = AsyncMock()
            bot.__aenter__.return_value = bot
            async def upload(**kwargs):
                uploads.append(kwargs)
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
            bot.send_animation.side_effect = upload
            bots.append(bot)
            return bot

        with patch("telegram.Bot", side_effect=make_bot), \
             patch("telegram.request.HTTPXRequest"), \
             patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
            self.uploads, self.bots, self.sleep = uploads, bots, sleep
            post_to_telegram("test-token", "@test-channel", str(self.path), "caption")

    @staticmethod
    def network_failure(cause):
        error = TimedOut() if isinstance(cause, httpx.TimeoutException) else NetworkError("offline")
        error.__cause__ = cause
        return error

    def test_write_timeout_retries_complete_file_and_closes_clients(self):
        self.send([self.network_failure(httpx.WriteTimeout("slow upload")), None])
        self.assertEqual(len(self.uploads), 2)
        for upload in self.uploads:
            self.assertEqual(upload["animation"].input_file_content, self.payload)
            self.assertEqual(upload["caption"], "caption")
            self.assertEqual(upload["write_timeout"], 300)
        for bot in self.bots:
            bot.__aexit__.assert_awaited_once()
        self.sleep.assert_awaited_once_with(5.0)

    def test_connection_failures_stop_after_three_attempts(self):
        with self.assertRaises(NetworkError):
            self.send([self.network_failure(httpx.ConnectError("offline")) for _ in range(3)])
        self.assertEqual(len(self.uploads), 3)
        self.assertEqual(self.sleep.await_count, 2)

    def test_rate_limit_retries_after_requested_delay(self):
        self.send([RetryAfter(2), None])
        self.sleep.assert_awaited_once_with(3.0)

    def test_timedelta_rate_limit_is_supported(self):
        error = RetryAfter(2)
        with patch.object(RetryAfter, "retry_after", timedelta(seconds=2)):
            self.send([error, None])
        self.sleep.assert_awaited_once_with(3.0)

    def test_long_rate_limit_does_not_hold_job_open(self):
        with self.assertRaises(RetryAfter):
            self.send([RetryAfter(600)])
        self.sleep.assert_not_awaited()

    def test_read_timeout_does_not_duplicate_a_possibly_delivered_post(self):
        with self.assertRaises(TimedOut):
            self.send([self.network_failure(httpx.ReadTimeout("reply lost"))])
        self.assertEqual(len(self.uploads), 1)
        self.sleep.assert_not_awaited()

    def test_permanent_error_is_not_retried(self):
        with self.assertRaises(BadRequest):
            self.send([BadRequest("invalid chat")])
        self.assertEqual(len(self.uploads), 1)
        self.sleep.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
