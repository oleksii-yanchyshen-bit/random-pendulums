# Random pendulums

A bot that posts a random chain-pendulum animation to
[@Rndom_pendulums](https://t.me/Rndom_pendulums) every day.
Inspired by [robolamp/3_body_problem_bot](https://github.com/robolamp/3_body_problem_bot).

Each post is a different setup: 1–3 springs, 2–5 chain pendulums per
spring, sometimes "bridge" chains between springs, sometimes pivots
that wiggle on their own. Everything is sampled fresh per post.

## Run locally

```bash
pip install -r requirements.txt
brew install ffmpeg          # or: sudo apt-get install ffmpeg

python generate_pendulum_simulation.py -V -o test.mp4
```

That writes `test.mp4`. Add `-T <token> -N @your_channel` to also post
it to Telegram.

## Deploy

The repo has a GitHub Action scheduled daily at 13:50 UTC (GitHub may delay
scheduled runs). To use it:

1. Push to GitHub.
2. Create a Telegram bot with [@BotFather](https://t.me/BotFather)
   and add it as **admin** of your channel (so it can post).
3. In the repo → Settings → Secrets and variables → Actions, add:
    - `TG_TOKEN` — the bot token
    - `TG_CHANNEL` — e.g. `@your_channel`
4. Trigger a first run manually: Actions → "Daily pendulum" →
   Run workflow.

That's it. Posts arrive once a day.

## Reliability and quality

The default solver is DOP853 with adaptive error control. The video still
uses 360 frames at 24 FPS, 2000 x 2000 pixels, and the same scene sampler,
trails, colours and masses. Numerical tolerances remain `rtol=1e-7`,
`atol=1e-9` in the CLI. The daily job simulates 16 seconds of physical time
and plays them in a 15-second video. `--method LSODA` (or `Radau`/`BDF`)
is available for comparisons or custom stiff configurations.

`--max-attempts` counts actual simulations; configurations rejected by
`--max-links` do not spend that budget. Sampling is bounded separately so
an impossible size limit cannot loop forever. Each integration has a
240-second default timeout on Unix. The log records the seed, accepted
sample number and solver statistics for reproducing failures.

Telegram credentials can also be supplied as `TG_TOKEN` and `TG_CHANNEL`
environment variables. Incomplete writes, connection failures and short
rate limits retry up to three times using the same completed video.
Read timeouts are not retried automatically because the post may already
have arrived. A failed upload leaves the video on disk; GitHub Actions
also retains it as an artifact for seven days.

Run the numerical and failure-recovery regression tests with:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

## What's in here

- `generate_pendulum_simulation.py` — main script
- `pendulum_system.py` — the simulator
- `.github/workflows/daily.yml` — the daily cron job
