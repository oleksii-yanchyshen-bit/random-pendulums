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

The repo has a GitHub Action that runs daily at 12:00 UTC. To use it:

1. Push to GitHub.
2. Create a Telegram bot with [@BotFather](https://t.me/BotFather)
   and add it as **admin** of your channel (so it can post).
3. In the repo → Settings → Secrets and variables → Actions, add:
    - `TG_TOKEN` — the bot token
    - `TG_CHANNEL` — e.g. `@your_channel`
4. Trigger a first run manually: Actions → "Daily pendulum" →
   Run workflow.

That's it. Posts arrive once a day.

## What's in here

- `generate_pendulum_simulation.py` — main script
- `pendulum_system.py` — the simulator
- `.github/workflows/daily.yml` — the daily cron job
