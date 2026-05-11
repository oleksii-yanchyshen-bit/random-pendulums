# Pendulum bot — daily Coupled-Pendulum simulation posted to Telegram.
#
# Build:
#   docker build -t pendulum-bot \
#     --build-arg TG_TOKEN=YOUR_BOT_TOKEN \
#     --build-arg TG_CHANNEL=@your_channel \
#     -f Dockerfile .
#
# Run (detached):
#   docker run -d --name pendulum-bot pendulum-bot
#
# Cron logs:
#   docker exec pendulum-bot tail -f /var/log/cron.log
#
# Trigger one run manually for testing:
#   docker exec pendulum-bot python3 /app/generate_pendulum_simulation.py \
#       -V --max-attempts 30
FROM python:3.11-slim

ARG TG_TOKEN=""
ARG TG_CHANNEL=""

# System deps: ffmpeg for video encoding, cron for scheduling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cron \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir 'python-telegram-bot>=20'

COPY pendulum.py coupled_pendulums.py pendulum_system.py generate_pendulum_simulation.py /app/

# Crontab: run every day at 12:00 UTC.
# Output and errors go to /var/log/cron.log so `docker logs` and
# `docker exec ... tail` show them.
RUN echo "0 12 * * * cd /app && /usr/local/bin/python3 generate_pendulum_simulation.py \
    -T '${TG_TOKEN}' -N '${TG_CHANNEL}' \
    --max-attempts 40 --min-score 0.8 --max-score 6.0 \
    -V >> /var/log/cron.log 2>&1" > /etc/cron.d/pendulum-bot \
    && chmod 0644 /etc/cron.d/pendulum-bot \
    && crontab /etc/cron.d/pendulum-bot \
    && touch /var/log/cron.log

# Run cron in the foreground; tail the log so `docker logs` picks it up.
CMD ["sh", "-c", "cron && tail -F /var/log/cron.log"]
