#!/bin/sh
# One IBKR paper cycle: pull the latest shadow decision, trade it on the paper
# gateway, commit the journal. Run by meridian-paper-cycle.timer; safe to rerun,
# the cycle skips a decision it has already acted on.
set -eu
cd "$(dirname "$0")/.."
# After a wake or boot the gateway takes a few minutes to log in; a missed
# firing replays immediately, so wait for it rather than fail the session.
tries=0
until nc -z 127.0.0.1 4002 2>/dev/null; do
  tries=$((tries + 1))
  if [ "$tries" -gt 20 ]; then
    echo "IB Gateway is not listening on 4002 after 10 minutes" >&2
    exit 1
  fi
  sleep 30
done
git pull --rebase --autostash --quiet
uv run python -m execution cycle --broker ibkr
if [ ! -d data/execution/ibkr ]; then
  exit 0
fi
git add data/execution/ibkr
if git diff --cached --quiet; then
  exit 0
fi
git commit -q -m "chore: paper trade $(date -u +%Y-%m-%d)"
git push -q
