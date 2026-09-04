#!/bin/sh
# One IBKR paper cycle: pull the latest shadow decision, trade it on the paper
# gateway, commit the journal. Run by meridian-paper-cycle.timer; safe to rerun,
# the cycle skips a decision it has already acted on.
set -eu
cd /home/harvey/projects/meridian
if ! nc -z 127.0.0.1 4002 2>/dev/null; then
  echo "IB Gateway is not listening on 4002" >&2
  exit 1
fi
git pull --rebase --quiet
uv run python -m execution cycle --broker ibkr
git add data/execution/ibkr
if git diff --cached --quiet; then
  exit 0
fi
git commit -q -m "chore: paper trade $(date -u +%Y-%m-%d)"
git push -q
