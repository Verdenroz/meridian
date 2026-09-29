#!/bin/sh
# Usage: unit-alert.sh <unit>. Best-effort: a missing webhook or a failed POST
# never fails the alert unit itself.
unit="$1"
[ -n "${MERIDIAN_ALERT_WEBHOOK:-}" ] || exit 0
journalctl --user -u "$unit" -n 20 --no-pager 2>/dev/null |
  curl -fsS -m 10 -H "X-Title: meridian unit failed: $unit" -H "X-Priority: high" \
    --data-binary @- "$MERIDIAN_ALERT_WEBHOOK" >/dev/null || true
