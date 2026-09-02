"""Outbound alert: one optional webhook from MERIDIAN_ALERT_WEBHOOK.

Same contract as the Rust bus: the JSON body is the journal row, an
X-Title header carries the one-line summary, and delivery is best-effort.
The journal row is the record; a failed webhook is logged, never raised.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

WEBHOOK_ENV = "MERIDIAN_ALERT_WEBHOOK"
SEND_TIMEOUT = 10


def alert(title: str, row: dict) -> bool:
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if not url:
        return False
    request = urllib.request.Request(
        url,
        data=json.dumps(row, sort_keys=True).encode(),
        headers={"Content-Type": "application/json", "X-Title": title},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=SEND_TIMEOUT):
            return True
    except (urllib.error.URLError, OSError) as exc:
        print(f"alert webhook failed ({exc}); journal row still recorded", file=sys.stderr)
        return False
