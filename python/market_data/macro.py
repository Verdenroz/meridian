"""Macro panel loaders: FRED series and the treasury curve, as clean date rows."""

from __future__ import annotations

TREASURY_FIRST_YEAR = 1990


def normalize_fred(observations: list[dict]) -> list[dict]:
    rows = [
        {"date": o["date"], "value": float(o["value"])}
        for o in observations
        if o.get("date") and o.get("value") is not None
    ]
    return sorted(rows, key=lambda r: r["date"])


def normalize_treasury(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        raw = row.get("date")
        if not raw:
            continue
        month, day, year = raw.split("/")
        date = f"{year}-{month}-{day}"
        out.append({"date": date, **{k: v for k, v in row.items() if k != "date"}})
    return sorted(out, key=lambda r: r["date"])
