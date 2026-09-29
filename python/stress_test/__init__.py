"""Prospective stress-episode test (pre-registration).

Registered 2026-09-29; frozen from the registering commit on. Never edit
GATE; a changed test is a new module.

The one test the research freeze leaves open. Every backtest verdict shares
the 1998-2026 span that has been raced ~30 times, so no further backtest can
add evidence. This asks the question only the future can answer: when a real
drawdown arrives, does the leading regime book (``leader``) actually defend
better than the untouched null, from its live point-in-time shadow rows?

Episode. The first session on or after ``start`` where SPY's total-return
close is at least ``trigger_drawdown`` below its trailing ``lookback`` -session
high. The episode window runs from that high's session to the first later
close back at or above it, or ``max_sessions`` after the trigger, whichever
comes first. Only the first episode is judged; later ones are descriptive.
Base rate (SPY 1993-2026, price history only): 0.42 triggers/yr, so about
72% odds of an episode before ``deadline``.

Books. Each shadow row's weights are held over the next session's
close-to-close return, the cash remainder earns the panel T-bill rate, and
weight changes pay the harness one-way cost. A missing row carries the prior
weights, as the paper book would. Rows are read exactly as logged.

Gate, judged once when the window closes. PASS iff both:
(a) tail: the leader's max drawdown inside the window is at most
    ``tail_ratio`` of the null's;
(b) timing, not timidity: the leader's total return over the window is at
    least the null's minus ``return_tolerance``, so cutting the fall does not
    count if it forfeits the recovery.

Consequences. PASS: the leader may replace the null on the paper book, then
climbs the trust ladder from there. FAIL: the leader is retired, the null
stays. No episode by ``deadline``: no evidence; the null stays.
"""

GATE: dict = {
    "leader": "spine7f_turn_3state",
    "null": "null_voltarget",
    "start": "2026-09-30",
    "deadline": "2029-09-28",
    "trigger_drawdown": 0.10,
    "lookback": 252,
    "max_sessions": 126,
    "tail_ratio": 2 / 3,
    "return_tolerance": 0.02,
}
