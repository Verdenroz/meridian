"""LLM scorer: 8-K text → structured event risk score, via chimeric.

The prompt is part of the score's identity: scorer_version = model@prompt-hash,
and a score is only reusable under the exact same pair. Editing the prompt
automatically invalidates nothing — old scores keep their version; new runs
write new rows.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_MODEL = "qwen3.6:35b-a3b"
DEFAULT_BASE_URL = "http://127.0.0.1:12434/v1"


class EventRiskScore(BaseModel):
    """Structured risk assessment of one filing (spec §5)."""

    severity: int = Field(ge=0, le=10, description="0 routine .. 10 existential")
    scope: Literal["asset", "market"] = Field(
        description="asset: affects mainly this issuer; market: systemic reach"
    )
    direction: Literal["risk_up", "risk_down", "neutral"] = Field(
        description="expected effect on forward volatility/risk for the issuer"
    )
    rationale: str = Field(max_length=600, description="one or two sentences")


PROMPT = """You are a risk analyst for a volatility-forecasting system. You will read \
an SEC Form 8-K (material event disclosure) and assess its risk content. You do NOT \
predict returns or judge whether news is good or bad for the stock price — only \
whether it raises or lowers *uncertainty and forward volatility* for the issuer.

Guidance:
- severity: 0-2 routine/administrative (earnings-date announcements, routine \
votes, immaterial amendments), 3-5 notable (guidance changes, financing, \
ordinary executive transitions), 6-8 serious (unexpected CEO/CFO departure, \
restatement, major litigation/regulatory action, large impairment, failed \
deal), 9-10 potentially solvency- or franchise-threatening.
- scope: "market" only if the event plausibly moves risk for the broad market \
or a whole sector (systemic credit events, sovereign-scale exposure); \
otherwise "asset".
- direction: "risk_up" if uncertainty about the issuer increases, "risk_down" \
if a known uncertainty is resolved (settlement closed, deal completed, \
guidance affirmed), "neutral" for routine disclosures.

Respond with the structured fields only. Filing follows.

SYMBOL: {symbol}
FILING DATE: {filing_date}
8-K TEXT:
{text}"""

PROMPT_VERSION = hashlib.sha256(PROMPT.encode()).hexdigest()[:12]


class EventScorer:
    """Scores filings through chimeric; inject `client` for tests."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: object | None = None,
    ) -> None:
        self.model = model
        self._base_url = base_url
        self._client = client

    @property
    def scorer_version(self) -> str:
        return f"{self.model}@{PROMPT_VERSION}"

    def _get_client(self) -> object:
        if self._client is None:
            from chimeric import Chimeric

            self._client = Chimeric(base_url=self._base_url, api_key="local")
        return self._client

    def score(self, symbol: str, filing_date: str, text: str) -> EventRiskScore:
        prompt = PROMPT.format(symbol=symbol, filing_date=filing_date, text=text)
        response = self._get_client().generate(  # type: ignore[attr-defined]
            self.model,
            prompt,
            response_model=EventRiskScore,
            temperature=0,
        )
        parsed = response.parsed
        if not isinstance(parsed, EventRiskScore):
            raise TypeError(f"scorer returned unparsed response: {response}")
        return parsed
