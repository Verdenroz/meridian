"""FinBERT sentiment sidecar.

Serves the exact `ProsusAI/finbert` model behind a `POST /predict` endpoint that
matches the contract `src/sentiment/mod.rs` expects:

    request:  {"inputs": "<text>"}              or  {"inputs": ["<t1>", "<t2>"]}
    response: [[{"label": "positive", "score": 0.9}, ...], ...]   # one list per input

Using `transformers` directly (rather than TEI) means no model surgery is needed:
the slow BERT tokenizer is built from `vocab.txt` and the PyTorch `.bin` weights load
as-is, so `ProsusAI/finbert` works unmodified. CPU-only (`device=-1`) — no GPU needed.

Model is overridable via `FINBERT_MODEL`; weights cache under `HF_HOME` (mount it to
persist across restarts).
"""

import os
from typing import Union

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)

MODEL_ID = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
# top_k=None → return the full score distribution over all labels (positive /
# negative / neutral) so the caller can compute P(pos) - P(neg). truncation=True
# guards FinBERT's 512-token cap on long filing text.
_pipe = TextClassificationPipeline(
    model=_model,
    tokenizer=_tokenizer,
    top_k=None,
    truncation=True,
    device=-1,
)

app = FastAPI(title="finbert-sentiment")


class PredictRequest(BaseModel):
    inputs: Union[str, list[str]]


@app.get("/info")
def info() -> dict:
    return {"model_id": MODEL_ID, "model_type": _model.config.model_type}


@app.post("/predict")
def predict(req: PredictRequest) -> list[list[dict]]:
    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    if not texts:
        return []
    with torch.inference_mode():
        results = _pipe(texts)
    # For a batch input the pipeline returns one list-of-{label,score} per text;
    # this is already the nested shape the Rust client decodes.
    return results
