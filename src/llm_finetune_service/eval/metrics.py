from __future__ import annotations

import math
import re
from typing import Any


FORMAL_MARKERS = [
    "i am writing to inform you",
    "please be advised",
    "kind regards",
    "thank you for your attention",
]
CASUAL_MARKERS = ["quick", "heads up", "fyi", "clarify", "can we", "good news", "update"]


def _contains_deadline(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["eod", "tomorrow", "friday", "monday", "tuesday", "wednesday", "afternoon"])


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def score_prediction(source_email: str, prediction: str, reference: str) -> dict[str, Any]:
    lowered_prediction = prediction.lower()
    source_tokens = _tokenize(source_email)
    prediction_tokens = _tokenize(prediction)
    overlap = len(source_tokens & prediction_tokens) / max(1, len(source_tokens))
    compression_ratio = len(prediction) / max(1, len(source_email))

    style_conformity = 1.0
    if any(marker in lowered_prediction for marker in FORMAL_MARKERS):
        style_conformity -= 0.5
    if any(marker in lowered_prediction for marker in CASUAL_MARKERS):
        style_conformity += 0.2
    style_conformity = max(0.0, min(1.0, style_conformity))

    brevity = 1.0 - min(1.0, abs(compression_ratio - 0.58))
    action_preservation = 1.0
    if _contains_deadline(source_email) and not _contains_deadline(prediction):
        action_preservation -= 0.5
    if "?" in reference and "?" not in prediction:
        action_preservation -= 0.2
    action_preservation = max(0.0, min(1.0, action_preservation))

    slack_likeness = 0.5
    if len(re.split(r"[.!?]+", prediction.strip())) <= 3:
        slack_likeness += 0.2
    if any(marker in lowered_prediction for marker in CASUAL_MARKERS):
        slack_likeness += 0.2
    if any(marker in lowered_prediction for marker in FORMAL_MARKERS):
        slack_likeness -= 0.3
    slack_likeness = max(0.0, min(1.0, slack_likeness))

    lexical_divergence = 1.0 - overlap
    overall = (style_conformity + brevity + action_preservation + slack_likeness + lexical_divergence) / 5.0

    return {
        "style_conformity": round(style_conformity, 4),
        "brevity": round(brevity, 4),
        "action_preservation": round(action_preservation, 4),
        "slack_likeness": round(slack_likeness, 4),
        "lexical_divergence": round(lexical_divergence, 4),
        "compression_ratio": round(compression_ratio, 4),
        "overall": round(overall, 4),
    }


def summarize_scores(scores: list[dict[str, Any]]) -> dict[str, float]:
    if not scores:
        return {}
    keys = [key for key in scores[0].keys() if isinstance(scores[0][key], (int, float))]
    return {key: round(sum(float(score[key]) for score in scores) / len(scores), 4) for key in keys}
