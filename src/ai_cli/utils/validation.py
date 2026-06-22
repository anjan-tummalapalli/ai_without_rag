from __future__ import annotations

import logging
from dataclasses import dataclass, field

try:
    from ai_cli.rag.vector_store import VectorStore  # type: ignore
except ModuleNotFoundError:
    VectorStore = None
# Local exceptions
from ai_cli.core.exceptions import ResponseValidationError

VectorStore = None

MIN_RESPONSE_LENGTH = 5

log = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result of hallucination risk evaluation.

    score: normalized risk in [0.0, 1.0]
    passed: True when risk is below threshold (default threshold: 0.5)
    reasons: human-readable labels for triggers
    """

    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class HallucinationDetector:
    """
    Heuristic-based hallucination risk estimator.

    Purpose:
        Lightweight triage to flag responses that warrant further verification.

    Signals:
        - very short responses
        - matches to suspicious phrases
        - presence of placeholder tokens like "TODO"
    """

    SUSPICIOUS_PATTERNS = [
        r"100% accurate",
        r"guaranteed",
        r"always works",
        r"never fails",
        r"trust me",
    ]

    def evaluate(self, response: str) -> HallucinationResult:
        """Evaluate response for hallucination risk."""
        score = 0.0
        reasons: list[str] = []

        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.lower() in response.lower():
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}")

        if "TODO" in response:
            score += 0.6
            reasons.append("placeholder content detected")

        score = min(score, 1.0)
        return HallucinationResult(score=score, passed=score < 0.5, reasons=reasons)


class ResponseValidator:
    """Simple response validation helper."""

    def validate(self, response: str) -> None:
        """Validate the response string.

        Raises:
            ResponseValidationError: If response is empty or too short.
        """
        if not response:
            raise ResponseValidationError("empty response")
        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            raise ResponseValidationError("response too short")