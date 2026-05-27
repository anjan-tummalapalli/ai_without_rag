from __future__ import annotations
import re
from dataclasses import dataclass, field
from ai_cli.core.exceptions import ResponseValidationError

MIN_RESPONSE_LENGTH = 5


@dataclass
class HallucinationResult:
    """Result of hallucination risk evaluation."""

    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class HallucinationDetector:
    """
    Heuristic-based hallucination risk estimator for model responses.

    Purpose:
        Provide a lightweight, rule-based estimate of whether a text response
        may contain hallucinated or unreliable content. This detector is intended
        as a triage tool to flag responses that warrant further review or more
        rigorous verification; it is not a definitive proof or verifier of truth.

    How it works:
        - Inspects a response string for simple signals such as:
            * Very short responses (below MIN_RESPONSE_LENGTH)
            * Matches to configurable suspicious phrases (SUSPICIOUS_PATTERNS)
            * Presence of placeholder tokens like "TODO"
        - Each signal contributes a fixed weight toward an aggregate risk score.
        - The final score is capped at 1.0 and translated into a boolean pass/fail
          using a threshold (passed when score < 0.5).
        - Returns both the numeric score and human-readable reasons for the score.

    End result:
        - A HallucinationResult containing:
            score (float): normalized risk in [0.0, 1.0]
            passed (bool): whether the response is considered low-risk (score < 0.5)
            reasons (list[str]): explanatory labels for the heuristics that triggered
        - Intended for use in logging, UI indicators, or automated triage pipelines;
          tune thresholds and patterns to fit your application's sensitivity needs.
    """
    """Heuristic-based hallucination risk estimator (not a verifier)."""

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

        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}")

        if "TODO" in response:
            score += 0.3
            reasons.append("placeholder content detected")

        score = min(score, 1.0)
        return HallucinationResult(
            score=score, passed=score < 0.5, reasons=reasons
        )


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
