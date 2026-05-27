from __future__ import annotations
import os
from typing import Optional


class SecretManager:
    """
    Utility for resolving secrets used by the application.

    File purpose:
    - Centralize secret retrieval logic so callers don't need to duplicate
        environment-or-file lookup patterns across the codebase.
    - Provide a minimal, predictable interface for obtaining sensitive values
        (API keys, tokens, credentials) from the environment or, optionally,
        from a UTF-8 encoded file.

    Behavior and end result:
    - get_secret(name, file_path=None) first checks the environment for the
        given variable name; if present, that value is returned.
    - If the environment variable is not set and a file_path is provided
        and exists, the file is read (UTF-8) and its contents are returned
        with surrounding whitespace stripped.
    - Any file read errors are handled internally and result in None,
        so callers receive either the secret string or None when the secret
        cannot be obtained.
    - Environment values take precedence over file contents.

    Return value:
    - Returns the secret as a str when found, otherwise returns None.

    Usage note:
    - Callers should treat None as "secret not available" and handle it
        explicitly (e.g., fail fast, prompt the user, or use a fallback).
    """
    """Resolve secret from environment or a file."""

    @staticmethod
    def get_secret(name: str, file_path: Optional[str] = None) -> Optional[str]:
        """Obtain a secret value."""
        val = os.getenv(name)
        if val:
            return val
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except Exception:
                return None
        return None


def is_kubernetes() -> bool:
    """Heuristic detection for running inside Kubernetes."""
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return True
    return os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")