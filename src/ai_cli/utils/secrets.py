from __future__ import annotations
import os
from typing import Optional


class SecretManager:
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