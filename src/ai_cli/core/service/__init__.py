"""Service layer public API.

Import ``AIService`` from here for both CLI and GUI consumers::

    from ai_cli.core.service import AIService

    svc = AIService(provider="openai", model="gpt-4o-mini")
    reply = svc.ask("Hello!")
"""

from ai_cli.core.service.ai_service import AIService

__all__ = ["AIService"]
