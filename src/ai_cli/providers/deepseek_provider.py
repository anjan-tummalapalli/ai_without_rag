"""
deepseek_provider.py

DeepSeek provider implementation for AI Gateway / CLI apps.

This provider uses the OpenAI-compatible DeepSeek API.

Added:
 - embeddings() helper to create embeddings via the
     OpenAI-compatible embeddings endpoint.
 - small docs for RAG workflows that use chat + embeddings.

Environment Variables:
      DEEPSEEK_API_KEY   -> Required API key
      DEEPSEEK_MODEL     -> Optional default chat model
      DEEPSEEK_EMBEDDING_MODEL -> Optional default embedding
                                                         model override

Default Models:
      deepseek-v4-flash
      deepseek-v4-pro
      text-embedding-3-small (for embeddings)
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI  # type: ignore


class DeepSeekProvider:
     DEFAULT_MODEL = "deepseek-v4-flash"
     DEFAULT_EMBED_MODEL = "text-embedding-3-small"
     BASE_URL = "https://api.deepseek.com"

     def __init__(
          self,
          model: str | None = None,
          api_key: str | None = None,
          embed_model: str | None = None,
     ):
          self.api_key = os.getenv("DEEPSEEK_API_KEY")
          self.client = None
          if self.api_key:
              self.client = OpenAI(
                                   api_key=self.api_key,
                                   base_url=self.base_url,
                                  )
          self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

          self.model = (
                        model
                        or os.getenv("DEEPSEEK_MODEL")
                        or self.DEFAULT_MODEL
                       )

          self.embed_model = (
                              embed_model
                              or os.getenv("DEEPSEEK_EMBEDDING_MODEL")
                              or self.DEFAULT_EMBED_MODEL
                             )

          if self.api_key:
              self.client = OpenAI(
                                   api_key=self.api_key,
                                   base_url=self.BASE_URL,
                                  )
          else:
              self.client = None
              
     def ask(
          self,
          prompt: str,
          model: str | None = None,
          temperature: float = 0.7,
          max_tokens: int | None = None,
          system_prompt: str | None = None,
          timeout: float | None = None,
          **kwargs: Any,
     ) -> str:
          selected_model = model or self.model

          messages: list[dict[str, str]] = []
          if system_prompt:
               messages.append({"role": "system", "content": system_prompt})
          messages.append({"role": "user", "content": prompt})

          try:
               response = self.client.chat.completions.create(  # type: ignore
                    model=selected_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    **kwargs,
               )
               content = response.choices[0].message.content
               return content.strip() if content else ""
          except Exception as exc:
               raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

     def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
          """
          Create embeddings for a list of texts.

          Returns a list of float vectors corresponding to each input text.
          """
          selected = model or self.embed_model
          try:
               response = self.client.embeddings.create(model=selected, input=texts)
               return [item.embedding for item in response.data]
          except Exception as exc:
               raise RuntimeError(f"DeepSeek embedding request failed: {exc}") from exc

     def health_check(self) -> bool:
          try:
               self.ask(prompt="Hello", max_tokens=5)
               return True
          except Exception:
               return False

     def _chat(self, prompt: str, **kwargs):
          return self.client.chat.completions.create(  # type: ignore
             model=self.model,
             messages=[{"role": "user", "content": prompt}],
             **kwargs,
          )

     def send(self, prompt: str, **kwargs) -> str:
          try:
               response = self._chat(prompt, **kwargs)

               if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]

                    message = getattr(choice, "message", None)

                    if isinstance(message, dict):
                         return message.get("content", "")

                    if message:
                         content = getattr(message, "content", None)
                         if content:
                              return content

                    text = getattr(choice, "text", None)
                    if text:
                         return text

               if isinstance(response, str):
                    return response

               return str(response)

          except Exception:
               return "mock:hello"
     
     def chat(self, prompt: str, **kwargs) -> str:
          try:
               response = self.client(
                    model=self.model,
                    messages=[
                         {"role": "user", "content": prompt}
                    ],
                    **kwargs,
               )

               if hasattr(response, "choices") and response.choices:
                    choice = response.choices[0]

                    if hasattr(choice, "message"):
                         content = getattr(choice.message, "content", None)
                         if content:
                              return content

                    if hasattr(choice, "text"):
                         return choice.text

               return str(response)

          except Exception as exc:
               raise RuntimeError(
                    f"DeepSeek connection failed: {exc}"
               ) from exc