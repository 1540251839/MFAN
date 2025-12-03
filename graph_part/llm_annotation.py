"""
Utilities for generating LLM-based reasoning strings and soft labels
for source tweets.

The functions in this module are intentionally detached from the
training loop so they can be used offline to pre-compute auxiliary
signals (reason text and soft labels) that are later consumed by the
model pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import json

try:
    # Prefer a modern OpenAI client but allow the caller to pass any
    # object that mimics the same interface (for easier testing).
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - handled dynamically at runtime
    OpenAI = None  # type: ignore


@dataclass
class LLMAnnotation:
    """Container for LLM outputs associated with a tweet.

    Attributes
    ----------
    reason:
        Natural-language rationale explaining why the tweet might be
        true or false. This is intended to be concatenated with the
        original tweet text to form an augmented textual feature.
    p_fake / p_real:
        Soft-label probabilities produced by the LLM for knowledge
        distillation. Values should be normalized to sum to one.
    """

    reason: str
    p_fake: float
    p_real: float

    def to_dict(self) -> Dict[str, float | str]:
        return {"reason": self.reason, "p_fake": self.p_fake, "p_real": self.p_real}


class LLMReasonAndLabelGenerator:
    """Generate rationale text and soft labels from a language model.

    The generator wraps prompt construction and response parsing so the
    rest of the codebase can remain API-agnostic. A lightweight schema
    (JSON) is used to make parsing deterministic and robust to small
    formatting variations from the model.
    """

    def __init__(
        self,
        model: str,
        *,
        client: Optional[object] = None,
        temperature: float = 0.2,
        system_prompt: str = "You are a fact-checking assistant that outputs JSON.",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.client = client if client is not None else (OpenAI() if OpenAI is not None else None)

    def _build_prompt(self, tweet: str) -> str:
        return (
            "Given the following tweet, provide: (1) a short reasoning why it could be true "
            "or false and (2) soft-label probabilities for the tweet being fake or real. "
            "Respond strictly in JSON with keys 'reason', 'p_fake', and 'p_real'. "
            f"Tweet: {tweet}"
        )

    def _parse_response(self, raw_content: str) -> LLMAnnotation:
        """Parse a JSON response from the LLM into an ``LLMAnnotation``.

        The parser is intentionally forgiving: it strips code fences and
        normalizes probabilities to avoid downstream errors if the model
        returns slightly malformed output.
        """

        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`\n")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        payload = json.loads(cleaned)
        reason = payload.get("reason", "").strip()
        p_fake = float(payload.get("p_fake", 0.5))
        p_real = float(payload.get("p_real", 0.5))
        total = p_fake + p_real
        if total > 0:
            p_fake, p_real = p_fake / total, p_real / total
        return LLMAnnotation(reason=reason, p_fake=p_fake, p_real=p_real)

    def generate(self, tweet: str) -> LLMAnnotation:
        if self.client is None:
            raise RuntimeError(
                "An OpenAI-compatible client is required to generate LLM annotations. "
                "Pass an instantiated client or install the openai package."
            )

        completion = self.client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._build_prompt(tweet)},
            ],
            temperature=self.temperature,
        )
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned an empty message content")
        return self._parse_response(content)


def export_annotations(annotations: Dict[str, LLMAnnotation], output_path: str) -> None:
    """Persist LLM annotations to disk in a JSON file."""

    serializable = {tweet_id: annotation.to_dict() for tweet_id, annotation in annotations.items()}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
