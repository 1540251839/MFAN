"""
Helper utilities to apply LLM annotations to the existing datasets.

This module focuses on:
1) Building augmented text sequences by concatenating source tweets with
   their LLM-provided reasoning strings.
2) Extracting soft labels (p_fake, p_real) so they can be consumed by
   knowledge-distillation losses.
"""
from __future__ import annotations

from typing import Dict, Tuple
import json
import os


class LLMDataAugmenter:
    """Load LLM annotations and expose helpers to align them with splits."""

    def __init__(self, annotation_path: str) -> None:
        self.annotation_path = annotation_path
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> Dict[str, Dict[str, float | str]]:
        if not os.path.exists(self.annotation_path):
            return {}
        with open(self.annotation_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_annotations(self) -> bool:
        return bool(self.annotations)

    def _lookup(self, mid: str) -> Tuple[str, float, float]:
        record = self.annotations.get(str(mid), {})
        reason = str(record.get("reason", "")).strip()
        p_fake = float(record.get("p_fake", 0.5))
        p_real = float(record.get("p_real", 0.5))
        total = p_fake + p_real
        if total > 0:
            p_fake, p_real = p_fake / total, p_real / total
        return reason, p_fake, p_real

    def build_augmented_text(self, newid2mid: Dict[int, str], mid2text: Dict[str, str]) -> Dict[int, str]:
        """Concatenate tweet text with the LLM reason for downstream featurization."""

        augmented = {}
        for newid, mid in newid2mid.items():
            reason, _, _ = self._lookup(mid)
            if reason:
                augmented[newid] = f"{mid2text.get(mid, '')} [LLM reason] {reason}"
        return augmented

    def build_soft_labels(self, newid2mid: Dict[int, str]) -> Dict[int, Dict[str, float]]:
        """Return KD soft labels aligned with the numeric tweet ids."""

        soft_labels: Dict[int, Dict[str, float]] = {}
        for newid, mid in newid2mid.items():
            _, p_fake, p_real = self._lookup(mid)
            soft_labels[newid] = {"p_fake": p_fake, "p_real": p_real}
        return soft_labels
