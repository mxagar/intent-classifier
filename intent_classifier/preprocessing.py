"""Text normalization and tokenizer artifact helpers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from intent_classifier.settings import BackboneConfig
from intent_classifier.utils import ensure_dir

logger = logging.getLogger(__name__)

_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Apply light normalization while preserving accents, punctuation, and numbers."""
    return _SPACE_RE.sub(" ", text.strip())


def export_tokenizer(backbone: BackboneConfig, output_dir: str | Path | None = None) -> Path:
    """Download/export tokenizer files once so production can load locally."""
    target = ensure_dir(output_dir or backbone.local_tokenizer_path)
    logger.info("Exporting tokenizer %s@%s to %s", backbone.name, backbone.revision, target)
    tokenizer = AutoTokenizer.from_pretrained(backbone.name, revision=backbone.revision)
    tokenizer.save_pretrained(target)
    return target


def load_tokenizer(path: str | Path, local_files_only: bool = True) -> PreTrainedTokenizerBase:
    """Load tokenizer from disk by default to avoid production network dependency."""
    logger.info("Loading tokenizer from %s", path)
    return AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int,
    truncation: bool = True,
    padding: str = "max_length",
    return_tensors: str | None = None,
) -> dict[str, Any]:
    normalized = [normalize_text(text) for text in texts]
    return tokenizer(
        normalized,
        truncation=truncation,
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )

