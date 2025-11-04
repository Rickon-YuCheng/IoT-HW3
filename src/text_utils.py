import re
from typing import Iterable


URL_PATTERN = re.compile(r"https?://\S+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")


def normalize_message(text: str) -> str:
    """Basic text normalization used before vectorization."""
    text = text.lower()
    text = URL_PATTERN.sub(" url ", text)
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_corpus(texts: Iterable[str]) -> list[str]:
    """Normalize an iterable of messages."""
    return [normalize_message(text) for text in texts]
