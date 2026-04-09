import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TokenizerClient:
    """Tokenizer for engram_tokens_saved accounting ONLY.

    This does NOT construct the request body. The restore-and-generate
    endpoint handles tokenization server-side. This client is used
    to estimate token savings for response metadata.

    Loads from HuggingFace model ID (lazy), with ENGRAM_TOKENIZER_PATH override.
    Falls back to character-based estimation if transformers is unavailable.
    """

    _cache: Dict[str, object] = {}

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._local_path = os.environ.get("ENGRAM_TOKENIZER_PATH")
        self._tokenizer = None
        self._load_attempted = False

    def _ensure_tokenizer(self):
        if self._load_attempted:
            return
        self._load_attempted = True

        cache_key = self._local_path or self._model_name
        if cache_key in TokenizerClient._cache:
            self._tokenizer = TokenizerClient._cache[cache_key]
            return

        try:
            from transformers import AutoTokenizer

            path = self._local_path or self._model_name
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            TokenizerClient._cache[cache_key] = self._tokenizer
            logger.debug("Loaded tokenizer for %s", cache_key)
        except Exception as e:
            logger.warning(
                "Failed to load tokenizer for %s: %s. "
                "Token savings estimates will use character-based approximation.",
                cache_key,
                e,
            )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string."""
        self._ensure_tokenizer()
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        # Fallback: ~4 chars per token
        return max(len(text) // 4, 1)

    def estimate_tokens_for_messages(self, messages: List[dict]) -> int:
        """Estimate token count for a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total += self.estimate_tokens(part["text"])
            # Per-message overhead (role, formatting)
            total += 4
        return total
