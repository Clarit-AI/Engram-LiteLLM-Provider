import hashlib
import json
import logging
from typing import Dict, List, Optional

from .types import ConversationState, PrefixMatch

logger = logging.getLogger(__name__)

MAX_LOOKAHEAD_TURNS = 5


class ContextDiffer:
    """Detects redundant context in incoming messages by prefix matching."""

    @staticmethod
    def find_prefix_match(
        conv_id: str,
        incoming: List[dict],
        stored: ConversationState,
    ) -> Optional[PrefixMatch]:
        """Find prefix match between incoming messages and stored state.

        Strips messages from the tail (1 to MAX_LOOKAHEAD_TURNS) and
        compares the hash of the remaining prefix against stored state.
        Scoped to a single conversation_id to prevent cross-contamination.
        """
        if not incoming or not stored:
            return None

        incoming_hash = _hash_messages(incoming)

        # Exact match — no new messages since last save
        if incoming_hash == stored.messages_hash:
            return PrefixMatch(
                conversation_id=conv_id,
                turn_number=stored.turn_number,
                new_messages=[],
                tokens_saved=_estimate_tokens(incoming),
            )

        # Strip from tail, looking for prefix match
        max_strip = min(MAX_LOOKAHEAD_TURNS, len(incoming) - 1)
        for n_new in range(1, max_strip + 1):
            prefix = incoming[:-n_new]
            if not prefix:
                break
            prefix_hash = _hash_messages(prefix)

            if prefix_hash == stored.messages_hash:
                delta = incoming[-n_new:]
                return PrefixMatch(
                    conversation_id=conv_id,
                    turn_number=stored.turn_number,
                    new_messages=delta,
                    tokens_saved=_estimate_tokens(prefix),
                )

        return None


def _hash_messages(messages: List[dict]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _estimate_tokens(messages: List[dict]) -> int:
    """Rough estimate: 4 chars per token across all message content."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_chars += len(part["text"])
    return max(total_chars // 4, 1)
