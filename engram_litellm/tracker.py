import hashlib
import json
import logging
import os
import threading
import uuid
from typing import Dict, List, Optional, Tuple

from .types import ConversationState

logger = logging.getLogger(__name__)


class ConversationTracker:
    """Thread-safe, process-local conversation state tracker.

    Tracks conversation history hashes for prefix detection in auto mode.
    Process-local only — logs a warning when multiple workers are detected.
    """

    _warned_distributed = False

    def __init__(self):
        self._state: Dict[str, ConversationState] = {}
        self._lock = threading.Lock()
        self._check_distributed()

    def _check_distributed(self):
        if ConversationTracker._warned_distributed:
            return
        worker_count = self._detect_workers()
        if worker_count > 1:
            logger.warning(
                "Engram ConversationTracker running with %d workers. "
                "Prefix matching in 'auto' mode is process-local and will fail "
                "across workers. Use explicit conversation_id for distributed "
                "deployments, or configure ENGRAM_TRACKER_REDIS_URL for shared state.",
                worker_count,
            )
            ConversationTracker._warned_distributed = True

    @staticmethod
    def _detect_workers() -> int:
        if gunicorn_workers := os.environ.get("GUNICORN_WORKERS"):
            try:
                return int(gunicorn_workers)
            except ValueError:
                pass
        if os.environ.get("WEB_CONCURRENCY"):
            try:
                return int(os.environ["WEB_CONCURRENCY"])
            except ValueError:
                pass
        return 1

    def get_or_create(
        self, messages: List[dict], user_id: Optional[str] = None
    ) -> Tuple[str, bool]:
        """Return (conversation_id, is_collision).

        Priority: user-provided > pseudo-ID from first 2 messages > UUID.
        On fingerprint collision, returns True so caller skips prefix match.
        """
        with self._lock:
            if user_id:
                return user_id, False

            if len(messages) >= 2:
                pseudo_id = self._fingerprint(messages[:2])

                if pseudo_id in self._state:
                    existing = self._state[pseudo_id]
                    current_hash = self._hash_messages(messages)
                    # Check if this is a continuation (prefix match) or collision
                    if not self._is_prefix_or_match(messages, existing):
                        logger.warning(
                            "Fingerprint collision on %s. "
                            "Skipping stateful optimization for this call.",
                            pseudo_id,
                        )
                        return pseudo_id, True

                return pseudo_id, False

            return str(uuid.uuid4()), False

    def _is_prefix_or_match(
        self, incoming: List[dict], stored: ConversationState
    ) -> bool:
        """Check if incoming is the same conversation (prefix or exact match)."""
        stored_msgs = stored.last_messages
        if len(incoming) < len(stored_msgs):
            return False
        # Check if stored messages are a prefix of incoming
        for i, msg in enumerate(stored_msgs):
            if i >= len(incoming):
                return False
            if self._hash_message(incoming[i]) != self._hash_message(msg):
                return False
        return True

    def record_turn(self, conv_id: str, messages: List[dict], turn: int):
        """Thread-safe state update."""
        with self._lock:
            self._state[conv_id] = ConversationState(
                turn_number=turn,
                messages_hash=self._hash_messages(messages),
                last_messages=[m.copy() for m in messages],
            )

    def get_state(self, conv_id: str) -> Optional[ConversationState]:
        with self._lock:
            return self._state.get(conv_id)

    def get_turn_number(self, conv_id: str) -> int:
        with self._lock:
            if conv_id in self._state:
                return self._state[conv_id].turn_number
            return 0

    def clear(self, conv_id: Optional[str] = None):
        with self._lock:
            if conv_id:
                self._state.pop(conv_id, None)
            else:
                self._state.clear()

    @staticmethod
    def _fingerprint(messages: List[dict]) -> str:
        canonical = json.dumps(messages[:2], sort_keys=True, ensure_ascii=True)
        return f"auto-{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    @staticmethod
    def _hash_messages(messages: List[dict]) -> str:
        canonical = json.dumps(messages, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _hash_message(message: dict) -> str:
        canonical = json.dumps(message, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
