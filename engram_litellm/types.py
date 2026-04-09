from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationState:
    turn_number: int
    messages_hash: str
    last_messages: List[Dict[str, Any]]


@dataclass
class PrefixMatch:
    conversation_id: str
    turn_number: int
    new_messages: List[Dict[str, Any]]
    tokens_saved: int


@dataclass
class SaveSnapshotResponse:
    conversation_id: str
    turn_number: int
    snapshot_id: str
    size_bytes: Optional[int] = None


@dataclass
class RestoreSnapshotResponse:
    conversation_id: str
    turn_number: int
    restore_time_ms: float


@dataclass
class SnapshotMetadata:
    conversation_id: str
    turn_number: int
    snapshot_id: str
    created_at: str
    size_bytes: Optional[int] = None
    tier: Optional[str] = None


@dataclass
class DeleteSnapshotResponse:
    conversation_id: str
    turn_number: Optional[int] = None
    deleted: bool = True


@dataclass
class EngramStateMetadata:
    """Metadata attached to responses for Engram state tracking."""
    conversation_id: Optional[str] = None
    turn_number: Optional[int] = None
    snapshot_id: Optional[str] = None
    tokens_saved: Optional[int] = None
    restore_time_ms: Optional[float] = None
    restore_failed: bool = False
    restore_error: Optional[str] = None
    auto_save: bool = True
