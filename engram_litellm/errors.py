class SnapshotError(Exception):
    """Base error for Engram snapshot operations."""

    def __init__(self, message: str, conversation_id: str = "", status_code: int = 0):
        self.conversation_id = conversation_id
        self.status_code = status_code
        super().__init__(message)


class RestoreError(SnapshotError):
    """Raised when snapshot restore fails."""


class SaveError(SnapshotError):
    """Raised when snapshot save fails."""
