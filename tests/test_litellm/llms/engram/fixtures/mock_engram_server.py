"""Mock Engram server for unit tests using httpx transport mocking."""

import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import httpx


@dataclass
class MockSnapshot:
    conversation_id: str
    turn_number: int
    snapshot_id: str
    created_at: str
    size_bytes: int = 56_000_000


@dataclass
class MockEngramState:
    """In-memory state for the mock server."""
    snapshots: Dict[str, List[MockSnapshot]] = field(default_factory=dict)
    restore_count: int = 0
    save_count: int = 0
    last_restore: Optional[str] = None
    last_save: Optional[str] = None
    fail_restore: bool = False
    fail_save: bool = False
    completions: List[dict] = field(default_factory=list)


class MockEngramTransport(httpx.AsyncBaseTransport):
    """httpx async transport that mocks Engram server endpoints."""

    def __init__(self, state: Optional[MockEngramState] = None):
        self.state = state or MockEngramState()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}

        if path == "/v1/chat/completions":
            return self._handle_completion(body)
        elif path == "/save_snapshot":
            return self._handle_save(body)
        elif path == "/restore_snapshot":
            return self._handle_restore(body)
        elif path == "/list_snapshots":
            return self._handle_list(body)
        elif path == "/get_snapshot_info":
            return self._handle_info(body)
        elif path == "/delete_snapshot":
            return self._handle_delete(body)
        else:
            return httpx.Response(404, json={"error": f"Unknown path: {path}"})

    def _handle_completion(self, body: dict) -> httpx.Response:
        self.state.completions.append(body)
        model = body.get("model", "test-model")
        content = f"Mock response to: {body.get('messages', [{}])[-1].get('content', '')}"

        resp = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        return httpx.Response(200, json=resp)

    def _handle_save(self, body: dict) -> httpx.Response:
        if self.state.fail_save:
            return httpx.Response(500, json={"error": "Save failed"})

        conv_id = body.get("conversation_id", "unknown")
        turn = body.get("turn_number", 1)
        snap_id = f"snap-{conv_id}-{turn}"

        snap = MockSnapshot(
            conversation_id=conv_id,
            turn_number=turn,
            snapshot_id=snap_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        if conv_id not in self.state.snapshots:
            self.state.snapshots[conv_id] = []
        self.state.snapshots[conv_id].append(snap)

        self.state.save_count += 1
        self.state.last_save = conv_id

        return httpx.Response(
            200,
            json={
                "conversation_id": conv_id,
                "turn_number": turn,
                "snapshot_id": snap_id,
                "size_bytes": snap.size_bytes,
            },
        )

    def _handle_restore(self, body: dict) -> httpx.Response:
        if self.state.fail_restore:
            return httpx.Response(500, json={"error": "Restore failed"})

        conv_id = body.get("conversation_id", "unknown")
        turn = body.get("turn_number")

        snaps = self.state.snapshots.get(conv_id, [])
        if not snaps:
            return httpx.Response(
                404, json={"error": f"No snapshots for {conv_id}"}
            )

        if turn is not None:
            matching = [s for s in snaps if s.turn_number == turn]
            if not matching:
                return httpx.Response(
                    404,
                    json={"error": f"No snapshot at turn {turn} for {conv_id}"},
                )

        self.state.restore_count += 1
        self.state.last_restore = conv_id

        return httpx.Response(
            200,
            json={
                "conversation_id": conv_id,
                "turn_number": turn or snaps[-1].turn_number,
                "restore_time_ms": 2.1,
            },
        )

    def _handle_list(self, body: dict) -> httpx.Response:
        conv_id = body.get("conversation_id", "unknown")
        snaps = self.state.snapshots.get(conv_id, [])
        return httpx.Response(
            200,
            json={
                "snapshots": [
                    {
                        "conversation_id": s.conversation_id,
                        "turn_number": s.turn_number,
                        "snapshot_id": s.snapshot_id,
                        "created_at": s.created_at,
                        "size_bytes": s.size_bytes,
                        "tier": "vram",
                    }
                    for s in snaps
                ]
            },
        )

    def _handle_info(self, body: dict) -> httpx.Response:
        conv_id = body.get("conversation_id", "unknown")
        turn = body.get("turn_number")
        snaps = self.state.snapshots.get(conv_id, [])

        if not snaps:
            return httpx.Response(404, json={"error": "Not found"})

        snap = snaps[-1]
        if turn is not None:
            matching = [s for s in snaps if s.turn_number == turn]
            if matching:
                snap = matching[0]

        return httpx.Response(
            200,
            json={
                "conversation_id": snap.conversation_id,
                "turn_number": snap.turn_number,
                "snapshot_id": snap.snapshot_id,
                "created_at": snap.created_at,
                "size_bytes": snap.size_bytes,
                "tier": "vram",
            },
        )

    def _handle_delete(self, body: dict) -> httpx.Response:
        conv_id = body.get("conversation_id", "unknown")
        turn = body.get("turn_number")

        if conv_id in self.state.snapshots:
            if turn is not None:
                self.state.snapshots[conv_id] = [
                    s
                    for s in self.state.snapshots[conv_id]
                    if s.turn_number != turn
                ]
            else:
                del self.state.snapshots[conv_id]

        return httpx.Response(
            200,
            json={
                "conversation_id": conv_id,
                "turn_number": turn,
                "deleted": True,
            },
        )


def create_mock_client(state: Optional[MockEngramState] = None) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient backed by the mock transport."""
    s = state or MockEngramState()
    transport = MockEngramTransport(s)
    return httpx.AsyncClient(transport=transport, base_url="http://engram-mock")
