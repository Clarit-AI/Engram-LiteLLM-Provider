import logging
from typing import Any, Dict, List, Optional

import httpx

from .errors import RestoreError, SaveError, SnapshotError
from .types import (
    DeleteSnapshotResponse,
    RestoreSnapshotResponse,
    SaveSnapshotResponse,
    SnapshotMetadata,
)

logger = logging.getLogger(__name__)


class SnapshotClient:
    """Async HTTP client for Engram snapshot endpoints."""

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._base_headers = headers or {}
        self._external_client = client

    def _get_headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        h.update(self._base_headers)
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _request(
        self, method: str, path: str, json_data: Optional[dict] = None
    ) -> Dict[str, Any]:
        url = f"{self._api_base}{path}"
        headers = self._get_headers()

        if self._external_client:
            resp = await self._external_client.request(
                method, url, json=json_data, headers=headers
            )
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.request(
                    method, url, json=json_data, headers=headers
                )

        resp.raise_for_status()
        return resp.json()

    async def save_snapshot(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
        branch_name: Optional[str] = None,
    ) -> SaveSnapshotResponse:
        payload: Dict[str, Any] = {"conversation_id": conversation_id}
        if turn_number is not None:
            payload["turn_number"] = turn_number
        if branch_name:
            payload["branch_name"] = branch_name

        try:
            data = await self._request("POST", "/save_snapshot", payload)
        except httpx.HTTPStatusError as e:
            raise SaveError(
                f"Save failed: {e}",
                conversation_id=conversation_id,
                status_code=e.response.status_code,
            ) from e

        return SaveSnapshotResponse(
            conversation_id=data.get("conversation_id", conversation_id),
            turn_number=data.get("turn_number", turn_number or 0),
            snapshot_id=data.get("snapshot_id", ""),
            size_bytes=data.get("size_bytes"),
        )

    async def restore_snapshot(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
    ) -> RestoreSnapshotResponse:
        payload: Dict[str, Any] = {"conversation_id": conversation_id}
        if turn_number is not None:
            payload["turn_number"] = turn_number

        try:
            data = await self._request("POST", "/restore_snapshot", payload)
        except httpx.HTTPStatusError as e:
            raise RestoreError(
                f"Restore failed: {e}",
                conversation_id=conversation_id,
                status_code=e.response.status_code,
            ) from e

        return RestoreSnapshotResponse(
            conversation_id=data.get("conversation_id", conversation_id),
            turn_number=data.get("turn_number", turn_number or 0),
            restore_time_ms=data.get("restore_time_ms", 0.0),
        )

    async def list_snapshots(
        self, conversation_id: str
    ) -> List[SnapshotMetadata]:
        payload = {"conversation_id": conversation_id}
        try:
            data = await self._request("POST", "/list_snapshots", payload)
        except httpx.HTTPStatusError as e:
            raise SnapshotError(
                f"List snapshots failed: {e}",
                conversation_id=conversation_id,
                status_code=e.response.status_code,
            ) from e

        snapshots = data.get("snapshots", [])
        return [
            SnapshotMetadata(
                conversation_id=s.get("conversation_id", conversation_id),
                turn_number=s.get("turn_number", 0),
                snapshot_id=s.get("snapshot_id", ""),
                created_at=s.get("created_at", ""),
                size_bytes=s.get("size_bytes"),
                tier=s.get("tier"),
            )
            for s in snapshots
        ]

    async def get_snapshot_info(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
    ) -> SnapshotMetadata:
        payload: Dict[str, Any] = {"conversation_id": conversation_id}
        if turn_number is not None:
            payload["turn_number"] = turn_number

        try:
            data = await self._request("POST", "/get_snapshot_info", payload)
        except httpx.HTTPStatusError as e:
            raise SnapshotError(
                f"Get snapshot info failed: {e}",
                conversation_id=conversation_id,
                status_code=e.response.status_code,
            ) from e

        return SnapshotMetadata(
            conversation_id=data.get("conversation_id", conversation_id),
            turn_number=data.get("turn_number", 0),
            snapshot_id=data.get("snapshot_id", ""),
            created_at=data.get("created_at", ""),
            size_bytes=data.get("size_bytes"),
            tier=data.get("tier"),
        )

    async def delete_snapshot(
        self,
        conversation_id: str,
        turn_number: Optional[int] = None,
    ) -> DeleteSnapshotResponse:
        payload: Dict[str, Any] = {"conversation_id": conversation_id}
        if turn_number is not None:
            payload["turn_number"] = turn_number

        try:
            data = await self._request("POST", "/delete_snapshot", payload)
        except httpx.HTTPStatusError as e:
            raise SnapshotError(
                f"Delete snapshot failed: {e}",
                conversation_id=conversation_id,
                status_code=e.response.status_code,
            ) from e

        return DeleteSnapshotResponse(
            conversation_id=data.get("conversation_id", conversation_id),
            turn_number=data.get("turn_number"),
            deleted=data.get("deleted", True),
        )
