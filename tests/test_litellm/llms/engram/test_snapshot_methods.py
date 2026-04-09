import pytest
import httpx
from litellm.llms.engram.snapshot.handler import SnapshotClient
from litellm.llms.engram.errors import RestoreError, SaveError, SnapshotError
from tests.test_litellm.llms.engram.fixtures.mock_engram_server import (
    MockEngramState,
    MockEngramTransport,
    create_mock_client,
)


@pytest.fixture
def mock_state():
    return MockEngramState()


@pytest.fixture
def mock_client(mock_state):
    transport = MockEngramTransport(mock_state)
    return httpx.AsyncClient(transport=transport, base_url="http://engram-mock")


@pytest.fixture
def snapshot_client(mock_client):
    return SnapshotClient(
        api_base="http://engram-mock",
        client=mock_client,
    )


class TestSnapshotClient:
    @pytest.mark.asyncio
    async def test_save_snapshot(self, snapshot_client, mock_state):
        resp = await snapshot_client.save_snapshot("conv-1", turn_number=1)
        assert resp.conversation_id == "conv-1"
        assert resp.turn_number == 1
        assert resp.snapshot_id == "snap-conv-1-1"
        assert mock_state.save_count == 1

    @pytest.mark.asyncio
    async def test_restore_snapshot(self, snapshot_client, mock_state):
        await snapshot_client.save_snapshot("conv-1", turn_number=1)
        resp = await snapshot_client.restore_snapshot("conv-1", turn_number=1)
        assert resp.conversation_id == "conv-1"
        assert resp.restore_time_ms == 2.1
        assert mock_state.restore_count == 1

    @pytest.mark.asyncio
    async def test_restore_not_found(self, snapshot_client):
        with pytest.raises(RestoreError):
            await snapshot_client.restore_snapshot("nonexistent")

    @pytest.mark.asyncio
    async def test_save_failure(self, snapshot_client, mock_state):
        mock_state.fail_save = True
        with pytest.raises(SaveError):
            await snapshot_client.save_snapshot("conv-1")

    @pytest.mark.asyncio
    async def test_list_snapshots(self, snapshot_client, mock_state):
        await snapshot_client.save_snapshot("conv-1", turn_number=1)
        await snapshot_client.save_snapshot("conv-1", turn_number=2)
        snapshots = await snapshot_client.list_snapshots("conv-1")
        assert len(snapshots) == 2
        assert snapshots[0].turn_number == 1
        assert snapshots[1].turn_number == 2

    @pytest.mark.asyncio
    async def test_get_snapshot_info(self, snapshot_client):
        await snapshot_client.save_snapshot("conv-1", turn_number=1)
        info = await snapshot_client.get_snapshot_info("conv-1", turn_number=1)
        assert info.conversation_id == "conv-1"
        assert info.snapshot_id == "snap-conv-1-1"

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, snapshot_client, mock_state):
        await snapshot_client.save_snapshot("conv-1", turn_number=1)
        resp = await snapshot_client.delete_snapshot("conv-1", turn_number=1)
        assert resp.deleted is True
        snapshots = await snapshot_client.list_snapshots("conv-1")
        assert len(snapshots) == 0

    @pytest.mark.asyncio
    async def test_restore_failure(self, snapshot_client, mock_state):
        mock_state.fail_restore = True
        await snapshot_client.save_snapshot("conv-1", turn_number=1)
        with pytest.raises(RestoreError):
            await snapshot_client.restore_snapshot("conv-1")
