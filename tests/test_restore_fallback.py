import pytest
import httpx
from engram_litellm.transformation import EngramChatConfig
from engram_litellm.types import EngramStateMetadata
from tests.fixtures.mock_engram_server import (
    MockEngramState,
    MockEngramTransport,
)


@pytest.fixture
def config():
    c = EngramChatConfig()
    c._tracker.clear()
    EngramChatConfig._pending_state.clear()
    return c


class TestRestoreFallback:
    @pytest.mark.asyncio
    async def test_restore_http_error_falls_back(self, config):
        """On restore HTTP error, should log warning and set restore_failed."""
        state = MockEngramState()
        state.fail_restore = True
        transport = MockEngramTransport(state)
        client = httpx.AsyncClient(transport=transport, base_url="http://mock")

        metadata = EngramStateMetadata()
        await config._attempt_restore(
            restore_target="conv-1:1",
            api_base="http://mock",
            api_key=None,
            headers={},
            client=client,
            metadata=metadata,
        )

        assert metadata.restore_failed is True
        assert metadata.restore_error is not None

    @pytest.mark.asyncio
    async def test_restore_success_records_time(self, config):
        """On successful restore, should record restore_time_ms."""
        state = MockEngramState()
        transport = MockEngramTransport(state)
        client = httpx.AsyncClient(transport=transport, base_url="http://mock")

        # Need a snapshot to restore
        from engram_litellm.snapshot import SnapshotClient
        snap_client = SnapshotClient(api_base="http://mock", client=client)
        await snap_client.save_snapshot("conv-1", turn_number=1)

        metadata = EngramStateMetadata()
        await config._attempt_restore(
            restore_target="conv-1:1",
            api_base="http://mock",
            api_key=None,
            headers={},
            client=client,
            metadata=metadata,
        )

        assert metadata.restore_failed is False
        assert metadata.restore_time_ms is not None
        assert metadata.restore_time_ms >= 0

    @pytest.mark.asyncio
    async def test_restore_target_parsing(self, config):
        """Test conv_id:turn_number parsing."""
        state = MockEngramState()
        transport = MockEngramTransport(state)
        client = httpx.AsyncClient(transport=transport, base_url="http://mock")

        from engram_litellm.snapshot import SnapshotClient
        snap_client = SnapshotClient(api_base="http://mock", client=client)
        await snap_client.save_snapshot("my-conv", turn_number=3)

        metadata = EngramStateMetadata()
        await config._attempt_restore(
            restore_target="my-conv:3",
            api_base="http://mock",
            api_key=None,
            headers={},
            client=client,
            metadata=metadata,
        )

        assert metadata.restore_failed is False
        assert state.last_restore == "my-conv"

    @pytest.mark.asyncio
    async def test_restore_no_turn_number(self, config):
        """Test restore_target without turn number."""
        state = MockEngramState()
        transport = MockEngramTransport(state)
        client = httpx.AsyncClient(transport=transport, base_url="http://mock")

        from engram_litellm.snapshot import SnapshotClient
        snap_client = SnapshotClient(api_base="http://mock", client=client)
        await snap_client.save_snapshot("conv-1", turn_number=1)

        metadata = EngramStateMetadata()
        await config._attempt_restore(
            restore_target="conv-1",
            api_base="http://mock",
            api_key=None,
            headers={},
            client=client,
            metadata=metadata,
        )

        assert metadata.restore_failed is False
