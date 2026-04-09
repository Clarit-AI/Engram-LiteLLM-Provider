import pytest
import httpx
from engram_litellm.transformation import EngramChatConfig
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


@pytest.fixture
def mock_state():
    return MockEngramState()


@pytest.fixture
def mock_client(mock_state):
    transport = MockEngramTransport(mock_state)
    return httpx.AsyncClient(transport=transport, base_url="http://engram-mock")


class TestEngramChatConfig:
    def test_validate_environment_with_key(self, config):
        headers = {}
        result = config.validate_environment(
            headers, "model", [{"role": "user", "content": "hi"}], {}, api_key="test-key"
        )
        assert result["Authorization"] == "Bearer test-key"
        assert result["Content-Type"] == "application/json"

    def test_validate_environment_no_key(self, config):
        headers = {}
        result = config.validate_environment(
            headers, "model", [{"role": "user", "content": "hi"}], {}
        )
        assert "Authorization" not in result

    def test_get_complete_url(self, config):
        url = config.get_complete_url("http://localhost:30000", "granite")
        assert url == "http://localhost:30000/v1/chat/completions"

    def test_get_complete_url_strips_trailing_slash(self, config):
        url = config.get_complete_url("http://localhost:30000/", "granite")
        assert url == "http://localhost:30000/v1/chat/completions"

    def test_transform_request_stateless(self, config):
        messages = [{"role": "user", "content": "hi"}]
        optional_params = {"extra_body": {"stateful_mode": "stateless"}}
        result = config.transform_request("engram/model", messages, optional_params, {})
        assert "messages" in result
        assert "_engram_request_id" not in result

    def test_transform_request_auto_first_turn(self, config):
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ]
        optional_params = {"extra_body": {"conversation_id": "chat-1"}}
        result = config.transform_request("engram/model", messages, optional_params, {})
        assert "_engram_request_id" in result
        req_id = result["_engram_request_id"]
        state = config._get_state(req_id)
        assert state["conversation_id"] == "chat-1"
        assert state["turn_number"] == 1

    def test_transform_request_auto_with_prefix_match(self, config):
        prefix = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ]
        # Record first turn
        config._tracker.record_turn("chat-1", prefix, 1)

        # Second turn with new message
        messages = prefix + [
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        optional_params = {"extra_body": {"conversation_id": "chat-1"}}
        result = config.transform_request("engram/model", messages, optional_params, {})
        req_id = result["_engram_request_id"]
        state = config._get_state(req_id)
        assert "_prefix_match" in state
        assert state["_restore_target"] == "chat-1:1"
        assert state["turn_number"] == 2

    def test_transform_request_extracts_stateful_params(self, config):
        messages = [{"role": "user", "content": "hi"}]
        optional_params = {
            "extra_body": {
                "conversation_id": "c1",
                "auto_save": False,
                "temperature": 0.5,
            },
            "temperature": 0.7,
        }
        result = config.transform_request("engram/model", messages, optional_params, {})
        # Stateful params should be extracted, not in request
        assert "conversation_id" not in result
        assert "auto_save" not in result
        # Non-stateful extra_body params should be preserved
        req_id = result["_engram_request_id"]
        state = config._get_state(req_id)
        assert state["conversation_id"] == "c1"
        assert state["auto_save"] is False

    def test_build_request_strips_provider_prefix(self, config):
        request = config._build_request(
            "engram/granite-4.0-h-tiny",
            [{"role": "user", "content": "test"}],
            {"temperature": 0.5},
        )
        assert request["model"] == "granite-4.0-h-tiny"

    def test_side_channel_cleanup(self, config):
        import time

        config._STATE_TTL_SECONDS = 0.01  # Very short TTL
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        optional_params = {"extra_body": {"conversation_id": "c"}}
        result = config.transform_request("engram/m", messages, optional_params, {})
        assert len(config._pending_state) == 1
        time.sleep(0.02)
        config._cleanup_orphaned_state()
        assert len(config._pending_state) == 0
        config._STATE_TTL_SECONDS = 300  # Reset


class TestAsyncCompletion:
    @pytest.mark.asyncio
    async def test_basic_completion(self, config, mock_client, mock_state):
        messages = [{"role": "user", "content": "hello"}]
        optional_params = {"extra_body": {"conversation_id": "c1"}}
        request_data = config.transform_request(
            "engram/test-model", messages, optional_params, {}
        )

        response, metadata = await config.async_completion(
            model="engram/test-model",
            messages=messages,
            api_base="http://engram-mock",
            api_key=None,
            headers={},
            optional_params={},
            request_data=request_data,
            client=mock_client,
        )

        assert response.status_code == 200
        assert metadata.conversation_id == "c1"
        assert metadata.turn_number == 1
        assert mock_state.save_count == 1

    @pytest.mark.asyncio
    async def test_completion_with_restore(self, config, mock_client, mock_state):
        # Setup: save a snapshot first
        from engram_litellm.snapshot import SnapshotClient

        snap_client = SnapshotClient(api_base="http://engram-mock", client=mock_client)
        await snap_client.save_snapshot("c1", turn_number=1)

        messages = [{"role": "user", "content": "hello"}]
        optional_params = {"extra_body": {"conversation_id": "c1", "restore_from": "c1:1"}}
        request_data = config.transform_request(
            "engram/test-model", messages, optional_params, {}
        )

        response, metadata = await config.async_completion(
            model="engram/test-model",
            messages=messages,
            api_base="http://engram-mock",
            api_key=None,
            headers={},
            optional_params={},
            request_data=request_data,
            client=mock_client,
        )

        assert response.status_code == 200
        assert mock_state.restore_count == 1
        assert metadata.restore_time_ms is not None
        assert metadata.restore_failed is False

    @pytest.mark.asyncio
    async def test_completion_restore_failure_fallback(
        self, config, mock_client, mock_state
    ):
        mock_state.fail_restore = True

        messages = [{"role": "user", "content": "hello"}]
        optional_params = {
            "extra_body": {"conversation_id": "c1", "restore_from": "c1:1"}
        }
        request_data = config.transform_request(
            "engram/test-model", messages, optional_params, {}
        )

        # Should NOT raise — falls back to full prefill
        response, metadata = await config.async_completion(
            model="engram/test-model",
            messages=messages,
            api_base="http://engram-mock",
            api_key=None,
            headers={},
            optional_params={},
            request_data=request_data,
            client=mock_client,
        )

        assert response.status_code == 200
        assert metadata.restore_failed is True
        assert metadata.restore_error is not None

    @pytest.mark.asyncio
    async def test_completion_auto_save_disabled(
        self, config, mock_client, mock_state
    ):
        messages = [{"role": "user", "content": "hello"}]
        optional_params = {
            "extra_body": {"conversation_id": "c1", "auto_save": False}
        }
        request_data = config.transform_request(
            "engram/test-model", messages, optional_params, {}
        )

        response, metadata = await config.async_completion(
            model="engram/test-model",
            messages=messages,
            api_base="http://engram-mock",
            api_key=None,
            headers={},
            optional_params={},
            request_data=request_data,
            client=mock_client,
        )

        assert response.status_code == 200
        assert mock_state.save_count == 0
