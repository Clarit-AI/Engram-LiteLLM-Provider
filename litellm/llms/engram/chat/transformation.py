import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx

from litellm.llms.engram.errors import RestoreError
from litellm.llms.engram.snapshot.handler import SnapshotClient
from litellm.llms.engram.state.differ import ContextDiffer
from litellm.llms.engram.state.tokenizer import TokenizerClient
from litellm.llms.engram.state.tracker import ConversationTracker
from litellm.llms.engram.streaming.wrapper import EngramStreamWrapper
from litellm.llms.engram.types import EngramStateMetadata, PrefixMatch

logger = logging.getLogger(__name__)

# Default env vars
DEFAULT_BASE_URL = "http://localhost:30000"
DEFAULT_AUTO_SAVE = True
DEFAULT_STATEFUL_MODE = "auto"

# Params extracted from extra_body for stateful operation
STATEFUL_PARAMS = frozenset(
    {
        "conversation_id",
        "restore_from",
        "auto_save",
        "stateful_mode",
        "branch_name",
    }
)


class EngramChatConfig:
    """Engram LiteLLM provider with automatic stateful inference.

    Modes:
    - 'auto' (default): Detect redundant context, use restore-and-generate
    - 'stateless': Standard OpenAI-compatible calls, no snapshot management
    - 'explicit': User manages snapshots via restore_from/conversation_id

    State is managed via a class-level side-channel keyed by request UUID,
    avoiding serialization of internal state into the HTTP request body.
    """

    # Class-level side-channel for passing state between transform_request
    # and async_completion without polluting the request body
    _pending_state: Dict[str, dict] = {}
    _state_lock = threading.Lock()
    _STATE_TTL_SECONDS = 300

    def __init__(self):
        self._tracker = ConversationTracker()
        self._tokenizers: Dict[str, TokenizerClient] = {}
        self._differ = ContextDiffer()

    # --- LiteLLM Provider Interface Methods ---

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[dict],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Setup headers for Engram server."""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.setdefault("Content-Type", "application/json")
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        endpoint: str = "chat/completions",
    ) -> str:
        base = api_base or os.environ.get("ENGRAM_BASE_URL", DEFAULT_BASE_URL)
        base = base.rstrip("/")
        return f"{base}/v1/{endpoint}"

    def transform_request(
        self,
        model: str,
        messages: List[dict],
        optional_params: dict,
        headers: dict,
        litellm_params: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """Transform request. Stateful params stored in side-channel."""
        # Extract stateful params from optional_params/extra_body
        stateful = self._extract_stateful_params(optional_params)

        # Determine conversation ID
        user_conv_id = stateful.get("conversation_id")
        mode = stateful.get("stateful_mode", DEFAULT_STATEFUL_MODE)

        if mode == "stateless":
            # No stateful tracking
            return self._build_request(model, messages, optional_params)

        conv_id, is_collision = self._tracker.get_or_create(messages, user_conv_id)
        stateful["conversation_id"] = conv_id
        stateful["_is_collision"] = is_collision

        # Check prefix match for auto mode
        if mode == "auto" and not is_collision and not stateful.get("restore_from"):
            stored = self._tracker.get_state(conv_id)
            if stored:
                match = self._differ.find_prefix_match(conv_id, messages, stored)
                if match and match.new_messages:
                    stateful["_prefix_match"] = match
                    stateful["_restore_target"] = (
                        f"{conv_id}:{match.turn_number}"
                    )

        # Determine turn number
        current_turn = self._tracker.get_turn_number(conv_id) + 1
        stateful["turn_number"] = current_turn

        # Store state in side-channel
        request_id = str(uuid.uuid4())
        with self._state_lock:
            self._pending_state[request_id] = {
                **stateful,
                "_created_at": time.time(),
                "_messages": messages,
            }

        # Build request with engram_request_id for retrieval
        request = self._build_request(model, messages, optional_params)
        request["_engram_request_id"] = request_id
        return request

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: Any,
        logging_obj: Any = None,
        request_data: Optional[dict] = None,
        messages: Optional[List[dict]] = None,
        optional_params: Optional[dict] = None,
        encoding: Any = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Process response and attach Engram metadata."""
        json_data = raw_response.json()

        if hasattr(model_response, "choices") and model_response.choices:
            if json_data.get("choices"):
                choice = json_data["choices"][0]
                msg = choice.get("message", {})
                if hasattr(model_response.choices[0], "message"):
                    model_response.choices[0].message.content = msg.get("content", "")
                    model_response.choices[0].message.role = msg.get("role", "assistant")
                model_response.choices[0].finish_reason = choice.get("finish_reason")

        model_response.model = json_data.get("model", model)

        if hasattr(model_response, "usage") and json_data.get("usage"):
            usage = json_data["usage"]
            model_response.usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return model_response

    async def async_completion(
        self,
        model: str,
        messages: List[dict],
        api_base: str,
        api_key: Optional[str],
        headers: dict,
        optional_params: dict,
        request_data: dict,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        **kwargs,
    ) -> Tuple[Any, EngramStateMetadata]:
        """Override to inject restore logic before the HTTP call.

        This is the correct interception point since OpenAILikeChatConfig
        does not expose an async_pre_api_call hook.
        """
        # Extract request ID and retrieve state from side-channel
        request_id = request_data.pop("_engram_request_id", None)
        state = self._get_state(request_id) if request_id else {}

        metadata = EngramStateMetadata(
            conversation_id=state.get("conversation_id"),
            turn_number=state.get("turn_number"),
            auto_save=state.get("auto_save", DEFAULT_AUTO_SAVE),
        )

        # Handle restore if needed
        restore_target = state.get("restore_from") or state.get("_restore_target")
        if restore_target:
            await self._attempt_restore(
                restore_target=restore_target,
                api_base=api_base,
                api_key=api_key,
                headers=headers,
                client=client,
                metadata=metadata,
            )

        # Make the actual completion call
        url = self.get_complete_url(api_base, model)
        request_headers = self.validate_environment(
            headers.copy(), model, messages, optional_params, api_key, api_base
        )

        async def _do_request() -> httpx.Response:
            if client:
                return await client.post(
                    url,
                    json=request_data,
                    headers=request_headers,
                    timeout=timeout,
                )
            else:
                async with httpx.AsyncClient() as c:
                    return await c.post(
                        url,
                        json=request_data,
                        headers=request_headers,
                        timeout=timeout,
                    )

        response = await _do_request()
        response.raise_for_status()

        # Record turn and auto-save
        if metadata.conversation_id and not state.get("stateful_mode") == "stateless":
            all_messages = state.get("_messages", messages)
            self._tracker.record_turn(
                metadata.conversation_id,
                all_messages,
                metadata.turn_number or 1,
            )

            if metadata.auto_save:
                snapshot_client = SnapshotClient(
                    api_base=api_base, api_key=api_key, headers=headers, client=client
                )
                try:
                    save_resp = await snapshot_client.save_snapshot(
                        conversation_id=metadata.conversation_id,
                        turn_number=metadata.turn_number,
                    )
                    metadata.snapshot_id = save_resp.snapshot_id
                except Exception as e:
                    logger.warning(
                        "Engram auto-save failed for %s:%s: %s",
                        metadata.conversation_id,
                        metadata.turn_number,
                        e,
                    )

        # Compute tokens saved estimate
        prefix_match: Optional[PrefixMatch] = state.get("_prefix_match")
        if prefix_match and not metadata.restore_failed:
            metadata.tokens_saved = prefix_match.tokens_saved

        return response, metadata

    async def async_streaming(
        self,
        model: str,
        messages: List[dict],
        api_base: str,
        api_key: Optional[str],
        headers: dict,
        optional_params: dict,
        request_data: dict,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        **kwargs,
    ):
        """Streaming completion with restore-before and save-after."""
        request_id = request_data.pop("_engram_request_id", None)
        state = self._get_state(request_id) if request_id else {}

        metadata = EngramStateMetadata(
            conversation_id=state.get("conversation_id"),
            turn_number=state.get("turn_number"),
            auto_save=state.get("auto_save", DEFAULT_AUTO_SAVE),
        )

        # Restore before streaming starts
        restore_target = state.get("restore_from") or state.get("_restore_target")
        if restore_target:
            await self._attempt_restore(
                restore_target=restore_target,
                api_base=api_base,
                api_key=api_key,
                headers=headers,
                client=client,
                metadata=metadata,
            )

        url = self.get_complete_url(api_base, model)
        request_headers = self.validate_environment(
            headers.copy(), model, messages, optional_params, api_key, api_base
        )
        request_data["stream"] = True

        # Create save function for the stream wrapper
        async def save_fn():
            if not metadata.conversation_id or not metadata.auto_save:
                return
            all_messages = state.get("_messages", messages)
            self._tracker.record_turn(
                metadata.conversation_id,
                all_messages,
                metadata.turn_number or 1,
            )
            snapshot_client = SnapshotClient(
                api_base=api_base, api_key=api_key, headers=headers, client=client
            )
            await snapshot_client.save_snapshot(
                conversation_id=metadata.conversation_id,
                turn_number=metadata.turn_number,
            )

        # Stream with wrapper
        if client:
            resp = await client.post(
                url,
                json=request_data,
                headers=request_headers,
                timeout=timeout,
            )
        else:
            async with httpx.AsyncClient() as c:
                resp = await c.post(
                    url,
                    json=request_data,
                    headers=request_headers,
                    timeout=timeout,
                )

        resp.raise_for_status()

        wrapped = EngramStreamWrapper(
            stream=resp.aiter_lines(),
            save_fn=save_fn,
            conv_id=metadata.conversation_id or "",
            turn_number=metadata.turn_number or 0,
        )

        return wrapped, metadata

    # --- Internal Methods ---

    async def _attempt_restore(
        self,
        restore_target: str,
        api_base: str,
        api_key: Optional[str],
        headers: dict,
        client: Optional[httpx.AsyncClient],
        metadata: EngramStateMetadata,
    ):
        """Attempt snapshot restore with fallback to full prefill."""
        parts = restore_target.split(":", 1)
        conv_id = parts[0]
        turn_number = int(parts[1]) if len(parts) > 1 else None

        snapshot_client = SnapshotClient(
            api_base=api_base, api_key=api_key, headers=headers, client=client
        )
        try:
            start = time.perf_counter()
            resp = await snapshot_client.restore_snapshot(
                conversation_id=conv_id, turn_number=turn_number
            )
            metadata.restore_time_ms = (time.perf_counter() - start) * 1000
        except (httpx.HTTPStatusError, httpx.RequestError, RestoreError) as e:
            logger.warning(
                "Engram restore failed for %s: %s. Falling back to full prefill.",
                restore_target,
                e,
            )
            metadata.restore_failed = True
            metadata.restore_error = str(e)

    def _extract_stateful_params(self, optional_params: dict) -> dict:
        """Extract Engram-specific params from optional_params."""
        stateful = {}
        # Check extra_body first
        extra_body = optional_params.pop("extra_body", None) or {}
        for key in STATEFUL_PARAMS:
            if key in extra_body:
                stateful[key] = extra_body.pop(key)
            elif key in optional_params:
                stateful[key] = optional_params.pop(key)
        # Put back remaining extra_body
        if extra_body:
            optional_params["extra_body"] = extra_body

        # Apply defaults from env
        stateful.setdefault(
            "auto_save",
            os.environ.get("ENGRAM_AUTO_SAVE", "true").lower() == "true",
        )
        stateful.setdefault(
            "stateful_mode",
            os.environ.get("ENGRAM_STATEFUL_MODE", DEFAULT_STATEFUL_MODE),
        )

        return stateful

    def _build_request(
        self, model: str, messages: List[dict], optional_params: dict
    ) -> dict:
        """Build the OpenAI-compatible request body."""
        request = {
            "model": model.split("/", 1)[-1] if "/" in model else model,
            "messages": messages,
        }
        # Forward standard OpenAI params
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "n",
        ):
            if key in optional_params:
                request[key] = optional_params[key]
        return request

    def _get_state(self, request_id: Optional[str]) -> dict:
        """Retrieve and remove state from side-channel."""
        if not request_id:
            return {}
        with self._state_lock:
            state = self._pending_state.pop(request_id, {})
        return state

    def _cleanup_orphaned_state(self):
        """Periodic cleanup for entries where hook never fired."""
        now = time.time()
        with self._state_lock:
            orphaned = [
                rid
                for rid, s in self._pending_state.items()
                if now - s.get("_created_at", 0) > self._STATE_TTL_SECONDS
            ]
            for rid in orphaned:
                del self._pending_state[rid]
                logger.debug("Cleaned orphaned state for request %s", rid)

    def get_tokenizer(self, model: str) -> TokenizerClient:
        if model not in self._tokenizers:
            self._tokenizers[model] = TokenizerClient(model)
        return self._tokenizers[model]
