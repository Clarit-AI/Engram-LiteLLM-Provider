import pytest
from litellm.llms.engram.state.differ import ContextDiffer, _hash_messages
from litellm.llms.engram.types import ConversationState


class TestContextDiffer:
    def _make_stored(self, messages, turn=3):
        return ConversationState(
            turn_number=turn,
            messages_hash=_hash_messages(messages),
            last_messages=messages.copy(),
        )

    def test_exact_match(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        stored = self._make_stored(msgs)
        match = ContextDiffer.find_prefix_match("conv-1", msgs, stored)
        assert match is not None
        assert match.conversation_id == "conv-1"
        assert match.new_messages == []
        assert match.tokens_saved > 0

    def test_single_new_message(self):
        prefix = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        stored = self._make_stored(prefix)
        incoming = prefix + [{"role": "user", "content": "how are you?"}]
        match = ContextDiffer.find_prefix_match("conv-1", incoming, stored)
        assert match is not None
        assert len(match.new_messages) == 1
        assert match.new_messages[0]["content"] == "how are you?"
        assert match.turn_number == 3

    def test_multiple_new_messages(self):
        prefix = [{"role": "user", "content": "hello"}]
        stored = self._make_stored(prefix, turn=1)
        incoming = prefix + [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        match = ContextDiffer.find_prefix_match("conv-1", incoming, stored)
        assert match is not None
        assert len(match.new_messages) == 2

    def test_no_match(self):
        stored_msgs = [{"role": "user", "content": "hello"}]
        stored = self._make_stored(stored_msgs)
        incoming = [{"role": "user", "content": "completely different"}]
        match = ContextDiffer.find_prefix_match("conv-1", incoming, stored)
        assert match is None

    def test_empty_incoming(self):
        stored = self._make_stored([{"role": "user", "content": "hello"}])
        match = ContextDiffer.find_prefix_match("conv-1", [], stored)
        assert match is None

    def test_exceeds_lookahead(self):
        prefix = [{"role": "user", "content": "start"}]
        stored = self._make_stored(prefix, turn=1)
        # Add 6 new messages (exceeds MAX_LOOKAHEAD_TURNS=5)
        incoming = prefix + [
            {"role": "assistant", "content": f"r{i}"} for i in range(6)
        ]
        match = ContextDiffer.find_prefix_match("conv-1", incoming, stored)
        assert match is None

    def test_within_lookahead(self):
        prefix = [{"role": "user", "content": "start"}]
        stored = self._make_stored(prefix, turn=1)
        incoming = prefix + [
            {"role": "assistant", "content": f"r{i}"} for i in range(5)
        ]
        match = ContextDiffer.find_prefix_match("conv-1", incoming, stored)
        assert match is not None
        assert len(match.new_messages) == 5

    def test_tokens_saved_estimate(self):
        prefix = [
            {"role": "user", "content": "a" * 400},
            {"role": "assistant", "content": "b" * 400},
        ]
        stored = self._make_stored(prefix)
        incoming = prefix + [{"role": "user", "content": "new"}]
        match = ContextDiffer.find_prefix_match("conv-1", incoming, stored)
        assert match is not None
        assert match.tokens_saved >= 200  # 800 chars / 4
