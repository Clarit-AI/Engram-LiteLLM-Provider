import pytest
from litellm.llms.engram.state.tracker import ConversationTracker


class TestCollisionDetection:
    def setup_method(self):
        self.tracker = ConversationTracker()
        ConversationTracker._warned_distributed = False

    def test_no_collision_same_conversation(self):
        msgs1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        conv_id, collision = self.tracker.get_or_create(msgs1)
        assert collision is False
        self.tracker.record_turn(conv_id, msgs1, 1)

        # Same messages = same conversation
        _, collision = self.tracker.get_or_create(msgs1)
        assert collision is False

    def test_no_collision_continuation(self):
        msgs1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        conv_id, _ = self.tracker.get_or_create(msgs1)
        self.tracker.record_turn(conv_id, msgs1, 1)

        # Extended messages = continuation, not collision
        msgs2 = msgs1 + [
            {"role": "assistant", "content": "hello!"},
            {"role": "user", "content": "how are you?"},
        ]
        _, collision = self.tracker.get_or_create(msgs2)
        assert collision is False

    def test_collision_different_conversation_same_fingerprint(self):
        msgs1 = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        conv_id, _ = self.tracker.get_or_create(msgs1)
        # Record with extra messages that DON'T start with msgs1
        self.tracker.record_turn(
            conv_id,
            [{"role": "system", "content": "TOTALLY DIFFERENT"}],
            1,
        )

        # Now msgs1 fingerprint matches conv_id, but stored messages
        # are different — this is a collision
        _, collision = self.tracker.get_or_create(msgs1)
        assert collision is True

    def test_user_provided_id_never_collides(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        conv_id, collision = self.tracker.get_or_create(msgs, "explicit-id")
        assert conv_id == "explicit-id"
        assert collision is False
