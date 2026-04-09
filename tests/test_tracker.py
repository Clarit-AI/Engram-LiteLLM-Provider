import threading
import pytest
from engram_litellm.tracker import ConversationTracker


class TestConversationTracker:
    def setup_method(self):
        self.tracker = ConversationTracker()
        ConversationTracker._warned_distributed = False

    def test_user_provided_id(self):
        msgs = [{"role": "user", "content": "hello"}]
        conv_id, collision = self.tracker.get_or_create(msgs, "my-chat-123")
        assert conv_id == "my-chat-123"
        assert collision is False

    def test_pseudo_id_from_messages(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        conv_id, collision = self.tracker.get_or_create(msgs)
        assert conv_id.startswith("auto-")
        assert collision is False

    def test_same_messages_same_pseudo_id(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        id1, _ = self.tracker.get_or_create(msgs)
        # Record a turn so the ID is stored
        self.tracker.record_turn(id1, msgs, 1)
        id2, _ = self.tracker.get_or_create(msgs)
        assert id1 == id2

    def test_uuid_fallback_single_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        conv_id, collision = self.tracker.get_or_create(msgs)
        assert not conv_id.startswith("auto-")
        assert collision is False

    def test_collision_detection(self):
        msgs1 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        # Record first conversation
        conv_id, _ = self.tracker.get_or_create(msgs1)
        self.tracker.record_turn(conv_id, msgs1, 1)

        # Second conversation with same first 2 messages but different continuation
        msgs2 = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "DIFFERENT CONTENT"},
            {"role": "user", "content": "something completely different"},
        ]
        # This should NOT be a collision because msgs2 is a continuation of msgs1
        # (msgs1 is a prefix of msgs2)
        _, collision = self.tracker.get_or_create(msgs2)
        assert collision is False

    def test_record_and_retrieve(self):
        msgs = [{"role": "user", "content": "test"}]
        self.tracker.record_turn("conv-1", msgs, 3)
        state = self.tracker.get_state("conv-1")
        assert state is not None
        assert state.turn_number == 3

    def test_get_turn_number_default(self):
        assert self.tracker.get_turn_number("nonexistent") == 0

    def test_get_turn_number_after_record(self):
        self.tracker.record_turn("c", [{"role": "user", "content": "x"}], 5)
        assert self.tracker.get_turn_number("c") == 5

    def test_clear_specific(self):
        self.tracker.record_turn("a", [{"role": "user", "content": "x"}], 1)
        self.tracker.record_turn("b", [{"role": "user", "content": "y"}], 1)
        self.tracker.clear("a")
        assert self.tracker.get_state("a") is None
        assert self.tracker.get_state("b") is not None

    def test_clear_all(self):
        self.tracker.record_turn("a", [{"role": "user", "content": "x"}], 1)
        self.tracker.record_turn("b", [{"role": "user", "content": "y"}], 1)
        self.tracker.clear()
        assert self.tracker.get_state("a") is None
        assert self.tracker.get_state("b") is None

    def test_thread_safety(self):
        errors = []

        def worker(conv_id, msgs, turn):
            try:
                self.tracker.record_turn(conv_id, msgs, turn)
                self.tracker.get_state(conv_id)
                self.tracker.get_or_create(msgs, conv_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            msgs = [{"role": "user", "content": f"msg-{i}"}]
            t = threading.Thread(target=worker, args=(f"conv-{i}", msgs, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
