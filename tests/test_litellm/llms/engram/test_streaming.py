import asyncio
import pytest
from litellm.llms.engram.streaming.wrapper import EngramStreamWrapper


class TestEngramStreamWrapper:
    def test_sync_iteration(self):
        chunks = ["chunk1", "chunk2", "chunk3"]
        saved = {"called": False}

        async def save_fn():
            saved["called"] = True

        wrapper = EngramStreamWrapper(
            stream=iter(chunks),
            save_fn=save_fn,
            conv_id="c1",
            turn_number=1,
        )

        collected = list(wrapper)
        assert collected == chunks
        # Give daemon thread time to run
        import time
        time.sleep(0.2)
        assert saved["called"] is True

    def test_sync_cancellation_skips_save(self):
        saved = {"called": False}

        async def save_fn():
            saved["called"] = True

        def gen():
            yield "chunk1"
            yield "chunk2"

        wrapper = EngramStreamWrapper(
            stream=gen(),
            save_fn=save_fn,
            conv_id="c1",
            turn_number=1,
        )

        it = iter(wrapper)
        next(it)  # Get first chunk
        it.close()  # Cancel
        import time
        time.sleep(0.1)
        assert saved["called"] is False

    @pytest.mark.asyncio
    async def test_async_iteration(self):
        saved = {"called": False}

        async def save_fn():
            saved["called"] = True

        async def async_gen():
            for chunk in ["a", "b", "c"]:
                yield chunk

        wrapper = EngramStreamWrapper(
            stream=async_gen(),
            save_fn=save_fn,
            conv_id="c1",
            turn_number=1,
        )

        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        assert collected == ["a", "b", "c"]
        await asyncio.sleep(0.1)
        assert saved["called"] is True

    @pytest.mark.asyncio
    async def test_save_failure_logged_not_raised(self):
        async def failing_save():
            raise RuntimeError("save failed!")

        async def async_gen():
            yield "chunk"

        wrapper = EngramStreamWrapper(
            stream=async_gen(),
            save_fn=failing_save,
            conv_id="c1",
            turn_number=1,
        )

        collected = []
        async for chunk in wrapper:
            collected.append(chunk)

        assert collected == ["chunk"]
        await asyncio.sleep(0.1)
        # Should not raise — failure is logged silently
