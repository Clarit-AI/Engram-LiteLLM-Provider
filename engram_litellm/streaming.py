import asyncio
import logging
import threading
from typing import Any, AsyncIterator, Callable, Coroutine, Iterator, Optional

logger = logging.getLogger(__name__)


class EngramStreamWrapper:
    """Wraps a stream with async save-after-completion.

    Detects sync vs async context for the fire-and-forget save.
    Skips save on mid-stream cancellation (GeneratorExit).
    """

    def __init__(
        self,
        stream: Any,
        save_fn: Callable[[], Coroutine],
        conv_id: str,
        turn_number: int,
    ):
        self._stream = stream
        self._save_fn = save_fn
        self._conv_id = conv_id
        self._turn = turn_number
        self._cancelled = False

    def __iter__(self) -> Iterator:
        try:
            for chunk in self._stream:
                yield chunk
        except GeneratorExit:
            self._cancelled = True
            raise
        finally:
            if not self._cancelled:
                self._fire_save()

    async def __aiter__(self) -> AsyncIterator:
        try:
            async for chunk in self._stream:
                yield chunk
        except GeneratorExit:
            self._cancelled = True
            raise
        finally:
            if not self._cancelled:
                self._fire_save()

    def _fire_save(self):
        """Fire-and-forget save, context-aware."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_and_log())
        except RuntimeError:
            # No event loop — use daemon thread
            t = threading.Thread(
                target=asyncio.run,
                args=(self._save_and_log(),),
                daemon=True,
            )
            t.start()

    async def _save_and_log(self):
        try:
            await self._save_fn()
        except Exception as e:
            logger.warning(
                "Engram snapshot save failed for %s:%d: %s",
                self._conv_id,
                self._turn,
                e,
            )
