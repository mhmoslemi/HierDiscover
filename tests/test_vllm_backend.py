import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from generation.gen_workers import VLLM_ENGINE_SHUTDOWN_TIMEOUT_S, _shutdown_vllm_engine


class VllmShutdownTest(unittest.TestCase):
    def test_grace_period_and_async_teardown(self):
        loop = asyncio.new_event_loop()
        cleanup = []

        async def output_handler():
            try:
                await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                cleanup.append("released")

        loop.create_task(output_handler())
        loop.run_until_complete(asyncio.sleep(0))
        shutdown = Mock()
        _shutdown_vllm_engine(SimpleNamespace(shutdown=shutdown), loop)

        shutdown.assert_called_once_with(timeout=VLLM_ENGINE_SHUTDOWN_TIMEOUT_S)
        self.assertEqual(cleanup, ["released"])
        self.assertTrue(loop.is_closed())

    def test_legacy_synchronous_engine_shuts_down(self):
        calls = []

        def shutdown():
            calls.append("stopped")

        core = SimpleNamespace(shutdown=shutdown)
        _shutdown_vllm_engine(SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core)), None)
        self.assertEqual(calls, ["stopped"])
