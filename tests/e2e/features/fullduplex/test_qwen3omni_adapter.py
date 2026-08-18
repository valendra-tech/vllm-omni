# tests/e2e/features/fullduplex/test_qwen3omni_adapter.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter tests for the Qwen3-Omni duplex adapter (stubbed chat service)."""

import asyncio
import json

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.core.adapter import DuplexCapability
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.qwen3omni.adapter import (
    Qwen3OmniDuplexAdapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.session import (
    INTERRUPTED_MARKER,
    Qwen3OmniServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sse(modality: str, chunks: list[str]) -> str:
    payload = {
        "modality": modality,
        "choices": [{"delta": {"content": c}, "finish_reason": None} for c in chunks],
    }
    return "data: " + json.dumps(payload) + "\n\n"


class _StubChatService:
    def __init__(self):
        self.requests = []

    async def create_chat_completion(self, request, raw_request=None):
        self.requests.append(request)
        out = []
        if request.modalities and "audio" in request.modalities:
            out = [_sse("audio", ["<audio-b64>"]), _sse("text", [" hi "])]
        else:
            out = [_sse("text", ["hello"])]
        for item in out:
            yield item


class _FailingChatService:
    async def create_chat_completion(self, request, raw_request=None):
        raise RuntimeError("boom")


class _ErrorPayloadChatService:
    async def create_chat_completion(self, request, raw_request=None):
        yield "data: " + json.dumps({"error": "no audio produced"}) + "\n\n"


def _make_adapter():
    chat = _StubChatService()
    adapter = Qwen3OmniDuplexAdapter(chat)
    session = DuplexSession(session_id="s1")
    state = Qwen3OmniServingSessionState()
    adapter._states["s1"] = state
    return adapter, session, chat


def test_capabilities_audio_text_proactive():
    adapter, _, _ = _make_adapter()
    caps = adapter.capabilities()
    assert caps == DuplexCapability(
        input_modalities=frozenset({"audio"}),
        output_modalities=frozenset({"audio", "text"}),
        proactive=True,
    )


def test_on_input_audio_buffers_pcm():
    adapter, session, _ = _make_adapter()
    asyncio.run(adapter.on_input(session, "audio", np.zeros(100, dtype=np.float32)))
    assert adapter._states["s1"].pcm.size == 100


def test_respond_builds_chat_request_with_policy_and_audio():
    adapter, session, chat = _make_adapter()
    adapter._states["s1"].append_pcm(np.zeros(48000, dtype=np.float32))

    async def drive():
        chunks = [c async for c in adapter.respond(session)]
        return chunks

    chunks = asyncio.run(drive())
    request = chat.requests[0]
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][-1]["role"] == "user"
    assert request["messages"][-1]["content"][0]["type"] == "input_audio"
    assert request["modalities"] == ["audio", "text"]
    assert [c.modality for c in chunks] == ["audio", "text"]
    assert chunks[0].data == "<audio-b64>"
    assert adapter._states["s1"].history[-1]["content"] == " hi "
    assert len(adapter._states["s1"].history) == 2


def test_respond_adds_interruption_note_after_barge_in():
    adapter, session, chat = _make_adapter()
    adapter._states["s1"].record_user_input("first")
    adapter._states["s1"].record_interrupted_turn()
    adapter._states["s1"].append_pcm(np.zeros(48000, dtype=np.float32))

    async def drive():
        return [c async for c in adapter.respond(session)]

    asyncio.run(drive())
    request = chat.requests[0]
    system_messages = [m["content"] for m in request["messages"] if m["role"] == "system"]
    assert any("interrupted" in c for c in system_messages)
    assert INTERRUPTED_MARKER in [m["content"] for m in request["messages"]]


def test_on_barge_in_marks_in_flight_turn_interrupted():
    adapter, session, chat = _make_adapter()
    adapter._states["s1"].record_user_input("hello")
    adapter._states["s1"].append_pcm(np.zeros(48000, dtype=np.float32))
    adapter._responding.add(session.session_id)

    asyncio.run(adapter.on_barge_in(session))

    assert adapter._states["s1"].history[-1] == {
        "role": "assistant",
        "content": INTERRUPTED_MARKER,
    }
    assert adapter._states["s1"].last_turn_interrupted is True


def test_history_records_audio_user_turn_once():
    adapter, session, chat = _make_adapter()
    adapter._states["s1"].append_pcm(np.zeros(48000, dtype=np.float32))

    async def drive():
        return [c async for c in adapter.respond(session)]

    asyncio.run(drive())
    history = adapter._states["s1"].history
    assert history[0] == {"role": "user", "content": "[audio]"}
    assert history[1] == {"role": "assistant", "content": " hi "}
    assert len(history) == 2


def test_respond_propagates_service_exception():
    adapter = Qwen3OmniDuplexAdapter(_FailingChatService())
    session = DuplexSession(session_id="s1")
    adapter._states["s1"] = Qwen3OmniServingSessionState()
    adapter._states["s1"].last_turn_interrupted = True
    adapter._states["s1"].append_pcm(np.zeros(48000, dtype=np.float32))

    async def drive():
        return [c async for c in adapter.respond(session)]

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(drive())
    # interrupted flag survives a failed turn
    assert adapter._states["s1"].last_turn_interrupted is True
    assert adapter._states["s1"].history == []


def test_respond_raises_on_error_payload():
    adapter = Qwen3OmniDuplexAdapter(_ErrorPayloadChatService())
    session = DuplexSession(session_id="s1")
    adapter._states["s1"] = Qwen3OmniServingSessionState()
    adapter._states["s1"].append_pcm(np.zeros(48000, dtype=np.float32))

    async def drive():
        return [c async for c in adapter.respond(session)]

    with pytest.raises(RuntimeError, match="no audio produced"):
        asyncio.run(drive())
    assert adapter._states["s1"].history == []
