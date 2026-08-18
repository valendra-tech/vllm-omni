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


def _make_adapter():
    chat = _StubChatService()
    adapter = Qwen3OmniDuplexAdapter(chat)
    session = DuplexSession(session_id="s1")
    state = Qwen3OmniServingSessionState()
    adapter._state = state
    return adapter, session, chat


def test_capabilities_audio_text_proactive():
    adapter, _, _ = _make_adapter()
    caps = adapter.capabilities()
    assert caps == DuplexCapability(
        input_modalities=frozenset({"audio", "text"}),
        output_modalities=frozenset({"audio", "text"}),
        proactive=True,
    )


def test_on_input_audio_buffers_pcm():
    adapter, session, _ = _make_adapter()
    asyncio.run(adapter.on_input(session, "audio", np.zeros(100, dtype=np.float32)))
    assert adapter._state.pcm.size == 100


def test_respond_builds_chat_request_with_policy_and_audio():
    adapter, session, chat = _make_adapter()
    adapter._state.append_pcm(np.zeros(48000, dtype=np.float32))

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
    assert adapter._state.history[-1]["content"] == " hi "
    assert len(adapter._state.history) == 2


def test_respond_adds_interruption_note_after_barge_in():
    adapter, session, chat = _make_adapter()
    adapter._state.record_user_input("first")
    adapter._state.record_interrupted_turn()
    adapter._state.append_pcm(np.zeros(48000, dtype=np.float32))

    async def drive():
        return [c async for c in adapter.respond(session)]

    asyncio.run(drive())
    request = chat.requests[0]
    system_messages = [m["content"] for m in request["messages"] if m["role"] == "system"]
    assert any("interrupted" in c for c in system_messages)
    assert INTERRUPTED_MARKER in [m["content"] for m in request["messages"]]
