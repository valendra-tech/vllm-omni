# tests/entrypoints/openai_api/test_duplex_handler_qwen3omni.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni duplex handler integration tests (require full vllm stack).

Runs on the GPU validation host; the local stub tree cannot import the
handler (vllm dependency).
"""

import pytest

from tests.entrypoints.openai_api.test_duplex_handler import (
    FakeChatService,
    FakeEngineClient,
    TimedWebSocket,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexSession,
    DuplexSessionConfig,
)
from vllm_omni.experimental.fullduplex.openai.serving import OmniDuplexSessionHandler
from vllm_omni.experimental.fullduplex.qwen3omni.policy import (
    INTERRUPTION_NOTE,
    SYSTEM_PROMPT,
)
from vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter import (
    Qwen3OmniServingRuntimeAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode(samples, sample_rate, fmt, speed=None):
    return "audio-b64"


def _qwen3_handler() -> OmniDuplexSessionHandler:
    return OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
        serving_runtime_adapter=Qwen3OmniServingRuntimeAdapter(_encode),
    )


def _session_create(session_id: str) -> dict[str, object]:
    return {
        "type": "session.create",
        "session_id": session_id,
        "session": {
            "model": "test-model",
            "modalities": ["text", "audio"],
            "idle_timeout_s": 1,
            "extra_body": {"session_mode": "duplex"},
        },
    }


def _message_pairs(request) -> list[tuple[str, object]]:
    return [(message["role"], message["content"]) for message in request.model_dump()["messages"]]


def test_build_chat_request_injects_policy_once():
    handler = _qwen3_handler()
    session = DuplexSession(
        session_id="sid-policy",
        config=DuplexSessionConfig(model="test-model", instructions="instr"),
    )
    session.append_text("hi")
    session.commit_user_input()
    state = handler._serving_runtime_adapter.session_state(session.session_id)

    request = handler._build_chat_request(session, "req-1")
    pairs = _message_pairs(request)
    assert pairs[0] == ("system", SYSTEM_PROMPT)
    assert ("system", "instr") in pairs
    assert ("user", "hi") in pairs
    assert state.last_turn_interrupted is False

    state.last_turn_interrupted = True
    request2 = handler._build_chat_request(session, "req-2")
    pairs2 = _message_pairs(request2)
    assert pairs2[0] == ("system", SYSTEM_PROMPT)
    assert pairs2[1] == ("system", INTERRUPTION_NOTE)
    assert ("system", "instr") in pairs2
    assert state.last_turn_interrupted is False

    request3 = handler._build_chat_request(session, "req-3")
    pairs3 = _message_pairs(request3)
    assert pairs3[0] == ("system", SYSTEM_PROMPT)
    assert pairs3[1] == ("system", "instr")
    assert ("system", INTERRUPTION_NOTE) not in pairs3


def test_build_chat_request_skips_policy_for_non_qwen3_adapter():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-policy-minicpmo",
        config=DuplexSessionConfig(model="test-model", instructions="instr"),
    )
    session.append_text("hi")
    session.commit_user_input()

    pairs = _message_pairs(handler._build_chat_request(session, "req-1"))
    assert pairs == [("system", "instr"), ("user", "hi")]


@pytest.mark.asyncio
async def test_qwen3omni_barge_in_marks_turn_interrupted_with_active_response():
    handler = _qwen3_handler()

    def on_send(ws: TimedWebSocket, data: dict[str, object]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-qwen-barge"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types().count("response.created") == 1
    state = handler._serving_runtime_adapter.session_states["sid-qwen-barge"]
    assert state.last_turn_interrupted is True


@pytest.mark.asyncio
async def test_qwen3omni_barge_in_without_active_response_does_not_mark_interrupted():
    handler = _qwen3_handler()
    ws = TimedWebSocket()
    ws.put(_session_create("sid-qwen-barge-none"))
    ws.put({"type": "input.cancel", "reason": "no_active_response"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    state = handler._serving_runtime_adapter.session_states["sid-qwen-barge-none"]
    assert state.last_turn_interrupted is False
