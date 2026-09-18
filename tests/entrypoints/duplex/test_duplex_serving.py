# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The thin websocket handler: handshake, command translation, event pump, resume/takeover."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import suppress
from dataclasses import replace
from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.events import AudioDelta, DuplexEvent, SessionClosed, SessionCreated
from vllm_omni.engine.duplex.fallback import DuplexFallbackRequest
from vllm_omni.engine.duplex.messages import (
    DuplexSessionError,
    DuplexSessionFallbackCancelMessage,
    DuplexSessionFallbackRequestMessage,
)
from vllm_omni.entrypoints.duplex.realtime_input import RealtimeEnvelope, parse_resume_request
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex.websocket import MAX_EVENT_BYTES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DISCONNECT = object()


class FakeWebSocket:
    def __init__(self, query: dict[str, str] | None = None) -> None:
        self.query_params = dict(query or {})
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self.closed: list[tuple[int, str]] = []
        self._send_failure: str | None = None
        self._inbound: asyncio.Queue[Any] = asyncio.Queue()

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.closed or self._send_failure is not None:
            raise RuntimeError(
                self._send_failure or "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
            )
        self.sent.append(json.loads(json.dumps(payload)))

    async def receive_text(self) -> str:
        item = await self._inbound.get()
        if item is _DISCONNECT:
            raise WebSocketDisconnect(code=1000)
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))
        # A closed socket ends the reader exactly like a client disconnect.
        self._inbound.put_nowait(_DISCONNECT)

    # ---- test helpers ----

    def feed(self, payload: dict[str, Any] | str) -> None:
        self._inbound.put_nowait(payload if isinstance(payload, str) else json.dumps(payload))

    def disconnect(self) -> None:
        self._inbound.put_nowait(_DISCONNECT)

    def break_sends(self) -> None:
        """Kill the write half only: the reader stays parked, as it does in practice."""
        self._send_failure = "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."

    def types(self) -> list[str]:
        return [payload["type"] for payload in self.sent]

    async def wait_for(self, wire_type: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            for payload in self.sent:
                if payload["type"] == wire_type:
                    return payload
            await asyncio.sleep(0.005)
        raise AssertionError(f"no {wire_type!r} in {self.types()}")


class FakeHandle:
    def __init__(self, session_id: str, capabilities: DuplexCapabilities, *, idle_timeout_s: float = 300) -> None:
        self.session_id = session_id
        self.capabilities = capabilities
        self.public_session: dict[str, Any] = {"id": session_id, "idle_timeout_s": idle_timeout_s}
        self.lease_generation = 0
        self.closed = False
        self.close_reasons: list[str] = []
        self.commands: list[commands.DuplexCommand] = []
        self._outbox: asyncio.Queue[DuplexEvent] = asyncio.Queue()

    def deliver(self, event: DuplexEvent) -> None:
        self._outbox.put_nowait(event)

    async def submit(self, command: commands.DuplexCommand) -> None:
        if self.closed:
            raise DuplexSessionError("closed", code="session_closed", session_id=self.session_id)
        self.commands.append(command)

    async def events(self):
        while True:
            event = await self._outbox.get()
            yield event
            if event.is_terminal:
                return

    async def close(self, *, reason: str = "client_close", timeout: float | None = None) -> None:
        self.close_reasons.append(reason)
        if self.closed:
            return
        self.closed = True
        self.deliver(SessionClosed(session_id=self.session_id, reason=reason))


class FakeOmni:
    def __init__(
        self, *, resumable: bool = True, replay_max_bytes: int = 64 * 1024, idle_timeout_s: float = 300
    ) -> None:
        self.engine = self
        self.model = "test-model"
        self.duplex_session_config = DuplexSessionRuntimeConfig(
            resume_replay_ttl_s=60.0, resume_replay_max_bytes_per_session=replay_max_bytes
        )
        self.capabilities = DuplexCapabilities(supports_session_resume=resumable)
        self.idle_timeout_s = idle_timeout_s
        self.opened: list[dict[str, Any]] = []
        self.handles: dict[str, FakeHandle] = {}
        self.resumed: list[tuple[str, int]] = []
        self.detached: list[str] = []
        self.open_error: DuplexSessionError | None = None
        self.fallback_sink = None
        self.fallback_started: list[tuple[str, str, str, int]] = []
        self.fallback_outputs: list[tuple[str, str, str, int, dict[str, object]]] = []
        self.fallback_failures: list[tuple[str, str, str, int, str, str]] = []

    def set_fallback_sink(self, sink) -> None:
        self.fallback_sink = sink

    async def submit_fallback_started_async(
        self, session_id: str, request_id: str, response_id: str, epoch: int
    ) -> None:
        self.fallback_started.append((session_id, request_id, response_id, epoch))

    async def submit_fallback_output_async(
        self,
        session_id: str,
        request_id: str,
        response_id: str,
        epoch: int,
        output: dict[str, object],
    ) -> None:
        self.fallback_outputs.append((session_id, request_id, response_id, epoch, output))

    async def submit_fallback_failed_async(
        self,
        session_id: str,
        request_id: str,
        response_id: str,
        epoch: int,
        error: str,
        error_code: str,
    ) -> None:
        self.fallback_failures.append((session_id, request_id, response_id, epoch, error, error_code))

    async def open_session(self, config: Any) -> FakeHandle:
        self.opened.append(dict(config))
        if self.open_error is not None:
            raise self.open_error
        session_id = f"duplex-{len(self.handles) + 1:032x}"
        handle = FakeHandle(session_id, self.capabilities, idle_timeout_s=self.idle_timeout_s)
        self.handles[session_id] = handle
        handle.deliver(SessionCreated(session_id=session_id, session={"id": session_id, "model": config.get("model")}))
        return handle

    def get_session(self, session_id: str) -> FakeHandle | None:
        return self.handles.get(session_id)

    async def resume_session(self, session_id: str, *, expected_lease_generation: int) -> FakeHandle:
        self.resumed.append((session_id, expected_lease_generation))
        handle = self.handles[session_id]
        handle.lease_generation = expected_lease_generation + 1
        return handle

    async def detach_session(self, session_id: str) -> None:
        self.detached.append(session_id)


def _handler(omni: FakeOmni, **kwargs: Any) -> OmniDuplexSessionHandler:
    kwargs.setdefault("config_timeout_s", 1.0)
    kwargs.setdefault("idle_timeout_s", 5.0)
    return OmniDuplexSessionHandler(duplex_omni=omni, **kwargs)


class FakeChatService:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.requests: list[Any] = []

    async def create_chat_completion(self, request: Any, raw_request: Any = None):
        del raw_request
        self.requests.append(request)

        async def stream():
            for chunk in self.chunks:
                yield chunk

        return stream()


class BlockingChatService(FakeChatService):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.release = asyncio.Event()

    async def create_chat_completion(self, request: Any, raw_request: Any = None):
        del raw_request
        self.requests.append(request)

        async def stream():
            self.started.set()
            try:
                await self.release.wait()
            finally:
                self.cancelled.set()
            if self.release.is_set():
                yield "data: [DONE]\n\n"

        return stream()


def _fallback_request() -> DuplexFallbackRequest:
    return DuplexFallbackRequest(
        session_id="sid",
        request_id="duplex-fallback-sid-0-1",
        response_id="resp-sid-0-1",
        epoch=0,
        history=({"role": "user", "content": "hello"},),
        response_config={"model": "qwen", "modalities": ["text"]},
        input_payload=None,
        policy_messages=(),
    )


def _session_update(**session: Any) -> dict[str, Any]:
    return {"type": "session.update", "session": {"model": "test-model", "modalities": ["audio", "text"], **session}}


@pytest.mark.asyncio
async def test_fallback_sink_runs_chat_stream_and_keeps_internal_messages_off_wire() -> None:
    omni = FakeOmni()
    chat_service = FakeChatService(
        [
            'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    assert len(chat_service.requests) == 1
    assert chat_service.requests[0].request_id == request.request_id
    assert omni.fallback_started == [(request.session_id, request.request_id, request.response_id, request.epoch)]
    assert [output[4]["text"] for output in omni.fallback_outputs if "text" in output[4]] == ["hello"]
    assert omni.fallback_outputs[-1][4]["end_of_turn"] is True
    assert omni.fallback_failures == []


@pytest.mark.asyncio
async def test_fallback_attaches_text_stream_to_audio_output_as_transcript() -> None:
    omni = FakeOmni()
    audio = base64.b64encode(b"\x00\x00" * 240).decode()
    chat_service = FakeChatService(
        [
            'data: {"choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}], "modality": "text"}\n\n',
            f'data: {{"choices": [{{"delta": {{"content": "{audio}"}}, "finish_reason": "stop"}}], "modality": "audio", "sample_rate_hz": 24000}}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    handler = _handler(omni, chat_service=chat_service)
    request = replace(
        _fallback_request(),
        response_config={"model": "qwen", "modalities": ["text", "audio"], "response_format": "pcm"},
    )

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    assert omni.fallback_outputs == [
        (
            request.session_id,
            request.request_id,
            request.response_id,
            request.epoch,
            {
                "audio": audio,
                "audio_format": "pcm",
                "data_plane_request_id": request.request_id,
                "sample_rate_hz": 24000,
                "audio_duration_ms": 10,
                "text": "hello",
            },
        ),
        (
            request.session_id,
            request.request_id,
            request.response_id,
            request.epoch,
            {"data_plane_request_id": request.request_id, "end_of_turn": True},
        ),
    ]
    assert omni.fallback_failures == []


@pytest.mark.asyncio
async def test_fallback_buffers_multiple_tool_call_fragments_until_done() -> None:
    omni = FakeOmni()
    first_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"city":"'},
                        },
                        {
                            "index": 1,
                            "id": "call-2",
                            "type": "function",
                            "function": {"name": "convert", "arguments": '{"unit":"'},
                        },
                    ]
                }
            }
        ]
    }
    second_chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": 'c"}'}},
                        {"index": 0, "function": {"arguments": 'Paris"}'}},
                    ]
                }
            }
        ]
    }
    chat_service = FakeChatService(
        [f"data: {json.dumps(first_chunk)}\n\n", f"data: {json.dumps(second_chunk)}\n\n", "data: [DONE]\n\n"]
    )
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    function_outputs = [output[4] for output in omni.fallback_outputs if output[4].get("function_call") is True]
    assert function_outputs == [
        {
            "function_call": True,
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"city":"Paris"}',
            "data_plane_request_id": request.request_id,
        },
        {
            "function_call": True,
            "call_id": "call-2",
            "name": "convert",
            "arguments": '{"unit":"c"}',
            "data_plane_request_id": request.request_id,
        },
    ]
    assert omni.fallback_outputs[-1][4] == {
        "data_plane_request_id": request.request_id,
        "end_of_turn": True,
    }
    assert omni.fallback_failures == []


@pytest.mark.asyncio
async def test_fallback_discards_buffered_tool_calls_without_done() -> None:
    omni = FakeOmni()
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }
    chat_service = FakeChatService([f"data: {json.dumps(chunk)}\n\n"])
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    assert not [output for output in omni.fallback_outputs if output[4].get("function_call") is True]
    assert not any(output[4].get("end_of_turn") is True for output in omni.fallback_outputs)
    assert omni.fallback_failures == [
        (
            request.session_id,
            request.request_id,
            request.response_id,
            request.epoch,
            "chat fallback stream ended before [DONE]",
            "fallback_stream_incomplete",
        )
    ]


@pytest.mark.asyncio
async def test_fallback_pcm_output_uses_negotiated_rate_for_duration() -> None:
    omni = FakeOmni()
    request = replace(
        _fallback_request(),
        response_config={
            "model": "qwen",
            "modalities": ["audio"],
            "response_format": "pcm",
            "extra_body": {
                "realtime_session_payload": {
                    "audio": {"output": {"format": "pcm16", "rate": 24_000}},
                },
            },
        },
    )
    audio = base64.b64encode(b"\x00\x00" * 240).decode()
    chat_service = FakeChatService(
        [
            f'data: {{"choices": [{{"delta": {{"audio": "{audio}"}}}}]}}\n\n',
            "data: [DONE]\n\n",
        ]
    )
    handler = _handler(omni, chat_service=chat_service)

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    audio_output = next(output[4] for output in omni.fallback_outputs if "audio" in output[4])
    assert audio_output["sample_rate_hz"] == 24_000
    assert audio_output["audio_duration_ms"] == 10


@pytest.mark.asyncio
async def test_fallback_ignores_records_after_done_marker() -> None:
    omni = FakeOmni()
    chat_service = FakeChatService(['data: [DONE]\n\ndata: {"choices": [{"delta": {"content": "late"}}]}\n\n'])
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    assert omni.fallback_outputs == [
        (
            request.session_id,
            request.request_id,
            request.response_id,
            request.epoch,
            {"data_plane_request_id": request.request_id, "end_of_turn": True},
        )
    ]
    assert omni.fallback_failures == []


@pytest.mark.asyncio
async def test_fallback_stream_without_done_fails_instead_of_synthesizing_success() -> None:
    omni = FakeOmni()
    chat_service = FakeChatService(['data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'])
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    assert [output[4]["text"] for output in omni.fallback_outputs if "text" in output[4]] == ["partial"]
    assert not any(output[4].get("end_of_turn") for output in omni.fallback_outputs)
    assert omni.fallback_failures == [
        (
            request.session_id,
            request.request_id,
            request.response_id,
            request.epoch,
            "chat fallback stream ended before [DONE]",
            "fallback_stream_incomplete",
        )
    ]


@pytest.mark.asyncio
async def test_malformed_fallback_sse_fails_without_synthesizing_success() -> None:
    omni = FakeOmni()
    chat_service = FakeChatService(["data: {not-json}\n\n"])
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(task, timeout=2.0)

    assert omni.fallback_outputs == []
    assert omni.fallback_failures == [
        (
            request.session_id,
            request.request_id,
            request.response_id,
            request.epoch,
            "malformed chat fallback SSE payload",
            "fallback_malformed_chunk",
        )
    ]


@pytest.mark.asyncio
async def test_fallback_cancel_stops_chat_stream_without_sending_late_output() -> None:
    omni = FakeOmni()
    chat_service = BlockingChatService()
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(chat_service.started.wait(), timeout=2.0)

    omni.fallback_sink(
        DuplexSessionFallbackCancelMessage(
            session_id=request.session_id,
            request_id=request.request_id,
            response_id=request.response_id,
            epoch=request.epoch,
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)

    assert task.cancelled()
    assert chat_service.cancelled.is_set()
    assert omni.fallback_started == [(request.session_id, request.request_id, request.response_id, request.epoch)]
    assert omni.fallback_outputs == []
    assert omni.fallback_failures == []


@pytest.mark.asyncio
async def test_handler_close_is_idempotent_and_ignores_late_fallback_messages() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    request = _fallback_request()
    sink = omni.fallback_sink

    assert sink is not None
    await handler.close()
    await handler.close()
    sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    await asyncio.sleep(0)

    assert handler._fallback_tasks == {}
    assert omni.fallback_failures == []


@pytest.mark.asyncio
async def test_stale_fallback_cancel_does_not_cancel_a_replacement_task() -> None:
    omni = FakeOmni()
    chat_service = BlockingChatService()
    handler = _handler(omni, chat_service=chat_service)
    request = _fallback_request()
    replacement = replace(request, request_id="duplex-fallback-sid-0-2", response_id="resp-sid-0-2")

    assert omni.fallback_sink is not None
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=request.session_id, request=request))
    first_task = handler._fallback_tasks[request.session_id]
    await asyncio.wait_for(chat_service.started.wait(), timeout=2.0)
    omni.fallback_sink(DuplexSessionFallbackRequestMessage(session_id=replacement.session_id, request=replacement))
    replacement_task = handler._fallback_tasks[request.session_id]
    await asyncio.sleep(0)

    omni.fallback_sink(
        DuplexSessionFallbackCancelMessage(
            session_id=request.session_id,
            request_id=request.request_id,
            response_id=request.response_id,
            epoch=request.epoch,
        )
    )
    await asyncio.sleep(0)

    assert first_task.cancelled() or first_task.done()
    assert not replacement_task.done()
    await handler.close()


async def _open(
    handler: OmniDuplexSessionHandler, omni: FakeOmni, **session: Any
) -> tuple[FakeWebSocket, FakeHandle, asyncio.Task]:
    ws = FakeWebSocket({"duplex": "1", "autostart": "0"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed(_session_update(**session))
    created = await ws.wait_for("session.created")
    handle = omni.handles[created["session"]["id"]]
    return ws, handle, task


# --------------------------------------------------------------------------- #
# Handshake and commands                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_session_update_opens_a_session_and_announces_the_server_id() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni, session_id="mine", instructions="hi")

    assert ws.accepted
    created = ws.sent[0]
    assert created["type"] == "session.created"
    assert created["session"]["id"] == handle.session_id != "mine"
    assert created["attachment_generation"] == 1
    assert isinstance(created["resume_token"], str) and created["resume_token"]
    assert "incarnation" not in created
    assert "server_event_seq" not in created
    # The whole session object went to open_session; the id inside it is ignored there.
    assert omni.opened == [
        {"model": "test-model", "modalities": ["audio", "text"], "session_id": "mine", "instructions": "hi"}
    ]

    ws.feed({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00\x01" * 8).decode("ascii")})
    ws.feed({"type": "input_audio_buffer.commit", "event_id": "evt-c"})
    await asyncio.sleep(0.05)
    assert [type(command) for command in handle.commands] == [commands.AppendAudio, commands.Commit]
    assert handle.commands[1].event_id == "evt-c"

    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    # A resumable session is detached (engine-owned grace), not closed.
    assert omni.detached == [handle.session_id]
    assert handle.close_reasons == []


@pytest.mark.asyncio
async def test_session_events_are_journaled_and_sent_in_order() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    delta = await ws.wait_for("response.output_audio.delta")
    assert delta["delta"] == "aGk=" and delta["server_event_seq"] == 1

    ws.feed({"type": "session.event_ack", "server_event_seq": 1})
    ws.feed({"type": "session.event_ack", "server_event_seq": 9, "event_id": "evt-ack"})
    error = await ws.wait_for("error")
    assert error["error"]["code"] == "invalid_event_ack" and error["error"]["event_id"] == "evt-ack"

    await handle.close(reason="client_close")
    closed = await ws.wait_for("session.closed")
    assert closed["reason"] == "client_close"
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.closed[0][1] == "client_close"


@pytest.mark.asyncio
async def test_envelope_errors_are_answered_locally_without_touching_the_session() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    ws.feed("not json")
    ws.feed("x" * (MAX_EVENT_BYTES + 1))
    ws.feed({"type": "totally.unknown", "event_id": "evt-u"})
    ws.feed({"type": "session.resume", "event_id": "evt-r"})
    ws.feed({"type": "playback.ack", "event_id": "evt-p"})  # played_ms missing
    await asyncio.sleep(0.1)

    codes = [payload["error"]["code"] for payload in ws.sent if payload["type"] == "error"]
    assert codes == ["invalid_json", "event_too_large", "unknown_event", "unsupported_session_resume", "bad_event"]
    errors = [payload for payload in ws.sent if payload["type"] == "error"]
    assert errors[2]["error"]["event_id"] == "evt-u"
    assert errors[4]["error"]["event_id"] == "evt-p"
    assert handle.commands == []
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_open_rejection_is_reported_and_the_socket_ends() -> None:
    omni = FakeOmni()
    omni.open_error = DuplexSessionError("no room", code="resource_exhausted", retryable=True)
    handler = _handler(omni)
    ws = FakeWebSocket({"duplex": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed(_session_update())
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.types() == ["error"]
    assert ws.sent[0]["error"]["code"] == "resource_exhausted"


@pytest.mark.asyncio
async def test_config_timeout_and_first_message_validation() -> None:
    omni = FakeOmni()
    handler = _handler(omni, config_timeout_s=0.05)
    ws = FakeWebSocket({"duplex": "1"})
    await asyncio.wait_for(handler.handle_realtime_session(ws), timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "config_timeout"

    ws = FakeWebSocket({"duplex": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed("{not json")
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "invalid_json"
    assert omni.opened == []


@pytest.mark.asyncio
async def test_idle_timeout_detaches_a_resumable_session_like_a_disconnect() -> None:
    omni = FakeOmni(idle_timeout_s=0.05)
    handler = _handler(omni, idle_timeout_s=0.05)
    ws, handle, task = await _open(handler, omni)

    await asyncio.wait_for(task, timeout=2.0)
    # Serving makes no session-lifetime decision: the engine lease decides
    # whether a silent, detached session expires.
    assert handle.close_reasons == []
    assert omni.detached == [handle.session_id]


@pytest.mark.asyncio
async def test_idle_timeout_closes_a_non_resumable_session() -> None:
    omni = FakeOmni(idle_timeout_s=0.05, resumable=False)
    handler = _handler(omni, idle_timeout_s=0.05)
    ws, handle, task = await _open(handler, omni)

    await asyncio.wait_for(task, timeout=2.0)
    assert handle.close_reasons == ["disconnect"]
    assert omni.detached == []


@pytest.mark.asyncio
async def test_transport_send_failure_detaches_the_session_instead_of_closing_it() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    # Only the write half dies, so the pump's send is the one and only report
    # of the broken socket (the reader is still parked on receive).
    ws.break_sends()
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    await asyncio.sleep(0.05)

    # The engine session is alive and resumable, so this is a disconnect of
    # the current attachment, not a session close; the pump keeps journaling.
    assert handle.close_reasons == []
    assert omni.detached == [handle.session_id]
    assert not handler._pumps[handle.session_id].done()

    # The reader sees the same broken socket a moment later. The attachment is
    # already gone, so this must not detach a second time: another detach would
    # restart the engine's disconnect grace window.
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    assert handle.close_reasons == []

    handle.deliver(SessionClosed(session_id=handle.session_id, reason="client_close"))
    await asyncio.sleep(0.05)
    assert handle.session_id not in handler._pumps


@pytest.mark.asyncio
async def test_non_resumable_session_is_closed_on_disconnect() -> None:
    omni = FakeOmni(resumable=False)
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    assert "resume_token" not in ws.sent[0]

    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert handle.close_reasons == ["disconnect"]
    assert omni.detached == []


@pytest.mark.asyncio
async def test_a_client_close_still_delivers_session_closed_before_the_socket_goes() -> None:
    """The read loop must not outrun the pump's terminal event.

    ``DuplexSessionHandle._deliver`` queues ``SessionClosed`` and marks the
    handle closed in one synchronous step, so ``handle.closed`` is already true
    while the event is still sitting in the outbox. ``_read_loop`` loops on
    exactly that flag: it returned, the endpoint returned, and the ASGI server
    tore the socket down with ``session.closed`` unsent -- the client saw an
    abrupt close (no close frame) instead of the terminal event it was waiting
    for.

    The teardown is what makes this observable, so it is modelled here: a real
    connection stops accepting writes the moment the endpoint returns, which is
    why a pump that is merely *still scheduled* is already too late.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws = FakeWebSocket({"duplex": "1", "autostart": "0"})

    async def serve() -> None:
        try:
            await handler.handle_realtime_session(ws)
        finally:
            # The ASGI server drops the connection when the endpoint returns;
            # anything the pump writes after this never reaches the client.
            ws.break_sends()

    task = asyncio.create_task(serve())
    ws.feed(_session_update())
    created = await ws.wait_for("session.created")
    handle = omni.handles[created["session"]["id"]]
    submit = handle.submit

    async def closing_submit(command: commands.DuplexCommand) -> None:
        await submit(command)
        if isinstance(command, commands.CloseSession):
            # Same order as the real handle: queue the event, then flip the flag.
            handle.deliver(SessionClosed(session_id=handle.session_id, reason="client_close"))
            handle.closed = True

    handle.submit = closing_submit  # type: ignore[method-assign]

    ws.feed({"type": "session.close"})
    await asyncio.wait_for(task, timeout=2.0)

    assert ws.types()[-1] == "session.closed", f"terminal event never reached the wire: {ws.types()}"
    assert ws.closed and ws.closed[0][0] == 1000, "the socket must close with a normal close frame"


# --------------------------------------------------------------------------- #
# Resume and takeover                                                         #
# --------------------------------------------------------------------------- #


async def _resume(
    handler: OmniDuplexSessionHandler,
    session_id: str,
    token: str,
    *,
    last_seq: int = 0,
) -> tuple[FakeWebSocket, asyncio.Task]:
    ws = FakeWebSocket({"duplex": "1", "resume": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed(
        {
            "type": "session.resume",
            "session_id": session_id,
            "resume_token": token,
            "last_received_server_event_seq": last_seq,
        }
    )
    return ws, task


@pytest.mark.asyncio
async def test_resume_after_disconnect_replays_missed_events_and_rotates_the_token() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="one"))
    await ws.wait_for("response.output_audio.delta")
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    # Emitted while detached: journaled for replay only.
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="two"))
    await asyncio.sleep(0.05)

    ws2, task2 = await _resume(handler, handle.session_id, token, last_seq=1)
    resumed = await ws2.wait_for("session.resumed")
    assert resumed["session_id"] == handle.session_id
    assert resumed["attachment_generation"] == 2
    assert resumed["resume_token"] != token
    assert "incarnation" not in resumed
    replayed = await ws2.wait_for("response.output_audio.delta")
    assert replayed["delta"] == "two" and replayed["server_event_seq"] == 2
    assert omni.resumed == [(handle.session_id, 0)]

    # The old token is revoked by the rotation.
    ws3, task3 = await _resume(handler, handle.session_id, token)
    await asyncio.wait_for(task3, timeout=2.0)
    assert ws3.sent[0]["error"]["code"] == "invalid_resume_token"

    ws2.feed({"type": "session.close"})
    await asyncio.sleep(0.05)
    ws2.disconnect()
    await asyncio.wait_for(task2, timeout=2.0)


@pytest.mark.asyncio
async def test_resume_takes_over_a_live_attachment() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]

    ws2, task2 = await _resume(handler, handle.session_id, token)
    await ws2.wait_for("session.resumed")
    replaced = await ws.wait_for("session.replaced")
    assert replaced["attachment_generation"] == 1
    assert ws.closed and ws.closed[0][1] == "session_replaced"
    # The replaced socket's later input is ignored; the new one drives the session.
    ws.feed({"type": "input_audio_buffer.clear"})
    ws2.feed({"type": "input_audio_buffer.clear", "event_id": "evt-new"})
    await asyncio.sleep(0.05)
    assert [command.event_id for command in handle.commands] == ["evt-new"]
    await asyncio.wait_for(task, timeout=2.0)

    ws2.disconnect()
    await asyncio.wait_for(task2, timeout=2.0)
    assert omni.detached == [handle.session_id]


@pytest.mark.asyncio
async def test_resume_validation_errors() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws = FakeWebSocket({"duplex": "1", "resume": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed({"type": "session.resume", "session_id": "duplex-x"})
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "invalid_session_resume"

    ws, task = await _resume(handler, "duplex-unknown", "token")
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "session_resume_expired"


@pytest.mark.asyncio
async def test_journal_overflow_degrades_to_live_delivery_with_resync_required() -> None:
    omni = FakeOmni(replay_max_bytes=256)
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="x" * 400))
    resync = await ws.wait_for("session.resync_required")
    assert resync["reason"] == "journal_overflow"
    delta = await ws.wait_for("response.output_audio.delta")
    assert "server_event_seq" not in delta
    assert ws.types().index("session.resync_required") < ws.types().index("response.output_audio.delta")

    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)


# --------------------------------------------------------------------------- #
# Envelope helpers                                                            #
# --------------------------------------------------------------------------- #


def test_realtime_envelope_query_rules() -> None:
    envelope = RealtimeEnvelope.from_query_params({"model": "m"})
    assert envelope.initial_open_payload() == {"model": "m"}
    assert envelope.initial_open_payload() is None  # autostart happens once

    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    assert envelope.resume_only is True
    assert envelope.initial_open_payload() is None
    assert RealtimeEnvelope.from_query_params({"resume": "1"}).resume_only is True
    assert RealtimeEnvelope.from_query_params({"session_id": "mine"}).default_session_payload() == {"model": None}


def test_realtime_envelope_first_message_classification() -> None:
    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    resume = envelope.first_message({"type": "session.resume", "session_id": "s"})
    assert resume.kind == "resume" and resume.resume_payload["session_id"] == "s"

    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    opened = envelope.first_message({"type": "session.update", "session": {"model": "x", "instructions": "hi"}})
    assert opened.kind == "open" and opened.session_payload == {"model": "x", "instructions": "hi"}
    assert opened.pending_command_payload is None

    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    autostarted = envelope.first_message({"type": "input_audio_buffer.commit"})
    assert autostarted.kind == "open" and autostarted.session_payload == {"model": "m"}
    assert autostarted.pending_command_payload == {"type": "input_audio_buffer.commit"}


def test_parse_resume_request_requires_the_three_fields_only() -> None:
    request = parse_resume_request({"session_id": "s", "resume_token": "t", "last_received_server_event_seq": 3})
    assert request is not None
    assert (request.session_id, request.resume_token, request.last_received_server_event_seq) == ("s", "t", 3)
    assert parse_resume_request({"session_id": "s", "resume_token": "t"}).last_received_server_event_seq == 0
    assert parse_resume_request({"session_id": "s", "resume_token": "t", "incarnation": 1}) is not None
    assert parse_resume_request({"session_id": "s"}) is None
    assert parse_resume_request({"session_id": "s", "resume_token": "t", "last_received_server_event_seq": -1}) is None


@pytest.mark.asyncio
async def test_a_resume_that_fails_to_activate_does_not_detach_the_live_attachment() -> None:
    """A rejected resume must roll back only what it owns.

    Two reconnects can both authenticate and both complete the engine resume;
    only one activates. The loser's activation raises, and the rollback used to
    detach the *session*, starting the engine's disconnect grace for the winner
    -- a grace ordinary heartbeats do not clear.

    The activation failure is injected rather than raced for: what is under test
    is the rollback's precondition, not the window that produces it.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry
    try:
        assert await registry.has_attachment(handle.session_id), "the opener is attached"
        detached_before = list(omni.detached)

        async def failing_resume(*args, **kwargs):
            raise RuntimeError("transport activation lost the race")

        original_resume = registry.resume
        registry.resume = failing_resume  # type: ignore[method-assign]
        try:
            ws_lose, task_lose = await _resume(handler, handle.session_id, token)
            error = await ws_lose.wait_for("error")
        finally:
            registry.resume = original_resume  # type: ignore[method-assign]

        assert error["error"]["code"] == "session_resume_conflict"
        assert omni.detached == detached_before, "the loser must not detach the winner"
        assert await registry.has_attachment(handle.session_id), "the winner is still attached"
    finally:
        for pending in (task, locals().get("task_lose")):
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pending
