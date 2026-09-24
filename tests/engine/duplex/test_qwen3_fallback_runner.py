# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine-resident Qwen3 chat-fallback lifecycle scenarios."""

from __future__ import annotations

import asyncio
import base64
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.commands import DuplexCommand
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import (
    DuplexStagePort,
    DuplexStageRequestContext,
    DuplexStageSubmission,
    DuplexStageSubmissionResult,
)
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexControlResultMessage,
    DuplexSessionCommandMessage,
    DuplexSessionEventMessage,
    DuplexSessionFallbackCancelMessage,
    DuplexSessionFallbackFailedMessage,
    DuplexSessionFallbackOutputMessage,
    DuplexSessionFallbackRequestMessage,
    DuplexSessionFallbackStartedMessage,
    OpenDuplexSessionMessage,
)
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.engine.duplex.turn_detection import TurnDetectionResult
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.model_executor.models.qwen3_omni.duplex.policy import INTERRUPTION_NOTE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION_ID = "qwen3-duplex-test"


class RecordingStagePort(DuplexStagePort):
    """Stage fake that proves Qwen fallback turns never reach native append."""

    def __init__(self) -> None:
        self.ensured: list[DuplexStageRequestContext] = []
        self.submissions: list[DuplexStageSubmission] = []
        self.cleanups: list[tuple[list[str], bool]] = []
        self.aborts: list[list[str]] = []

    @property
    def stage_count(self) -> int:
        return 1

    def sampling_defaults(self) -> tuple[object, ...]:
        return (None,)

    def ensure_request(self, context: DuplexStageRequestContext) -> None:
        self.ensured.append(context)

    async def submit(self, submission: DuplexStageSubmission) -> DuplexStageSubmissionResult:
        self.submissions.append(submission)
        return DuplexStageSubmissionResult(
            request_id=submission.context.request_id,
            stage_id=submission.context.stage_id,
            replica_id=0,
        )

    async def cleanup(self, request_ids: list[str], *, abort: bool = False) -> None:
        self.cleanups.append((list(request_ids), abort))

    async def abort_requests(self, request_ids: list[str]) -> None:
        self.aborts.append(list(request_ids))


def _encode_audio(audio: object, sample_rate_hz: int, response_format: str, speed: float | None) -> str | None:
    del audio, sample_rate_hz, response_format, speed
    return None


def _pcm_f32(value: float = 0.05, samples: int = 16_000) -> bytes:
    return struct.pack(f"<{samples}f", *([value] * samples))


class _ScriptedDetector:
    """Return deterministic VAD results without loading the Silero model."""

    def __init__(self, results: list[TurnDetectionResult]) -> None:
        self._results = list(results)

    def process(
        self,
        base64_audio: str,
        *,
        fmt: str,
        sample_rate_hz: int | None,
        audio_end_ms: int | None = None,
    ) -> TurnDetectionResult:
        del base64_audio, fmt, sample_rate_hz, audio_end_ms
        return self._results.pop(0)

    def reset(self) -> None:
        pass


def _speech_then_stop_detector() -> _ScriptedDetector:
    return _ScriptedDetector(
        [
            TurnDetectionResult(is_speech=True, speech_active=True, speech_started=True, speech_probability=0.9),
            TurnDetectionResult(
                is_speech=False,
                speech_active=False,
                speech_stopped=True,
                speech_probability=0.05,
                should_commit=True,
                create_response=True,
            ),
        ]
    )


@dataclass
class Harness:
    manager: DuplexSessionManager
    port: RecordingStagePort
    output: asyncio.Queue[Any]
    results: asyncio.Queue[Any]
    runner: DuplexSessionRunner
    events: list[Any] = field(default_factory=list)
    fallback_requests: list[DuplexSessionFallbackRequestMessage] = field(default_factory=list)
    fallback_cancels: list[DuplexSessionFallbackCancelMessage] = field(default_factory=list)

    @property
    def session(self):
        return self.runner.session

    def submit(self, command: DuplexCommand) -> None:
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=SESSION_ID, command=command))

    async def settle(self, *, timeout_s: float = 3.0) -> list[Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        collected: list[Any] = []
        quiet_since: float | None = None
        while True:
            drained = False
            while not self.output.empty():
                message = self.output.get_nowait()
                drained = True
                if isinstance(message, DuplexSessionEventMessage):
                    collected.append(message.event)
                elif isinstance(message, DuplexSessionFallbackRequestMessage):
                    self.fallback_requests.append(message)
                elif isinstance(message, DuplexSessionFallbackCancelMessage):
                    self.fallback_cancels.append(message)
                else:
                    raise AssertionError(f"unexpected engine output: {message!r}")
            busy = (
                drained
                or not self.runner._mailbox.empty()
                or any(not task.done() for task in self.runner._background_tasks)
            )
            now = loop.time()
            if busy:
                quiet_since = None
            elif quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= 0.05:
                break
            if now >= deadline:
                raise AssertionError("duplex harness did not settle")
            await asyncio.sleep(0.005)
        self.events.extend(collected)
        return collected

    async def run(self, command: DuplexCommand) -> list[Any]:
        self.submit(command)
        return await self.settle()

    async def start_fallback_turn(self) -> DuplexSessionFallbackRequestMessage:
        await self.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        await self.run(commands.Commit())
        assert self.fallback_requests, "commit did not emit an internal fallback request"
        return self.fallback_requests.pop(0)

    async def send_fallback_started(self, request: DuplexSessionFallbackRequestMessage) -> list[Any]:
        self.manager.dispatch(
            DuplexSessionFallbackStartedMessage(
                session_id=SESSION_ID,
                request_id=request.request.request_id,
                response_id=request.request.response_id,
                epoch=request.request.epoch,
            )
        )
        return await self.settle()

    async def send_fallback_output(
        self,
        request: DuplexSessionFallbackRequestMessage,
        output: dict[str, object],
        *,
        epoch: int | None = None,
    ) -> list[Any]:
        self.manager.dispatch(
            DuplexSessionFallbackOutputMessage(
                session_id=SESSION_ID,
                request_id=request.request.request_id,
                response_id=request.request.response_id,
                epoch=request.request.epoch if epoch is None else epoch,
                output=output,
            )
        )
        return await self.settle()

    async def send_fallback_failed(
        self,
        request: DuplexSessionFallbackRequestMessage,
        *,
        error: str = "provider failed",
        error_code: str = "provider_error",
    ) -> list[Any]:
        self.manager.dispatch(
            DuplexSessionFallbackFailedMessage(
                session_id=SESSION_ID,
                request_id=request.request.request_id,
                response_id=request.request.response_id,
                epoch=request.request.epoch,
                error=error,
                error_code=error_code,
            )
        )
        return await self.settle()


async def open_harness(
    *,
    initial_user_text: str | None = None,
    modalities: Sequence[str] = ("text",),
) -> Harness:
    port = RecordingStagePort()
    output: asyncio.Queue[Any] = asyncio.Queue()
    results: asyncio.Queue[Any] = asyncio.Queue()
    manager = DuplexSessionManager(
        plugin=Qwen3OmniDuplexPlugin(_encode_audio),
        stage_port=port,
        output_sink=output,
        result_sink=results,
        runtime_config=DuplexSessionRuntimeConfig(),
        model_config=None,
    )
    await manager.handle(
        OpenDuplexSessionMessage(
            control_id="open-qwen3",
            session_id=SESSION_ID,
            session_config=DuplexSessionConfig(
                model="qwen3-omni",
                modalities=list(modalities),
                initial_user_text=initial_user_text,
            ),
        )
    )
    result = await asyncio.wait_for(results.get(), timeout=2.0)
    assert isinstance(result, DuplexControlResultMessage) and result.ok, result
    runner = manager.runners[SESSION_ID]
    harness = Harness(manager, port, output, results, runner)
    await harness.settle()
    return harness


async def close_harness(harness: Harness) -> None:
    await harness.manager.handle(
        CloseDuplexSessionMessage(control_id="close-qwen3", session_id=SESSION_ID, reason="test_cleanup")
    )
    await harness.results.get()
    await harness.settle()
    await harness.manager.shutdown()


def _event(events: Sequence[Any], event_type: str) -> Any:
    for event in events:
        if event.type == event_type:
            return event
    raise AssertionError(f"missing {event_type}: {[event.type for event in events]}")


@pytest.mark.asyncio
async def test_qwen_commit_emits_internal_fallback_request_and_no_native_stage_submission() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        created = _event(harness.events, "response.created")

        assert request.request.session_id == SESSION_ID
        assert request.request.request_id == f"duplex-fallback-{SESSION_ID}-0-1"
        assert request.request.response_id == created.response_id
        assert request.request.history[-1]["role"] == "user"
        assert harness.port.submissions == []
        assert not any(event.type == "duplex_session_fallback_request" for event in harness.events)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_history_keeps_audio_per_turn_without_public_metadata() -> None:
    harness = await open_harness()
    try:
        first = await harness.start_fallback_turn()
        first_audio = first.request.history[-1]["_duplex_fallback_audio"]["audio"]
        assert "_duplex_fallback_audio" not in harness.session.history[-1]

        await harness.send_fallback_started(first)
        await harness.send_fallback_output(
            first,
            {"text": "first", "data_plane_request_id": first.request.request_id, "end_of_turn": True},
        )
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(0.1),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        await harness.run(commands.Commit())

        assert len(harness.fallback_requests) == 1
        second = harness.fallback_requests.pop()
        audio_messages = [
            message for message in second.request.history if message.get("_duplex_fallback_audio") is not None
        ]
        assert len(audio_messages) == 2
        assert [message["_duplex_fallback_audio"]["audio"] for message in audio_messages] == [
            first_audio,
            base64.b64encode(_pcm_f32(0.1)).decode(),
        ]
        assert (
            audio_messages[0]["_duplex_fallback_audio"]["audio"] != audio_messages[1]["_duplex_fallback_audio"]["audio"]
        )
        assert all("_duplex_fallback_audio" not in message for message in harness.session.history)
        assert all("_duplex_fallback_audio" not in str(event.to_realtime()) for event in harness.events)
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_output_projects_text_done_and_history() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        events = await harness.send_fallback_output(
            request,
            {
                "text": "hello",
                "data_plane_request_id": request.request.request_id,
                "end_of_turn": True,
            },
        )

        assert _event(events, "response.text.delta").delta == "hello"
        assert not any(event.type == "response.output_audio_transcript.delta" for event in events)
        assert _event(events, "response.done").status == "completed"
        assert harness.session.active_response_id is None
        assert harness.session.active_request_id is None
        assert harness.runner.run.fallback_request_id is None
        assert harness.runner.plugin.data_plane.is_terminal(request.request.request_id)
        assert harness.session.history[-1] == {"role": "assistant", "content": "hello"}

        late_events = await harness.send_fallback_output(
            request,
            {"text": "late", "data_plane_request_id": request.request.request_id},
        )
        assert late_events == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_audio_output_projects_its_transcript() -> None:
    harness = await open_harness(modalities=("text", "audio"))
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        audio = base64.b64encode(b"\x00\x00" * 240).decode()
        events = await harness.send_fallback_output(
            request,
            {
                "audio": audio,
                "audio_format": "pcm",
                "sample_rate_hz": 24_000,
                "audio_duration_ms": 10,
                "text": "hello",
                "data_plane_request_id": request.request.request_id,
            },
        )

        assert _event(events, "response.output_audio_transcript.delta").delta == "hello"

        terminal_events = await harness.send_fallback_output(
            request,
            {"data_plane_request_id": request.request.request_id, "end_of_turn": True},
        )
        assert _event(terminal_events, "response.output_audio_transcript.done").transcript == "hello"
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_done_marker_does_not_emit_an_empty_text_delta() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        text_events = await harness.send_fallback_output(
            request,
            {"text": "hello", "data_plane_request_id": request.request.request_id},
        )
        assert _event(text_events, "response.text.delta").delta == "hello"

        terminal_events = await harness.send_fallback_output(
            request,
            {"data_plane_request_id": request.request.request_id, "end_of_turn": True},
        )

        assert not any(event.type == "response.text.delta" for event in terminal_events)
        assert _event(terminal_events, "response.done").status == "completed"
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_explicit_false_commit_does_not_use_qwen_default_auto_response() -> None:
    harness = await open_harness()
    try:
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        events = await harness.run(commands.Commit(create_response=False))

        assert not harness.fallback_requests
        assert not any(event.type == "response.created" for event in events)
        assert harness.session.active_response_id is None
        assert harness.session.active_request_id is None
        assert harness.runner.model_state.committed_audio_payload is not None
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_qwen_server_vad_stop_commits_the_buffered_fallback_turn() -> None:
    harness = await open_harness()
    try:
        harness.runner.control._detector = _speech_then_stop_detector()
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        events = await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(value=0.0, samples=8_000),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=False,
            )
        )

        assert "input_audio_buffer.speech_stopped" in [event.type for event in events]
        assert "input_audio_buffer.committed" in [event.type for event in events]
        assert "response.created" in [event.type for event in events]
        assert len(harness.fallback_requests) == 1
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_qwen_fallback_accepts_response_create_options() -> None:
    harness = await open_harness()
    try:
        await harness.run(
            commands.AppendAudio(
                audio=_pcm_f32(),
                format="pcm_f32le",
                sample_rate_hz=16_000,
                is_speech=True,
            )
        )
        await harness.run(commands.Commit(create_response=False))
        await harness.run(
            commands.CreateResponse(
                options={
                    "instructions": "answer briefly",
                    "temperature": 0.2,
                    "max_tokens": 12,
                }
            )
        )

        assert len(harness.fallback_requests) == 1
        request = harness.fallback_requests[0].request
        assert request.response_config["instructions"] == "answer briefly"
        assert request.response_config["temperature"] == 0.2
        assert request.response_config["max_tokens"] == 12
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_initial_user_text_is_separated_from_fallback_history_snapshot() -> None:
    harness = await open_harness(initial_user_text="start with this")
    try:
        await harness.run(commands.CreateResponse())

        assert len(harness.fallback_requests) == 1
        request = harness.fallback_requests[0].request
        assert request.response_config["initial_user_text"] == "start with this"
        assert list(request.history).count({"role": "user", "content": "start with this"}) == 0
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_fallback_failure_closes_response_and_drops_late_output() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        events = await harness.send_fallback_failed(request)

        assert _event(events, "error").code == "provider_error"
        assert _event(events, "response.done").status == "failed"
        assert harness.session.active_response_id is None
        assert harness.session.active_request_id is None
        assert harness.runner.run.fallback_request_id is None
        assert harness.runner.plugin.data_plane.is_terminal(request.request.request_id)

        late_events = await harness.send_fallback_output(
            request,
            {"text": "late", "data_plane_request_id": request.request.request_id},
        )
        assert late_events == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_barge_in_cancels_fallback_and_drops_stale_output() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)
        old_epoch = harness.session.epoch

        events = await harness.run(commands.BargeIn())

        assert harness.session.epoch == old_epoch + 1
        assert harness.session.active_response_id is None
        assert [message.request_id for message in harness.fallback_cancels] == [request.request.request_id]
        assert harness.port.aborts == []
        assert _event(events, "response.done").status == "cancelled"

        late_events = await harness.send_fallback_output(
            request,
            {
                "text": "late",
                "data_plane_request_id": request.request.request_id,
                "end_of_turn": True,
            },
            epoch=old_epoch,
        )
        assert late_events == []
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_vad_turn_detected_cancels_fallback_and_marks_qwen_interrupted() -> None:
    harness = await open_harness()
    try:
        request = await harness.start_fallback_turn()
        await harness.send_fallback_started(request)

        cancelled = await harness.runner._cancel_active_response(None, reason="turn_detected")
        await harness.settle()

        assert cancelled is True
        assert harness.runner.model_state.last_turn_interrupted is True
        assert {"role": "system", "content": INTERRUPTION_NOTE} in harness.runner.plugin.fallback_policy_messages(
            harness.runner.model_state
        )
        assert [message.request_id for message in harness.fallback_cancels] == [request.request.request_id]
        assert harness.fallback_cancels[0].reason == "turn_detected"
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_committed_audio_fallback_consumes_pending_initial_user_item() -> None:
    harness = await open_harness(initial_user_text="start with this")
    try:
        request = await harness.start_fallback_turn()

        assert harness.session.unanswered_user_items() == 0
        assert [message["role"] for message in harness.session.history] == ["user", "user"]
        assert harness.session.history[0] == {"role": "user", "content": "start with this"}

        await harness.send_fallback_started(request)
        await harness.send_fallback_output(
            request,
            {"text": "answer", "data_plane_request_id": request.request.request_id, "end_of_turn": True},
        )
        events = await harness.run(commands.CreateResponse())

        assert not harness.fallback_requests
        assert _event(events, "error").code == "response_create_without_input"
        assert harness.session.history[0] == {"role": "user", "content": "start with this"}
    finally:
        await close_harness(harness)


@pytest.mark.asyncio
async def test_close_cancels_an_active_qwen_fallback_request() -> None:
    harness = await open_harness()
    request = await harness.start_fallback_turn()

    await close_harness(harness)

    assert len(harness.fallback_cancels) == 1
    cancel = harness.fallback_cancels[0]
    assert cancel.request_id == request.request.request_id
    assert cancel.response_id == request.request.response_id
    assert cancel.reason == "test_cleanup"


@pytest.mark.asyncio
async def test_expiry_cancels_an_active_qwen_fallback_request() -> None:
    harness = await open_harness()
    request = await harness.start_fallback_turn()
    harness.session.lease.last_activity = 0

    try:
        assert await harness.manager.reap_expired(now=301) == 1
        await harness.settle()

        assert len(harness.fallback_cancels) == 1
        cancel = harness.fallback_cancels[0]
        assert cancel.request_id == request.request.request_id
        assert cancel.response_id == request.request.response_id
        assert cancel.reason == "idle_ttl_expired"
    finally:
        await harness.manager.shutdown()
