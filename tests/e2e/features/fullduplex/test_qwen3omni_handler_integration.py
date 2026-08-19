# tests/e2e/features/fullduplex/test_qwen3omni_handler_integration.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Omni session state satisfies the handler's ServingRuntimeSessionState."""

import pytest

from vllm_omni.experimental.fullduplex.qwen3omni.policy import (
    INTERRUPTION_NOTE,
    SYSTEM_PROMPT,
)
from vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter import (
    Qwen3OmniServingRuntimeAdapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.session import (
    Qwen3OmniServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_session_state_satisfies_protocol_fields():
    state = Qwen3OmniServingSessionState()
    assert isinstance(state.audio_buffer, object)
    assert state.audio_buffer.has_pending() is False
    assert state.audio_buffer.has_reserved() is False
    state.audio_buffer.clear()
    assert state.input_since_commit is False
    assert state.speech_since_commit is False
    assert state.committed_audio_payload is None
    assert state.committed_audio_operation_id is None
    assert state.committed_audio_reserved_bytes == 0
    assert state.deferred_response_create is False
    assert state.deferred_precreate_response is False
    assert state.data_plane_task is None
    assert state.data_plane_restart_requested is False
    assert state.continuation_owner_id is None
    assert state.continuation_units == 0
    assert state.pending_silence_task is None
    assert state.pending_silence_owner_id is None
    assert state.silence_continuation_scheduler is None
    assert state.clear_committed_audio() == 0
    state.retain_committed_audio({"a": 1}, operation_id="op", reserved_bytes=4)
    assert state.committed_audio_payload == {"a": 1}
    assert state.committed_audio_operation_id == "op"
    assert state.committed_audio_reserved_bytes == 4
    assert state.clear_committed_audio() == 4
    state.clear_continuation()


def test_session_state_satisfies_protocol_structural():
    state = Qwen3OmniServingSessionState()
    for attr in (
        "audio_buffer",
        "input_since_commit",
        "speech_since_commit",
        "committed_audio_payload",
        "committed_audio_operation_id",
        "committed_audio_reserved_bytes",
        "deferred_response_create",
        "deferred_precreate_response",
        "data_plane_task",
        "data_plane_restart_requested",
        "continuation_owner_id",
        "continuation_units",
        "pending_silence_task",
        "pending_silence_owner_id",
        "silence_continuation_scheduler",
    ):
        assert hasattr(state, attr), f"missing field: {attr}"
    for meth in ("retain_committed_audio", "clear_committed_audio", "clear_continuation"):
        assert callable(getattr(state, meth)), f"missing method: {meth}"


def _encode(samples, sample_rate, fmt, speed=None):
    return "audio-b64"


def test_turn_policy_messages_injects_prompts_once():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    state = adapter.session_state("s1")
    assert adapter.turn_policy_messages(state) == [{"role": "system", "content": SYSTEM_PROMPT}]
    state.last_turn_interrupted = True
    messages = adapter.turn_policy_messages(state)
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": INTERRUPTION_NOTE},
    ]
    assert state.last_turn_interrupted is False, "flag consumed after note injection"
