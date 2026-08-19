# tests/e2e/features/fullduplex/test_qwen3omni_session.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session state tests for the Qwen3-Omni duplex adapter."""

import numpy as np
import pytest

from vllm_omni.experimental.fullduplex.qwen3omni.session import (
    Qwen3OmniServingSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

INTERRUPTED = {"role": "assistant", "content": "[interrupted]"}


def test_buffer_accumulates_pcm():
    state = Qwen3OmniServingSessionState()
    state.append_pcm(np.zeros(100, dtype=np.float32))
    state.append_pcm(np.ones(50, dtype=np.float32))
    assert state.pcm.size == 150
    assert state.pcm.sum() == 50.0


def test_drain_clears_buffer():
    state = Qwen3OmniServingSessionState()
    state.append_pcm(np.zeros(64, dtype=np.float32))
    drained = state.drain_pcm()
    assert drained.size == 64
    assert state.pcm.size == 0


def test_commit_pending_audio_builds_data_uri():
    state = Qwen3OmniServingSessionState()
    state.append_pcm(np.zeros(48000, dtype=np.float32))
    part = state.build_audio_content_part()
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "wav"
    assert part["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_history_records_completed_turn():
    state = Qwen3OmniServingSessionState()
    state.record_user_input("hello")
    state.record_completed_turn("hi there")
    assert state.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_records_interrupted_turn():
    state = Qwen3OmniServingSessionState()
    state.record_user_input("hello")
    state.record_interrupted_turn()
    assert state.history[-1] == INTERRUPTED
    assert len(state.history) == 2


def test_history_never_stores_partial_output():
    state = Qwen3OmniServingSessionState()
    state.record_user_input("hello")
    state.record_partial_text("hi there, I was")
    state.record_interrupted_turn()
    assert all("I was" not in str(msg.get("content", "")) for msg in state.history)


def test_was_interrupted_flag_cleared_on_new_turn():
    state = Qwen3OmniServingSessionState()
    state.record_user_input("hello")
    state.record_interrupted_turn()
    assert state.last_turn_interrupted is True
    state.record_user_input("new input")
    assert state.last_turn_interrupted is False
