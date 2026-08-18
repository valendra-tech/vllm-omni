# tests/e2e/features/fullduplex/test_qwen3omni_serving_adapter.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving runtime adapter tests for Qwen3-Omni duplex."""

import pytest

from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    load_serving_runtime_adapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter import (
    Qwen3OmniServingRuntimeAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATH = (
    "vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter."
    "Qwen3OmniServingRuntimeAdapter"
)


def _encode(samples, sample_rate, fmt, speed=None):
    return "audio-b64"


def test_adapter_id_and_capabilities():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    assert adapter.adapter_id == "qwen3omni"
    caps = adapter.capabilities(max_sessions=1)
    assert caps.supports_model_native_turn_policy is False
    assert caps.supports_client_commit is True
    assert caps.supports_barge_in is True
    assert caps.supports_realtime_endpoint is True


def test_load_serving_runtime_adapter_validates():
    adapter = load_serving_runtime_adapter(_PATH, _encode)
    assert adapter.adapter_id == "qwen3omni"
    adapter.create_session_state()
    assert "s1" not in adapter.session_states
    state = adapter.session_state("s1")
    assert state is not None
    adapter.remove_session_state("s1")
    assert "s1" not in adapter.session_states


def test_is_enabled_and_private_keys():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    assert adapter.is_enabled({"session_mode": "duplex"}) is True
    assert adapter.is_enabled({"session_mode": "turn_commit_only"}) is False
    assert "auto_commit_silence_ms" in adapter.private_runtime_config_keys


def test_data_plane_context():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    ctx = adapter.data_plane_context(
        epoch=0,
        turn_id=1,
        active_response_turn_id=None,
        active_response_id=None,
        auto_responds=True,
        response_format="pcm16",
        speed=None,
        modalities=("audio", "text"),
    )
    assert ctx.auto_responds is True
