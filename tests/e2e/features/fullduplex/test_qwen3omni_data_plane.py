# tests/e2e/features/fullduplex/test_qwen3omni_data_plane.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data-plane projection tests for the Qwen3-Omni duplex adapter."""

import pytest

from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
    Qwen3OmniDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode(samples, sample_rate, fmt, speed=None):
    return "audio-b64"


def test_project_maps_audio_and_text_chunks():
    plane = Qwen3OmniDataPlaneSession(_encode)
    plane.begin_request("req-1")

    def results():
        yield type(
            "R",
            (),
            {
                "request_id": "req-1",
                "chunks": [
                    type("C", (), {"modality": "audio", "data": "raw-pcm"}),
                    type("C", (), {"modality": "text", "data": "hi"}),
                ],
            },
        )()

    events = list(plane.project(next(results())))
    assert events[0]["type"] == "response.audio.delta"
    assert events[0]["audio"] == "audio-b64"
    assert events[1]["type"] == "response.audio_transcript.delta"
    assert events[1]["delta"] == "hi"
    assert plane.is_terminal("req-1") is False


def test_terminal_lifecycle():
    plane = Qwen3OmniDataPlaneSession(_encode)
    plane.begin_request("req-1")
    plane.mark_terminal("req-1")
    assert plane.is_terminal("req-1") is True
    plane.close_session("s1", active_request_id="req-1")
    assert plane.is_terminal("req-1") is True


def _chunk_result(*items):
    return type(
        "R", (), {"request_id": "req-1", "chunks": [type("C", (), {"modality": m, "data": d}) for m, d in items]}
    )()


def test_project_fails_closed_on_unknown_modality():
    plane = Qwen3OmniDataPlaneSession(_encode)
    events = list(plane.project(_chunk_result(("image", "img-b64"))))
    assert events == []


def test_project_skips_audio_when_encoder_returns_none():
    plane = Qwen3OmniDataPlaneSession(lambda samples, sample_rate, fmt, speed=None: None)
    events = list(plane.project(_chunk_result(("audio", "raw-pcm"))))
    assert events == []


def test_is_terminal_none_is_terminal():
    plane = Qwen3OmniDataPlaneSession(_encode)
    assert plane.is_terminal(None) is True


def test_close_stream_marks_terminal():
    plane = Qwen3OmniDataPlaneSession(_encode)
    plane.close_stream("req-1")
    assert plane.is_terminal("req-1") is True
