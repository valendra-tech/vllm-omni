# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure request construction and output projection for duplex chat fallback."""

from __future__ import annotations

import base64
import struct

from vllm_omni.engine.duplex.fallback import FALLBACK_AUDIO_PAYLOAD_KEY, DuplexFallbackRequest
from vllm_omni.entrypoints.duplex.chat_fallback import (
    _fallback_output_sample_rate,
    build_chat_request,
    project_chat_payload,
)


def test_build_chat_request_replaces_engine_audio_placeholder() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
            },
        ),
        response_config={"model": "qwen", "modalities": ["audio"], "response_format": "wav"},
        input_payload={
            "audio": base64.b64encode(struct.pack("<2f", 0.1, 0.2)).decode(),
            "format": "pcm_f32le",
            "sample_rate_hz": 16_000,
        },
        policy_messages=({"role": "system", "content": "policy"},),
    )
    chat_request = build_chat_request(request, model="qwen")

    assert chat_request.messages[0]["content"] == "policy"
    assert chat_request.messages[-1]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_build_chat_request_does_not_duplicate_current_history_audio() -> None:
    payload = {
        "audio": base64.b64encode(struct.pack("<2f", 0.1, 0.2)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(
            {
                "role": "user",
                "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}],
                FALLBACK_AUDIO_PAYLOAD_KEY: payload,
            },
        ),
        response_config={"model": "qwen", "modalities": ["audio"], "response_format": "wav"},
        input_payload=payload,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")
    audio_messages = [
        message
        for message in chat_request.messages
        if isinstance(message.get("content"), list)
        and message["content"]
        and message["content"][0].get("type") == "audio_url"
    ]

    assert len(audio_messages) == 1
    assert audio_messages[0]["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")


def test_build_chat_request_includes_initial_user_text_once_in_user_order() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(
            {"role": "assistant", "content": "previous answer"},
            {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "native-duplex:input"}}]},
        ),
        response_config={"model": "qwen", "initial_user_text": "start with this"},
        input_payload=None,
        policy_messages=({"role": "system", "content": "policy"},),
    )

    first = build_chat_request(request, model="qwen")
    second = build_chat_request(request, model="qwen")
    first_user_messages = [message for message in first.messages if message["role"] == "user"]
    second_user_messages = [message for message in second.messages if message["role"] == "user"]

    assert [message["content"] for message in first_user_messages] == [
        "start with this",
        [{"type": "audio_url", "audio_url": {"url": "native-duplex:input"}}],
    ]
    assert [message["role"] for message in first.messages] == ["system", "user", "assistant", "user"]
    assert first_user_messages == second_user_messages


def test_build_chat_request_preserves_audio_voice_and_speed() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={
            "model": "qwen",
            "modalities": ["audio"],
            "response_format": "pcm16",
            "voice": "Vivian",
            "speed": 1.25,
        },
        input_payload=None,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")

    assert chat_request.audio == {"format": "pcm", "voice": "Vivian", "speed": 1.25}


def test_build_chat_request_normalizes_frozen_tools_and_tool_choice() -> None:
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-1",
        response_id="resp",
        epoch=0,
        history=(),
        response_config={
            "model": "qwen",
            "extra_body": {
                "realtime_response_tools": [
                    {"type": "function", "function": {"name": "lookup"}},
                ],
                "realtime_response_tool_choice": {
                    "type": "function",
                    "function": {"name": "lookup"},
                },
            },
        },
        input_payload=None,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")

    assert chat_request.tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert chat_request.tool_choice == {"type": "function", "function": {"name": "lookup"}}


def test_build_chat_request_keeps_each_history_audio_payload_distinct() -> None:
    first_payload = {
        "audio": base64.b64encode(struct.pack("<2f", 0.1, 0.1)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    second_payload = {
        "audio": base64.b64encode(struct.pack("<2f", 0.2, 0.2)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
    }
    placeholder = {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}
    request = DuplexFallbackRequest(
        session_id="sid",
        request_id="fallback-sid-0-2",
        response_id="resp-2",
        epoch=1,
        history=(
            {"role": "user", "content": [placeholder], "_duplex_fallback_audio": first_payload},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": [placeholder], "_duplex_fallback_audio": second_payload},
            {"role": "user", "content": "text follow-up"},
        ),
        response_config={"model": "qwen"},
        input_payload=None,
        policy_messages=(),
    )

    chat_request = build_chat_request(request, model="qwen")
    audio_urls = [
        message["content"][0]["audio_url"]["url"]
        for message in chat_request.messages
        if isinstance(message.get("content"), list)
        and message["content"]
        and message["content"][0].get("type") == "audio_url"
    ]

    assert len(audio_urls) == 2
    assert audio_urls[0].startswith("data:audio/wav;base64,")
    assert audio_urls[0] != audio_urls[1]
    assert all("native-duplex:" not in url for url in audio_urls)
    assert all("_duplex_fallback_audio" not in message for message in chat_request.messages)
    assert chat_request.messages[-1]["content"] == "text follow-up"


def test_fallback_output_sample_rate_reads_realtime_session_payload() -> None:
    assert (
        _fallback_output_sample_rate(
            {
                "extra_body": {
                    "realtime_session_payload": {
                        "audio": {"output": {"format": "pcm16", "sample_rate": 24_000}},
                    },
                },
            }
        )
        == 24_000
    )


def test_project_chat_payload_returns_model_neutral_text_output() -> None:
    outputs = project_chat_payload(
        {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
        request_id="fallback-sid-0-1",
    )

    assert outputs == [{"text": "hello", "data_plane_request_id": "fallback-sid-0-1"}]


def test_project_chat_payload_preserves_streamed_tool_call_fragments() -> None:
    first_outputs = project_chat_payload(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"city":"'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        request_id="fallback-sid-0-1",
    )
    continuation_outputs = project_chat_payload(
        {
            "choices": [
                {
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'Paris"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        request_id="fallback-sid-0-1",
    )

    assert first_outputs == [
        {
            "function_call": True,
            "_duplex_tool_call_fragment": True,
            "_duplex_tool_call_index": 0,
            "call_id": "call-1",
            "name": "lookup",
            "arguments": '{"city":"',
            "data_plane_request_id": "fallback-sid-0-1",
        }
    ]
    assert continuation_outputs == [
        {
            "function_call": True,
            "_duplex_tool_call_fragment": True,
            "_duplex_tool_call_index": 0,
            "arguments": 'Paris"}',
            "data_plane_request_id": "fallback-sid-0-1",
        }
    ]


def test_project_chat_payload_returns_provider_error() -> None:
    outputs = project_chat_payload(
        {"error": {"message": "rejected", "type": "BadRequestError"}},
        request_id="fallback-sid-0-1",
    )

    assert outputs == [{"error": "rejected", "error_code": "BadRequestError"}]


def test_project_chat_payload_marks_stream_completion_as_end_of_turn() -> None:
    outputs = project_chat_payload("[DONE]", request_id="fallback-sid-0-1")

    assert outputs == [{"data_plane_request_id": "fallback-sid-0-1", "end_of_turn": True}]
