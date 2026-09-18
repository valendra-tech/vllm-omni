# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pure helpers for the API-side chat-completion fallback."""

from __future__ import annotations

import base64
import json
import wave
from collections.abc import Mapping
from typing import Any

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

from vllm_omni.engine.duplex.audio import pcm_f32le_payload_to_wav, wav_payload_to_pcm16
from vllm_omni.engine.duplex.fallback import FALLBACK_AUDIO_PAYLOAD_KEY, DuplexFallbackRequest

_TOOL_CALL_FRAGMENT_KEY = "_duplex_tool_call_fragment"
_TOOL_CALL_INDEX_KEY = "_duplex_tool_call_index"


class ChatFallbackStreamError(ValueError):
    """The API-side chat provider returned an invalid SSE record."""


def _copy_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _copy_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_copy_json(item) for item in value]
    return value


def _replace_audio_placeholder(value: object, data_uri: str) -> tuple[object, bool]:
    if isinstance(value, Mapping):
        replaced = False
        result: dict[str, object] = {}
        for key, item in value.items():
            copied, item_replaced = _replace_audio_placeholder(item, data_uri)
            result[str(key)] = copied
            replaced = replaced or item_replaced
        audio_url = result.get("audio_url")
        if isinstance(audio_url, dict) and isinstance(audio_url.get("url"), str):
            if audio_url["url"].startswith("native-duplex:"):
                result["audio_url"] = {**audio_url, "url": data_uri}
                replaced = True
        return result, replaced
    if isinstance(value, list | tuple):
        result = []
        replaced = False
        for item in value:
            copied, item_replaced = _replace_audio_placeholder(item, data_uri)
            result.append(copied)
            replaced = replaced or item_replaced
        return result, replaced
    return value, False


def _audio_data_uri(input_payload: Mapping[str, object]) -> str:
    audio = input_payload.get("audio", input_payload.get("data"))
    sample_rate_hz = input_payload.get("sample_rate_hz")
    if not isinstance(audio, str) or not isinstance(sample_rate_hz, int | float):
        raise ValueError("fallback audio payload must contain base64 audio and sample_rate_hz")
    wav_audio, fmt, _ = pcm_f32le_payload_to_wav(audio, sample_rate_hz)
    return f"data:audio/{fmt};base64,{wav_audio}"


def _realtime_extra_body(response_config: Mapping[str, object]) -> dict[str, object]:
    extra_body = response_config.get("extra_body")
    model_extra = dict(extra_body) if isinstance(extra_body, Mapping) else {}
    tools = model_extra.pop("realtime_response_tools", model_extra.pop("realtime_tools", None))
    tool_choice = model_extra.pop(
        "realtime_response_tool_choice",
        model_extra.pop("realtime_tool_choice", None),
    )
    for key in list(model_extra):
        if key.startswith("realtime_") or key in {
            "native_duplex",
            "minicpmo45_native_duplex",
            "auto_response",
            "full_duplex",
            "auto_commit_silence_ms",
        }:
            model_extra.pop(key)
    if isinstance(tools, list | tuple):
        model_extra["tools"] = _copy_json(tools)
    if isinstance(tool_choice, str | Mapping):
        model_extra["tool_choice"] = _copy_json(tool_choice)
    return model_extra


def _fallback_history_messages(
    history: tuple[Mapping[str, object], ...],
    *,
    initial_user_text: object,
) -> list[dict[str, object]]:
    """Build fallback history from an engine snapshot and insert its seed once.

    Engine fallback snapshots omit the seeded initial user item from ``history``;
    the session configuration carries it separately so repeated requests do not
    mutate or duplicate the canonical native history.
    """
    messages = [dict(_copy_json(message)) for message in history]
    if not isinstance(initial_user_text, str) or not initial_user_text:
        return messages
    messages.insert(0, {"role": "user", "content": initial_user_text})
    return messages


def _restore_history_audio(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    """Replace each engine-only audio marker with that message's own payload."""
    restored: list[dict[str, object]] = []
    for message in messages:
        payload = message.pop(FALLBACK_AUDIO_PAYLOAD_KEY, None)
        if payload is not None:
            if not isinstance(payload, Mapping):
                raise ValueError("fallback history audio payload must be a mapping")
            message, _ = _replace_audio_placeholder(message, _audio_data_uri(payload))
        restored.append(message)
    return restored


def _fallback_output_sample_rate(response_config: Mapping[str, object]) -> int | None:
    def positive_rate(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            return None
        return int(value)

    for key in ("sample_rate_hz", "sample_rate", "output_sample_rate_hz"):
        rate = positive_rate(response_config.get(key))
        if rate is not None:
            return rate

    extra_body = response_config.get("extra_body")
    session_payload = extra_body.get("realtime_session_payload") if isinstance(extra_body, Mapping) else None
    if not isinstance(session_payload, Mapping):
        return None
    audio_config = session_payload.get("audio")
    audio_output = audio_config.get("output") if isinstance(audio_config, Mapping) else None
    if isinstance(audio_output, Mapping):
        for key in ("rate", "sample_rate_hz", "sample_rate", "output_sample_rate_hz"):
            rate = positive_rate(audio_output.get(key))
            if rate is not None:
                return rate
    return None


def build_chat_request(request: DuplexFallbackRequest, *, model: str) -> ChatCompletionRequest:
    """Build a normal chat request from an immutable engine fallback snapshot."""
    response_config = request.response_config
    history = [dict(_copy_json(message)) for message in request.history]
    input_payload = request.input_payload
    input_is_in_history = input_payload is not None and any(
        message.get(FALLBACK_AUDIO_PAYLOAD_KEY) == input_payload for message in history
    )
    messages: list[dict[str, object]] = [dict(_copy_json(message)) for message in request.policy_messages]
    instructions = response_config.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    messages.extend(
        _fallback_history_messages(
            tuple(history),
            initial_user_text=response_config.get("initial_user_text"),
        )
    )
    messages = _restore_history_audio(messages)

    if input_payload is not None and not input_is_in_history:
        data_uri = _audio_data_uri(input_payload)
        messages, replaced = _replace_audio_placeholder(messages, data_uri)
        if not replaced:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": data_uri}}],
                }
            )

    kwargs: dict[str, Any] = {
        "model": response_config.get("model") if isinstance(response_config.get("model"), str) else model,
        "messages": messages,
        "stream": True,
    }
    temperature = response_config.get("temperature")
    if isinstance(temperature, int | float):
        kwargs["temperature"] = temperature
    max_tokens = response_config.get("max_tokens")
    if isinstance(max_tokens, int):
        kwargs["max_tokens"] = max_tokens
    kwargs.update(_realtime_extra_body(response_config))

    chat_request = ChatCompletionRequest(**kwargs)
    modalities = response_config.get("modalities")
    if isinstance(modalities, list | tuple):
        object.__setattr__(chat_request, "modalities", list(modalities))
    if "audio" in (modalities if isinstance(modalities, list | tuple) else ()):
        response_format = response_config.get("response_format")
        audio_format = str(response_format or "wav").lower()
        if audio_format == "pcm16":
            audio_format = "pcm"
        audio_request: dict[str, object] = {"format": audio_format}
        voice = response_config.get("voice")
        if isinstance(voice, str) and voice:
            audio_request["voice"] = voice
        speed = response_config.get("speed")
        if isinstance(speed, int | float):
            audio_request["speed"] = speed
        object.__setattr__(chat_request, "audio", audio_request)
    object.__setattr__(chat_request, "request_id", request.request_id)
    object.__setattr__(
        chat_request,
        "chat_template_kwargs",
        {"use_tts_template": bool(response_config.get("use_tts_template", True))},
    )
    return chat_request


def parse_sse_payloads(raw_chunk: str) -> list[dict[str, object] | str]:
    """Parse JSON ``data:`` records from one chat-completion stream chunk."""
    payloads: list[dict[str, object] | str] = []
    for line in raw_chunk.splitlines():
        data = line.strip()
        if not data.startswith("data:"):
            continue
        data = data[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            payloads.append(data)
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ChatFallbackStreamError("malformed chat fallback SSE payload") from exc
        if not isinstance(parsed, dict):
            raise ChatFallbackStreamError("malformed chat fallback SSE payload")
        payloads.append(parsed)
    return payloads


def _audio_metadata(
    audio_base64: str,
    *,
    fmt: str,
    sample_rate_hz: int | None = None,
) -> tuple[int, int | None]:
    """Return one audio chunk's duration and effective sample rate."""
    try:
        raw = base64.b64decode(audio_base64, validate=False)
        normalized = fmt.lower()
        if normalized == "wav":
            pcm16, wav_rate = wav_payload_to_pcm16(raw)
            if pcm16 is None or not wav_rate:
                return 0, sample_rate_hz
            return round(len(pcm16) * 1000 / (2 * wav_rate)), int(wav_rate)
        if normalized in {"pcm", "pcm16", "pcm_s16le", "s16le"} and sample_rate_hz is not None:
            return round(len(raw) * 1000 / (2 * sample_rate_hz)), sample_rate_hz
    except (ValueError, wave.Error):
        return 0, sample_rate_hz
    return 0, sample_rate_hz


def _audio_format(response_format: object) -> str:
    normalized = str(response_format or "wav").lower()
    return "pcm" if normalized == "pcm16" else normalized


def _choice_source(choice: Mapping[str, object]) -> Mapping[str, object]:
    delta = choice.get("delta")
    message = choice.get("message")
    source = delta if isinstance(delta, Mapping) else message if isinstance(message, Mapping) else {}
    return source if isinstance(source, Mapping) else {}


def _choice_content(choice: Mapping[str, object]) -> tuple[object, str]:
    source = _choice_source(choice)
    audio = source.get("audio")
    if isinstance(audio, Mapping):
        audio = audio.get("data", audio.get("content"))
    if isinstance(audio, str):
        return audio, "audio"
    return source.get("content"), "text"


def project_chat_payload(
    payload: dict[str, object] | str,
    *,
    request_id: str,
    response_format: str = "wav",
    sample_rate_hz: int | None = None,
) -> list[dict[str, object]]:
    """Project a chat chunk into model-neutral output dictionaries.

    This function deliberately has no session or playback side effects; the
    engine applies the returned dictionaries through ``ModelChannel``.
    """
    if payload == "[DONE]":
        return [{"data_plane_request_id": request_id, "end_of_turn": True}]
    if isinstance(payload, str):
        return []
    error = payload.get("error")
    if isinstance(error, Mapping):
        return [
            {
                "error": str(error.get("message") or error),
                "error_code": str(error.get("type") or error.get("code") or "chat_error"),
            }
        ]
    modality = payload.get("modality")
    if modality not in {None, "text", "audio"}:
        return [
            {"error": f"Unsupported chat response modality: {modality}", "error_code": "unsupported_response_modality"}
        ]
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []

    results: list[dict[str, object]] = []
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        source = _choice_source(choice)
        tool_calls = source.get("tool_calls")
        if isinstance(tool_calls, list):
            for fallback_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                function = tool_call.get("function")
                function = function if isinstance(function, Mapping) else {}
                result: dict[str, object] = {
                    "function_call": True,
                    "data_plane_request_id": request_id,
                }
                call_id = tool_call.get("id")
                name = function.get("name")
                arguments = function.get("arguments")
                tool_call_index = tool_call.get("index")
                if not isinstance(tool_call_index, int) or isinstance(tool_call_index, bool):
                    tool_call_index = call_id if isinstance(call_id, str) and call_id else fallback_index
                result[_TOOL_CALL_FRAGMENT_KEY] = True
                result[_TOOL_CALL_INDEX_KEY] = tool_call_index
                if isinstance(call_id, str):
                    result["call_id"] = call_id
                if isinstance(name, str):
                    result["name"] = name
                if isinstance(arguments, str):
                    result["arguments"] = arguments
                results.append(result)
        content, inferred_modality = _choice_content(choice)
        output_modality = "audio" if modality == "audio" or inferred_modality == "audio" else "text"
        if not isinstance(content, str) or not content:
            continue
        if output_modality == "audio":
            output_fmt = _audio_format(response_format)
            hinted_rate = sample_rate_hz
            for source in (choice, payload, payload.get("metrics")):
                if not isinstance(source, Mapping):
                    continue
                for key in ("sample_rate_hz", "audio_sample_rate", "sample_rate", "sr"):
                    value = source.get(key)
                    if isinstance(value, int | float) and int(value) > 0:
                        hinted_rate = int(value)
                        break
                if hinted_rate is not None:
                    break
            duration_ms, output_rate = _audio_metadata(content, fmt=output_fmt, sample_rate_hz=hinted_rate)
            result: dict[str, object] = {
                "audio": content,
                "audio_format": output_fmt,
                "data_plane_request_id": request_id,
            }
            if output_rate is not None:
                result["sample_rate_hz"] = output_rate
            if duration_ms > 0:
                result["audio_duration_ms"] = duration_ms
            results.append(result)
        else:
            results.append({"text": content, "data_plane_request_id": request_id})
    return results


__all__ = [
    "build_chat_request",
    "ChatFallbackStreamError",
    "parse_sse_payloads",
    "project_chat_payload",
]
