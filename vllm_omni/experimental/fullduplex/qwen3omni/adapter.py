# vllm_omni/experimental/fullduplex/qwen3omni/adapter.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn-based DuplexAdapter for Qwen3-Omni.

Delegates response generation to an injected chat-service callable
(``OmniOpenAIServingChat.create_chat_completion`` in production, a stub in
tests) and applies the turn policy from :mod:`policy`.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from vllm_omni.experimental.fullduplex.core.adapter import (
    DuplexAdapter,
    DuplexCapability,
    OutputChunk,
)
from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.qwen3omni.policy import (
    INTERRUPTION_NOTE,
    SYSTEM_PROMPT,
)
from vllm_omni.experimental.fullduplex.qwen3omni.session import (
    Qwen3OmniServingSessionState,
    USER_AUDIO_MARKER,
)

ChatServiceCall = Callable[[dict[str, Any], Any], Any]


class _ChatRequestDict(dict[str, Any]):
    """Chat request dict that also exposes attribute access.

    The injected chat service reads ``request.modalities`` while tests read
    ``request["modalities"]``; a plain dict supports only the latter.
    """

    @property
    def modalities(self) -> list[str] | None:
        return self.get("modalities")


class Qwen3OmniDuplexAdapter(DuplexAdapter):
    """Turn-based duplex adapter delegating to an OpenAI-compatible chat service."""

    def __init__(self, chat_service: ChatServiceCall, *, sample_rate: int = 24000) -> None:
        self._chat_service = chat_service
        self._sample_rate = sample_rate
        self._states: dict[str, Qwen3OmniServingSessionState] = {}
        self._responding: set[str] = set()

    def capabilities(self) -> DuplexCapability:
        return DuplexCapability(
            input_modalities=frozenset({"audio", "text"}),
            output_modalities=frozenset({"audio", "text"}),
            proactive=True,
        )

    def _get_state(self, session: DuplexSession) -> Qwen3OmniServingSessionState:
        injected = self.__dict__.get("_state")
        if isinstance(injected, Qwen3OmniServingSessionState):
            return injected
        state = self._states.get(session.session_id)
        if state is None:
            state = Qwen3OmniServingSessionState(sample_rate=self._sample_rate)
            self._states[session.session_id] = state
        return state

    async def on_input(self, session: DuplexSession, modality: str, data: Any) -> None:
        state = self._get_state(session)
        if modality == "audio":
            state.append_pcm(data)
        elif modality == "text":
            state.record_user_input(str(data))

    def should_respond(self, session: DuplexSession) -> bool:
        state = self._get_state(session)
        return state.pcm.size > 0

    async def on_barge_in(self, session: DuplexSession) -> None:
        if session.session_id in self._responding:
            self._get_state(session).record_interrupted_turn()

    async def respond(self, session: DuplexSession) -> AsyncIterator[OutputChunk]:
        state = self._get_state(session)
        self._responding.add(session.session_id)
        try:
            had_audio = state.pcm.size > 0
            request = _ChatRequestDict(self._build_request(state))
            result = self._chat_service.create_chat_completion(request, raw_request=None)
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "__aiter__"):
                async for raw_chunk in result:
                    for payload in _parse_sse_payloads(raw_chunk):
                        if payload == "[DONE]":
                            continue
                        if isinstance(payload, dict):
                            chunk = _extract_content_chunk(payload)
                            if chunk is None:
                                continue
                            modality, content = chunk
                            if modality == "audio":
                                yield OutputChunk("audio", content)
                            else:
                                state.record_partial_text(content)
                                yield OutputChunk("text", content)
            else:
                payload = result.model_dump(mode="json", exclude_unset=True) if hasattr(result, "model_dump") else {}
                chunk = _extract_content_chunk(payload)
                if chunk is not None:
                    modality, content = chunk
                    if modality == "audio":
                        yield OutputChunk("audio", content)
                    else:
                        state.record_partial_text(content)
                        yield OutputChunk("text", content)
            final_text = state._assistant_text()
            if had_audio:
                state.record_user_input(USER_AUDIO_MARKER)
            state.record_completed_turn(final_text)
        finally:
            self._responding.discard(session.session_id)

    def _build_request(self, state: Qwen3OmniServingSessionState) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if state.last_turn_interrupted:
            messages.append({"role": "system", "content": INTERRUPTION_NOTE})
            state.last_turn_interrupted = False
        messages.extend(state.history)
        user_part: list[dict[str, Any]] = []
        if state.pcm.size:
            user_part.append(state.build_audio_content_part())
        if not user_part:
            user_part.append({"type": "text", "text": ""})
        messages.append({"role": "user", "content": user_part})
        return {
            "model": "qwen3-omni",
            "messages": messages,
            "stream": True,
            "modalities": ["audio", "text"],
        }


def _parse_sse_payloads(raw_chunk: str) -> list[dict[str, Any] | str]:
    payloads: list[dict[str, Any] | str] = []
    for line in raw_chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            payloads.append(data)
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _extract_content_chunk(payload: dict[str, Any]) -> tuple[str, str] | None:
    modality = payload.get("modality", "text")
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            return modality, content
    return None
