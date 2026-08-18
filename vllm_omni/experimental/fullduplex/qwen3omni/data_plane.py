# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Realtime data-plane projection for the Qwen3-Omni duplex adapter."""

from __future__ import annotations

from collections.abc import Callable, Iterable

EncodeAudio = Callable[[object, int, str, float | None], str | None]


class Qwen3OmniDataPlaneSession:
    """Project Qwen3-Omni duplex output into Realtime contract events."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._terminal: set[str] = set()

    def begin_request(self, request_id: str) -> None:
        pass

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id is None or request_id in self._terminal

    def mark_terminal(self, request_id: str) -> None:
        self._terminal.add(request_id)

    def close_stream(self, request_id: str) -> None:
        self.mark_terminal(request_id)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self.mark_terminal(active_request_id)

    def project(self, result: object, *, context: object | None = None) -> Iterable[dict[str, object]]:
        chunks = getattr(result, "chunks", [])
        events: list[dict[str, object]] = []
        for chunk in chunks:
            modality = getattr(chunk, "modality", "text")
            data = getattr(chunk, "data", "")
            if modality == "audio":
                encoded = self._encode_audio(data, 24000, "pcm16")
                if encoded is not None:
                    events.append({"type": "response.audio.delta", "audio": encoded})
            elif modality == "text":
                events.append({"type": "response.audio_transcript.delta", "delta": data})
        return events
