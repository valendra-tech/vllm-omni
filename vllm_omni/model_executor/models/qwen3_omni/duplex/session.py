# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni model-owned duplex session state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from vllm_omni.engine.duplex.plugin import DuplexModelSessionState
from vllm_omni.model_executor.models.qwen3_omni.duplex.input import Qwen3OmniPcmAppendBuffer


@dataclass(slots=True)
class Qwen3OmniDuplexSessionState(DuplexModelSessionState):
    """Mutable Qwen policy and committed-input state for one engine session."""

    audio_buffer: Qwen3OmniPcmAppendBuffer = field(default_factory=Qwen3OmniPcmAppendBuffer)
    input_since_commit: bool = False
    speech_since_commit: bool = False
    context_locked: bool = False
    committed_audio_payload: dict[str, object] | None = None
    committed_audio_operation_id: str | None = None
    committed_audio_reserved_bytes: int = 0
    deferred_response_create: bool = False
    deferred_precreate_response: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    last_turn_interrupted: bool = False

    def retain_committed_audio(
        self,
        payload: dict[str, object],
        *,
        operation_id: str | None,
        reserved_bytes: int = 0,
    ) -> None:
        self.committed_audio_payload = payload
        self.committed_audio_operation_id = operation_id
        self.committed_audio_reserved_bytes += max(0, int(reserved_bytes))

    def clear_committed_audio(self) -> int:
        reserved_bytes = self.committed_audio_reserved_bytes
        self.committed_audio_payload = None
        self.committed_audio_operation_id = None
        self.committed_audio_reserved_bytes = 0
        self.deferred_response_create = False
        self.deferred_precreate_response = False
        return reserved_bytes

    def clear_continuation(self) -> None:
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None


Qwen3OmniServingSessionState = Qwen3OmniDuplexSessionState

__all__ = ["Qwen3OmniDuplexSessionState", "Qwen3OmniServingSessionState"]
