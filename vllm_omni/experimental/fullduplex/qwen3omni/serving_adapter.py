# vllm_omni/experimental/fullduplex/qwen3omni/serving_adapter.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving runtime adapter for the Qwen3-Omni duplex path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vllm_omni.experimental.fullduplex.openai.protocol import DuplexCapabilities
from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
    Qwen3OmniDataPlaneSession,
)
from vllm_omni.experimental.fullduplex.qwen3omni.session import (
    Qwen3OmniServingSessionState,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]

PRIVATE_RUNTIME_CONFIG_KEYS = frozenset({"auto_commit_silence_ms"})


class Qwen3OmniDataPlaneContext:
    def __init__(
        self,
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> None:
        self.epoch = epoch
        self.turn_id = turn_id
        self.active_response_turn_id = active_response_turn_id
        self.active_response_id = active_response_id
        self.auto_responds = auto_responds
        self.response_format = response_format
        self.speed = speed
        self.modalities = modalities


class Qwen3OmniServingRuntimeAdapter:
    """Qwen3-Omni serving state, input packing, and output projection."""

    adapter_id = "qwen3omni"
    clean_response_done_prefix = ""
    interrupted_tts_prefix = ""
    private_runtime_config_keys = PRIVATE_RUNTIME_CONFIG_KEYS

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self.session_states: dict[str, Qwen3OmniServingSessionState] = {}
        self.data_plane = Qwen3OmniDataPlaneSession(encode_audio)

    def create_session_state(self) -> Qwen3OmniServingSessionState:
        return Qwen3OmniServingSessionState()

    def session_state(self, session_id: str) -> Qwen3OmniServingSessionState:
        state = self.session_states.get(session_id)
        if state is None:
            state = self.create_session_state()
            self.session_states[session_id] = state
        return state

    def remove_session_state(self, session_id: str) -> None:
        self.session_states.pop(session_id, None)

    @staticmethod
    def is_enabled(config: object) -> bool:
        mode = None
        if isinstance(config, Mapping):
            mode = config.get("session_mode")
        return mode == "duplex"

    @staticmethod
    def capabilities(*, max_sessions: int) -> DuplexCapabilities:
        return DuplexCapabilities(
            supports_model_native_turn_policy=False,
            supports_barge_in=True,
            supports_client_commit=True,
            supports_input_append=True,
            supports_reencode_context=False,
            supports_turn_commit_only=True,
            supports_realtime_endpoint=True,
            supports_multi_session=False,
            supports_multi_session_same_replica=False,
            supports_session_lease=False,
            supports_session_resume=False,
            session_admission_mode="serving_managed",
            implementation_level="serving_session_adapter",
            adapter_patterns=["turn_commit_only"],
            input_modes=["turn_commit_only"],
            signal_sources=["client_event", "server_policy"],
            chunk_period_ms=None,
            target_barge_in_latency_ms=1000,
        )

    @staticmethod
    def validate_client_extra_body(extra_body: object) -> None:
        if not isinstance(extra_body, Mapping):
            return
        for key in PRIVATE_RUNTIME_CONFIG_KEYS:
            if key in extra_body:
                raise ValueError(f"client cannot set private runtime key: {key}")

    @staticmethod
    async def prepare_runtime_config(config: object, *, model_config: Any) -> dict[str, object]:
        runtime: dict[str, object] = {}
        if isinstance(config, Mapping):
            for key in ("auto_commit_silence_ms",):
                if key in config:
                    runtime[key] = config[key]
        return runtime

    @staticmethod
    def runtime_config_for_update(
        config: object,
        current: Mapping[str, object],
    ) -> dict[str, object]:
        merged = dict(current)
        if isinstance(config, Mapping):
            for key in ("auto_commit_silence_ms",):
                if key in config:
                    merged[key] = config[key]
        return merged

    @staticmethod
    def data_plane_context(
        *,
        epoch: int,
        turn_id: int,
        active_response_turn_id: int | None,
        active_response_id: str | None,
        auto_responds: bool,
        response_format: str,
        speed: float | None,
        modalities: tuple[str, ...],
    ) -> Qwen3OmniDataPlaneContext:
        return Qwen3OmniDataPlaneContext(
            epoch=epoch,
            turn_id=turn_id,
            active_response_turn_id=active_response_turn_id,
            active_response_id=active_response_id,
            auto_responds=auto_responds,
            response_format=response_format,
            speed=speed,
            modalities=modalities,
        )
