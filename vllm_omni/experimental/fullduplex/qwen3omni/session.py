# vllm_omni/experimental/fullduplex/qwen3omni/session.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session state for the Qwen3-Omni duplex adapter."""

from __future__ import annotations

import base64
import wave
from io import BytesIO

import numpy as np

INTERRUPTED_MARKER = "[interrupted]"
SAMPLE_RATE = 24000


class Qwen3OmniServingSessionState:
    """PCM buffer and turn history for one Qwen3-Omni duplex session."""

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.pcm = np.zeros(0, dtype=np.float32)
        self.history: list[dict[str, str]] = []
        self.last_turn_interrupted = False
        self._partial_text: list[str] = []

    def append_pcm(self, samples: np.ndarray) -> None:
        samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self.pcm = np.concatenate([self.pcm, samples]) if self.pcm.size else samples

    def drain_pcm(self) -> np.ndarray:
        drained, self.pcm = self.pcm, np.zeros(0, dtype=np.float32)
        return drained

    def build_audio_content_part(self) -> dict[str, object]:
        samples = self.drain_pcm()
        wav_bytes = _pcm_to_wav(samples, self.sample_rate)
        encoded = base64.b64encode(wav_bytes).decode("ascii")
        return {
            "type": "input_audio",
            "input_audio": {"data": f"data:audio/wav;base64,{encoded}", "format": "wav"},
        }

    def record_user_input(self, text: str | None = None) -> None:
        if text:
            self.history.append({"role": "user", "content": text})
        self.last_turn_interrupted = False
        self._partial_text = []

    def record_partial_text(self, delta: str) -> None:
        self._partial_text.append(delta)

    def record_completed_turn(self, final_text: str) -> None:
        self.history.append({"role": "assistant", "content": final_text})
        self._partial_text = []

    def record_interrupted_turn(self) -> None:
        self.history.append({"role": "assistant", "content": INTERRUPTED_MARKER})
        self.last_turn_interrupted = True
        self._partial_text = []

    def _assistant_text(self) -> str:
        return "".join(self._partial_text)


def _pcm_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = np.clip(samples, -1.0, 1.0)
    pcm16 = (pcm16 * 32767).astype("<i2")
    with BytesIO() as buf:
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm16.tobytes())
        return buf.getvalue()
