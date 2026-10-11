# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw float PCM must be packaged as a real audio file for chat consumers."""

import io
import wave

import numpy as np
import pybase64 as base64
import pytest

from vllm_omni.protocol.realtime import audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("sample_rate_hz", [16_000, 24_000])
def test_normalized_pcm16_turn_becomes_a_decodable_wav(sample_rate_hz: int) -> None:
    samples = np.array([-32768, -16384, 0, 16384, 32767], dtype="<i2")
    payload = base64.b64encode(samples.tobytes()).decode("ascii")
    normalized, fmt, rate = audio.convert_input_audio_with_rate(
        payload,
        "pcm16",
        sample_rate_hz=sample_rate_hz,
        target_sample_rate_hz=sample_rate_hz,
    )
    assert isinstance(normalized, str)
    assert isinstance(rate, int)
    assert fmt == "pcm_f32le"
    # Normalization alone has no file header and is not a decodable WAV.
    assert audio.wav_payload_to_pcm16(base64.b64decode(normalized)) == (None, None)

    wrapped, fmt, rate = audio.pcm_f32le_payload_to_wav(normalized, rate)

    raw_wav = base64.b64decode(wrapped)
    decoded, decoded_rate = audio.wav_payload_to_pcm16(raw_wav)
    assert (fmt, rate, decoded_rate) == ("wav", sample_rate_hz, sample_rate_hz)
    assert decoded == samples.tobytes()
    with wave.open(io.BytesIO(raw_wav), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (1, 2, samples.size)


def test_float_pcm_wav_packaging_reuses_the_shared_encoder_and_clips_samples() -> None:
    samples = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype="<f4")
    payload = base64.b64encode(samples.tobytes()).decode("ascii")

    wrapped, fmt, rate = audio.pcm_f32le_payload_to_wav(f"  {payload}  ", 16_000.0)

    assert (fmt, rate) == ("wav", 16_000)
    assert wrapped == audio.encode_float32_mono_wav_base64(samples, sample_rate_hz=16_000)
    decoded, decoded_rate = audio.wav_payload_to_pcm16(base64.b64decode(wrapped))
    expected = np.array([-32768, -16384, 0, 16384, 32767], dtype="<i2")
    assert decoded == expected.tobytes()
    assert decoded_rate == 16_000


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"\0",
        b"\0" * 6,
        np.array([np.nan], dtype="<f4").tobytes(),
        np.array([np.inf], dtype="<f4").tobytes(),
        np.array([-np.inf], dtype="<f4").tobytes(),
    ],
)
def test_float_pcm_wav_packaging_rejects_invalid_samples(raw: bytes) -> None:
    with pytest.raises(ValueError, match="pcm_f32le"):
        audio.pcm_f32le_payload_to_wav(base64.b64encode(raw).decode("ascii"), 16_000)


@pytest.mark.parametrize("payload", ["not!base64", "/////", "non-ascii-ñ"])
def test_float_pcm_wav_packaging_rejects_invalid_base64(payload: str) -> None:
    with pytest.raises(ValueError, match="base64"):
        audio.pcm_f32le_payload_to_wav(payload, 16_000)


@pytest.mark.parametrize("rate", [-1, 7999, 192001, 16000.5, float("inf"), float("nan"), True])
def test_float_pcm_wav_packaging_rejects_invalid_sample_rates(rate: int | float) -> None:
    payload = base64.b64encode(np.zeros(1, dtype="<f4").tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="sample_rate_hz"):
        audio.pcm_f32le_payload_to_wav(payload, rate)
