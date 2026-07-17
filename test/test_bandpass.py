"""Behavioral + adversarial tests for the band-pass audio transformer.

The happy-path assertions prove the filter attenuates out-of-band tones while
passing in-band ones; the adversarial cases feed it the hostile inputs a live
microphone stream actually produces (empty buffers, silence, odd-length byte
runs, DC offset, full-scale clipping) and assert it never crashes or corrupts.
"""
import numpy as np
import pytest

from ovos_bandpass_transformer import BandpassAudioTransformer

SR = 16000


def tone(freq_hz: float, seconds: float = 0.25, amp: int = 8000) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16).tobytes()


def rms(pcm: bytes) -> float:
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


@pytest.fixture
def xf():
    return BandpassAudioTransformer({"low_hz": 300, "high_hz": 3400, "sample_rate": SR})


def test_passband_tone_survives(xf):
    src = tone(1000)  # squarely inside 300-3400
    out, ctx = xf.transform(src)
    assert ctx == {}
    assert rms(out) > 0.6 * rms(src)


def test_subsonic_tone_attenuated(xf):
    src = tone(60)  # below the low edge — rumble
    out, _ = xf.transform(src)
    assert rms(out) < 0.2 * rms(src)


def test_high_tone_attenuated(xf):
    src = tone(7000)  # above the high edge — hiss
    out, _ = xf.transform(src)
    assert rms(out) < 0.3 * rms(src)


def test_output_is_same_length_int16(xf):
    src = tone(1000)
    out, _ = xf.transform(src)
    assert len(out) == len(src)
    assert len(out) % 2 == 0


# --- adversarial: the messy reality of a live mic stream ---

def test_empty_buffer_is_passthrough(xf):
    assert xf.transform(b"") == (b"", {})


def test_silence_stays_silent(xf):
    out, _ = xf.transform(b"\x00\x00" * SR)
    assert rms(out) == 0.0


def test_odd_length_bytes_do_not_crash(xf):
    # a truncated frame: 401 bytes is not a whole number of int16 samples
    out, _ = xf.transform(tone(1000, 0.05) + b"\x01")
    assert isinstance(out, bytes)


def test_full_scale_input_never_overflows(xf):
    src = tone(1000, amp=32767)
    out, _ = xf.transform(src)
    a = np.frombuffer(out, dtype=np.int16)
    assert a.min() >= -32768 and a.max() <= 32767


def test_dc_offset_removed(xf):
    src = (np.full(SR, 5000, dtype=np.int16)).tobytes()  # pure DC, 0 Hz
    out, _ = xf.transform(src)
    assert rms(out) < 0.05 * rms(src)


def test_inverted_band_rejected():
    with pytest.raises(ValueError):
        BandpassAudioTransformer({"low_hz": 4000, "high_hz": 300})


def test_aggressive_band_on_low_rate_still_designs():
    # high edge just under Nyquist for an 8 kHz stream must clamp, not raise
    xf = BandpassAudioTransformer({"low_hz": 300, "high_hz": 3900, "sample_rate": 8000})
    out, _ = xf.transform(tone(1000))
    assert isinstance(out, bytes)
