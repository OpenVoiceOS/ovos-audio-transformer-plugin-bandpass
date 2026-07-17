"""Band-pass audio transformer for the OVOS listener.

Runs on the captured speech audio before STT and attenuates energy outside a
configurable pass-band. The default 300-3400 Hz band is the telephone speech
band: it removes low-frequency rumble (HVAC, handling noise) and high-frequency
hiss that carry no phonetic information, which can improve recognition on noisy
inputs without touching the speech formants.
"""
from typing import Dict, Optional, Tuple

import numpy as np
from ovos_plugin_manager.templates.transformers import AudioTransformer
from ovos_utils.log import LOG
from scipy.signal import butter, sosfilt


class BandpassAudioTransformer(AudioTransformer):
    """Butterworth band-pass filter applied to 16-bit mono PCM speech audio."""

    def __init__(self, config: Optional[dict] = None):
        super().__init__("ovos-audio-transformer-plugin-bandpass", 10, config)
        self.low_hz: float = self.config.get("low_hz", 300.0)
        self.high_hz: float = self.config.get("high_hz", 3400.0)
        self.sample_rate: int = self.config.get("sample_rate", 16000)
        self.order: int = self.config.get("order", 4)
        if self.low_hz >= self.high_hz:
            raise ValueError(
                f"low_hz ({self.low_hz}) must be below high_hz ({self.high_hz})")
        # A second-order-sections design stays numerically stable at higher
        # orders where a single transfer-function polynomial would not.
        self._sos = self._design(self.sample_rate)

    def _design(self, sample_rate: int) -> np.ndarray:
        nyquist = 0.5 * sample_rate
        # Clamp into the open interval (0, 1) so an aggressive band on a low
        # sample rate can never ask butter() for a normalized edge at/above 1.
        low = max(self.low_hz / nyquist, 1e-4)
        high = min(self.high_hz / nyquist, 1.0 - 1e-4)
        return butter(self.order, [low, high], btype="band", output="sos")

    def transform(self, audio_data: bytes) -> Tuple[bytes, Dict]:
        """Filter one utterance's PCM bytes; returns the filtered audio + empty context."""
        if not audio_data:
            return audio_data, {}
        # A live capture can hand over a truncated final frame; int16 needs an
        # even byte count, so drop a trailing odd byte rather than raising.
        if len(audio_data) % 2:
            audio_data = audio_data[:-1]
        samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        filtered = sosfilt(self._sos, samples)
        out = np.clip(np.round(filtered), -32768, 32767).astype(np.int16).tobytes()
        LOG.debug(
            f"band-pass [{self.low_hz:.0f}-{self.high_hz:.0f} Hz, order {self.order}] "
            f"filtered {len(audio_data)} bytes of {self.sample_rate} Hz audio")
        return out, {}

    def feed_speech_utterance(self, chunk: bytes) -> bytes:
        """Filter a full speech utterance, returning only the audio."""
        return self.transform(chunk)[0]

    @property
    def default_config(self) -> Dict:
        return {"low_hz": 300.0, "high_hz": 3400.0, "sample_rate": 16000, "order": 4}
