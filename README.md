# ovos-audio-transformer-plugin-bandpass

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

A band-pass filter **audio transformer** for OpenVoiceOS. It runs on the captured
speech audio before STT and attenuates energy outside a configurable pass-band.

The default 300–3400 Hz band is the telephone speech band: it removes
low-frequency rumble (HVAC, handling noise) and high-frequency hiss that carry no
phonetic information, which can improve recognition on noisy inputs without
touching the speech formants.

## Install

```bash
pip install ovos-audio-transformer-plugin-bandpass
```

## Configure

Enable it in `mycroft.conf` under `audio_transformers`. Any component that runs
the audio-transformer pipeline, such as the listener or the STT server, applies it:

```json
{
  "audio_transformers": {
    "ovos-audio-transformer-plugin-bandpass": {
      "low_hz": 300,
      "high_hz": 3400,
      "order": 4,
      "sample_rate": 16000
    }
  }
}
```

| Option | Default | Meaning |
|--------|---------|---------|
| `low_hz` | `300` | Lower edge of the pass-band, in Hz. |
| `high_hz` | `3400` | Upper edge of the pass-band, in Hz. |
| `order` | `4` | Butterworth filter order. Higher is a steeper roll-off. |
| `sample_rate` | `16000` | Sample rate of the incoming PCM audio, in Hz. |

Edges are clamped into the open interval below Nyquist, so an aggressive band on a
low sample rate degrades gracefully instead of failing.

## Related projects

- [OpenVoiceOS/ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): loads audio transformer plugins through the `opm.transformer.audio` entry point.
- [OpenVoiceOS/ovos-audio-transformer-plugin-speechbrain-langdetect](https://github.com/OpenVoiceOS/ovos-audio-transformer-plugin-speechbrain-langdetect): another audio transformer plugin, for language detection.

## License

Apache-2.0. See [LICENSE](LICENSE).
