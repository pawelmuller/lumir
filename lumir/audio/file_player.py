"""Audio source that reads a file and yields chunks."""

from __future__ import annotations

import time
from typing import Iterator

import librosa

from lumir.audio.base import AudioSource
from lumir.config import CHUNK_SAMPLES, PLAYBACK_SAMPLE_RATE
from lumir.models import AudioChunk


class FilePlayer(AudioSource):
    """Loads an audio file upfront, then yields fixed-size chunks.

    Loads in high-quality (44.1 kHz stereo) to ensure speakers get
    excellent audio, while the analysis pipeline downsamples internally.
    """

    def __init__(
            self,
            file_path: str,
            sample_rate: int = PLAYBACK_SAMPLE_RATE,
            chunk_samples: int = CHUNK_SAMPLES,
            realtime: bool = False,
    ) -> None:
        # Load file at 44100 Hz, preserving stereo channels
        self._samples, self._sample_rate = librosa.load(file_path, sr=sample_rate, mono=False)
        self._chunk_samples = chunk_samples
        self._realtime = realtime

    def iter_chunks(self) -> Iterator[AudioChunk]:
        chunk_duration = self._chunk_samples / self._sample_rate
        start_wall = time.perf_counter()

        total_samples = self._samples.shape[-1] if self._samples.ndim > 1 else len(self._samples)

        for i, start in enumerate(range(0, total_samples - self._chunk_samples + 1, self._chunk_samples)):
            end = start + self._chunk_samples
            if self._realtime:
                target_time = start_wall + i * chunk_duration
                sleep_for = target_time - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            chunk_samples = self._samples[:, start:end] if self._samples.ndim > 1 else self._samples[start:end]
            yield AudioChunk(
                samples=chunk_samples,
                sample_rate=self._sample_rate,
                timestamp=start / self._sample_rate,
            )
