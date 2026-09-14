"""Audio source that captures from microphone or virtual audio loopback device."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import sounddevice as sd

from lumir.audio.base import AudioSource
from lumir.config import CHUNK_SAMPLES, PLAYBACK_SAMPLE_RATE
from lumir.models import AudioChunk

logger = logging.getLogger(__name__)


class LiveStream(AudioSource):
    """Captures real-time audio from an input device (mic or virtual loopback)."""

    def __init__(
            self,
            device_name: str | None = None,
            sample_rate: int = PLAYBACK_SAMPLE_RATE,
            chunk_samples: int = CHUNK_SAMPLES,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_samples = chunk_samples
        self._device_name = device_name
        self._device_id = None

        # Find the device index by name
        if device_name is not None:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and device_name.lower() in dev['name'].lower():
                    self._device_id = idx
                    logger.info(f"Using input device: {dev['name']} (ID: {idx})")
                    break

            if self._device_id is None:
                logger.warning(f"Device containing '{device_name}' not found. Falling back to default input.")

        if self._device_id is None:
            default_device = sd.query_devices(kind='input')
            logger.info(f"Using default input device: {default_device['name']}")

    def iter_chunks(self) -> Iterator[AudioChunk]:
        # Open synchronous input stream with sounddevice
        stream = sd.InputStream(
            device=self._device_id,
            samplerate=self._sample_rate,
            channels=1,  # mono is sufficient for analysis
            dtype="float32",
            blocksize=self._chunk_samples,
        )
        stream.start()

        start_time = time.monotonic()
        try:
            while True:
                # Read blocksize samples from the stream (blocks until samples are ready)
                samples, overflowed = stream.read(self._chunk_samples)
                if overflowed:
                    # Input overflow occurred (usually safe to ignore in real-time mapping)
                    pass

                # Reshape samples from (chunk_samples, 1) to (chunk_samples,)
                samples_mono = samples[:, 0]

                timestamp = time.monotonic() - start_time
                yield AudioChunk(
                    samples=samples_mono,
                    sample_rate=self._sample_rate,
                    timestamp=timestamp,
                )
        finally:
            stream.stop()
            stream.close()
