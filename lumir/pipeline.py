"""Clock-paced pipeline orchestration."""

from __future__ import annotations

import logging
import os
import time

from lumir.analysis.mir import Analyzer
from lumir.audio.base import AudioSource
from lumir.audio.live_stream import LiveStream
from lumir.config import AUDIO_INPUT_DEVICE_NAME
from lumir.mapping.rules import Mapper
from lumir.output.base import Output
from lumir.telemetry import TelemetryDumper

logger = logging.getLogger(__name__)


def run(source: AudioSource, output: Output, *, playback: bool = False, track_id: str = "") -> None:
    """Run the pipeline: read chunks at real-time pace, analyse, map, send.

    When *playback* is True, chunks are played through the default audio
    output — this also replaces time.sleep() as the pacing mechanism.

    Blocks until the source is exhausted or KeyboardInterrupt.
    """
    analyzer: Analyzer | None = None
    mapper = Mapper()
    stream = None
    telemetry = TelemetryDumper(track_id=track_id)

    start_wall = time.monotonic()

    try:
        for chunk in source.iter_chunks():
            if analyzer is None:
                analyzer = Analyzer()
                if playback:
                    import sounddevice as sd
                    channels = chunk.samples.shape[0] if chunk.samples.ndim > 1 else 1

                    stream = sd.OutputStream(
                        samplerate=chunk.sample_rate,
                        channels=channels,
                        dtype="float32",
                    )
                    stream.start()

            start_time = time.perf_counter()
            features = analyzer.process(chunk)
            if features.is_section_boundary:
                logger.info(f"New track section: {features.current_section.name}")
            command = mapper.translate(features)
            output.send(command)
            end_time = time.perf_counter()

            processing_time_ms = (end_time - start_time) * 1000
            telemetry.record(features, command, processing_time_ms)

            if stream is not None:
                # write() blocks until the device consumes the samples → pacing
                samples_to_write = chunk.samples.T if chunk.samples.ndim > 1 else chunk.samples.reshape(-1, 1)
                stream.write(samples_to_write)
            else:
                deadline = start_wall + chunk.timestamp
                sleep_for = deadline - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        telemetry.close()
        if stream is not None:
            stream.stop()
            stream.close()
        output.close()


def play(
        file_path: str | None = None,
        *,
        live: bool = False,
        playback: bool = True,
        output_backend: str = "console",
        file_realtime: bool = False,
) -> None:
    """Run the pipeline on a specified audio file or in LIVE mode (Apple Music / Spotify)."""
    if live:
        logger.info("Starting in LIVE mode (capturing system audio / microphone)")
        source = LiveStream(device_name=AUDIO_INPUT_DEVICE_NAME)
        playback = False  # in live mode, audio is played directly in Spotify/Apple Music
    else:
        if not file_path:
            raise ValueError("In file playback mode, you must provide a file_path.")
        logger.info(f"Loading {file_path}")
        from lumir.audio.file_player import FilePlayer
        source = FilePlayer(file_path, realtime=file_realtime)

    if output_backend == "avolites":
        from lumir.output.avolites import AvolitesTitanOutput
        output = AvolitesTitanOutput()
    else:
        from lumir.output.console_logger import ConsoleLogger
        output = ConsoleLogger()

    track_id = os.path.basename(file_path) if file_path else "live"

    try:
        run(source, output, playback=playback, track_id=track_id)
    except KeyboardInterrupt:
        pass
    logger.info("Done.")
