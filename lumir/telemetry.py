"""Telemetry module – collects pipeline data in memory, dumps to CSV on close."""

from __future__ import annotations

import csv
import logging
import math
from datetime import datetime
from pathlib import Path

from lumir.config import CENTROID_FREQUENCY_MINIMUM_HERTZ, CENTROID_FREQUENCY_MAXIMUM_HERTZ
from lumir.models import AnalysisResult, LightCommand

logger = logging.getLogger(__name__)

# Pre-computed log-scale centroid bounds for Hz denormalization
_LOG_CENTROID_MIN = math.log2(CENTROID_FREQUENCY_MINIMUM_HERTZ)
_LOG_CENTROID_MAX = math.log2(CENTROID_FREQUENCY_MAXIMUM_HERTZ)

_COLUMNS = [
    "timestamp",
    "processing_time_ms",
    "track_id",
    "rms",
    "rms_bass",
    "rms_mid",
    "rms_high",
    "spectral_centroid",
    "spectral_centroid_hz",
    "onset_strength",
    "bpm",
    "is_beat",
    "is_raw_beat",
    "is_beat_bass",
    "is_beat_high",
    "is_section_boundary",
    "current_section",
    "energy_state",
    "dimmer",
    "strobe_bass",
]


class TelemetryDumper:
    """Buffers telemetry records in memory and writes them to a CSV file on close()."""

    def __init__(self, track_id: str = "") -> None:
        self._records: list[dict] = []
        self._track_id = track_id
        self._start_time = datetime.now()

    def record(self, features: AnalysisResult, command: LightCommand, processing_time_ms: float) -> None:
        centroid_hz = 2 ** (features.spectral_centroid * (_LOG_CENTROID_MAX - _LOG_CENTROID_MIN) + _LOG_CENTROID_MIN)

        self._records.append({
            "timestamp": features.timestamp,
            "processing_time_ms": round(processing_time_ms, 4),
            "track_id": self._track_id,
            "rms": features.rms,
            "rms_bass": features.rms_bass,
            "rms_mid": features.rms_mid,
            "rms_high": features.rms_high,
            "spectral_centroid": features.spectral_centroid,
            "spectral_centroid_hz": round(centroid_hz, 2),
            "onset_strength": features.onset_strength,
            "bpm": features.bpm,
            "is_beat": features.is_beat,
            "is_raw_beat": features.is_raw_beat,
            "is_beat_bass": features.is_beat_bass,
            "is_beat_high": features.is_beat_high,
            "is_section_boundary": features.is_section_boundary,
            "current_section": features.current_section.value,
            "energy_state": command.energy_state,
            "dimmer": command.dimmer,
            "strobe_bass": command.strobe_bass,
        })

    def close(self) -> None:
        if not self._records:
            return

        filename = f"telemetry_{self._start_time.strftime('%Y%m%d_%H%M%S')}.csv"
        path = Path(filename)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(self._records)

        logger.info(f"Saved {len(self._records)} records to {path}")
        self._records.clear()
