"""Data models passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EnergyStateStrategy(Enum):
    RMS_ONLY = "rms_only"
    RMS_AND_CENTROID = "rms_and_centroid"
    RMS_AND_ONSET_DENSITY = "rms_and_onset_density"
    MULTI_FACTOR = "multi_factor"


class EnergyState(Enum):
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class TrackSection(Enum):
    """Abstract labels for detected musical sections."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"


@dataclass
class AudioChunk:
    """Raw audio window emitted by an AudioSource."""

    samples: np.ndarray  # mono float32, shape (N,)
    sample_rate: int
    timestamp: float  # chunk start time in the track (seconds)


@dataclass
class AnalysisResult:
    """MIR features extracted from a single AudioChunk."""

    timestamp: float
    rms: float  # overall normalized energy (0.0-1.0)
    rms_bass: float  # low frequency band (0-200 Hz) energy (0.0-1.0)
    rms_mid: float  # mid frequency band (200-2000 Hz) energy (0.0-1.0)
    rms_high: float  # high frequency band (2000+ Hz) energy (0.0-1.0)
    onset_strength: float
    is_beat: bool  # logical OR of bass and high beats (for structural trigger compatibility)
    is_raw_beat: bool  # raw analytical beat before cooldown and floor logic
    is_beat_bass: bool  # low frequency onset beat
    is_beat_high: bool  # high frequency onset beat
    is_beat_strobe: bool  # dense high-frequency strobe hits
    bpm: float
    spectral_centroid: float  # 0.0–1.0 (normalised)

    # Structural Segmentation
    is_section_boundary: bool
    current_section: TrackSection


@dataclass
class LightCommand:
    """Abstract lighting command produced by the mapper."""

    timestamp: float
    dimmer: float  # overall dimmer (0.0–1.0)
    dimmer_bass: float  # dimmer for bass wash lights (0.0-1.0)
    dimmer_mid: float  # dimmer for mid spots/fixtures (0.0-1.0)
    dimmer_high: float  # dimmer for high strobes/fixtures (0.0-1.0)

    strobe_bass: bool  # trigger bass flashes
    strobe_high: bool  # trigger high flashes
    trigger_strobe: bool  # trigger dense strobe based on onset density

    # Special Effects
    blinder_active: bool

    bpm_intensity: float
    bpm_movement: float

    # State-based energy mapping (split into Intensity and Movement layers)
    active_intensity_playback: int | None = None
    previous_intensity_playback_to_kill: int | None = None
    active_movement_playback: int | None = None
    previous_movement_playback_to_kill: int | None = None
    active_position_playback: int | None = None
    size_master_level: float = 0.0
    tilt_level: float = 0.0

    # Cuelist-based color mapping
    active_color_playback: int | None = None
    trigger_color_go: bool = False
    energy_state: str = ""
