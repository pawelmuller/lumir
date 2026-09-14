from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np

from lumir.config import (
    ENERGY_PERCENTILE_HIGH,
    ENERGY_PERCENTILE_LOW,
    ENERGY_STATE_STRATEGY,
    ENERGY_THRESHOLD_HIGH,
    ENERGY_THRESHOLD_LOW,
)
from lumir.models import EnergyState, EnergyStateStrategy


@dataclass
class SmoothedFeatures:
    """Features required by energy classifiers."""

    volume: float
    brightness: float
    rhythm: float
    # Raw normalized values needed for some legacy 1D strategies
    raw_rms: float
    raw_centroid: float
    raw_beat_density: float


class HysteresisThreshold:
    """A Schmitt Trigger implementation to prevent state bouncing."""

    def __init__(self, base_threshold: float, margin: float = 0.05) -> None:
        self.base = base_threshold
        self.margin = margin

    def evaluate(self, value: float, currently_active: bool) -> bool:
        if currently_active:
            return value >= (self.base - self.margin)
        else:
            return value >= (self.base + self.margin)


class EnergyClassifier(ABC):
    @abstractmethod
    def update_thresholds(self) -> None:
        """Periodically update percentile-based internal thresholds."""
        pass

    @abstractmethod
    def classify(
            self, features: SmoothedFeatures, current_state: EnergyState,
            timestamp: float = 0.0,
    ) -> EnergyState:
        """Evaluates features and returns the targeted EnergyState."""
        pass


class MultiFactorClassifier(EnergyClassifier):
    """Multi-factor energy classification using volume, rhythm density, and spectral brightness."""

    def __init__(self) -> None:
        self.v_high = HysteresisThreshold(0.70)
        self.v_low = HysteresisThreshold(0.30)
        self.b_high = HysteresisThreshold(0.65)
        self.r_high = HysteresisThreshold(0.60)
        self.r_low = HysteresisThreshold(0.25)

        self._vol_history: deque[float] = deque(maxlen=430)
        self._bright_history: deque[float] = deque(maxlen=430)
        self._rhythm_history: deque[float] = deque(maxlen=430)

        self._last_high_score_time: float | None = None

    def update_thresholds(self) -> None:
        if len(self._vol_history) >= 200:
            self.v_high.base = max(
                0.40, min(float(np.percentile(self._vol_history, ENERGY_PERCENTILE_HIGH)), 0.80)
            )
            self.v_low.base = max(
                0.18, min(float(np.percentile(self._vol_history, ENERGY_PERCENTILE_LOW)), 0.38)
            )
            self.b_high.base = max(
                0.50, min(float(np.percentile(self._bright_history, 70)), 0.75)
            )
            self.r_high.base = max(
                0.45, min(float(np.percentile(self._rhythm_history, 70)), 0.75)
            )
            self.r_low.base = max(
                0.15, min(float(np.percentile(self._rhythm_history, ENERGY_PERCENTILE_LOW)), 0.35)
            )

    def reset_history(self, features: SmoothedFeatures) -> None:
        for _ in range(self._vol_history.maxlen or 0):
            self._vol_history.append(features.raw_rms)
        for _ in range(self._bright_history.maxlen or 0):
            self._bright_history.append(features.raw_centroid)
        for _ in range(self._rhythm_history.maxlen or 0):
            self._rhythm_history.append(features.raw_beat_density)

    # Thresholds for HIGH score gate (with Schmitt trigger)
    _SCORE_HIGH_THRESH = HysteresisThreshold(0.50, margin=0.04)

    # How long (seconds) to sustain HIGH after score last confirmed it
    _HIGH_SUSTAIN_SECONDS = 4.0

    def _compute_score(self, vol: float, density: float, brightness: float) -> float:
        """Weighted energy score combining volume, rhythm density and spectral brightness."""
        return 0.5 * vol + 0.3 * density + 0.2 * brightness

    def evaluate_instant(self, features: SmoothedFeatures) -> EnergyState:
        score = self._compute_score(
            features.raw_rms, features.raw_beat_density, features.raw_centroid
        )
        if features.raw_rms < self.v_low.base:
            return EnergyState.LOW
        if score >= (self._SCORE_HIGH_THRESH.base - 0.05):
            return EnergyState.HIGH
        if features.raw_beat_density < self.r_low.base and features.raw_rms < 0.5:
            return EnergyState.LOW
        return EnergyState.MID

    def classify(
            self, features: SmoothedFeatures, current_state: EnergyState,
            timestamp: float = 0.0,
    ) -> EnergyState:
        self._vol_history.append(features.volume)
        self._bright_history.append(features.brightness)
        self._rhythm_history.append(features.rhythm)

        # Compute weighted score from smoothed features
        score = self._compute_score(features.volume, features.rhythm, features.brightness)

        # Volume floor gate: hard exit regardless of sustain
        if features.volume < self.v_low.base:
            self._last_high_score_time = None
            return EnergyState.LOW

        # Update sustain clock whenever score is above the exit threshold (base - margin)
        exit_threshold = self._SCORE_HIGH_THRESH.base - self._SCORE_HIGH_THRESH.margin
        if score >= exit_threshold:
            self._last_high_score_time = timestamp

        # Schmitt trigger for entry into HIGH (requires score >= base + margin)
        if self._SCORE_HIGH_THRESH.evaluate(score, current_state == EnergyState.HIGH):
            return EnergyState.HIGH

        # Sustain HIGH if score was confirmed recently (recency guard)
        if (
                current_state == EnergyState.HIGH
                and self._last_high_score_time is not None
                and (timestamp - self._last_high_score_time) <= self._HIGH_SUSTAIN_SECONDS
        ):
            return EnergyState.HIGH

        if features.rhythm < self.r_low.base and features.volume < 0.5:
            return EnergyState.LOW

        return EnergyState.MID


class LegacyClassifier(EnergyClassifier):
    """Fallback classifier for legacy 1D energy classification strategies."""

    def __init__(self, strategy: EnergyStateStrategy) -> None:
        self.strategy = strategy
        self.energy_high = HysteresisThreshold(ENERGY_THRESHOLD_HIGH)
        self.energy_low = HysteresisThreshold(ENERGY_THRESHOLD_LOW)
        self._energy_history: deque[float] = deque(maxlen=32)
        self._energy_score_history: deque[float] = deque(maxlen=430)

    def update_thresholds(self) -> None:
        if len(self._energy_score_history) >= 200:
            self.energy_high.base = max(
                0.58,
                min(float(np.percentile(self._energy_score_history, ENERGY_PERCENTILE_HIGH)), 0.82),
            )
            self.energy_low.base = max(
                0.18,
                min(float(np.percentile(self._energy_score_history, ENERGY_PERCENTILE_LOW)), 0.38),
            )

    def classify(
            self, features: SmoothedFeatures, current_state: EnergyState
    ) -> EnergyState:
        match self.strategy:
            case EnergyStateStrategy.RMS_ONLY:
                decision_metric = features.raw_rms
            case EnergyStateStrategy.RMS_AND_CENTROID:
                decision_metric = 0.7 * features.raw_rms + 0.3 * features.raw_centroid
            case EnergyStateStrategy.RMS_AND_ONSET_DENSITY:
                decision_metric = 0.6 * features.raw_rms + 0.4 * features.raw_beat_density
            case _:
                decision_metric = features.raw_rms

        self._energy_history.append(decision_metric)
        smoothed_metric = float(np.mean(self._energy_history))
        self._energy_score_history.append(smoothed_metric)

        is_high = self.energy_high.evaluate(
            smoothed_metric, current_state == EnergyState.HIGH
        )
        is_low = smoothed_metric < self.energy_low.base

        if is_low:
            return EnergyState.LOW
        if is_high:
            return EnergyState.HIGH
        return EnergyState.MID


def get_classifier() -> EnergyClassifier:
    """Factory to retrieve the currently active classifier."""
    if ENERGY_STATE_STRATEGY == EnergyStateStrategy.MULTI_FACTOR:
        return MultiFactorClassifier()
    return LegacyClassifier(ENERGY_STATE_STRATEGY)
