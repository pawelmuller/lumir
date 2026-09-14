"""Translation rules: MIR features → lighting commands."""

from __future__ import annotations

import logging
import math
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

from lumir.config import (
    AVOLITES_PLAYBACKS_INTENSITY,
    AVOLITES_PLAYBACKS_MOVEMENT,
    AVOLITES_PLAYBACKS_POSITION,
    AVOLITES_BPM_MULTIPLIER_INTENSITY_LOW,
    AVOLITES_BPM_MULTIPLIER_INTENSITY_MID,
    AVOLITES_BPM_MULTIPLIER_INTENSITY_HIGH,
    AVOLITES_BPM_MULTIPLIER_MOVEMENT_LOW,
    AVOLITES_BPM_MULTIPLIER_MOVEMENT_MID,
    AVOLITES_BPM_MULTIPLIER_MOVEMENT_HIGH,
    SIZE_MASTER_LOW,
    SIZE_MASTER_MID,
    SIZE_MASTER_HIGH,
    BASE_DIMMER_SMOOTHING_COEFFICIENT,
    FLASH_DECAY_COEFFICIENT,
    MAX_BASE_DIMMER,
    PATTERN_CHANGE_INTERVAL_SECONDS,
    TILT_SMOOTHING_COEFFICIENT,
    AVOLITES_PLAYBACK_COLOR_WARM_ACOUSTIC,
    AVOLITES_PLAYBACK_COLOR_WARM_DRIVE,
    AVOLITES_PLAYBACK_COLOR_COLD_AMBIENT,
    AVOLITES_PLAYBACK_COLOR_COLD_IMPACT,
    STROBE_DENSITY_WINDOW_SECONDS,
    STROBE_DENSITY_THRESHOLD_BEATS,
    STROBE_RMS_GATE,
    STROBE_HYSTERESIS_SECONDS,
)
from lumir.mapping.classification import get_classifier, SmoothedFeatures
from lumir.models import AnalysisResult, LightCommand, EnergyState


class PaletteManager:
    """Manages cuelist-based colour palette mood state and phrase advancement."""

    def __init__(self) -> None:
        self.active_playback: int | None = None
        self.last_change_time = 0.0

    def evaluate(self, norm_centroid: float, norm_rms: float, is_section_boundary: bool) -> int | None:
        # Only evaluate mood changes on section boundaries (or at startup)
        if self.active_playback is None or is_section_boundary:
            # Determine target mood based on audio features
            if norm_centroid < 0.5:
                target = AVOLITES_PLAYBACK_COLOR_WARM_ACOUSTIC if norm_rms < 0.5 else AVOLITES_PLAYBACK_COLOR_WARM_DRIVE
            else:
                target = AVOLITES_PLAYBACK_COLOR_COLD_AMBIENT if norm_rms < 0.5 else AVOLITES_PLAYBACK_COLOR_COLD_IMPACT

            if target != self.active_playback:
                self.active_playback = target
                logger.info(f"Mood changed! Firing Playback {self.active_playback}")

        return self.active_playback


class StrobeManager:
    """Evaluates onset density to trigger a strobe effect."""

    def __init__(self) -> None:
        self.beat_timestamps: deque[float] = deque()
        self.condition_met_since: float | None = None

    def evaluate(self, timestamp: float, is_beat_strobe: bool, norm_rms: float) -> bool:
        if is_beat_strobe:
            self.beat_timestamps.append(timestamp)

        # Remove old timestamps outside the window
        while self.beat_timestamps and (timestamp - self.beat_timestamps[0] > STROBE_DENSITY_WINDOW_SECONDS):
            self.beat_timestamps.popleft()

        density = len(self.beat_timestamps)
        raw_condition = (density >= STROBE_DENSITY_THRESHOLD_BEATS) and (norm_rms > STROBE_RMS_GATE)

        if raw_condition:
            if self.condition_met_since is None:
                self.condition_met_since = timestamp

            if (timestamp - self.condition_met_since) >= STROBE_HYSTERESIS_SECONDS:
                return True
        else:
            self.condition_met_since = None

        return False


class Mapper:
    """Translates an AnalysisResult into a LightCommand using fixed rules."""

    def __init__(self) -> None:
        self.palette_manager = PaletteManager()
        self.strobe_manager = StrobeManager()

        # Base dimmer history states
        self._previous_dimmer = 0.0
        self._previous_dimmer_bass = 0.0
        self._previous_dimmer_mid = 0.0
        self._previous_dimmer_high = 0.0

        # Beat flash states (for Python-side mapping fallback)
        self._flash_dimmer = 0.0

        # Size Master and Tilt smoothing states
        self._previous_size = 0.15
        self._previous_tilt = 0.0

        # Rolling history for dynamic RMS and spectral features normalization (~10 seconds window)
        self._rms_history: deque[float] = deque(maxlen=107)  # ~10s window at ~10.7 fps (93ms/frame)
        self._rms_bass_history: deque[float] = deque(maxlen=107)
        self._rms_mid_history: deque[float] = deque(maxlen=107)
        self._rms_high_history: deque[float] = deque(maxlen=107)
        self._centroid_history: deque[float] = deque(maxlen=107)

        # Look/Pattern rotation states (independent of energy state)
        self._active_intensity_playback: int | None = None
        self._active_movement_playback: int | None = None
        self._active_position_playback: int | None = None
        self._last_pattern_change_timestamp = 0.0
        self._beat_counter = 0
        self._intensity_index = 0
        self._movement_index = 0
        self._position_index = 0
        self._onset_history: deque[float] = deque(maxlen=100)

        # 3-second smoothing queues for Multi-Factor parameters (32 frames ≈ 3s at 93ms/frame)
        self._volume_history: deque[float] = deque(maxlen=32)
        self._brightness_history: deque[float] = deque(maxlen=32)
        self._rhythm_history: deque[float] = deque(maxlen=32)

        # Backward compatibility / legacy 1D history tracking
        self._beat_history: deque[float] = deque(maxlen=32)

        # State tracking
        self._current_state = EnergyState.LOW
        self._macro_state = EnergyState.LOW
        self.classifier = get_classifier()
        self._last_threshold_update_time = 0.0

        # Special Effects Timers
        self._blinder_timer = 0.0

    def _normalize_rolling(self, value: float, history: deque[float]) -> float:
        """Applies dynamic min-max normalization based on a rolling history window."""
        history.append(value)
        if len(history) < 10:
            return 0.5  # default neutral normalized value during early startup

        min_value = min(history)
        max_value = max(history)

        if (max_value - min_value) > 1e-4:
            return (value - min_value) / (max_value - min_value)
        return 0.0

    def translate(self, result: AnalysisResult) -> LightCommand:
        # 1. Dynamically normalize RMS and Spectral features relative to recent song section dynamics
        norm_rms = self._normalize_rolling(result.rms, self._rms_history)
        norm_rms_bass = self._normalize_rolling(result.rms_bass, self._rms_bass_history)
        norm_rms_mid = self._normalize_rolling(result.rms_mid, self._rms_mid_history)
        norm_rms_high = self._normalize_rolling(result.rms_high, self._rms_high_history)
        norm_centroid = self._normalize_rolling(result.spectral_centroid, self._centroid_history)

        trigger_strobe = self.strobe_manager.evaluate(result.timestamp, result.is_beat_strobe, norm_rms)

        # Track beat history for density strategy
        self._beat_history.append(1.0 if result.is_beat else 0.0)

        blinder_active = result.timestamp < self._blinder_timer
        trigger_color_go = False

        # 2. Overall Dimmer (for Active Intensity Playback Level): Base + Beat Flash
        # Apply perceptual gamma power-law expansion (Stevens' power law, gamma = 2.2)
        gamma = 2.2  # perceptual gamma (Stevens' power law)
        perceptual_rms = math.pow(norm_rms, gamma) if norm_rms > 0 else 0.0
        base_dimmer_target = perceptual_rms * MAX_BASE_DIMMER
        base_dimmer = self._previous_dimmer + BASE_DIMMER_SMOOTHING_COEFFICIENT * (
                base_dimmer_target - self._previous_dimmer)
        self._previous_dimmer = base_dimmer

        if result.is_beat:
            self._flash_dimmer = 1.0
        else:
            self._flash_dimmer *= (1.0 - FLASH_DECAY_COEFFICIENT)
        dimmer = max(base_dimmer, self._flash_dimmer)

        # 3. Bass Dimmer: Base (Symmetrically and slowly smoothed background)
        perceptual_rms_bass = math.pow(norm_rms_bass, gamma) if norm_rms_bass > 0 else 0.0
        base_dimmer_bass_target = perceptual_rms_bass * MAX_BASE_DIMMER
        dimmer_bass = self._previous_dimmer_bass + BASE_DIMMER_SMOOTHING_COEFFICIENT * (
                base_dimmer_bass_target - self._previous_dimmer_bass)
        self._previous_dimmer_bass = dimmer_bass

        # 4. Mid-Dimmer (smooth volume movement)
        if self._current_state in [EnergyState.MID, EnergyState.HIGH]:
            dimmer_mid = 0.0
            self._previous_dimmer_mid = 0.0
        else:
            perceptual_rms_mid = math.pow(norm_rms_mid, gamma) if norm_rms_mid > 0 else 0.0
            dimmer_mid_target = perceptual_rms_mid * MAX_BASE_DIMMER
            dimmer_mid = self._previous_dimmer_mid + BASE_DIMMER_SMOOTHING_COEFFICIENT * (
                    dimmer_mid_target - self._previous_dimmer_mid)
            self._previous_dimmer_mid = dimmer_mid

        # 5. High Dimmer (smooth high frequency dynamics)
        perceptual_rms_high = math.pow(norm_rms_high, gamma) if norm_rms_high > 0 else 0.0
        dimmer_high_target = perceptual_rms_high * MAX_BASE_DIMMER
        dimmer_high = self._previous_dimmer_high + BASE_DIMMER_SMOOTHING_COEFFICIENT * (
                dimmer_high_target - self._previous_dimmer_high)
        self._previous_dimmer_high = dimmer_high

        # 6. Smooth all Multi-Factor classification parameters independently
        self._volume_history.append(norm_rms)
        smoothed_volume = float(np.mean(self._volume_history))

        self._brightness_history.append(norm_centroid)
        smoothed_brightness = float(np.mean(self._brightness_history))

        # rhythm_score counts the density of hits in a 3s window (rescaled for range 0.0 - 1.0)
        beat_count = sum(self._beat_history)
        norm_beat_density = min(beat_count / 5.0, 1.0)  # 5 beats in 3s = maximum density
        self._rhythm_history.append(norm_beat_density)
        smoothed_rhythm = float(np.mean(self._rhythm_history))

        # --- Periodic Threshold Update ---
        if result.timestamp - self._last_threshold_update_time >= 1.0:
            self.classifier.update_thresholds()
            self._last_threshold_update_time = result.timestamp

        features = SmoothedFeatures(
            volume=smoothed_volume,
            brightness=smoothed_brightness,
            rhythm=smoothed_rhythm,
            raw_rms=norm_rms,
            raw_centroid=norm_centroid,
            raw_beat_density=norm_beat_density,
        )

        # --- Boundary Reset & Macro-State Logic ---
        if result.is_section_boundary:
            # Clear mapper buffers to remove inertia
            for _ in range(self._volume_history.maxlen or 0):
                self._volume_history.append(norm_rms)
            for _ in range(self._brightness_history.maxlen or 0):
                self._brightness_history.append(norm_centroid)
            for _ in range(self._rhythm_history.maxlen or 0):
                self._rhythm_history.append(norm_beat_density)

            if hasattr(self.classifier, 'reset_history'):
                self.classifier.reset_history(features)

            # Evaluate new macro state
            if hasattr(self.classifier, 'evaluate_instant'):
                new_macro_state = self.classifier.evaluate_instant(features)
                logger.info(f"MACRO-STATE Check: Was {self._macro_state.name}, is {new_macro_state.name}")

                # Check for escalation / de-escalation
                if self._macro_state in [EnergyState.LOW, EnergyState.MID] and new_macro_state == EnergyState.HIGH:
                    logger.info(f"ESCALATION (Drop) detected! Firing Blinders!")
                    self._intensity_index += 1
                    self._movement_index += 1
                    self._blinder_timer = result.timestamp + 2.0  # 2 seconds of Blinders
                elif self._macro_state == EnergyState.HIGH and new_macro_state in [EnergyState.LOW, EnergyState.MID]:
                    logger.info(f"DE-ESCALATION (Breakdown) detected!")
                    self._active_movement_playback = None

                self._macro_state = new_macro_state

        # --- Energy State Decision Process ---
        if result.rms < 0.005:
            # Noise gate / Silence check: if audio stops, drop to LOW state instantly
            target_state = EnergyState.LOW
        else:
            target_state = self.classifier.classify(features, self._current_state, result.timestamp)

            # Apply Macro-State Constraints (soft: only prevent hard drops HIGH→LOW)
            if self._macro_state == EnergyState.HIGH and target_state == EnergyState.LOW:
                target_state = EnergyState.MID

        # Apply state transitions
        state_changed = False
        if target_state != self._current_state:
            self._current_state = target_state
            state_changed = True

        # 7. Select state-dependent BPM multipliers and Size Master targets
        if self._current_state == EnergyState.LOW:
            mult_intensity = AVOLITES_BPM_MULTIPLIER_INTENSITY_LOW
            mult_movement = AVOLITES_BPM_MULTIPLIER_MOVEMENT_LOW
            target_size = SIZE_MASTER_LOW
        elif self._current_state == EnergyState.MID:
            mult_intensity = AVOLITES_BPM_MULTIPLIER_INTENSITY_MID
            mult_movement = AVOLITES_BPM_MULTIPLIER_MOVEMENT_MID
            target_size = SIZE_MASTER_MID
        else:
            mult_intensity = AVOLITES_BPM_MULTIPLIER_INTENSITY_HIGH
            mult_movement = AVOLITES_BPM_MULTIPLIER_MOVEMENT_HIGH
            target_size = SIZE_MASTER_HIGH

        # Calculate final BPM target for each layer
        bpm_intensity = result.bpm * mult_intensity
        bpm_movement = result.bpm * mult_movement

        # Smooth Size Master transitions
        size_master_level = self._previous_size + BASE_DIMMER_SMOOTHING_COEFFICIENT * (
                target_size - self._previous_size)
        self._previous_size = size_master_level

        # Smooth Tilt Level (height mapping) based on Spectral Centroid / Brightness
        # Restrict dynamic height sweeps strictly to the HIGH energy state (drops/choruses)
        # In LOW/MID states, fade back slowly and freeze at 0.0 (safe default position)
        if self._current_state == EnergyState.HIGH:
            tilt_target = norm_centroid
        else:
            tilt_target = 0.0

        tilt_level = self._previous_tilt + TILT_SMOOTHING_COEFFICIENT * (tilt_target - self._previous_tilt)
        self._previous_tilt = tilt_level

        # 8. Beat counting, Strobe filtering and Look selection logic
        strobe_bass = False
        strobe_high = False
        previous_intensity_playback_to_kill: int | None = None
        previous_movement_playback_to_kill: int | None = None

        # Record onset strength to history for dynamic accent percentile calculation
        self._onset_history.append(result.onset_strength)

        # Handle active playbacks setup/destruction during state changes
        if state_changed or self._active_intensity_playback is None:
            if self._current_state in (EnergyState.LOW, EnergyState.MID):
                # In LOW/MID state: activate intensity chase, kill movement chase
                if self._active_intensity_playback is None and AVOLITES_PLAYBACKS_INTENSITY:
                    self._active_intensity_playback = AVOLITES_PLAYBACKS_INTENSITY[
                        self._intensity_index % len(AVOLITES_PLAYBACKS_INTENSITY)]
                    self._intensity_index += 1
                    self._last_pattern_change_timestamp = result.timestamp
                if self._active_movement_playback is not None:
                    previous_movement_playback_to_kill = self._active_movement_playback
                    self._active_movement_playback = None
            elif self._current_state == EnergyState.HIGH:
                # In HIGH state: activate both intensity and movement chases
                if self._active_intensity_playback is None and AVOLITES_PLAYBACKS_INTENSITY:
                    self._active_intensity_playback = AVOLITES_PLAYBACKS_INTENSITY[
                        self._intensity_index % len(AVOLITES_PLAYBACKS_INTENSITY)]
                    self._intensity_index += 1
                    self._last_pattern_change_timestamp = result.timestamp
                if self._active_movement_playback is None and AVOLITES_PLAYBACKS_MOVEMENT:
                    self._active_movement_playback = AVOLITES_PLAYBACKS_MOVEMENT[
                        self._movement_index % len(AVOLITES_PLAYBACKS_MOVEMENT)]
                    self._movement_index += 1

        # Process beat strobes and pattern rotation only in MID and HIGH states
        if self._current_state != EnergyState.LOW:
            if result.is_beat:
                self._beat_counter += 1
                strobe_bass = result.is_beat_bass

                strobe_high = result.is_beat_high if self._current_state == EnergyState.HIGH else False

                # Look Rotation logic (occurs on multiples of 8 beats and on a strong accent)
                time_elapsed = result.timestamp - self._last_pattern_change_timestamp
                is_phrase_boundary = (self._beat_counter >= 8 and self._beat_counter % 8 == 0)

                if is_phrase_boundary:
                    trigger_color_go = True

                recent_onsets = np.array(self._onset_history)
                threshold_accent = np.percentile(recent_onsets, 70) if len(recent_onsets) >= 10 else 5.0
                is_strong_accent = (result.onset_strength >= threshold_accent)
                force_transition = (
                            time_elapsed >= PATTERN_CHANGE_INTERVAL_SECONDS + 15.0)  # forced transition after extended interval

                if (time_elapsed >= PATTERN_CHANGE_INTERVAL_SECONDS and is_phrase_boundary) and (
                        is_strong_accent or force_transition):
                    # Rotate Intensity Chase
                    previous_intensity_playback_to_kill = self._active_intensity_playback
                    self._last_pattern_change_timestamp = result.timestamp

                    if AVOLITES_PLAYBACKS_INTENSITY:
                        self._active_intensity_playback = AVOLITES_PLAYBACKS_INTENSITY[
                            self._intensity_index % len(AVOLITES_PLAYBACKS_INTENSITY)]
                        self._intensity_index += 1

                    # Rotate Position
                    if AVOLITES_PLAYBACKS_POSITION:
                        self._active_position_playback = AVOLITES_PLAYBACKS_POSITION[
                            self._position_index % len(AVOLITES_PLAYBACKS_POSITION)]
                        self._position_index += 1

                    # Rotate Movement Shape ONLY IF we are in the HIGH state
                    if self._current_state == EnergyState.HIGH and AVOLITES_PLAYBACKS_MOVEMENT:
                        previous_movement_playback_to_kill = self._active_movement_playback
                        self._active_movement_playback = AVOLITES_PLAYBACKS_MOVEMENT[
                            self._movement_index % len(AVOLITES_PLAYBACKS_MOVEMENT)]
                        self._movement_index += 1
        else:
            # Explicitly force strobe states and intensity level to 0 in LOW state
            strobe_bass = False
            strobe_high = False
            dimmer = 0.0

        is_section_boundary = result.is_section_boundary
        active_color_playback = self.palette_manager.evaluate(norm_centroid, norm_rms, is_section_boundary)

        # Minimum wash floor (30%) — keeps stage lit during music, fades in silence
        if result.rms >= 0.005:
            dimmer_bass = max(dimmer_bass, 0.30)  # 30% minimum wash floor to keep stage lit
            dimmer_mid = max(dimmer_mid, 0.30)
            dimmer_high = max(dimmer_high, 0.30)

        return LightCommand(
            timestamp=result.timestamp,
            dimmer=dimmer,
            dimmer_bass=dimmer_bass,
            dimmer_mid=dimmer_mid,
            dimmer_high=dimmer_high,
            active_color_playback=active_color_playback,
            trigger_color_go=trigger_color_go,
            strobe_bass=strobe_bass,
            strobe_high=strobe_high,
            trigger_strobe=trigger_strobe,
            blinder_active=blinder_active,
            bpm_intensity=bpm_intensity,
            bpm_movement=bpm_movement,
            active_intensity_playback=self._active_intensity_playback,
            previous_intensity_playback_to_kill=previous_intensity_playback_to_kill,
            active_movement_playback=self._active_movement_playback,
            previous_movement_playback_to_kill=previous_movement_playback_to_kill,
            active_position_playback=self._active_position_playback,
            size_master_level=size_master_level,
            tilt_level=tilt_level,
            energy_state=self._current_state.name,
        )
