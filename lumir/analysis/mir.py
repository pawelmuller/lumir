"""MIR feature extraction from audio chunks."""

from __future__ import annotations

import logging
import math
from collections import deque

import librosa
import librosa.feature.rhythm
import numpy as np
import scipy.signal

from lumir.config import (
    ANALYSIS_SAMPLE_RATE,
    ANALYSIS_WINDOW_SAMPLES,
    BASS_MAX_FREQUENCY_HERTZ,
    MID_MAX_FREQUENCY_HERTZ,
    BPM_UPDATE_INTERVAL_SECONDS,
    BPM_ANALYSIS_WINDOW_SECONDS,
    CENTROID_FREQUENCY_MAXIMUM_HERTZ,
    CENTROID_FREQUENCY_MINIMUM_HERTZ,
    DEFAULT_BPM,
    HOP_LENGTH,
    PLAYBACK_SAMPLE_RATE,
    CHUNK_SAMPLES,
    MINIMUM_ONSET_ABSOLUTE_THRESHOLD,
    MINIMUM_RMS_STROBE_THRESHOLD,
    ONSET_MAX_FREQUENCY_HERTZ,
    FLASH_ONSET_THRESHOLD_FACTOR,
    FLASH_COOLDOWN_MULTIPLIER,
    STROBE_MIN_FREQUENCY_HERTZ,
)
from lumir.models import AnalysisResult, AudioChunk, TrackSection

logger = logging.getLogger(__name__)


class Analyzer:
    """Extracts MIR features from successive AudioChunks.

    Maintains internal state for BPM tracking and adaptive onset
    thresholding across chunks.
    """

    _N_FFT = 2048  # STFT window size (frequency resolution)

    def __init__(self, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

        # Sliding audio buffer for wider analysis context
        self.window_samples = ANALYSIS_WINDOW_SAMPLES
        self.audio_buffer = np.zeros(self.window_samples, dtype=np.float32)

        # --- Incremental STFT cache ---
        # Periodic Hann window matching librosa.stft defaults
        self._hann_window = scipy.signal.get_window('hann', self._N_FFT, fftbins=True)
        n_freqs = 1 + self._N_FFT // 2  # 1025 bins for n_fft=2048
        self._total_stft_frames = 1 + (self.window_samples - self._N_FFT) // HOP_LENGTH
        self._spec_cache = np.zeros((n_freqs, self._total_stft_frames))
        self._spec_valid = False

        # Pre-computed frequency bin boundaries (avoids repeated division per frame)
        self._bin_bass_onset = int(np.ceil(250.0 * self._N_FFT / self.sample_rate))
        self._bin_high_onset = int(np.ceil(2000.0 * self._N_FFT / self.sample_rate))
        self._bin_strobe = int(np.ceil(STROBE_MIN_FREQUENCY_HERTZ * self._N_FFT / self.sample_rate))
        self._bin_main_onset = int(np.ceil(ONSET_MAX_FREQUENCY_HERTZ * self._N_FFT / self.sample_rate))
        self._bin_bass_rms = int(np.ceil(BASS_MAX_FREQUENCY_HERTZ * self._N_FFT / self.sample_rate))
        self._bin_mid_rms = int(np.ceil(MID_MAX_FREQUENCY_HERTZ * self._N_FFT / self.sample_rate))

        # BPM tracking — accumulate onset envelope over time
        bpm_history_frames = int(BPM_ANALYSIS_WINDOW_SECONDS * sample_rate / HOP_LENGTH)
        self.onset_history: deque[float] = deque(maxlen=bpm_history_frames)
        self.bpm = DEFAULT_BPM
        self.samples_since_bpm_update = 0

        # Frequency-separated beat detection states (Bass and High)
        self.recent_onsets_bass: deque[float] = deque(maxlen=200)  # ~20s window at ~10 fps (93ms/frame)
        self.recent_onsets_high: deque[float] = deque(maxlen=200)
        self.recent_onsets_strobe: deque[float] = deque(maxlen=200)

        self._recent_onsets_long_bass: deque[float] = deque(maxlen=215)  # ~20s extended window for floor calculation
        self._recent_onsets_long_high: deque[float] = deque(maxlen=215)
        self._recent_onsets_long_strobe: deque[float] = deque(maxlen=215)

        self.last_beat_timestamp_bass = 0.0
        self.last_beat_timestamp_high = 0.0

        # Centroid normalisation bounds (log-scale)
        self.log_centroid_minimum_frequency = np.log2(CENTROID_FREQUENCY_MINIMUM_HERTZ)
        self.log_centroid_maximum_frequency = np.log2(CENTROID_FREQUENCY_MAXIMUM_HERTZ)

        # Resampling state for scipy.signal.resample_poly
        self._resample_history = np.zeros(0, dtype=np.float32)

        # BPM tracking smoothing
        self.bpm_estimates = deque(maxlen=5)

        # Structural Segmentation (Novelty Detection)
        chunks_per_second = PLAYBACK_SAMPLE_RATE / CHUNK_SAMPLES

        # short window ~2s, long window ~10s
        self.mfcc_history_short: deque[np.ndarray] = deque(maxlen=int(2.0 * chunks_per_second))
        self.mfcc_history_long: deque[np.ndarray] = deque(maxlen=int(10.0 * chunks_per_second))
        self.novelty_history: deque[float] = deque(maxlen=int(15.0 * chunks_per_second))
        self.last_boundary_time = 0.0
        self.section_cooldown_seconds = 10.0  # minimum seconds between detected section boundaries

        self.section_letters = list(TrackSection)
        self.current_section_idx = 0

    # ------------------------------------------------------------------
    # Optimised DSP helpers
    # ------------------------------------------------------------------

    def _update_spectrogram(self, new_sample_count: int) -> np.ndarray:
        """Incrementally update the cached STFT magnitude spectrogram.

        Uses center=False framing (no zero-padding at signal edges) so
        that cached columns can be reused exactly when the chunk size is
        a multiple of HOP_LENGTH.  Only computes FFT for frames whose
        analysis windows overlap with newly arrived audio samples.

        For the standard configuration (CHUNK_SAMPLES=4096 at 44100 Hz →
        2048 analysis samples, HOP_LENGTH=512), this reuses previously
        computed columns and only calculates FFT for newly arrived frames.
        """
        shift_frames = new_sample_count // HOP_LENGTH
        aligned = (new_sample_count % HOP_LENGTH == 0)

        if self._spec_valid and aligned and 0 < shift_frames < self._total_stft_frames:
            reuse_count = self._total_stft_frames - shift_frames
            # Shift cached columns left (numpy handles overlapping views safely since 1.13)
            self._spec_cache[:, :reuse_count] = self._spec_cache[:, shift_frames:]
            compute_start = reuse_count
        else:
            # First call, misaligned chunk, or chunk larger than window — full recompute
            compute_start = 0
            self._spec_valid = True

        # Compute only the new STFT columns via batched FFT
        frames_to_compute = self._total_stft_frames - compute_start
        if frames_to_compute > 0:
            first_sample = compute_start * HOP_LENGTH
            buf_slice = self.audio_buffer[first_sample:]
            item_stride = buf_slice.strides[0]

            # Zero-copy strided view: (frames_to_compute, n_fft) with hop_length stride
            frames_view = np.lib.stride_tricks.as_strided(
                buf_slice,
                shape=(frames_to_compute, self._N_FFT),
                strides=(HOP_LENGTH * item_stride, item_stride),
            )
            windowed = frames_view * self._hann_window
            self._spec_cache[:, compute_start:] = np.abs(np.fft.rfft(windowed, axis=1)).T

        return self._spec_cache

    @staticmethod
    def _band_onset_envelope(log_spec_band: np.ndarray) -> np.ndarray:
        """Half-wave rectified spectral flux for a sub-band log-spectrogram.

        Computes the core of onset detection (positive energy increase
        averaged across frequency bins) without the overhead of librosa's
        mel filter bank construction and median reference subtraction.

        Returns an array with the same frame count as the input
        (a leading zero is prepended to compensate for np.diff).

        Note: The absolute scale differs from librosa.onset.onset_strength
        (which applies median centering per-bin).  The adaptive
        thresholding in process() is relative to recent history, so it
        self-calibrates to the new scale within a few chunks.
        """
        n_frames = log_spec_band.shape[1]
        if n_frames < 2:
            return np.zeros(max(n_frames, 1))

        diff = np.diff(log_spec_band, axis=1)
        np.maximum(diff, 0, out=diff)  # half-wave rectify in-place
        env = np.mean(diff, axis=0)

        # Prepend a zero so result length matches the spectrogram frame count
        result = np.empty(n_frames)
        result[0] = 0.0
        result[1:] = env
        return result

    def _detect_onset(
            self,
            onset_value: float,
            recent_history: deque[float],
            long_history: deque[float],
            rms: float,
    ) -> bool:
        """Adaptive thresholding for sub-band onset detection."""
        recent_history.append(onset_value)
        if len(recent_history) >= 5:
            threshold = float(
                np.mean(recent_history)
                + FLASH_ONSET_THRESHOLD_FACTOR * np.std(recent_history)
            )
        else:
            threshold = onset_value + 1.0
        long_history.append(onset_value)
        floor = max(MINIMUM_ONSET_ABSOLUTE_THRESHOLD, 0.35 * max(long_history))
        return (
                onset_value > threshold
                and onset_value > floor
                and rms > MINIMUM_RMS_STROBE_THRESHOLD
        )

    # ------------------------------------------------------------------

    def process(self, chunk: AudioChunk) -> AnalysisResult:
        raw_samples = chunk.samples

        # Convert chunk samples to mono if stereo
        if raw_samples.ndim > 1:
            mono_samples = np.mean(raw_samples, axis=0)
        else:
            mono_samples = raw_samples

        # Downsample to ANALYSIS_SAMPLE_RATE (22050 Hz)
        if chunk.sample_rate == 44100:
            analysis_samples = mono_samples[::2]
        elif chunk.sample_rate != 22050:
            target_sr = 22050
            g = math.gcd(target_sr, chunk.sample_rate)
            up = target_sr // g
            down = chunk.sample_rate // g

            # Overlap-save resampling with scipy.signal.resample_poly
            # Use a large margin (e.g. 2048 samples) to cover the resample_poly FIR filter settling time
            margin = 2048
            if len(self._resample_history) == 0:
                combined = np.pad(mono_samples, (margin, 0), mode='constant')
            else:
                combined = np.concatenate((self._resample_history, mono_samples))

            # Store the current samples for the next chunk's history
            if len(mono_samples) >= margin:
                self._resample_history = mono_samples[-margin:]
            else:
                self._resample_history = np.concatenate((self._resample_history, mono_samples))[-margin:]

            resampled_combined = scipy.signal.resample_poly(combined, up, down)

            # Extract the mathematically valid portion (without the history margin)
            expected_len = int(np.round(len(mono_samples) * target_sr / chunk.sample_rate))
            analysis_samples = resampled_combined[-expected_len:].astype(np.float32)
        else:
            analysis_samples = mono_samples

        # Update sliding window buffer (in-place shift, no allocation)
        n = len(analysis_samples)
        if n >= self.window_samples:
            self.audio_buffer[:] = analysis_samples[-self.window_samples:]
        else:
            self.audio_buffer[:-n] = self.audio_buffer[n:]
            self.audio_buffer[-n:] = analysis_samples

        # How many frames correspond to the current chunk
        chunk_frame_count = int(np.ceil(len(analysis_samples) / HOP_LENGTH))

        # --- Shared STFT calculation (incremental, cached) ---
        spectrogram = self._update_spectrogram(n)

        # --- RMS energy (direct scalar computation on current chunk) ---
        rms = float(np.sqrt(np.mean(analysis_samples ** 2)))

        # --- Multiband RMS energy (on current chunk spectrogram) ---
        spectrogram_current = spectrogram[:, -chunk_frame_count:]

        bass_band = spectrogram_current[:self._bin_bass_rms, :]
        mid_band = spectrogram_current[self._bin_bass_rms:self._bin_mid_rms, :]
        high_band = spectrogram_current[self._bin_mid_rms:, :]

        # Calibration factor (0.022) scales frequency-domain energy to match time-domain RMS
        calibration_factor = 0.022  # empirically tuned to scale frequency-domain energy to time-domain RMS range
        rms_bass = float(np.sqrt(np.mean(bass_band ** 2))) * calibration_factor if bass_band.size > 0 else 0.0
        rms_mid = float(np.sqrt(np.mean(mid_band ** 2))) * calibration_factor if mid_band.size > 0 else 0.0
        rms_high = float(np.sqrt(np.mean(high_band ** 2))) * calibration_factor if high_band.size > 0 else 0.0

        # Convert the full spectrogram to log-amplitude (dB) once to avoid redundant computations
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

        # --- Sub-band onset envelopes (vectorized spectral flux) ---
        # 1. Bass Onsets (frequencies < 250 Hz)
        onset_env_bass = self._band_onset_envelope(log_spectrogram[:self._bin_bass_onset, :])
        curr_onset_bass = (
            onset_env_bass[-chunk_frame_count:]
            if len(onset_env_bass) >= chunk_frame_count
            else onset_env_bass
        )
        onset_bass = float(np.max(curr_onset_bass)) if len(curr_onset_bass) > 0 else 0.0

        # 2. High Onsets (frequencies > 2000 Hz)
        onset_env_high = self._band_onset_envelope(log_spectrogram[self._bin_high_onset:, :])
        curr_onset_high = (
            onset_env_high[-chunk_frame_count:]
            if len(onset_env_high) >= chunk_frame_count
            else onset_env_high
        )
        onset_high = float(np.max(curr_onset_high)) if len(curr_onset_high) > 0 else 0.0

        # 2.5. Strobe Onsets (frequencies > STROBE_MIN_FREQUENCY_HERTZ)
        log_spec_strobe = log_spectrogram[self._bin_strobe:, :]
        if log_spec_strobe.size > 0:
            onset_env_strobe = self._band_onset_envelope(log_spec_strobe)
            curr_onset_strobe = (
                onset_env_strobe[-chunk_frame_count:]
                if len(onset_env_strobe) >= chunk_frame_count
                else onset_env_strobe
            )
            onset_strobe = float(np.max(curr_onset_strobe)) if len(curr_onset_strobe) > 0 else 0.0
        else:
            onset_strobe = 0.0

        # 3. Main Onset Envelope (up to 800 Hz) for global BPM tracking
        onset_env_main = self._band_onset_envelope(log_spectrogram[:self._bin_main_onset, :])
        curr_onset_main = (
            onset_env_main[-chunk_frame_count:]
            if len(onset_env_main) >= chunk_frame_count
            else onset_env_main
        )
        onset_main = float(np.max(curr_onset_main)) if len(curr_onset_main) > 0 else 0.0

        # --- Adaptive onset thresholding ---
        is_beat_bass = self._detect_onset(onset_bass, self.recent_onsets_bass, self._recent_onsets_long_bass, rms)
        is_beat_high = self._detect_onset(onset_high, self.recent_onsets_high, self._recent_onsets_long_high, rms)
        is_beat_strobe = self._detect_onset(onset_strobe, self.recent_onsets_strobe, self._recent_onsets_long_strobe,
                                            rms)

        # Apply FLASH_COOLDOWN_MULTIPLIER for selective beat accents
        cooldown_flash = float(np.clip(FLASH_COOLDOWN_MULTIPLIER * (60.0 / self.bpm), 0.50,
                                       1.80))  # cooldown range: 0.5s minimum to 1.8s maximum

        if is_beat_bass:
            if chunk.timestamp - self.last_beat_timestamp_bass >= cooldown_flash:
                self.last_beat_timestamp_bass = chunk.timestamp
            else:
                is_beat_bass = False

        if is_beat_high:
            if chunk.timestamp - self.last_beat_timestamp_high >= cooldown_flash:
                self.last_beat_timestamp_high = chunk.timestamp
            else:
                is_beat_high = False

        # Raw beats before cooldown and absolute floor restrictions
        threshold_bass = float(
            np.mean(self.recent_onsets_bass) + FLASH_ONSET_THRESHOLD_FACTOR * np.std(self.recent_onsets_bass)) if len(
            self.recent_onsets_bass) >= 5 else onset_bass + 1.0
        threshold_high = float(
            np.mean(self.recent_onsets_high) + FLASH_ONSET_THRESHOLD_FACTOR * np.std(self.recent_onsets_high)) if len(
            self.recent_onsets_high) >= 5 else onset_high + 1.0

        is_raw_beat_bass = onset_bass > threshold_bass
        is_raw_beat_high = onset_high > threshold_high
        is_raw_beat = is_raw_beat_bass or is_raw_beat_high

        # Logical OR for structural phrase change compatibility
        is_beat = is_beat_bass or is_beat_high

        # --- BPM tracking (accumulate main onset envelope) ---
        self.onset_history.extend(curr_onset_main.tolist())
        self.samples_since_bpm_update += len(analysis_samples)
        bpm_update_samples = int(BPM_UPDATE_INTERVAL_SECONDS * self.sample_rate)
        if self.samples_since_bpm_update >= bpm_update_samples:
            if len(self.onset_history) > 20:
                tempo = librosa.feature.rhythm.tempo(
                    onset_envelope=np.array(self.onset_history),
                    sr=self.sample_rate,
                    hop_length=HOP_LENGTH,
                )
                raw_bpm = float(tempo[0])
                # Octave regularization to the standard 90-180 BPM human range (safe check to prevent infinite loop)
                if raw_bpm > 30.0:
                    while raw_bpm < 90.0:
                        raw_bpm *= 2.0
                    while raw_bpm > 180.0:
                        raw_bpm /= 2.0
                else:
                    raw_bpm = 120.0

                # Apply median filter to smooth tempo estimates
                self.bpm_estimates.append(raw_bpm)
                self.bpm = float(np.median(self.bpm_estimates))
            self.samples_since_bpm_update = 0

        # --- Spectral centroid ---
        spectral_centroids = librosa.feature.spectral_centroid(
            S=spectrogram, sr=self.sample_rate, hop_length=HOP_LENGTH,
        )[0]
        current_spectral_centroids = (
            spectral_centroids[-chunk_frame_count:]
            if len(spectral_centroids) >= chunk_frame_count
            else spectral_centroids
        )
        centroid_hertz = (
            float(np.mean(current_spectral_centroids))
            if len(current_spectral_centroids) > 0
            else CENTROID_FREQUENCY_MINIMUM_HERTZ
        )
        log_centroid_hertz = np.log2(max(centroid_hertz, 1.0))
        centroid_norm = float(
            np.clip(
                (log_centroid_hertz - self.log_centroid_minimum_frequency)
                / (self.log_centroid_maximum_frequency - self.log_centroid_minimum_frequency),
                0.0,
                1.0,
            )
        )

        # --- Structural Segmentation (Novelty) ---
        mfcc = librosa.feature.mfcc(S=log_spectrogram, sr=self.sample_rate, n_mfcc=13)
        mfcc_chunk = np.mean(mfcc[:, -chunk_frame_count:], axis=1) if mfcc.shape[1] > 0 else np.zeros(13)

        self.mfcc_history_short.append(mfcc_chunk)
        self.mfcc_history_long.append(mfcc_chunk)

        is_boundary = False
        if len(self.mfcc_history_long) >= self.mfcc_history_long.maxlen * 0.4:  # At least 4 seconds of data
            short_avg = np.mean(self.mfcc_history_short, axis=0)
            long_avg = np.mean(self.mfcc_history_long, axis=0)

            norm_s = np.linalg.norm(short_avg)
            norm_l = np.linalg.norm(long_avg)

            if norm_s > 0 and norm_l > 0:
                cos_sim = np.dot(short_avg, long_avg) / (norm_s * norm_l)
                novelty = 1.0 - cos_sim
            else:
                novelty = 0.0

            self.novelty_history.append(novelty)

            # Dynamic threshold based on moving average + std dev
            threshold = np.mean(self.novelty_history) + 3.0 * np.std(self.novelty_history)
            threshold = max(threshold, 0.01)  # Lowered minimum noise floor to 0.01

            if novelty > threshold and (chunk.timestamp - self.last_boundary_time) > self.section_cooldown_seconds:
                is_boundary = True
                self.last_boundary_time = chunk.timestamp
                self.current_section_idx = (self.current_section_idx + 1) % len(self.section_letters)

        return AnalysisResult(
            timestamp=chunk.timestamp,
            rms=rms,
            rms_bass=rms_bass,
            rms_mid=rms_mid,
            rms_high=rms_high,
            onset_strength=onset_main,
            is_beat=is_beat,
            is_raw_beat=is_raw_beat,
            is_beat_bass=is_beat_bass,
            is_beat_high=is_beat_high,
            is_beat_strobe=is_beat_strobe,
            bpm=self.bpm,
            spectral_centroid=centroid_norm,
            is_section_boundary=is_boundary,
            current_section=self.section_letters[self.current_section_idx],
        )
