"""Avolites Titan v19 Output Backend.

Sends lighting commands to Avolites Titan Web API over HTTP.
Uses a ThreadPoolExecutor to run requests asynchronously, preventing network latency
from blocking the real-time audio playback loop.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from lumir.config import (
    AVOLITES_PLAYBACK_BASS_USER_NUMBER,
    AVOLITES_PLAYBACK_HIGH_USER_NUMBER,
    AVOLITES_PLAYBACK_MID_USER_NUMBER,
    AVOLITES_PLAYBACK_BASS_FLASH_NUMBER,
    AVOLITES_PLAYBACK_HIGH_FLASH_NUMBER,
    AVOLITES_PLAYBACK_TILT_USER_NUMBER,
    AVOLITES_SIZE_MASTER_TITAN_ID,
    AVOLITES_BPM_MASTER_INTENSITY_TITAN_ID,
    AVOLITES_BPM_MASTER_MOVEMENT_TITAN_ID,
    AVOLITES_TITAN_IP_ADDRESS,
    AVOLITES_TITAN_PORT,
    AVOLITES_PLAYBACK_BLINDER_NUMBER,
    AVOLITES_PLAYBACK_STROBE,
)
from lumir.models import LightCommand
from lumir.output.base import Output

logger = logging.getLogger(__name__)


class AvolitesTitanOutput(Output):
    """Sends dimmer levels, active playbacks, tilt offset, and masters to Titan console."""

    def __init__(self) -> None:
        self.ip_address = AVOLITES_TITAN_IP_ADDRESS
        self.port = AVOLITES_TITAN_PORT
        self.base_url = f"http://{self.ip_address}:{self.port}/titan"

        # Continuous background playbacks (smooth swells)
        self.bass_user_number = AVOLITES_PLAYBACK_BASS_USER_NUMBER
        self.mid_user_number = AVOLITES_PLAYBACK_MID_USER_NUMBER
        self.high_user_number = AVOLITES_PLAYBACK_HIGH_USER_NUMBER

        # Momentary flash playbacks (console-side fade-outs using Release Time)
        self.bass_flash_number = AVOLITES_PLAYBACK_BASS_FLASH_NUMBER
        self.high_flash_number = AVOLITES_PLAYBACK_HIGH_FLASH_NUMBER
        self.blinder_number = AVOLITES_PLAYBACK_BLINDER_NUMBER
        self.strobe_playbacks = AVOLITES_PLAYBACK_STROBE

        # Tilt offset (height mapping) playback fader
        self.tilt_user_number = AVOLITES_PLAYBACK_TILT_USER_NUMBER

        # Masters
        self.size_master_titan_id = AVOLITES_SIZE_MASTER_TITAN_ID
        self.bpm_master_intensity_id = AVOLITES_BPM_MASTER_INTENSITY_TITAN_ID
        self.bpm_master_movement_id = AVOLITES_BPM_MASTER_MOVEMENT_TITAN_ID

        # Persistent HTTP session for Keep-Alive connection reuse
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

        # Thread pool for non-blocking asynchronous HTTP requests
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Debouncing state
        self.debounce_threshold = 0.005
        self.debounce_interval = 0.05
        self._last_sent_values: dict[str, float] = {}
        self._last_sent_times: dict[str, float] = {}

        # State tracking to minimize unnecessary API requests
        self.last_sent_bpm_intensity: float | None = None
        self.last_sent_bpm_movement: float | None = None
        self.last_strobe_bass_state: bool = False
        self.last_strobe_high_state: bool = False
        self.last_blinder_state: bool = False
        self.is_strobe_active: bool = False
        self.current_strobe_playback: int | None = None

        # Look selection tracking
        self.current_intensity_playback: int | None = None
        self.current_movement_playback: int | None = None
        self.current_position_playback: int | None = None
        self.current_color_playback: int | None = None

        logger.info(f"Connected to Titan API at {self.ip_address}:{self.port}")

    def _send_request(self, url: str, params: dict, timeout: float) -> None:
        """Executes the HTTP GET request. Runs inside the ThreadPoolExecutor."""
        try:
            # Overriding passed timeout to use relaxed connect/read tuple (0.1, 0.25)
            response = self.session.get(url, params=params, timeout=(0.1, 0.25))
            response.raise_for_status()
        except Exception as error:
            # Print error but don't raise to avoid crashing background threads
            logger.warning(f"API request failed: {error}")

    def _set_playback_level(self, user_number: int | None, level: float, force: bool = False) -> None:
        """Submit playback fader level change to background thread pool."""
        if user_number is None:
            return
        clamped_level = max(0.0, min(level, 1.0))

        key = f"playback_{user_number}"
        current_time = time.time()

        last_val = self._last_sent_values.get(key)
        last_time = self._last_sent_times.get(key, 0.0)

        if not force and last_val is not None:
            # Only send if absolute change exceeds threshold AND interval has passed
            if abs(clamped_level - last_val) < self.debounce_threshold or (
                    current_time - last_time) < self.debounce_interval:
                return

        self._last_sent_values[key] = clamped_level
        self._last_sent_times[key] = current_time

        url = f"{self.base_url}/script/Playbacks/FirePlaybackAtLevel"
        params = {
            "userNumber": user_number,
            "level": f"{clamped_level:.4f}",
            "bool": "false"  # alwaysRefire = false
        }
        self.executor.submit(self._send_request, url, params, 0.1)

    def _kill_playback(self, user_number: int | None) -> None:
        """Submit playback release (kill) command to background thread pool."""
        if user_number is None:
            return
        url = f"{self.base_url}/script/Playbacks/KillPlayback"
        params = {"userNumber": user_number}
        self.executor.submit(self._send_request, url, params, 0.1)

    def _trigger_go(self, user_number: int | None) -> None:
        """Submit CueList Play (Go) command to background thread pool."""
        if user_number is None:
            return
        url = f"{self.base_url}/script/CueLists/Play"
        params = {"userNumber": user_number}
        self.executor.submit(self._send_request, url, params, 0.1)

    def _set_bpm_master(self, titan_id: int, bpm: float) -> None:
        """Submit BPM master update command to background thread pool."""
        url = f"{self.base_url}/script/2/Masters/SetMaster"
        params = {
            "handle_titanId": titan_id,
            "value": f"{bpm:.2f}"
        }
        self.executor.submit(self._send_request, url, params, 0.2)

    def _set_size_master(self, titan_id: int, level: float, force: bool = False) -> None:
        """Submit Size Master level change to background thread pool."""
        clamped_level = max(0.0, min(level, 1.0))

        key = f"size_master_{titan_id}"
        current_time = time.time()

        last_val = self._last_sent_values.get(key)
        last_time = self._last_sent_times.get(key, 0.0)

        if not force and last_val is not None:
            if abs(clamped_level - last_val) < self.debounce_threshold or (
                    current_time - last_time) < self.debounce_interval:
                return

        self._last_sent_values[key] = clamped_level
        self._last_sent_times[key] = current_time

        url = f"{self.base_url}/script/2/Masters/SetMaster"
        params = {
            "handle_titanId": titan_id,
            "value": f"{clamped_level:.4f}"
        }
        self.executor.submit(self._send_request, url, params, 0.1)

    def send(self, command: LightCommand) -> None:
        # Update continuous background playbacks
        self._set_playback_level(self.bass_user_number, command.dimmer_bass)
        self._set_playback_level(self.mid_user_number, command.dimmer_mid)
        self._set_playback_level(self.high_user_number, command.dimmer_high)

        # Update Size Master level
        if self.size_master_titan_id is not None:
            self._set_size_master(self.size_master_titan_id, command.size_master_level)

        # Update Tilt Offset fader
        self._set_playback_level(self.tilt_user_number, command.tilt_level)

        # Update active Intensity playback level
        if command.previous_intensity_playback_to_kill is not None:
            self._kill_playback(command.previous_intensity_playback_to_kill)
            if command.active_intensity_playback is None:
                self.current_intensity_playback = None

        if command.active_intensity_playback is not None:
            # Fire/Update fader level of the active intensity chase
            self._set_playback_level(command.active_intensity_playback, command.dimmer)
            self.current_intensity_playback = command.active_intensity_playback

        # Update active Movement playback level
        if command.previous_movement_playback_to_kill is not None:
            self._kill_playback(command.previous_movement_playback_to_kill)
            if command.active_movement_playback is None:
                self.current_movement_playback = None

        if command.active_movement_playback is not None:
            if command.active_movement_playback != self.current_movement_playback:
                self._set_playback_level(command.active_movement_playback, 1.0, force=True)
                self.current_movement_playback = command.active_movement_playback

        # Update active Position playback level (kept at 100%)
        if command.active_position_playback is not None:
            if command.active_position_playback != self.current_position_playback:
                self._set_playback_level(command.active_position_playback, 1.0, force=True)
                self.current_position_playback = command.active_position_playback

        # Handle Bass Flash
        if command.strobe_bass:
            if not self.last_strobe_bass_state:
                self._set_playback_level(self.bass_flash_number, 1.0, force=True)
                self.last_strobe_bass_state = True
        else:
            if self.last_strobe_bass_state:
                self._kill_playback(self.bass_flash_number)
                self.last_strobe_bass_state = False

        # Handle High Flash
        if command.strobe_high:
            if not self.last_strobe_high_state:
                self._set_playback_level(self.high_flash_number, 1.0, force=True)
                self.last_strobe_high_state = True
        else:
            if self.last_strobe_high_state:
                self._kill_playback(self.high_flash_number)
                self.last_strobe_high_state = False

        # Handle Blinders
        if command.blinder_active:
            if not self.last_blinder_state:
                self._set_playback_level(self.blinder_number, 1.0, force=True)
                self.last_blinder_state = True
        else:
            if self.last_blinder_state:
                self._kill_playback(self.blinder_number)
                self.last_blinder_state = False

        # Handle Strobe
        if getattr(command, 'trigger_strobe', False):
            if not self.is_strobe_active:
                if self.strobe_playbacks:
                    self.current_strobe_playback = random.choice(self.strobe_playbacks)
                    self._set_playback_level(self.current_strobe_playback, 1.0, force=True)
                self.is_strobe_active = True
        else:
            if self.is_strobe_active:
                if self.current_strobe_playback is not None:
                    self._kill_playback(self.current_strobe_playback)
                    self.current_strobe_playback = None
                self.is_strobe_active = False

        # Handle Color Cuelist
        if command.active_color_playback is not None:
            if command.active_color_playback != self.current_color_playback:
                self._set_playback_level(command.active_color_playback, 1.0, force=True)
                self.current_color_playback = command.active_color_playback

            if command.trigger_color_go:
                self._trigger_go(self.current_color_playback)

        # Update BPM Masters
        current_bpm_intensity = round(command.bpm_intensity, 1)
        if self.bpm_master_intensity_id is not None:
            if self.last_sent_bpm_intensity is None or current_bpm_intensity != self.last_sent_bpm_intensity:
                self._set_bpm_master(self.bpm_master_intensity_id, current_bpm_intensity)
                self.last_sent_bpm_intensity = current_bpm_intensity

        current_bpm_movement = round(command.bpm_movement, 1)
        if self.bpm_master_movement_id is not None:
            if self.last_sent_bpm_movement is None or current_bpm_movement != self.last_sent_bpm_movement:
                self._set_bpm_master(self.bpm_master_movement_id, current_bpm_movement)
                self.last_sent_bpm_movement = current_bpm_movement

    def close(self) -> None:
        """Release resources and perform blackout on exit."""
        # Blackout on exit: release all playbacks (washes, strobes, active patterns, tilt)
        playbacks_to_kill = [
            self.bass_user_number,
            self.mid_user_number,
            self.high_user_number,
            self.bass_flash_number,
            self.high_flash_number,
            self.tilt_user_number,
        ]
        if self.current_intensity_playback is not None:
            playbacks_to_kill.append(self.current_intensity_playback)
        if self.current_movement_playback is not None:
            playbacks_to_kill.append(self.current_movement_playback)
        if self.current_position_playback is not None:
            playbacks_to_kill.append(self.current_position_playback)
        if self.current_color_playback is not None:
            playbacks_to_kill.append(self.current_color_playback)
        if self.current_strobe_playback is not None:
            playbacks_to_kill.append(self.current_strobe_playback)

        for user_number in playbacks_to_kill:
            if user_number is None:
                continue
            try:
                # Use synchronous requests here to make sure they complete before exit
                self.session.get(
                    f"{self.base_url}/script/Playbacks/KillPlayback",
                    params={"userNumber": user_number},
                    timeout=0.2
                )
            except Exception:
                pass

        # Reset Size Master to 0.0
        if self.size_master_titan_id is not None:
            try:
                self.session.get(
                    f"{self.base_url}/script/2/Masters/SetMaster",
                    params={"handle_titanId": self.size_master_titan_id, "value": "0.0000"},
                    timeout=0.2
                )
            except Exception:
                pass

        self.executor.shutdown(wait=False)
        self.session.close()
        logger.info("Blackout complete, session closed and executor shutdown.")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
