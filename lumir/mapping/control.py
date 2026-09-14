import random

from lumir import config
from lumir.models import AnalysisResult, LightCommand


class RandomControlMapper:
    def __init__(self, target_bpm: float):
        self.target_bpm = target_bpm
        self.beat_interval = 60.0 / target_bpm
        self.last_beat_time = 0.0
        self.beat_count = 0

        self.colors = [
            config.AVOLITES_PLAYBACK_COLOR_WARM_ACOUSTIC,
            config.AVOLITES_PLAYBACK_COLOR_WARM_DRIVE,
            config.AVOLITES_PLAYBACK_COLOR_COLD_AMBIENT,
            config.AVOLITES_PLAYBACK_COLOR_COLD_IMPACT,
            config.AVOLITES_PLAYBACK_COLOR_TENSION,
        ]

        self.current_color = random.choice(self.colors)
        self.current_intensity = random.choice(config.AVOLITES_PLAYBACKS_INTENSITY)
        self.current_movement = random.choice(config.AVOLITES_PLAYBACKS_MOVEMENT)

    def translate(self, result: AnalysisResult) -> LightCommand:
        # Generate an artificial beat based on timestamp and target_bpm
        is_beat = False
        if result.timestamp - self.last_beat_time >= self.beat_interval:
            is_beat = True
            self.last_beat_time = result.timestamp
            self.beat_count += 1

            # Randomize colors, intensity, and movement every 16 beats (4 bars)
            if self.beat_count % 16 == 0:
                self.current_color = random.choice(self.colors)
                self.current_intensity = random.choice(config.AVOLITES_PLAYBACKS_INTENSITY)
                self.current_movement = random.choice(config.AVOLITES_PLAYBACKS_MOVEMENT)

        # Randomize dimmer based on beat (e.g., high on beat, low between)
        dimmer = 1.0 if is_beat else 0.3

        return LightCommand(
            timestamp=result.timestamp,
            dimmer=dimmer,
            dimmer_bass=dimmer,
            dimmer_mid=dimmer,
            dimmer_high=dimmer,
            strobe_bass=False,
            strobe_high=False,
            trigger_strobe=False,
            blinder_active=False,
            bpm_intensity=self.target_bpm,
            bpm_movement=self.target_bpm,
            active_intensity_playback=self.current_intensity,
            active_movement_playback=self.current_movement,
            active_position_playback=None,
            size_master_level=0.5,
            tilt_level=0.5,
            active_color_playback=self.current_color,
            trigger_color_go=is_beat and (self.beat_count % 16 == 0),
            energy_state="CONTROL",
        )
