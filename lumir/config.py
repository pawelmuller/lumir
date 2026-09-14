"""Pipeline configuration constants with fully descriptive names."""

from lumir.models import EnergyStateStrategy

# Audio Input Configuration (for live mode)
AUDIO_INPUT_DEVICE_NAME = "BlackHole 2ch"  # Virtual audio loopback device name. Set to None to use the default microphone.

PLAYBACK_SAMPLE_RATE = 44100
ANALYSIS_SAMPLE_RATE = 22050
CHUNK_SAMPLES = 4096  # ~93 ms at 44100 Hz
ANALYSIS_WINDOW_SAMPLES = 8192  # ~372 ms window at 22050 Hz for features
HOP_LENGTH = 512

# Analysis
DEFAULT_BPM = 120.0
BPM_UPDATE_INTERVAL_SECONDS = 2.0
BPM_ANALYSIS_WINDOW_SECONDS = 12.0
ONSET_THRESHOLD_FACTOR = 1.5  # adaptive: mean + factor * std
FLASH_ONSET_THRESHOLD_FACTOR = 1.8  # selective threshold factor for strobe accents (higher = fewer flashes)
FLASH_COOLDOWN_MULTIPLIER = 1.5  # minimum beats between flashes (e.g. 1.5 beats to restrict to accents)
AVOLITES_FLASH_INTERVAL_BEATS = 4  # trigger flashes only on multiples of this beat count (e.g. 4 = downbeats/first beat of bar)
MINIMUM_ONSET_ABSOLUTE_THRESHOLD = 3.5  # avoid strobe on quiet/soft transients (vocals, synth pads)
MINIMUM_RMS_STROBE_THRESHOLD = 0.01  # avoid strobe in very quiet sections (volume gate)
ONSET_MAX_FREQUENCY_HERTZ = 800.0  # focus onset detection on lower bands (bass/kick/snare) to filter out vocals

# Multiband Frequency Divisions
BASS_MAX_FREQUENCY_HERTZ = 200.0
MID_MAX_FREQUENCY_HERTZ = 2000.0

# Mapping Gains
RMS_GAIN = 3.0
BASS_GAIN = 3.0
MID_GAIN = 4.5
HIGH_GAIN = 8.0

# Mapping Constants
CENTROID_FREQUENCY_MINIMUM_HERTZ = 200.0
CENTROID_FREQUENCY_MAXIMUM_HERTZ = 8000.0
BASE_BPM = 120.0  # baseline tempo for speed normalization (1.0 speed)
DIMMER_DECAY_COEFFICIENT = 0.15  # dimmer decay speed (0.0 = never decays, 1.0 = instant decay/no smoothing)
BASE_DIMMER_SMOOTHING_COEFFICIENT = 0.05  # slow smoothing for base background light
FLASH_DECAY_COEFFICIENT = 0.35  # beat flash decay speed (0.0 = never decays, 1.0 = instant decay)
MAX_BASE_DIMMER = 0.4  # maximum fader level for background volume (leaves headroom for beat flashes)

# Avolites Titan Web API Configuration
AVOLITES_TITAN_IP_ADDRESS = "192.168.0.1"
AVOLITES_TITAN_PORT = 4430

# Avolites Playbacks (User Numbers)
AVOLITES_PLAYBACK_BASS_USER_NUMBER = 7
AVOLITES_PLAYBACK_MID_USER_NUMBER = 6
AVOLITES_PLAYBACK_HIGH_USER_NUMBER = None  # Temporarily disabled (was 8)
AVOLITES_PLAYBACK_BASS_FLASH_NUMBER = 9
AVOLITES_PLAYBACK_HIGH_FLASH_NUMBER = 10
AVOLITES_PLAYBACK_BLINDER_NUMBER = 2
AVOLITES_PLAYBACK_STROBE = [1, 3, 4, 5, 18]

# Avolites Color Playbacks (Moods)
AVOLITES_PLAYBACK_COLOR_WARM_ACOUSTIC = 23
AVOLITES_PLAYBACK_COLOR_WARM_DRIVE = 24
AVOLITES_PLAYBACK_COLOR_COLD_AMBIENT = 25
AVOLITES_PLAYBACK_COLOR_COLD_IMPACT = 26
AVOLITES_PLAYBACK_COLOR_TENSION = 22
COLOR_MOOD_DEBOUNCE_SECONDS = 2.5

# Avolites Masters
AVOLITES_BPM_MASTER_INTENSITY_TITAN_ID = 1612
AVOLITES_BPM_MASTER_MOVEMENT_TITAN_ID = 1613
AVOLITES_SIZE_MASTER_TITAN_ID = 1621  # Titan ID of global Size Master (e.g. 1620) if using global master

# State-dependent BPM multipliers for Intensity
AVOLITES_BPM_MULTIPLIER_INTENSITY_LOW = 0.25  # quarter-tempo pulse in quiet parts
AVOLITES_BPM_MULTIPLIER_INTENSITY_MID = 0.5  # half-tempo in verse
AVOLITES_BPM_MULTIPLIER_INTENSITY_HIGH = 0.5  # half-tempo in drop (keeps chases readable and clean)

# State-dependent BPM multipliers for Movement
AVOLITES_BPM_MULTIPLIER_MOVEMENT_LOW = 0.0625  # extremely slow background drift
AVOLITES_BPM_MULTIPLIER_MOVEMENT_MID = 0.0625  # very slow verse movements
AVOLITES_BPM_MULTIPLIER_MOVEMENT_HIGH = 0.125  # slow movement on drop (quarter/eighth tempo)

# Energy state thresholds (smoothed normalized RMS bounds - fallbacks during startup)
ENERGY_THRESHOLD_LOW = 0.30
ENERGY_THRESHOLD_HIGH = 0.75

# Energy state percentile thresholds (for dynamic, self-calibrating classification)
ENERGY_PERCENTILE_LOW = 30  # bottom 30% of energy history is LOW state
ENERGY_PERCENTILE_HIGH = 65  # top 35% of energy history is HIGH state

# Size Master target levels for each energy state
SIZE_MASTER_LOW = 0.20
SIZE_MASTER_MID = 0.65
SIZE_MASTER_HIGH = 1.00

# Strategy for low/mid/high energy state classification
ENERGY_STATE_STRATEGY = EnergyStateStrategy.MULTI_FACTOR

# Avolites Playbacks lists for dynamic selection (not state-dependent)
AVOLITES_PLAYBACKS_INTENSITY = [11, 13, 14, 33]
AVOLITES_PLAYBACKS_MOVEMENT = [15, 16, 31, 32]
AVOLITES_PLAYBACKS_POSITION = [20, 21, 27, 28, 29, 30]

# Interval to rotate active playbacks from the lists (seconds)
PATTERN_CHANGE_INTERVAL_SECONDS = 30.0
STATE_TRANSITION_COOLDOWN_SECONDS = 3.0

# Tilt offset (height mapping) configuration
AVOLITES_PLAYBACK_TILT_USER_NUMBER = 17
TILT_SMOOTHING_COEFFICIENT = 0.02  # extremely slow, smooth motor glide (e.g. 0.02)

# Strobe Detection via Onset Density
STROBE_MIN_FREQUENCY_HERTZ = 1500.0
STROBE_DENSITY_WINDOW_SECONDS = 0.8
STROBE_DENSITY_THRESHOLD_BEATS = 5
STROBE_RMS_GATE = 0.50
STROBE_HYSTERESIS_SECONDS = 0.25
