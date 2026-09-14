"""Console output — prints LightCommands with multiband details to the terminal."""

from __future__ import annotations

from lumir.models import LightCommand
from lumir.output.base import Output


class ConsoleLogger(Output):
    """Human-readable one-line-per-command terminal output with multiband values."""

    def send(self, command: LightCommand) -> None:
        strobe_bass_str = "BASS_STROBE" if command.strobe_bass else "           "
        strobe_high_str = "HIGH_STROBE" if command.strobe_high else "           "
        print(
            f"[{command.timestamp:6.2f}s] "
            f"bass={command.dimmer_bass:.2f}  "
            f"mid={command.dimmer_mid:.2f}  "
            f"high={command.dimmer_high:.2f}  "
            f"{strobe_bass_str} | {strobe_high_str}  "
            f"bpm_int={command.bpm_intensity:.1f}  "
            f"bpm_mov={command.bpm_movement:.1f}"
        )
