"""Entry point: python -m lumir [audio_file]"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lumir",
        description="Real-time music → lighting pipeline (research prototype)",
    )
    parser.add_argument("file", nargs="?", help="Path to an audio file (mp3, wav, flac, …)")
    parser.add_argument("-l", "--live", action="store_true",
                        help="Capture live system audio or microphone in real-time")
    parser.add_argument("-p", "--playback", action="store_true",
                        help="Play audio through speakers while processing (file mode only)")
    parser.add_argument(
        "-o",
        "--output",
        choices=["console", "avolites"],
        default="console",
        help="Output backend (default: console)",
    )
    args = parser.parse_args()

    if not args.live and not args.file:
        print("Error: You must either specify a file to play OR use the --live flag to capture system audio.",
              file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    from lumir.pipeline import play
    play(
        file_path=args.file,
        live=args.live,
        playback=args.playback,
        output_backend=args.output,
    )


if __name__ == "__main__":
    main()
