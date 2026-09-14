"""Working entrypoint for the lumir pipeline."""

from lumir.pipeline import play

if __name__ == "__main__":
    play(
        live=True,
        output_backend="console"
    )
