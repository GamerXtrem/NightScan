"""Utility to generate the Wi-Fi wake up tone."""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import array
import math

RATE = 22050
FREQ = 2100
DURATION = 1.0


def generate_tone(out_path: Path, freq: int = FREQ, duration: float = DURATION, rate: int = RATE) -> None:
    """Write a ``.wav`` file containing a sine wave tone."""
    out_path = Path(out_path)
    count = int(duration * rate)
    scale = int(0.5 * 32767)
    samples = array.array(
        "h",
        (
            int(scale * math.sin(2 * math.pi * freq * i / rate))
            for i in range(count)
        ),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wake-up tone")
    parser.add_argument("out", nargs="?", default="wake_tone.wav")
    args = parser.parse_args()
    generate_tone(Path(args.out))


if __name__ == "__main__":
    main()
