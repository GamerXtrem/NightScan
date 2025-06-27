"""Audio recording utilities for NightScanPi."""
from __future__ import annotations

import wave
from pathlib import Path

import pyaudio


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def record_segment(duration: int, out_path: Path) -> None:
    """Record ``duration`` seconds of audio and write a WAV file."""
    out_path = Path(out_path)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames: list[bytes] = []
    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))


if __name__ == "__main__":
    record_segment(8, Path("capture.wav"))
