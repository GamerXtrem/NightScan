import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import generate_wake_tone


def test_generate_tone(tmp_path):
    out = tmp_path / "tone.wav"
    generate_wake_tone.generate_tone(out)
    data = out.read_bytes()
    assert data.startswith(b"RIFF")
    # Expect roughly 1 second of audio at 22050Hz, 16-bit mono
    # 44 byte header + 22050 * 2 bytes
    assert abs(len(data) - (44 + 22050 * 2)) < 100
