import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from NightScanPi.Program import audio_capture


def test_record_segment(tmp_path, monkeypatch):
    class DummyStream:
        def __init__(self):
            self.read_calls = 0
        def read(self, chunk, exception_on_overflow=False):
            self.read_calls += 1
            return b"\x00" * chunk
        def stop_stream(self):
            pass
        def close(self):
            pass
    class DummyPyAudio:
        def __init__(self):
            self.stream = DummyStream()
        def open(self, format, channels, rate, input=True, frames_per_buffer=0):
            return self.stream
        def get_sample_size(self, fmt):
            return 2
        def terminate(self):
            pass
    dummy = types.SimpleNamespace(paInt16=1, PyAudio=lambda: DummyPyAudio())
    monkeypatch.setattr(audio_capture, "pyaudio", dummy)

    out_path = tmp_path / "test.wav"
    audio_capture.record_segment(1, out_path)
    assert out_path.exists()
    assert out_path.read_bytes().startswith(b"RIFF")

