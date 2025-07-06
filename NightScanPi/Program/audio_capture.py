"""Audio recording utilities for NightScanPi."""
from __future__ import annotations

import wave
import logging
from pathlib import Path

import pyaudio

logger = logging.getLogger(__name__)

# Default audio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050


def get_optimal_chunk_size() -> int:
    """Get optimal chunk size for current system."""
    try:
        from .utils.pi_zero_optimizer import optimize_audio_buffer_size
        return optimize_audio_buffer_size(CHUNK)
    except ImportError:
        return CHUNK


def record_segment(duration: int, out_path: Path) -> None:
    """Record ``duration`` seconds of audio and write a WAV file with Pi Zero optimization."""
    out_path = Path(out_path)
    
    # Get optimized chunk size for Pi Zero
    optimal_chunk = get_optimal_chunk_size()
    
    try:
        # Use memory-optimized context for Pi Zero
        from .utils.pi_zero_optimizer import memory_optimized_operation
        
        with memory_optimized_operation():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT, 
                channels=CHANNELS, 
                rate=RATE, 
                input=True, 
                frames_per_buffer=optimal_chunk
            )

            frames: list[bytes] = []
            total_frames = int(RATE / optimal_chunk * duration)
            
            logger.debug(f"Recording {duration}s audio with {optimal_chunk} chunk size ({total_frames} frames)")
            
            for i in range(total_frames):
                data = stream.read(optimal_chunk, exception_on_overflow=False)
                frames.append(data)
                
                # Periodic memory cleanup for long recordings on Pi Zero
                if i % 100 == 0 and i > 0:  # Every ~4.5 seconds at default settings
                    try:
                        from .utils.pi_zero_optimizer import cleanup_memory
                        cleanup_memory()
                    except ImportError:
                        pass

            stream.stop_stream()
            stream.close()
            p.terminate()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(frames))
            
            logger.info(f"Audio recorded successfully: {out_path}")
    
    except ImportError:
        # Fallback to standard recording if optimizer not available
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=optimal_chunk)

        frames: list[bytes] = []
        for _ in range(int(RATE / optimal_chunk * duration)):
            data = stream.read(optimal_chunk, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))


if __name__ == "__main__":
    record_segment(8, Path("capture.wav"))
