"""Audio recording utilities for NightScanPi with ReSpeaker Lite support."""
from __future__ import annotations

import wave
import logging
from pathlib import Path
from typing import Optional, Dict

import pyaudio

logger = logging.getLogger(__name__)

# Default audio settings (fallback)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

# ReSpeaker Lite optimal settings
RESPEAKER_RATE = 16000  # ReSpeaker Lite maximum sample rate
RESPEAKER_CHANNELS = 2  # Dual microphone array


def get_audio_config() -> Dict:
    """Get optimal audio configuration, preferring ReSpeaker Lite if available."""
    try:
        from .respeaker_detector import get_respeaker_config
        
        # Try to get ReSpeaker configuration
        respeaker_config = get_respeaker_config()
        if respeaker_config:
            logger.info(f"Using ReSpeaker Lite configuration: {respeaker_config['sample_rate']}Hz, {respeaker_config['channels']} channels")
            return {
                'device_id': respeaker_config.get('device_id'),
                'sample_rate': respeaker_config['sample_rate'],
                'channels': respeaker_config['channels'],
                'format': FORMAT,
                'chunk_size': respeaker_config.get('chunk_size', CHUNK),
                'is_respeaker': True
            }
    except ImportError:
        logger.debug("ReSpeaker detector not available")
    except Exception as e:
        logger.warning(f"Failed to detect ReSpeaker Lite: {e}")
    
    # Fallback to default configuration
    try:
        from .utils.pi_zero_optimizer import optimize_audio_buffer_size
        chunk_size = optimize_audio_buffer_size(CHUNK)
    except ImportError:
        chunk_size = CHUNK
    
    logger.info(f"Using default audio configuration: {RATE}Hz, {CHANNELS} channels")
    return {
        'device_id': None,  # Use default device
        'sample_rate': RATE,
        'channels': CHANNELS,
        'format': FORMAT,
        'chunk_size': chunk_size,
        'is_respeaker': False
    }


def get_optimal_chunk_size() -> int:
    """Get optimal chunk size for current system."""
    config = get_audio_config()
    return config['chunk_size']


def record_segment(duration: int, out_path: Path) -> None:
    """Record audio with automatic ReSpeaker Lite support and Pi Zero optimization."""
    out_path = Path(out_path)
    
    # Get optimal audio configuration (ReSpeaker Lite or fallback)
    audio_config = get_audio_config()
    
    device_id = audio_config['device_id']
    sample_rate = audio_config['sample_rate']
    channels = audio_config['channels']
    chunk_size = audio_config['chunk_size']
    is_respeaker = audio_config['is_respeaker']
    
    logger.info(f"Recording {duration}s audio: {sample_rate}Hz, {channels}ch, {chunk_size} buffer")
    if is_respeaker:
        logger.info("Using ReSpeaker Lite microphone array")
    
    try:
        # Use memory-optimized context for Pi Zero
        from .utils.pi_zero_optimizer import memory_optimized_operation
        
        with memory_optimized_operation():
            p = pyaudio.PyAudio()
            
            # Configure stream parameters
            stream_kwargs = {
                'format': FORMAT,
                'channels': channels,
                'rate': sample_rate,
                'input': True,
                'frames_per_buffer': chunk_size
            }
            
            # Use specific device if ReSpeaker detected
            if device_id is not None:
                stream_kwargs['input_device_index'] = device_id
            
            stream = p.open(**stream_kwargs)

            frames: list[bytes] = []
            total_frames = int(sample_rate / chunk_size * duration)
            
            logger.debug(f"Recording {total_frames} frames with {chunk_size} samples each")
            
            for i in range(total_frames):
                try:
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Periodic memory cleanup for long recordings on Pi Zero
                    if i % 100 == 0 and i > 0:  # Every ~4.5 seconds at default settings
                        try:
                            from .utils.pi_zero_optimizer import cleanup_memory
                            cleanup_memory()
                        except ImportError:
                            pass
                except Exception as e:
                    logger.warning(f"Frame {i} read error: {e}")
                    # Continue with next frame

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Convert stereo to mono if ReSpeaker but need mono output
            if is_respeaker and channels == 2:
                frames = convert_stereo_to_mono(frames, p.get_sample_size(FORMAT))
                output_channels = 1
                logger.debug("Converted ReSpeaker stereo to mono")
            else:
                output_channels = channels

            # Save the recording
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(output_channels)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(sample_rate)
                wf.writeframes(b"".join(frames))
            
            logger.info(f"Audio recorded successfully: {out_path}")
            logger.info(f"File specs: {sample_rate}Hz, {output_channels}ch, {len(frames)} frames")
    
    except ImportError:
        # Fallback to standard recording if optimizer not available
        logger.debug("Pi Zero optimizer not available, using standard recording")
        record_segment_standard(duration, out_path, audio_config)


def convert_stereo_to_mono(frames: list[bytes], sample_width: int) -> list[bytes]:
    """Convert stereo audio frames to mono by averaging channels."""
    import struct
    
    mono_frames = []
    
    if sample_width == 2:  # 16-bit samples
        format_char = 'h'  # signed short
    elif sample_width == 1:  # 8-bit samples
        format_char = 'b'  # signed char
    elif sample_width == 4:  # 32-bit samples
        format_char = 'i'  # signed int
    else:
        # Unsupported format, return original
        logger.warning(f"Unsupported sample width {sample_width}, keeping stereo")
        return frames
    
    for frame in frames:
        # Unpack stereo samples
        samples = struct.unpack(f'<{len(frame) // sample_width}{format_char}', frame)
        
        # Convert to mono by averaging left and right channels
        mono_samples = []
        for i in range(0, len(samples), 2):
            if i + 1 < len(samples):
                # Average left and right channel
                mono_sample = (samples[i] + samples[i + 1]) // 2
                mono_samples.append(mono_sample)
            else:
                # Odd sample, just use it
                mono_samples.append(samples[i])
        
        # Pack mono samples back to bytes
        mono_frame = struct.pack(f'<{len(mono_samples)}{format_char}', *mono_samples)
        mono_frames.append(mono_frame)
    
    return mono_frames


def record_segment_standard(duration: int, out_path: Path, audio_config: Dict) -> None:
    """Standard recording without Pi Zero optimizations."""
    device_id = audio_config['device_id']
    sample_rate = audio_config['sample_rate']
    channels = audio_config['channels']
    chunk_size = audio_config['chunk_size']
    
    p = pyaudio.PyAudio()
    
    stream_kwargs = {
        'format': FORMAT,
        'channels': channels,
        'rate': sample_rate,
        'input': True,
        'frames_per_buffer': chunk_size
    }
    
    if device_id is not None:
        stream_kwargs['input_device_index'] = device_id
    
    stream = p.open(**stream_kwargs)

    frames: list[bytes] = []
    total_frames = int(sample_rate / chunk_size * duration)
    
    for _ in range(total_frames):
        data = stream.read(chunk_size, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert stereo to mono if needed
    if channels == 2:
        frames = convert_stereo_to_mono(frames, p.get_sample_size(FORMAT))
        output_channels = 1
    else:
        output_channels = channels

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(output_channels)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))


if __name__ == "__main__":
    record_segment(8, Path("capture.wav"))
