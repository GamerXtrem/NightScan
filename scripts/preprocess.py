from pathlib import Path
import argparse

from pydub import AudioSegment


def isolate_cries(segment: AudioSegment) -> AudioSegment:
    """Return the segment containing animal cries.

    This is a placeholder implementation that simply returns the input
    audio without modification. Replace it with your own processing if
    needed.
    """
    return segment


def is_silent(segment: AudioSegment, threshold_db: float = -60.0) -> bool:
    """Check if an audio segment is silent.

    Parameters
    ----------
    segment : AudioSegment
        The audio segment to analyze.
    threshold_db : float, optional
        Silence threshold in dBFS. Defaults to -60 dBFS.

    Returns
    -------
    bool
        ``True`` if the segment has no RMS energy or its dBFS is below the
        threshold.
    """
    return segment.rms == 0 or segment.dBFS < threshold_db


def process_audio_files(input_dir: Path, output_dir: Path, pattern: str = "*.wav") -> None:
    """Process all audio files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in input_dir.glob(pattern):
        audio = AudioSegment.from_file(audio_path)

        # Apply cry isolation
        processed = isolate_cries(audio)

        wav_path = output_dir / (audio_path.stem + ".wav")
        if is_silent(processed):
            wav_path.unlink(missing_ok=True)
            continue
        processed.export(wav_path, format="wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess audio files")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing raw audio")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store processed WAVs")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern for audio files")
    args = parser.parse_args()

    process_audio_files(args.input_dir, args.output_dir, args.pattern)


if __name__ == "__main__":
    main()
