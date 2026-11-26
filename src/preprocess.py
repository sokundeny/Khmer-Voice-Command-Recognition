from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 16000
LABELS = ("yes", "no", "go", "stop")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def preprocess_audio(
    file_path: str | Path | BinaryIO,
    target_sr: int = TARGET_SR,
    trim_silence: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Load and standardize a voice recording.

    - Convert to mono
    - Resample to 16 kHz
    - Optionally trim leading/trailing silence
    - Return audio in memory (do not save WAV files)
    """

    name = "audio upload"
    if isinstance(file_path, (str, Path)):
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        name = file_path.name

    # Load audio and standardize sample rate/channels.
    audio, sr = librosa.load(
        file_path,
        sr=target_sr,
        mono=True,
    )

    # Check for invalid audio.
    if len(audio) == 0:
        raise ValueError(f"Empty recording: {name}")

    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Invalid audio values: {name}")

    duration = len(audio) / sr

    if duration < 0.3:
        raise ValueError(
            f"Recording too short ({duration:.2f}s): {name}"
        )

    if trim_silence:
        if not np.any(audio):
            raise ValueError(f"Silent recording: {name}")
        audio, _ = librosa.effects.trim(audio, top_db=25)

    return audio.astype(np.float32), sr


def validate_dataset():
    """
    Check all WAV files in data/raw without modifying them.
    """

    total = 0
    failed = 0

    for label in LABELS:
        folder = RAW_DIR / label

        if not folder.exists():
            print(f"[WARNING] Missing folder: {folder}")
            continue

        files = sorted(folder.glob("*.wav"))

        print(f"\n[{label.upper()}] {len(files)} recordings")

        for file_path in files:
            try:
                info = sf.info(file_path)

                audio, sr = preprocess_audio(file_path, trim_silence=True)

                duration = len(audio) / sr

                print(
                    f"  OK {file_path.name} "
                    f"| original={info.samplerate}Hz "
                    f"| processed={sr}Hz "
                    f"| duration={duration:.2f}s"
                )

                total += 1

            except Exception as error:
                failed += 1
                print(f"  ERROR {file_path.name}: {error}")

    print("\n--- Dataset Summary ---")
    print(f"Valid recordings: {total}")
    print(f"Failed recordings: {failed}")


if __name__ == "__main__":
    validate_dataset()
