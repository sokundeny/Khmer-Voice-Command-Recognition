import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import librosa
import numpy as np
import pandas as pd

from src.preprocess import preprocess_audio, RAW_DIR, LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "mfcc_features.csv"

@dataclass(frozen=True)
class FeatureConfig:
    """
    Settings for MFCC feature extraction.

    n_mfcc       Number of MFCC coefficients computed per frame.
    n_mels       Number of mel filters.
    n_fft        Frame length in samples (400 = 25 ms at 16 kHz).
    hop_length   Step between frames in samples (160 = 10 ms at 16 kHz).
    n_segments   Split each recording into this many equal time segments
                 and summarize each one separately, so the features keep
                 the order of sounds in the word. 1 = whole recording.
    """

    n_mfcc:     int = 13
    n_mels:     int = 40
    n_fft:      int = 400
    hop_length: int = 160
    n_segments: int = 1

    def __post_init__(self):
        if self.n_segments < 1:
            raise ValueError("n_segments must be at least 1")


# Feature sets.
# baseline:  mean and std of 13 MFCCs over the whole recording
#            = 13 x 2 = 26 features (the proposed pipeline).
# segments4: the same, computed separately for 4 time segments
#            = 13 x 2 x 4 = 104 features (tested improvement).
PRESETS = {
    "baseline": FeatureConfig(),
    "segments4": FeatureConfig(n_segments=4),
}


def feature_names(config: FeatureConfig = FeatureConfig()) -> list[str]:
    """
    Column names in the same order as compute_features() output.

    Coefficients are numbered from 0, matching c0, c1, ... and the rows
    of the MFCC matrix: mfcc_0_mean, mfcc_0_std, ..., mfcc_12_std.
    With several segments, names get a prefix: seg1_mfcc_0_mean, ...
    """

    names = []
    for segment in range(1, config.n_segments + 1):
        prefix = f"seg{segment}_" if config.n_segments > 1 else ""

        for coefficient in range(config.n_mfcc):
            names.append(f"{prefix}mfcc_{coefficient}_mean")
            names.append(f"{prefix}mfcc_{coefficient}_std")

    return names


def compute_features(
    audio: np.ndarray,
    sr: int,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """
    Turn preprocessed audio into a fixed-length MFCC feature vector.
    """

    # Pad very short audio so there is at least one frame per segment.
    min_length = config.n_fft + config.hop_length * (config.n_segments - 1)
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)))

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.n_mfcc,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        center=False,
    )

    # mfcc has shape (n_mfcc, frames). Split the frames into equal
    # segments, e.g. 43 frames and 4 segments -> [0, 10, 21, 32, 43].
    frames = mfcc.shape[1]
    boundaries = np.linspace(0, frames, config.n_segments + 1).astype(int)

    parts = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = mfcc[:, start:end]

        # Mean and std of each coefficient over the segment's frames,
        # paired per coefficient to match feature_names():
        # [mean_0, std_0, mean_1, std_1, ...]
        summary = np.stack(
            [segment.mean(axis=1), segment.std(axis=1)],
            axis=1,
        )
        parts.append(summary.ravel())

    return np.concatenate(parts).astype(np.float32)


def extract_mfcc(
    file_path: str | Path | BinaryIO,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """
    Load one recording and extract its MFCC feature vector.

    With the default config this returns the original 26 features:
    mean and standard deviation of 13 MFCC coefficients.
    """

    audio, sr = preprocess_audio(file_path, trim_silence=True)
    return compute_features(audio, sr, config)


def parse_filename(file_path: Path) -> str:
    """
    Return the speaker ID from {speaker_id}_{word}_{number}.wav.

    The word portion can contain underscores, e.g. yes_bart.
    """

    parts = file_path.stem.split("_")

    if len(parts) < 4 or parts[0] != "speaker" or not all(parts):
        raise ValueError(f"Invalid filename: {file_path.name}")

    return "_".join(parts[:2])


def list_recordings() -> list[tuple[Path, str]]:
    """
    Return (file_path, label) for every WAV file in data/raw.
    """

    recordings = []

    for label in LABELS:
        folder = RAW_DIR / label

        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        for file_path in sorted(folder.glob("*.wav")):
            recordings.append((file_path, label))

    return recordings


def extract_feature_table(
    config: FeatureConfig = FeatureConfig(),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features from all recordings in data/raw.

    Returns one row per recording with the columns filename, speaker_id,
    label, then one column per feature (see feature_names()).
    """

    names = feature_names(config)
    records = []

    for file_path, label in list_recordings():
        try:
            speaker_id = parse_filename(file_path)
            features = extract_mfcc(file_path, config)

            record = {
                "filename": file_path.name,
                "speaker_id": speaker_id,
                "label": label,
            }
            record.update(zip(names, features.tolist()))

            records.append(record)

            if verbose:
                print(f"Extracted: {label}/{file_path.name}")

        except Exception as error:
            print(f"Failed: {file_path.name} - {error}")

    if not records:
        raise RuntimeError("No valid recordings found.")

    return pd.DataFrame(records)


def build_feature_dataset(
    config: FeatureConfig = FeatureConfig(),
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Extract MFCC features from all recordings and save to CSV.
    """

    names = feature_names(config)
    df = extract_feature_table(config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"\nSaved {len(df)} recordings x {len(names)} features "
        f"to: {output_path}"
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract MFCC features to a CSV file."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="baseline",
        help="Feature set to extract (default: baseline).",
    )
    args = parser.parse_args()

    if args.preset == "baseline":
        output_path = OUTPUT_PATH
    else:
        output_path = PROCESSED_DIR / f"mfcc_features_{args.preset}.csv"

    build_feature_dataset(PRESETS[args.preset], output_path)


if __name__ == "__main__":
    main()
