
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.preprocess import preprocess_audio, RAW_DIR, LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mfcc_features.csv"


def extract_mfcc(file_path: str | Path) -> np.ndarray:
    """
    Extract 26 MFCC-based features from one recording:
    - Mean of 13 MFCC coefficients
    - Standard deviation of 13 MFCC coefficients
    """

    audio, sr = preprocess_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_mels=40,
        n_fft=400,
        hop_length=160,
        center=False,
    )

    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
    ])

    return features.astype(np.float32)


def build_feature_dataset() -> pd.DataFrame:
    """
    Extract MFCC features from all recordings and save to CSV.
    """

    records = []

    for label in LABELS:
        folder = RAW_DIR / label

        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue

        for file_path in sorted(folder.glob("*.wav")):
            try:
                features = extract_mfcc(file_path)

                # The word portion can contain underscores, e.g. yes_bart.
                parts = file_path.stem.split("_")

                if len(parts) < 4 or parts[0] != "speaker" or not all(parts):
                    raise ValueError(
                        f"Invalid filename: {file_path.name}"
                    )

                speaker_id = "_".join(parts[:2])

                record = {
                    "filename": file_path.name,
                    "speaker_id": speaker_id,
                    "label": label,
                }

                for i in range(13):
                    record[f"mfcc_{i + 1}_mean"] = float(features[i])
                    record[f"mfcc_{i + 1}_std"] = float(features[i + 13])

                records.append(record)

                print(f"Extracted: {label}/{file_path.name}")

            except Exception as error:
                print(f"Failed: {file_path.name} - {error}")

    if not records:
        raise RuntimeError("No valid recordings found.")

    df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} recordings to: {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    build_feature_dataset()
