"""
Reduce MFCC features to 4 values for the SVM and quantum models:
StandardScaler -> PCA (4) -> rescale to [0, pi] (angles for Ry gates).

Run: uv run python -m src.reduce
"""

import argparse
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.features import (
    PRESETS,
    PROCESSED_DIR,
    extract_feature_table,
    extract_mfcc,
    feature_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REDUCER_PATH = MODELS_DIR / "reducer.joblib"

DEFAULT_PRESET = "baseline"
DEFAULT_COMPONENTS = 4
ANGLE_RANGE = (0.0, np.pi)


def reduced_csv_path(n_components: int = DEFAULT_COMPONENTS) -> Path:
    return PROCESSED_DIR / f"features_pca{n_components}.csv"


def make_reducer(n_components: int = DEFAULT_COMPONENTS) -> Pipeline:
    """Standardize -> PCA -> rescale to [0, pi] (clip keeps new data in range)."""

    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("angle", MinMaxScaler(feature_range=ANGLE_RANGE, clip=True)),
    ])


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """
    Return True for test recordings. Holds out whole speakers if there
    are 2+, otherwise splits at random keeping the class balance.
    """

    indices = np.arange(len(df))
    speakers = df["speaker_id"].to_numpy()

    if len(np.unique(speakers)) >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=seed
        )
        _, test_idx = next(splitter.split(indices, groups=speakers))
    else:
        _, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=df["label"],
            random_state=seed,
        )

    is_test = np.zeros(len(df), dtype=bool)
    is_test[test_idx] = True
    return is_test


def fit_and_save(
    preset: str = DEFAULT_PRESET,
    n_components: int = DEFAULT_COMPONENTS,
    test_size: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit the reducer on training data and save it with the reduced features."""

    config = PRESETS[preset]
    names = feature_names(config)

    print(f"Extracting '{preset}' features ({len(names)} per recording)...")
    df = extract_feature_table(config, verbose=False)

    is_test = split_train_test(df, test_size, seed)
    X = df[names].to_numpy()

    reducer = make_reducer(n_components).fit(X[~is_test])
    X_reduced = reducer.transform(X)

    explained = reducer.named_steps["pca"].explained_variance_ratio_

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": reducer,
            "preset": preset,
            "feature_config": config,
            "n_components": n_components,
        },
        REDUCER_PATH,
    )

    out = df[["filename", "speaker_id", "label"]].copy()
    out["split"] = np.where(is_test, "test", "train")
    for i in range(n_components):
        out[f"pc{i + 1}"] = X_reduced[:, i]

    csv_path = reduced_csv_path(n_components)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)

    train_speakers = sorted(out.loc[~is_test, "speaker_id"].unique())
    test_speakers = sorted(out.loc[is_test, "speaker_id"].unique())

    print(f"\nTrain: {(~is_test).sum()} recordings, speakers {train_speakers}")
    print(f"Test:  {is_test.sum()} recordings, speakers {test_speakers}")
    if train_speakers == test_speakers:
        print("Note: same speaker(s) in train and test, so scores will be "
              "optimistic until more speakers are recorded.")

    print(f"\nPCA variance kept per component: "
          f"{', '.join(f'{v:.1%}' for v in explained)} "
          f"(total {explained.sum():.1%})")
    print(f"Values rescaled to [0, pi] for Ry(x) angle encoding.")
    print(f"\nSaved reducer to:  {REDUCER_PATH}")
    print(f"Saved features to: {csv_path}")

    return out


def load_reducer(path: Path = REDUCER_PATH) -> dict:
    """Load the reducer saved by fit_and_save()."""

    if not Path(path).is_file():
        raise FileNotFoundError(
            f"No reducer at {path}. Run: uv run python -m src.reduce"
        )
    return joblib.load(path)


def transform_audio(
    file_path: str | Path | BinaryIO,
    reducer: dict,
) -> np.ndarray:
    """Turn one recording into its 4 reduced features (web app, inference)."""

    features = extract_mfcc(file_path, reducer["feature_config"])
    return reducer["pipeline"].transform(features.reshape(1, -1))[0]


def load_reduced_features(n_components: int = DEFAULT_COMPONENTS):
    """Return X_train, X_test, y_train, y_test. All models use this split."""

    csv_path = reduced_csv_path(n_components)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"No reduced features at {csv_path}. "
            "Run: uv run python -m src.reduce"
        )

    df = pd.read_csv(csv_path)
    columns = [f"pc{i + 1}" for i in range(n_components)]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    return (
        train[columns].to_numpy(),
        test[columns].to_numpy(),
        train["label"].to_numpy(),
        test["label"].to_numpy(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fit PCA reduction for the SVM and quantum models."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help=f"Feature set to reduce (default: {DEFAULT_PRESET}).",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=DEFAULT_COMPONENTS,
        help="Number of PCA components, i.e. qubits (default: 4).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fit_and_save(args.preset, args.components, args.test_size, args.seed)


if __name__ == "__main__":
    main()
