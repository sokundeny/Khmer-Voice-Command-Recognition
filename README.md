# Khmer Voice Command Recognition

A Khmer-language voice command classification project comparing **classical machine learning** and **quantum machine learning** approaches.

The project aims to recognize four basic Khmer voice commands using audio feature extraction and three classification models: **SVM, QSVC, and VQC**.

## 1. Project Overview

This project investigates how classical and quantum machine learning models perform on a limited-vocabulary Khmer speech recognition task.

### Supported Commands

| Class | Khmer Word | Meaning |
|---|---|---|
| YES | បាទ, ចាស | Yes |
| NO | ទេ | No |
| GO | ទៅ | Go |
| STOP | ឈប់ | Stop |

There are **four classification classes** and **five spoken words**, since both បាទ and ចាស belong to the YES class.

### Objectives

- Build a Khmer voice command dataset with recordings from multiple speakers.
- Preprocess audio and extract Mel-Frequency Cepstral Coefficients (MFCCs).
- Train a classical Support Vector Machine (SVM) baseline.
- Implement Quantum Support Vector Classifier (QSVC) and Variational Quantum Classifier (VQC).
- Compare classification performance and robustness across the three models.

## 2. Project Structure

```text
khmer-voice-recognition/
├── data/
│   ├── raw/                    # Downloaded Hugging Face dataset
│   └── processed/              # Extracted features (planned)
├── scripts/
│   └── download_dataset.py
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── features.py
│   ├── reduce.py               # PCA reduction to qubit-sized features
│   ├── train_svm.py            # Planned
│   ├── train_qsvc.py           # Planned
│   ├── train_vqc.py            # Planned
│   └── evaluate.py             # Planned
├── notebooks/
│   └── exploration.ipynb       # Planned
├── models/                     # Saved trained models
├── results/                    # Evaluation outputs
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

The audio dataset is maintained separately on Hugging Face. Downloaded data, generated features, and trained model files should not be committed to GitHub.

## 3. Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) — Python package and project manager
- Git
- Hugging Face account with access to the dataset

## 4. Installation and Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_ORG/khmer-voice-recognition.git
cd khmer-voice-recognition
```

Replace `YOUR_GITHUB_ORG` with the actual GitHub organization or username.

### Step 2 — Install uv

**Windows (PowerShell):**

```powershell
winget install --id astral-sh.uv -e
```

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal if necessary and verify:

```bash
uv --version
```

### Step 3 — Install Project Dependencies

```bash
uv sync
```

This command automatically:

- Uses the project's configured Python version.
- Creates a local `.venv/` virtual environment.
- Installs the dependencies from `pyproject.toml` and `uv.lock`.

You do not need to activate the virtual environment manually when running commands through `uv run`.

## 5. Dataset

The dataset is hosted on Hugging Face:

**[Khmer Voice Commands Dataset](https://huggingface.co/datasets/school-project/khmer-voice-commands)**

It contains WAV recordings organized into four class folders:

```text
yes/
no/
go/
stop/
```

Recordings use the filename convention:

`{speaker_id}_{word}_{recording_number}.wav`

For recording requirements and contribution instructions, refer to the README in the Hugging Face dataset repository.

### Step 1 — Authenticate with Hugging Face

For a private dataset, authenticate with a token that has read access:

```bash
uv run hf auth login
```

### Step 2 — Download the Dataset

```bash
uv run python scripts/download_dataset.py
```

The script downloads the dataset into `data/raw/`.

Expected local structure:

```text
data/raw/
├── yes/
│   ├── speaker_001_bat_001.wav
│   └── speaker_001_chas_001.wav
├── no/
├── go/
├── stop/
└── README.md
```

The downloaded README is the Hugging Face dataset card.

**Note:** `data/raw/` is created automatically by the download process. You do not need to create it manually.

Run the same command again whenever you need to retrieve new recordings contributed by teammates.

## 6. Dataset Contribution

Team members can contribute new voice recordings to the Hugging Face dataset.

The contribution workflow is:

1. Obtain a unique speaker ID and write access to the Hugging Face dataset.
2. Download the latest dataset using `scripts/download_dataset.py`.
3. Add new WAV recordings to the correct folders inside `data/raw/`.
4. Upload the new recordings to Hugging Face.
5. Verify the contribution in **Files and versions**.

For example, from the project root:

```bash
uv run hf upload school-project/khmer-voice-commands ./data/raw . --repo-type dataset --commit-message "Add speaker_002 recordings"
```

This uploads the local dataset directory and creates a commit on Hugging Face automatically.

**Important:** Uploading the entire `data/raw/` directory may overwrite remote files if existing recordings have been modified locally. Contributors should avoid changing existing files and preferably upload only their own new recordings.

For detailed instructions, see the Hugging Face dataset README.

## 7. Machine Learning Pipeline

The planned processing pipeline is:

```text
Raw Khmer WAV recordings
          |
          v
Audio preprocessing
(resampling, silence trimming, normalization)
          |
          v
MFCC feature extraction
          |
          v
Speaker-independent dataset splitting
          |
          v
Dimensionality reduction
(PCA -> 4 features, rescaled to [0, pi])
          |
          +-------------------+
          |                   |
          v                   v
     Classical SVM       Quantum models
                          QSVC, VQC
          |                   |
          +---------+---------+
                    |
                    v
          Model evaluation
      (Accuracy, Precision,
        Recall, F1-score)
```

### Audio Preprocessing

Audio will be standardized to 16 kHz mono and processed to reduce inconsistencies between recordings.

### Feature Extraction

MFCCs are used to represent audio recordings numerically (`src/features.py`).

Base MFCC settings:

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| MFCC coefficients | 13 |
| Mel filters | 40 |
| FFT size | 400 samples (25 ms) |
| Hop length | 160 samples (10 ms) |
| Summary statistics | Mean and standard deviation |

Each MFCC matrix (13 coefficients × N frames) is summarized into a fixed-length vector. There are two feature sets:

| Preset | Description | Features |
|---|---|---|
| `baseline` (default) | Mean and std of 13 MFCCs over the whole recording | 13 × 2 = 26 |
| `segments4` | The same, computed separately for 4 equal time segments of the word | 13 × 2 × 4 = 104 |

Averaging over the whole recording loses the order of the sounds in a word. `segments4` keeps it by summarizing each quarter of the word separately. It is included as an optional alternative; the proposed pipeline uses `baseline`.

Extract a feature set to `data/processed/`:

```bash
uv run python -m src.features                        # baseline -> mfcc_features.csv
uv run python -m src.features --preset segments4     # -> mfcc_features_segments4.csv
```

### Dimensionality Reduction

The quantum models take one feature per qubit, so features are reduced before training (`src/reduce.py`):

```text
MFCC features (baseline, 26) -> StandardScaler -> PCA (4) -> rescale to [0, pi]
```

The final rescaling turns each value into a rotation angle for `Ry(x)` angle encoding. The SVM uses the same 4 values so all models are compared on identical inputs.

```bash
uv run python -m src.reduce                         # baseline, 4 components
uv run python -m src.reduce --preset segments4      # use the 104 segment features instead
```

This creates:

- `models/reducer.joblib`: the fitted transform and the feature settings it expects.
- `data/processed/features_pca4.csv`: reduced features with a `split` column (`train` / `test`).

The transform is fitted on the training recordings only. In training scripts:

```python
from src.reduce import load_reduced_features

X_train, X_test, y_train, y_test = load_reduced_features()
```

For a single recording (for example in the web app, or to measure inference time):

```python
from src.reduce import load_reducer, transform_audio

reducer = load_reducer()
x = transform_audio("recording.wav", reducer)  # 4 angles in [0, pi]
```

The first call takes a few seconds while librosa compiles; later calls take about 10 ms. Run one call to warm up before measuring inference time.

### Dataset Splitting

`src/reduce.py` holds out 20% of the data for testing. With two or more speakers, whole speakers are held out so that no speaker appears in both training and testing. With only one speaker, recordings are split at random with the same class balance in both sets.

All models should use the split from `load_reduced_features()` so results are comparable. For validation (for example, tuning the SVM's `C`), use cross-validation on the training set only and keep the test set for the final comparison.

## 8. Models

### Support Vector Machine (SVM)

The classical baseline will use scikit-learn's SVM implementation with the reduced MFCC features.

### Quantum Support Vector Classifier (QSVC)

QSVC will use a quantum kernel computed from the reduced features.

### Variational Quantum Classifier (VQC)

VQC will use a parameterized quantum circuit optimized for the four-class classification task.

## 9. Evaluation

Models will be compared using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Performance under background noise

Experiments will use consistent data splits to support fair comparisons.

## 10. Development Commands

Install or synchronize dependencies:

```bash
uv sync
```

Add a dependency:

```bash
uv add package-name
```

Run a Python script:

```bash
uv run python path/to/script.py
```

Download the latest dataset:

```bash
uv run python scripts/download_dataset.py
```

When dependencies change, commit both `pyproject.toml` and `uv.lock`.

## 11. Team Collaboration

- **Hugging Face:** Shared audio dataset and recording contributions
- **GitHub:** Source code, experiments, documentation, and collaboration
- **uv:** Python dependency and environment management

All contributors should follow the repository structure and keep generated datasets and model artifacts outside Git version control.