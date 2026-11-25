---
language:
  - km
pretty_name: Khmer Voice Commands
task_categories:
  - audio-classification
tags:
  - khmer
  - speech
  - voice-commands
---

# Khmer Voice Commands Dataset

A collaborative Khmer voice command dataset for classical and quantum machine learning research.

This dataset contains four command classes: **YES, NO, GO, and STOP**, covering five Khmer spoken words.

## 1. Dataset Structure

```text
khmer-voice-commands/
├── yes/
│   ├── speaker_001_bat_001.wav
│   └── speaker_001_chas_001.wav
├── no/
│   └── speaker_001_no_001.wav
├── go/
│   └── speaker_001_go_001.wav
├── stop/
│   └── speaker_001_stop_001.wav
└── README.md
```

## 2. Commands and Recording Requirements

| Label | Khmer Word | Filename Keyword |
|---|---|---|
| YES | បាទ | yes_bart |
| YES | ចាស | yes_jas |
| NO | ទេ | no |
| GO | ទៅ | go |
| STOP | ឈប់ | stop |


Audio requirements:

- Format: WAV (`.wav`)
- Sample rate: 16,000 Hz
- Channels: Mono
- Recommended duration: 1–2 seconds
- Record one command per file
- Prefer a quiet recording environment

## 3. Filename Convention

Use:

`{speaker_id}_{word}_{recording_number}.wav`

Examples:

```text
speaker_001_bat_001.wav
speaker_001_chas_001.wav
speaker_001_no_001.wav
speaker_001_go_001.wav
speaker_001_stop_001.wav
```

Each contributor must use a unique speaker ID, such as `speaker_001`, `speaker_002`, or `speaker_003`.

Keep the same speaker ID across all recordings. Do not include personal names in filenames.

## 4. How to Contribute

### Step 1 — Join the organization

1. Create a Hugging Face account at https://huggingface.co/join.
2. Send your Hugging Face username to the dataset maintainer.
3. Wait for the maintainer to add you to the organization with write access.
4. Obtain your assigned speaker ID.

### Step 2 — Record your voice commands

Record 40 WAV files following the recording requirements above.

Organize them locally into four folders:

```text
my-recordings/
├── yes/
│   ├── speaker_002_bat_001.wav
│   └── speaker_002_chas_001.wav
├── no/
│   └── speaker_002_no_001.wav
├── go/
│   └── speaker_002_go_001.wav
└── stop/
    └── speaker_002_stop_001.wav
```

The filenames shown are examples; complete the required number of recordings for each word.

### Step 3 — Upload recordings through the website

This is the easiest method for contributors who do not use Git or Python.

1. Open the team's Hugging Face dataset repository.
2. Navigate to **Files and versions**.
3. Open the correct folder (`yes`, `no`, `go`, or `stop`).
4. Select **Add file → Upload files**.
5. Select your WAV recordings.
6. Commit the changes.
7. Repeat for the remaining command folders.

Verify that your uploaded files appear in the correct folders.

### Alternative — Upload using the CLI

Install the Hugging Face CLI using uv:

```bash
uv tool install huggingface_hub
```

Authenticate:

```bash
hf auth login
```

Download the existing dataset **5**

Open the downloaded khmer-voice-commands/ folder and add your WAV files to the existing yes/, no/, go/, and stop/ folders and add your recordings

Use your assigned speaker ID and follow the filename convention.

```bash
hf upload school-project/khmer-voice-commands ./data/raw . --repo-type dataset --commit-message "Add speaker_002 recordings"
```

## 5. How to Download the Dataset

### Option A — Download using the CLI

Install the Hugging Face CLI:

```bash
uv tool install huggingface_hub
```

If the dataset is private, authenticate:

```bash
hf auth login
```

Download the dataset:

```bash
hf download school-project/khmer-voice-commands --repo-type dataset --local-dir data/raw
```

This creates the local `data/raw/` directory containing:

```text
data/raw/
├── yes/
├── no/
├── go/
├── stop/
└── README.md
```

The README is included because it is part of the dataset repository.

### Option B — Load directly in Python

Install the required package:

```bash
pip install "datasets[audio]"
```

If the dataset is private, authenticate using `hf auth login` first.

```bash
uv run python scripts/download_dataset.py       
```

The `train` split initially represents the uploaded collection. It is not necessarily the final training split used for experiments.

Folder names are used as classification labels when the dataset is loaded through the AudioFolder format.

## 6. Contribution Guidelines

- Use your assigned speaker ID.
- Follow the required filename format.
- Place recordings in the correct label folder.
- Avoid uploading duplicate or mislabeled files.
- Do not delete or overwrite other contributors' recordings.
- Obtain the speaker's permission before uploading voice recordings.
- Notify the maintainer when your upload is complete.

## 7. Intended Use

This dataset supports Khmer voice command classification using classical and quantum machine learning methods, including SVM, QSVC, and VQC.

For model evaluation, recordings should be split by speaker so that the same speaker does not appear in both training and testing datasets.

## 8. Dataset Status

Data collection is ongoing. Additional recordings will be added as contributors submit their audio files.
