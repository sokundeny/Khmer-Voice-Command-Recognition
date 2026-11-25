from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "school-project/khmer-voice-commands"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DATA_DIR,
)

print(f"Dataset downloaded to: {DATA_DIR}")
