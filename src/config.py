from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_DIR = PROJECT_ROOT / "samples"
TEST_EVAL_DIR = PROJECT_ROOT / "test_evaluate"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

DATA_DIR = Path(os.environ.get("DATA_DIR", SAMPLES_DIR))

ASR_MODEL_DIR = Path(os.environ.get("ASR_MODEL_DIR", CHECKPOINTS_DIR / "epoch_20"))

AUDIO_EXT = ".wav"
TEXT_EXT = ".txt"

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "openai/whisper-medium")

MAX_MEL_FRAMES = int(os.environ.get("MAX_MEL_FRAMES", 3000))
MAX_TOKEN_LEN = int(os.environ.get("MAX_TOKEN_LEN", 448))
