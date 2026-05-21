from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCES_DIR = PROJECT_ROOT / "resources"
MODELS_DIR = RESOURCES_DIR / "models"

PT_MODEL_DIR = MODELS_DIR / "piano_transcription_inference_data"
PT_MODEL_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
PT_MODEL_PATH = PT_MODEL_DIR / PT_MODEL_NAME

BP_MODEL_DIR = MODELS_DIR / "basic_pitch" / "icassp_2022"
BP_MODEL_PATH = BP_MODEL_DIR / "nmp"

RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
BP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
