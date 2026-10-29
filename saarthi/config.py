import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INTENTS_FILE = DATA_DIR / "intents.json"
MODEL_FILE = MODELS_DIR / "saarthi_intent_model.keras"
TOKENIZER_FILE = MODELS_DIR / "tokenizer.pickle"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pickle"
METRICS_PLOT_FILE = BASE_DIR / "metrics_plot.png"
NOTES_FILE = BASE_DIR / "saarthi_notes.txt"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Deep Learning Hyperparameters
MAX_VOCAB_SIZE = 1500
EMBEDDING_DIM = 64
MAX_SEQUENCE_LENGTH = 18
EPOCHS = 120
BATCH_SIZE = 8
CONFIDENCE_THRESHOLD = 0.35  # Realistic threshold for 29-class Softmax distribution

# TTS & Voice Configurations
DEFAULT_VOICE_RATE = 175    # Words per minute (natural pacing)
DEFAULT_VOICE_VOLUME = 1.0  # 0.0 to 1.0
ASSISTANT_NAME = "Saarthi"
WAKE_WORDS = ["saarthi", "hey saarthi", "sarthi", "hello saarthi"]

# Common Windows App Executables mapping
APP_MAPPING = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "browser": "start msedge",
    "edge": "msedge.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "paint": "mspaint.exe",
    "task manager": "taskmgr.exe",
    "vs code": "code",
    "vscode": "code",
    "settings": "start ms-settings:",
    "terminal": "wt.exe"
}
