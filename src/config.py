from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "musan"
MODELS_DIR = BASE_DIR / "models"

SCALER_PATH = MODELS_DIR / "scaler.joblib"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.joblib"
MLP_MODEL_PATH = MODELS_DIR / "mlp_model.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

SAMPLE_RATE = 16000
PCM_MAX_VALUE = 32768.0

SEGMENT_DURATION = 3.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)

N_WIN = 512
HOP_LENGTH = 256

SUBBAND_EDGES = [
    (0.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
]

CATEGORIES = ["musica", "voz", "ruido", "silencio"]
MUSAN_FOLDER_MAP = {
    "music": "musica",
    "speech": "voz",
    "noise": "ruido",
}

LABEL_TO_ID = {label: idx for idx, label in enumerate(CATEGORIES)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(CATEGORIES)}

SILENCE_SCALE_MIN = 0.001
SILENCE_SCALE_MAX = 0.01
LOW_ENERGY_PERCENTILE = 10.0

TEST_SIZE = 0.2
RANDOM_STATE = 42

KNN_NEIGHBORS = 5
MLP_HIDDEN_LAYERS = (64, 32)
MLP_MAX_ITER = 300

EPSILON = 1e-12
