"""Parametros del sistema. Entrenamiento, inferencia y modo en vivo leen todo de aca."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "musan"
REAL_DIR = BASE_DIR / "data" / "real"
TESTBANK_PATH = BASE_DIR / "data" / "banco_prueba.npz"
MODELS_DIR = BASE_DIR / "models"
LEGACY_DIR = MODELS_DIR / "legacy"
FEATURES_PATH = BASE_DIR / "features" / "musan_features.npz"
FIGURES_DIR = BASE_DIR / "docs" / "figuras"
REPORTS_DIR = BASE_DIR / "docs" / "resultados"

MODEL_BUNDLE_PATH = MODELS_DIR / "modelo.joblib"
METRICS_PATH = MODELS_DIR / "metricas.json"

# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------
# 16 kHz: Nyquist en 8 kHz cubre los formantes de la voz y casi todo el contenido
# util para distinguir musica de ruido. Es ademas la tasa nativa de MUSAN.
SAMPLE_RATE = 16000
PCM_MAX_VALUE = 32768.0

# ---------------------------------------------------------------------------
# Acondicionamiento (src/filters.py)
# ---------------------------------------------------------------------------
# Pasa-altos Butterworth: 70 Hz queda por debajo de la F0 masculina mas baja
# (~85 Hz) y por encima del rumble de manipulacion, viento y HVAC (< 50 Hz).
# Orden 4: 24 dB/octava, -12 dB en 50 Hz y -43 dB en 20 Hz por pasada, con perdida
# < 0.25 dB en 100 Hz. Con filtfilt las atenuaciones se duplican en dB.
HP_CUTOFF_HZ = 70.0
HP_ORDER = 4
# Notch de red electrica (Uruguay: 50 Hz). Q = f0/BW = 10 -> BW = 5 Hz; tolera
# desvios de la red de +-0.5 Hz con mas de 14 dB de atenuacion por pasada.
# No se agregan armonicos (100, 150 Hz): con el mismo Q borrarian bandas de 10-15 Hz
# justo en el rango de F0 de la voz.
NOTCH_FREQS_HZ = (50.0,)
NOTCH_Q = 10.0
# Anti-alias del remuestreo: Kaiser, rechazo de 80 dB desde la nueva Nyquist y banda
# de transicion del 10 % por debajo de ella (7.2-8 kHz al bajar a 16 kHz).
AA_ATTENUATION_DB = 80.0
AA_TRANSITION_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Analisis de corto plazo (src/features.py)
# ---------------------------------------------------------------------------
# Frame de 30 ms (480 muestras), salto de 10 ms (160) y FFT de 512 con relleno de
# ceros. Hann periodica con salto N/3 cumple COLA exactamente, y 10 ms divide
# exacto al salto de ventana (0.5 s = 50 frames) y a la ventana (3 s = 300 frames).
N_WIN = 480
HOP_LENGTH = 160
N_FFT = 512

SUBBAND_EDGES = [
    (0.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
]
N_MELS = 40
MEL_FMIN = 20.0
MEL_FMAX = 8000.0
N_MFCC = 13
DELTA_WIDTH = 2          # delta por regresion sobre +-2 frames
ROLLOFF_FRACTION = 0.85
LOW_ENERGY_FACTOR = 0.5  # frame "de baja energia" si RMS < 0.5 * RMS medio de la ventana
FFT_BACKEND = "numpy"    # "numpy" o "propia" (src/fft.py, mismo resultado)

# ---------------------------------------------------------------------------
# Ventana de decision y segmentacion (src/segmenter.py)
# ---------------------------------------------------------------------------
SEGMENT_DURATION = 3.0
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
WINDOW_HOP = 0.5                 # salto de la ventana deslizante en inferencia
TRAIN_HOP = 3.0                  # salto al extraer del dataset (sin solape)
NOISE_TRAIN_HOP = 1.5            # la clase ruido es escasa: se extrae con solape
SMOOTHING = "viterbi"            # "ninguno", "mediana" o "viterbi"
MEDIAN_KERNEL = 5                # celdas de 0.5 s
TRANSITION_PENALTY = 2.0         # costo de un cambio de clase, en nats, por celda de 0.5 s
MIN_SEGMENT_DURATION = 1.5       # s; segmentos mas cortos se absorben
LIVE_STEP = 1.0                  # s entre decisiones en el modo en vivo

CATEGORIES = ["musica", "voz", "ruido", "silencio"]
MUSAN_FOLDER_MAP = {
    "music": "musica",
    "speech": "voz",
    "noise": "ruido",
}
LABEL_TO_ID = {label: idx for idx, label in enumerate(CATEGORIES)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(CATEGORIES)}

# ---------------------------------------------------------------------------
# Dataset y augmentation (src/dataset.py, src/augment.py)
# ---------------------------------------------------------------------------
N_FOLDS = 5          # particion por archivo; el pliegue TEST_FOLD queda de prueba
TEST_FOLD = 4
PER_CLASS_PER_FOLD = 3000
CLIP_BANK_PER_CLASS = 900   # recortes crudos por clase y pliegue para augmentation
TESTBANK_PER_CLASS = 300    # recortes crudos del pliegue de prueba para robustez
# Niveles. Un celular con control automatico de ganancia deja el ruido de una
# pieza en silencio alrededor de -41 dBFS ya acondicionado (medido en data/real),
# asi que el silencio llega hasta -36 dBFS, medido sobre la senal acondicionada.
# Para no pisarse, el ruido de MUSAN por debajo de -34 dBFS (RMS de la ventana)
# no se usa como clase ruido. Voz y musica conservan un piso mas bajo: a ellas
# las separa el espectro, no el nivel.
ACTIVE_MIN_DBFS = -50.0     # voz y musica por debajo de esto son silencio real
NOISE_MIN_DBFS = -34.0      # ruido por debajo de esto es ruido de fondo = silencio
SILENCE_DBFS_RANGE = (-75.0, -36.0)
FILE_GAIN_DB_RANGE = (-10.0, 6.0)
MIX_SNR_DB_RANGE = (0.0, 20.0)
NOISE_BG_SNR_DB_RANGE = (8.0, 20.0)   # ruido por encima de voz/musica de fondo
RANDOM_STATE = 42

EPSILON = 1e-12
