"""Extraccion de caracteristicas.

Dos niveles:

1. `frame_features(x)`: matriz (n_frames x 38) con descriptores de corto plazo de la
   senal YA acondicionada (frames de 30 ms cada 10 ms).
2. `window_features(F, ...)`: agrega esa matriz sobre ventanas de 3 s (300 frames)
   y devuelve un vector de 64 valores por ventana.

Entrenamiento (`src/dataset.py`), inferencia (`src/segmenter.py`) y modo en vivo usan
`signal_to_windows`, que es la unica puerta de entrada: acondicionamiento ->
frames -> ventanas. Asi no puede haber diferencias de preprocesamiento entre etapas.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import (
    DELTA_WIDTH,
    EPSILON,
    FFT_BACKEND,
    HOP_LENGTH,
    LOW_ENERGY_FACTOR,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    N_MFCC,
    N_WIN,
    ROLLOFF_FRACTION,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    SUBBAND_EDGES,
)
from src.fft import rfft_radix2
from src.filters import condition_offline
from src.preprocessing import frame_signal, n_frames_for

FRAME_FEATURE_NAMES = (
    ["rms", "zcr", "centroide", "ancho_banda"]
    + [f"subbanda_{i + 1}" for i in range(len(SUBBAND_EDGES))]
    + ["rolloff", "planitud", "flujo"]
    + [f"mfcc_{i}" for i in range(N_MFCC)]
    + [f"delta_mfcc_{i}" for i in range(N_MFCC)]
)
N_BASE = 9          # rms, zcr, centroide, ancho de banda y 5 subbandas
N_SPECTRAL = 3      # rolloff, planitud, flujo
SL_MFCC = slice(N_BASE + N_SPECTRAL, N_BASE + N_SPECTRAL + N_MFCC)
SL_DELTA = slice(SL_MFCC.stop, SL_MFCC.stop + N_MFCC)
WINDOW_FRAMES = n_frames_for(SEGMENT_SAMPLES)  # 298 frames en 3 s


# ---------------------------------------------------------------------------
# Banco de filtros mel y DCT, hechos a mano
# ---------------------------------------------------------------------------

def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE,
                   fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    """Filtros triangulares equiespaciados en mel, (n_mels x n_fft/2+1).

    Los n_mels + 2 bordes se reparten uniformemente en la escala mel y se vuelven a
    Hz. El filtro m sube linealmente de f[m-1] a f[m] y baja de f[m] a f[m+1]. Se
    normalizan por area (2 / ancho) para que filtros anchos de alta frecuencia no
    pesen mas que los angostos de baja.
    """
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size))
    for m in range(n_mels):
        lo, center, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (freqs - lo) / (center - lo)
        falling = (hi - freqs) / (hi - center)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling)) * (2.0 / (hi - lo))
    return fb


def dct_matrix(n_out: int = N_MFCC, n_in: int = N_MELS) -> np.ndarray:
    """DCT-II ortonormal: C[k, n] = s_k cos(pi k (2n + 1) / (2 N))."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    c = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    c[0] *= np.sqrt(1.0 / n_in)
    c[1:] *= np.sqrt(2.0 / n_in)
    return c


_MEL_FB = mel_filterbank()
_DCT = dct_matrix()
_FREQS = np.arange(N_FFT // 2 + 1) * SAMPLE_RATE / N_FFT
_BAND_MASKS = [
    (_FREQS >= lo) & ((_FREQS <= hi) if i == len(SUBBAND_EDGES) - 1 else (_FREQS < hi))
    for i, (lo, hi) in enumerate(SUBBAND_EDGES)
]


# ---------------------------------------------------------------------------
# Descriptores por frame
# ---------------------------------------------------------------------------

def compute_spectrum(windowed: np.ndarray, backend: str = FFT_BACKEND) -> np.ndarray:
    if backend == "propia":
        return rfft_radix2(windowed, n=N_FFT)
    return np.fft.rfft(windowed, n=N_FFT, axis=1)


def compute_rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(frames), axis=1))


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)


def compute_centroid_bandwidth(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = mag.sum(axis=1) + EPSILON
    centroid = (mag @ _FREQS) / total
    var = np.sum(mag * np.square(_FREQS[None, :] - centroid[:, None]), axis=1) / total
    return centroid, np.sqrt(var)


def compute_subbands(power: np.ndarray) -> np.ndarray:
    total = power.sum(axis=1) + EPSILON
    return np.column_stack([power[:, m].sum(axis=1) / total for m in _BAND_MASKS])


def compute_rolloff(power: np.ndarray, fraction: float = ROLLOFF_FRACTION) -> np.ndarray:
    cum = np.cumsum(power, axis=1)
    thresh = fraction * cum[:, -1:]
    idx = np.argmax(cum >= thresh, axis=1)
    idx[cum[:, -1] <= EPSILON] = 0
    return _FREQS[idx]


def compute_flatness(power: np.ndarray) -> np.ndarray:
    """Media geometrica / media aritmetica del espectro de potencia (bins 1..N/2).

    Vale 1 para ruido blanco y tiende a 0 para un espectro de pocas lineas.
    """
    p = power[:, 1:] + EPSILON
    return np.exp(np.mean(np.log(p), axis=1)) / np.mean(p, axis=1)


def compute_flux(mag: np.ndarray) -> np.ndarray:
    """Norma L2 de la diferencia entre espectros consecutivos normalizados."""
    norm = mag / (mag.sum(axis=1, keepdims=True) + EPSILON)
    flux = np.zeros(mag.shape[0])
    flux[1:] = np.sqrt(np.sum(np.square(np.diff(norm, axis=0)), axis=1))
    return flux


def compute_mfcc(power: np.ndarray) -> np.ndarray:
    mel_energy = power @ _MEL_FB.T
    return np.log(mel_energy + 1e-10) @ _DCT.T


def compute_deltas(feats: np.ndarray, width: int = DELTA_WIDTH) -> np.ndarray:
    """d_t = sum_n n (c_{t+n} - c_{t-n}) / (2 sum_n n^2), bordes replicados."""
    padded = np.pad(feats, ((width, width), (0, 0)), mode="edge")
    num = np.zeros_like(feats)
    t = feats.shape[0]
    for n in range(1, width + 1):
        num += n * (padded[width + n: width + n + t] - padded[width - n: width - n + t])
    return num / (2.0 * sum(n * n for n in range(1, width + 1)))


def frame_features(x: np.ndarray, backend: str = FFT_BACKEND) -> np.ndarray:
    """Descriptores de corto plazo de una senal ya acondicionada."""
    raw = frame_signal(x, N_WIN, HOP_LENGTH, apply_window=False)
    windowed = frame_signal(x, N_WIN, HOP_LENGTH, apply_window=True)
    spec = compute_spectrum(windowed, backend)
    mag = np.abs(spec)
    power = np.square(mag)
    centroid, bandwidth = compute_centroid_bandwidth(mag)
    mfcc = compute_mfcc(power)
    return np.column_stack([
        compute_rms(raw),
        compute_zcr(raw),
        centroid,
        bandwidth,
        compute_subbands(power),
        compute_rolloff(power),
        compute_flatness(power),
        compute_flux(mag),
        mfcc,
        compute_deltas(mfcc),
    ])


# ---------------------------------------------------------------------------
# Agregacion en ventanas de 3 s
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {}


def _build_names() -> list[str]:
    base = FRAME_FEATURE_NAMES[:N_BASE]
    spectral = FRAME_FEATURE_NAMES[N_BASE:N_BASE + N_SPECTRAL]
    mfcc = FRAME_FEATURE_NAMES[SL_MFCC]
    delta = FRAME_FEATURE_NAMES[SL_DELTA]
    groups = {
        "base": [f"{n}_media" for n in base] + [f"{n}_desvio" for n in base],
        "espectrales": [f"{n}_media" for n in spectral] + [f"{n}_desvio" for n in spectral],
        "mfcc": [f"{n}_media" for n in mfcc] + [f"{n}_desvio" for n in mfcc],
        "deltas": [f"{n}_desvio" for n in delta],
        "dinamica": ["fraccion_baja_energia"],
    }
    FEATURE_GROUPS.update(groups)
    return [name for g in groups.values() for name in g]


FEATURE_NAMES = _build_names()


def get_feature_names() -> list[str]:
    return list(FEATURE_NAMES)


def group_indices(groups: list[str]) -> np.ndarray:
    wanted = {n for g in groups for n in FEATURE_GROUPS[g]}
    return np.array([i for i, n in enumerate(FEATURE_NAMES) if n in wanted])


def window_starts(n_frames: int, hop_frames: int, include_tail: bool) -> np.ndarray:
    """Frame inicial de cada ventana. Con `include_tail` se agrega una ventana final
    alineada al ultimo frame para que la cola del audio tambien se clasifique."""
    if n_frames < WINDOW_FRAMES:
        return np.zeros(0, dtype=np.int64)
    starts = np.arange(0, n_frames - WINDOW_FRAMES + 1, hop_frames)
    last = n_frames - WINDOW_FRAMES
    if include_tail and starts[-1] != last:
        starts = np.append(starts, last)
    return starts


def window_features(frames: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Media y desvio de cada descriptor sobre los 298 frames de cada ventana."""
    if starts.size == 0:
        return np.zeros((0, len(FEATURE_NAMES)))
    view = sliding_window_view(frames, WINDOW_FRAMES, axis=0)[starts]  # (W, F, 298)
    means = view.mean(axis=2)
    stds = view.std(axis=2)
    rms = view[:, 0, :]
    low_energy = np.mean(rms < LOW_ENERGY_FACTOR * rms.mean(axis=1, keepdims=True), axis=1)
    base = slice(0, N_BASE)
    spec = slice(N_BASE, N_BASE + N_SPECTRAL)
    return np.column_stack([
        means[:, base], stds[:, base],
        means[:, spec], stds[:, spec],
        means[:, SL_MFCC], stds[:, SL_MFCC],
        stds[:, SL_DELTA],
        low_energy,
    ])


def signal_to_windows(
    x: np.ndarray,
    hop_seconds: float,
    include_tail: bool = True,
    condition: bool = True,
    backend: str = FFT_BACKEND,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unica cadena de procesamiento: acondicionar -> frames -> ventanas.

    Devuelve (X, inicios_en_segundos, matriz_de_frames). Las senales mas cortas
    que una ventana se repiten periodicamente hasta completar 3 s, lo que conserva
    sus estadisticas (rellenar con ceros las sesgaria hacia silencio).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), np.zeros((0, len(FRAME_FEATURE_NAMES)))
    if x.size < SEGMENT_SAMPLES:
        x = np.resize(x, SEGMENT_SAMPLES)
    y = condition_offline(x) if condition else x
    frames = frame_features(y, backend)
    hop_frames = max(1, int(round(hop_seconds * SAMPLE_RATE / HOP_LENGTH)))
    starts = window_starts(frames.shape[0], hop_frames, include_tail)
    return window_features(frames, starts), starts * HOP_LENGTH / SAMPLE_RATE, frames


def extract_segment_features(segment: np.ndarray, condition: bool = True) -> np.ndarray:
    """Vector de una sola ventana de 3 s (recortes sinteticos del dataset)."""
    X, _, _ = signal_to_windows(segment[:SEGMENT_SAMPLES], hop_seconds=3.0, include_tail=False,
                                condition=condition)
    return X[0]
