from pathlib import Path
import numpy as np
from scipy.io import wavfile
from numpy.lib.stride_tricks import sliding_window_view

from src.config import (
    HOP_LENGTH,
    N_WIN,
    PCM_MAX_VALUE,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
)


def scale_pcm16(signal: np.ndarray) -> np.ndarray:
    if np.issubdtype(signal.dtype, np.integer):
        return signal.astype(np.float64) / PCM_MAX_VALUE
    return signal.astype(np.float64)


def load_wav(file_path: str | Path, expected_sr: int = SAMPLE_RATE) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(Path(file_path))
    if sr != expected_sr:
        raise ValueError(f"Frecuencia de muestreo invalida: {sr} Hz. Se esperaba {expected_sr} Hz.")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    scaled = scale_pcm16(data)
    return sr, scaled


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return signal
    return signal - np.mean(signal)


def hann_window(n_win: int = N_WIN) -> np.ndarray:
    n = np.arange(n_win, dtype=np.float64)
    return 0.5 * (1.0 - np.cos((2.0 * np.pi * n) / (n_win - 1)))


def frame_signal(
    segment: np.ndarray,
    n_win: int = N_WIN,
    hop_length: int = HOP_LENGTH,
    apply_window: bool = True,
) -> np.ndarray:
    if segment.shape[0] < n_win:
        raise ValueError(f"La longitud del segmento ({segment.shape[0]}) es menor que N_win ({n_win}).")
    frames = sliding_window_view(segment, window_shape=n_win)[::hop_length].copy()
    if apply_window:
        window = hann_window(n_win)
        frames = frames * window
    return frames


def split_into_segments(
    signal: np.ndarray,
    segment_samples: int = SEGMENT_SAMPLES,
    remove_dc: bool = True,
) -> np.ndarray:
    total_samples = signal.shape[0]
    n_segments = total_samples // segment_samples
    if n_segments == 0:
        return np.empty((0, segment_samples), dtype=np.float64)
    trimmed = signal[: n_segments * segment_samples]
    segments = trimmed.reshape(n_segments, segment_samples).copy()
    if remove_dc:
        segments = segments - np.mean(segments, axis=1, keepdims=True)
    return segments


def preprocess_segment(
    segment: np.ndarray,
    n_win: int = N_WIN,
    hop_length: int = HOP_LENGTH,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_segment = remove_dc_offset(segment)
    raw_frames = frame_signal(clean_segment, n_win=n_win, hop_length=hop_length, apply_window=False)
    windowed_frames = frame_signal(clean_segment, n_win=n_win, hop_length=hop_length, apply_window=True)
    return clean_segment, raw_frames, windowed_frames
