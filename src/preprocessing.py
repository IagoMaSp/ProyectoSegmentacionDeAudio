"""Lectura, decodificacion, remuestreo y entramado de la senal."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile

from src.config import HOP_LENGTH, N_WIN, PCM_MAX_VALUE, SAMPLE_RATE
from src.filters import resample

# ---------------------------------------------------------------------------
# Entrada de audio
# ---------------------------------------------------------------------------


def ffmpeg_path() -> str:
    """ffmpeg del sistema o, si no esta instalado, el binario de imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("No se encontro ffmpeg. Instalar ffmpeg o `pip install imageio-ffmpeg`.") from exc


def scale_pcm(data: np.ndarray) -> np.ndarray:
    """Enteros PCM a punto flotante en [-1, 1)."""
    if data.dtype == np.uint8:
        return (data.astype(np.float64) - 128.0) / 128.0
    if np.issubdtype(data.dtype, np.integer):
        full_scale = float(2 ** (8 * data.dtype.itemsize - 1))
        return data.astype(np.float64) / full_scale
    return data.astype(np.float64)


def scale_pcm16(signal: np.ndarray) -> np.ndarray:
    if np.issubdtype(signal.dtype, np.integer):
        return signal.astype(np.float64) / PCM_MAX_VALUE
    return signal.astype(np.float64)


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1) if data.ndim > 1 else data


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sr, data = wavfile.read(Path(path))
    return int(sr), to_mono(scale_pcm(data))


def decode_with_ffmpeg(path: str | Path) -> tuple[int, np.ndarray]:
    """Decodifica cualquier formato a WAV mono float32 en su frecuencia original.

    No se le pide a ffmpeg que cambie la frecuencia de muestreo: el remuestreo lo
    hace `src.filters.resample` con el anti-alias disenado en el proyecto.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decodificado.wav"
        result = subprocess.run(
            [ffmpeg_path(), "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1",
             "-c:a", "pcm_f32le", str(out)],
            capture_output=True,
        )
        if result.returncode != 0 or not out.exists():
            raise ValueError("No se pudo decodificar el audio: " + result.stderr.decode(errors="replace").strip())
        return read_wav(out)


def decode_bytes(data: bytes, suffix: str = "") -> tuple[int, np.ndarray]:
    # Se escribe en un directorio temporal y se cierra el archivo antes de llamar
    # a ffmpeg: en Windows un NamedTemporaryFile abierto no lo puede leer otro proceso.
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"entrada{suffix}"
        src.write_bytes(data)
        return decode_with_ffmpeg(src)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Carga un archivo cualquiera y lo deja mono a `target_sr` con anti-alias."""
    path = Path(path)
    try:
        sr, x = read_wav(path)
    except (ValueError, OSError):
        sr, x = decode_with_ffmpeg(path)
    return resample(x, sr, target_sr)


def load_wav(file_path: str | Path, expected_sr: int = SAMPLE_RATE) -> tuple[int, np.ndarray]:
    """Lectura estricta usada para MUSAN (ya viene a 16 kHz)."""
    sr, data = read_wav(file_path)
    if sr != expected_sr:
        data = resample(data, sr, expected_sr)
    return expected_sr, data


def write_wav_bytes(x: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    import io
    pcm = np.clip(np.asarray(x), -1.0, 1.0 - 1.0 / 32768.0)
    buf = io.BytesIO()
    wavfile.write(buf, sr, (pcm * 32768.0).astype(np.int16))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Entramado
# ---------------------------------------------------------------------------


def hann_window(n_win: int = N_WIN, periodic: bool = True) -> np.ndarray:
    """Ventana de Hann.

    La version periodica (denominador N) es la que cumple COLA: sum_m w[n - mH] es
    constante para H = N/R con R entero >= 2. La simetrica (denominador N - 1) es la
    adecuada para disenar filtros FIR, pero con solape no suma una constante exacta.
    """
    n = np.arange(n_win, dtype=np.float64)
    denom = n_win if periodic else n_win - 1
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / denom))


def n_frames_for(n_samples: int, n_win: int = N_WIN, hop_length: int = HOP_LENGTH) -> int:
    if n_samples < n_win:
        return 0
    return 1 + (n_samples - n_win) // hop_length


def frame_signal(
    signal: np.ndarray,
    n_win: int = N_WIN,
    hop_length: int = HOP_LENGTH,
    apply_window: bool = True,
) -> np.ndarray:
    if signal.shape[0] < n_win:
        raise ValueError(f"La longitud de la senal ({signal.shape[0]}) es menor que N_win ({n_win}).")
    frames = sliding_window_view(signal, window_shape=n_win)[::hop_length]
    if apply_window:
        return frames * hann_window(n_win)
    return frames.copy()
