"""Generacion de datos: silencio sintetico, mezclas a SNR y canal de celular.

Todas las funciones trabajan sobre audio crudo a 16 kHz, ANTES del
acondicionamiento: el filtro forma parte del sistema de medida y tiene que ver el
mismo tipo de senal que vera en una grabacion real.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal as sps

from src.config import EPSILON, SAMPLE_RATE, SEGMENT_SAMPLES
from src.filters import resample
from src.preprocessing import ffmpeg_path, read_wav

# ---------------------------------------------------------------------------
# Niveles
# ---------------------------------------------------------------------------


def rms_dbfs(x: np.ndarray) -> float:
    """Nivel RMS en dBFS (0 dBFS corresponde a valor eficaz 1.0)."""
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + EPSILON))


def set_level(x: np.ndarray, target_dbfs: float) -> np.ndarray:
    current = np.sqrt(np.mean(np.square(x))) + EPSILON
    return x * (10 ** (target_dbfs / 20) / current)


def apply_gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    return np.clip(x * 10 ** (gain_db / 20), -1.0, 1.0)


def mix_at_snr(foreground: np.ndarray, background: np.ndarray, snr_db: float) -> np.ndarray:
    """foreground + a * background con 10 log10(P_fg / P_bg') = snr_db."""
    n = foreground.size
    bg = np.resize(background, n)
    p_fg = np.mean(np.square(foreground)) + EPSILON
    p_bg = np.mean(np.square(bg)) + EPSILON
    a = np.sqrt(p_fg / (p_bg * 10 ** (snr_db / 10)))
    y = foreground + a * bg
    peak = np.max(np.abs(y))
    return y / peak * 0.99 if peak >= 1.0 else y


# ---------------------------------------------------------------------------
# Ruidos de color y silencio
# ---------------------------------------------------------------------------


def colored_noise(n: int, exponent: float, rng: np.random.Generator, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Ruido con densidad espectral S(f) ~ 1/f^exponent (0 blanco, 1 rosa, 2 marron).

    Se genera en frecuencia: espectro blanco complejo multiplicado por f^(-exponent/2)
    y transformada inversa. Por debajo de 20 Hz se satura la pendiente para no
    generar una deriva enorme de muy baja frecuencia.
    """
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = f[1]
    spec *= np.maximum(f, 20.0) ** (-exponent / 2)
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + EPSILON)


def mic_hiss(n: int, rng: np.random.Generator, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Siseo de preamplificador: blanco con enfasis en agudos (pasa-altos de 1er orden a 2 kHz)."""
    b, a = sps.butter(1, 2000, btype="highpass", fs=fs)
    x = sps.lfilter(b, a, rng.standard_normal(n)) + 0.3 * rng.standard_normal(n)
    return x / (np.std(x) + EPSILON)


def mains_hum(n: int, rng: np.random.Generator, fs: int = SAMPLE_RATE, f0: float = 50.0) -> np.ndarray:
    """Zumbido de red: 50 Hz con armonicos impares y un leve desvio de frecuencia."""
    t = np.arange(n) / fs
    f = f0 + rng.uniform(-0.3, 0.3)
    x = np.zeros(n)
    for k, amp in ((1, 1.0), (2, 0.25), (3, 0.5), (5, 0.2)):
        x += amp * np.sin(2 * np.pi * k * f * t + rng.uniform(0, 2 * np.pi))
    return x / (np.std(x) + EPSILON)


def dithered_digital_zero(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ceros digitales con dither TPDF de +-1 LSB y cuantizados a 16 bits."""
    lsb = 1.0 / 32768.0
    d = (rng.uniform(-0.5, 0.5, n) + rng.uniform(-0.5, 0.5, n)) * lsb
    return np.round(d / lsb) * lsb


SILENCE_KINDS = ("sala", "rosa", "marron", "siseo", "dither", "pausas_voz", "zumbido", "muy_bajo")


def make_silence(kind: str, rng: np.random.Generator, level_range: tuple[float, float],
                 noise_bank: np.ndarray | None = None, speech_pauses: np.ndarray | None = None,
                 n: int = SEGMENT_SAMPLES) -> np.ndarray:
    lo, hi = level_range
    level = rng.uniform(lo, hi)
    if kind == "sala" and noise_bank is not None and len(noise_bank):
        base = np.resize(noise_bank[rng.integers(len(noise_bank))].astype(np.float64), n)
        # tono de sala: ruido real filtrado pasa-bajos (las paredes absorben agudos)
        b, a = sps.butter(2, rng.uniform(1500, 5000), fs=SAMPLE_RATE)
        x = sps.lfilter(b, a, base)
    elif kind == "pausas_voz" and speech_pauses is not None and speech_pauses.size >= n:
        start = rng.integers(0, speech_pauses.size - n + 1)
        x = speech_pauses[start:start + n].astype(np.float64)
    elif kind == "rosa":
        x = colored_noise(n, 1.0, rng)
    elif kind == "marron":
        x = colored_noise(n, 2.0, rng)
    elif kind == "siseo":
        x = mic_hiss(n, rng)
    elif kind == "dither":
        return dithered_digital_zero(n, rng)
    elif kind == "zumbido":
        x = mains_hum(n, rng) + 10 ** (-rng.uniform(6, 20) / 20) * colored_noise(n, 1.0, rng)
    elif kind == "muy_bajo":
        x = colored_noise(n, rng.uniform(0.0, 1.5), rng)
        level = rng.uniform(lo - 10, lo + 5)
    else:
        x = colored_noise(n, 1.0, rng)
    x = set_level(x - np.mean(x), level)
    # cuantizar a 16 bits como haria el conversor A/D
    return np.round(x * 32768.0) / 32768.0


# ---------------------------------------------------------------------------
# Canal de celular
# ---------------------------------------------------------------------------


def synthetic_reverb(x: np.ndarray, rng: np.random.Generator, rt60: float | None = None,
                     wet: float | None = None, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Convolucion con una respuesta al impulso sintetica.

    h[n] = delta[n] + g * e[n] * exp(-6.91 n / (RT60 fs)), con e[n] ruido blanco:
    el decaimiento de 60 dB en RT60 segundos sale de exp(-6.91) = 10^-3.
    """
    rt60 = rng.uniform(0.15, 0.7) if rt60 is None else rt60
    wet = rng.uniform(0.2, 0.6) if wet is None else wet
    n_ir = int(min(1.0, 1.2 * rt60) * fs)
    t = np.arange(n_ir) / fs
    tail = rng.standard_normal(n_ir) * np.exp(-6.91 * t / rt60)
    tail[: int(0.003 * fs)] = 0.0  # primeras reflexiones a partir de 3 ms
    tail /= np.sqrt(np.sum(tail ** 2)) + EPSILON
    h = np.zeros(n_ir)
    h[0] = 1.0
    h += wet * tail
    y = sps.fftconvolve(x, h)[: x.size]
    return y * (np.std(x) + EPSILON) / (np.std(y) + EPSILON)


def phone_bandpass(x: np.ndarray, rng: np.random.Generator, narrowband: bool, fs: int = SAMPLE_RATE) -> np.ndarray:
    if narrowband:
        lo, hi = rng.uniform(250, 350), rng.uniform(3200, 3500)
    else:
        lo, hi = rng.uniform(80, 150), rng.uniform(6500, 7500)
    sos = sps.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sps.sosfilt(sos, x)


CODECS = {
    # nombre: (encoder ffmpeg, extension, bitrate, frecuencia de trabajo)
    "mp3_128k": ("libmp3lame", "mp3", "128k", 16000),
    "mp3_64k": ("libmp3lame", "mp3", "64k", 16000),
    "mp3_32k": ("libmp3lame", "mp3", "32k", 16000),
    "aac_64k": ("aac", "m4a", "64k", 16000),
    "aac_32k": ("aac", "m4a", "32k", 16000),
    "opus_16k": ("libopus", "ogg", "16k", 16000),
    "amr_nb": ("libopencore_amrnb", "amr", "12.2k", 8000),
}


def codec_roundtrip(x: np.ndarray, codec: str, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Codifica y decodifica con ffmpeg y realinea compensando el retardo del codec."""
    encoder, ext, bitrate, fs_codec = CODECS[codec]
    x_in = resample(x, fs, fs_codec) if fs_codec != fs else np.asarray(x, dtype=np.float64)
    x_in = np.clip(x_in, -1.0, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "in.f32"
        enc = Path(tmp) / f"enc.{ext}"
        dec = Path(tmp) / "dec.wav"
        x_in.astype("<f4").tofile(raw)
        cmd_enc = [ffmpeg_path(), "-v", "error", "-y", "-f", "f32le", "-ar", str(fs_codec), "-ac", "1",
                   "-i", str(raw), "-c:a", encoder, "-b:a", bitrate, str(enc)]
        if encoder == "libopencore_amrnb":
            cmd_enc[cmd_enc.index("-b:a") + 1] = "12.2k"
        r = subprocess.run(cmd_enc, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg ({codec}): " + r.stderr.decode(errors="replace"))
        r = subprocess.run([ffmpeg_path(), "-v", "error", "-y", "-i", str(enc), "-ac", "1",
                            "-ar", str(fs_codec), "-c:a", "pcm_f32le", str(dec)], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg ({codec}): " + r.stderr.decode(errors="replace"))
        _, y = read_wav(dec)
    lag = estimate_lag(x_in, y)
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), y])
    y = np.resize(y, x_in.size) if y.size < x_in.size else y[: x_in.size]
    return resample(y, fs_codec, fs)[: x.size] if fs_codec != fs else y


def estimate_lag(ref: np.ndarray, y: np.ndarray, max_lag: int = 6000) -> int:
    n = min(ref.size, y.size, 16000 * 8)
    a = ref[:n] - ref[:n].mean()
    b = y[: n + max_lag] - y[: n + max_lag].mean()
    size = 1 << int(np.ceil(np.log2(a.size + b.size)))
    corr = np.fft.irfft(np.fft.rfft(b, size) * np.conj(np.fft.rfft(a, size)), size)
    return int(np.argmax(corr[: max_lag + 1]))


def codec_batch(clips: np.ndarray, codec: str, gap: int = 4000) -> np.ndarray:
    """Pasa un lote de recortes por el codec en una sola llamada a ffmpeg.

    Se concatenan con 0.25 s de silencio entre ellos para que el solape de tramas
    del codec no mezcle recortes vecinos.
    """
    n_clips, n = clips.shape
    stride = n + gap
    long = np.zeros(n_clips * stride + gap)
    for i, c in enumerate(clips):
        long[gap + i * stride: gap + i * stride + n] = c
    out = codec_roundtrip(long, codec)
    return np.stack([out[gap + i * stride: gap + i * stride + n] for i in range(n_clips)])


def phone_channel(clips: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """Simulacion del camino de un celular para un lote de recortes.

    Por recorte: reverb de sala (p = 0.6), pasa-banda de banda angosta (300-3400 Hz,
    con AMR-NB a 8 kHz) o ancha (100-7000 Hz), ganancia de -12 a +6 dB y, a veces,
    saturacion suave (AGC agresivo). Despues un codec con perdida por lote.
    """
    out = np.empty_like(clips, dtype=np.float64)
    narrow = rng.random(len(clips)) < 0.4
    tags = []
    for i, c in enumerate(clips.astype(np.float64)):
        y = synthetic_reverb(c, rng) if rng.random() < 0.6 else c
        y = phone_bandpass(y, rng, bool(narrow[i]))
        y = y * 10 ** (rng.uniform(-12, 6) / 20)
        if rng.random() < 0.15:
            drive = rng.uniform(1.5, 4.0)
            y = np.tanh(drive * y) / drive
        out[i] = np.clip(y, -1.0, 1.0)
    wide_codecs = ["mp3_64k", "mp3_32k", "aac_64k", "aac_32k", "opus_16k", None]
    for mask, codecs in ((narrow, ["amr_nb", None]), (~narrow, wide_codecs)):
        idx = np.flatnonzero(mask)
        rng.shuffle(idx)
        for k, part in enumerate(np.array_split(idx, len(codecs))):
            codec = codecs[k]
            if part.size == 0:
                continue
            if codec is not None:
                out[part] = codec_batch(out[part], codec)
            for j in part:
                tags.append((j, f"celular_{'angosta' if narrow[j] else 'ancha'}_{codec or 'sin_codec'}"))
    tags.sort()
    return out, [t for _, t in tags]
