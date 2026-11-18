"""Clasificacion en vivo desde el microfono.

El navegador manda bloques de PCM float32 a la frecuencia de su placa de audio
(tipicamente 44.1 o 48 kHz). Por cada bloque:

1. `StreamingResampler`: anti-alias + remuestreo a 16 kHz, sin cortes entre bloques.
2. `CausalConditioner`: el mismo pasa-altos + notch, dos pasadas causales (|H|^2).
3. Columnas de espectrograma de los frames nuevos (mismos frames de 30 ms / 10 ms).
4. Cada LIVE_STEP segundos: vector de caracteristicas de los ultimos 3 s y
   probabilidades del modelo.
5. Filtrado HMM hacia adelante (version causal del Viterbi): misma matriz de
   transicion, elevada a la cantidad de celdas de 0.5 s que dura un paso.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.config import (
    CATEGORIES,
    HOP_LENGTH,
    LIVE_STEP,
    N_WIN,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    TRANSITION_PENALTY,
    WINDOW_HOP,
)
from src.features import compute_spectrum, signal_to_windows
from src.filters import CausalConditioner, StreamingResampler
from src.preprocessing import hann_window
from src.segmenter import predict_proba

K = len(CATEGORIES)
SPEC_BINS = 128


def transition_matrix(penalty: float = TRANSITION_PENALTY, steps: int = 1) -> np.ndarray:
    """Matriz del HMM equivalente a la penalidad del Viterbi, para `steps` celdas.

    De penalty = log((1 - e)(K - 1) / e) sale e = (K - 1) / (exp(penalty) + K - 1).
    """
    e = (K - 1) / (np.exp(penalty) + K - 1)
    T = np.full((K, K), e / (K - 1))
    np.fill_diagonal(T, 1 - e)
    return np.linalg.matrix_power(T, steps)


def spectrogram_columns(x: np.ndarray) -> np.ndarray:
    """Columnas en dB (128 bandas de 62.5 Hz) de los frames completos de `x`."""
    if x.size < N_WIN:
        return np.zeros((0, SPEC_BINS))
    from src.preprocessing import frame_signal
    frames = frame_signal(x, N_WIN, HOP_LENGTH, apply_window=True)
    power = np.abs(compute_spectrum(frames)) ** 2
    power = power[:, : 2 * SPEC_BINS].reshape(-1, SPEC_BINS, 2).mean(axis=2)
    scale = (hann_window(N_WIN).sum() / 2) ** 2
    return 10 * np.log10(power / scale + 1e-12)


class LiveSession:
    def __init__(self, fs_in: int, bundle: dict[str, Any], step: float = LIVE_STEP):
        self.bundle = bundle
        self.resampler = StreamingResampler(fs_in, SAMPLE_RATE)
        self.conditioner = CausalConditioner(SAMPLE_RATE)
        self.step = int(step * SAMPLE_RATE)
        self.buf = np.zeros(0)          # ultimos 3 s ya acondicionados
        self.spec_pending = np.zeros(0)  # muestras que todavia no completan un frame
        self.total = 0
        self.next_decision = SAMPLE_RATE  # primera decision con 1 s de audio
        self.alpha = np.full(K, 1.0 / K)
        self.T = transition_matrix(steps=max(1, int(round(step / WINDOW_HOP))))
        self.last_seen = time.time()

    def push(self, chunk: np.ndarray) -> dict[str, Any]:
        self.last_seen = time.time()
        y = self.conditioner.process(self.resampler.process(chunk))
        self.total += y.size
        self.buf = np.concatenate([self.buf, y])[-SEGMENT_SAMPLES:]

        pending = np.concatenate([self.spec_pending, y])
        cols = spectrogram_columns(pending)
        consumed = cols.shape[0] * HOP_LENGTH
        self.spec_pending = pending[consumed:]

        out: dict[str, Any] = {
            "t": round(self.total / SAMPLE_RATE, 3),
            "columns": np.round(cols, 1).tolist(),
            "level_db": round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)), 1) if y.size else None,
            "decisions": [],
        }
        while self.total >= self.next_decision:
            X, _, _ = signal_to_windows(self.buf, hop_seconds=3.0, include_tail=False, condition=False)
            p = predict_proba(self.bundle, X)[0]
            prior = self.T.T @ self.alpha
            alpha = p * prior
            self.alpha = alpha / alpha.sum()
            k = int(np.argmax(self.alpha))
            out["decisions"].append({
                "t": round(self.next_decision / SAMPLE_RATE, 3),
                "ventana_s": round(min(self.total, SEGMENT_SAMPLES) / SAMPLE_RATE, 2),
                "crudo": {c: round(float(v), 4) for c, v in zip(CATEGORIES, p)},
                "suavizado": {c: round(float(v), 4) for c, v in zip(CATEGORIES, self.alpha)},
                "label": CATEGORIES[k],
                "label_crudo": CATEGORIES[int(np.argmax(p))],
            })
            self.next_decision += self.step
        return out
