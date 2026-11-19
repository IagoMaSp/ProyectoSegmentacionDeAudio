"""Reproduccion exacta de la version anterior del sistema, solo para comparar.

Bloques fijos de 3 s, resta de la media por bloque, frames de 512 muestras con salto
de 256, Hann simetrica, 18 caracteristicas (media y desvio de RMS, ZCR, centroide,
ancho de banda y 5 subbandas), sin filtrado. Modelo: models/legacy/mlp_model.joblib
con su scaler (el que estaba en produccion).
"""
from __future__ import annotations

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.config import EPSILON, LEGACY_DIR, SAMPLE_RATE, SUBBAND_EDGES

N_WIN, HOP = 512, 256
_W = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_WIN) / (N_WIN - 1)))
_F = np.fft.rfftfreq(N_WIN, 1.0 / SAMPLE_RATE)


def legacy_features(segment: np.ndarray) -> np.ndarray:
    x = segment - np.mean(segment)
    raw = sliding_window_view(x, N_WIN)[::HOP]
    win = raw * _W
    mag = np.abs(np.fft.rfft(win, n=N_WIN, axis=1))
    power = mag ** 2
    rms = np.sqrt(np.mean(raw ** 2, axis=1))
    zcr = np.mean(np.signbit(raw[:, 1:]) != np.signbit(raw[:, :-1]), axis=1)
    s = mag.sum(axis=1) + EPSILON
    cen = (mag * _F).sum(axis=1) / s
    bw = np.sqrt((mag * (_F[None] - cen[:, None]) ** 2).sum(axis=1) / s)
    tot = power.sum(axis=1) + EPSILON
    bands = []
    for i, (lo, hi) in enumerate(SUBBAND_EDGES):
        m = (_F >= lo) & ((_F <= hi) if i == len(SUBBAND_EDGES) - 1 else (_F < hi))
        bands.append(power[:, m].sum(axis=1) / tot)
    F = np.column_stack([rms, zcr, cen, bw] + bands)
    return np.concatenate([F.mean(axis=0), F.std(axis=0)])


class LegacyModel:
    def __init__(self, name: str = "mlp_model"):
        self.scaler = joblib.load(LEGACY_DIR / "scaler.joblib")
        self.model = joblib.load(LEGACY_DIR / f"{name}.joblib")

    def predict_clips(self, clips: np.ndarray) -> np.ndarray:
        X = np.vstack([legacy_features(c[:3 * SAMPLE_RATE]) for c in clips])
        return self.model.predict(self.scaler.transform(X))
