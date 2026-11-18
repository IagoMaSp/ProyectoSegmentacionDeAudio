"""Segmentacion de audio continuo.

Cadena: senal -> ventanas de 3 s cada 0.5 s -> P(clase | ventana) -> probabilidad
por celda de 0.5 s (promedio de las ventanas que la cubren) -> suavizado temporal
(mediana o Viterbi) -> duracion minima -> intervalos fusionados con confianza.

Uso:  python -m src.segmenter archivo.(wav|mp3|m4a|...) [--suavizado viterbi] [--salto 0.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.ndimage import median_filter

from src.config import (
    CATEGORIES,
    MEDIAN_KERNEL,
    MIN_SEGMENT_DURATION,
    MODEL_BUNDLE_PATH,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    SMOOTHING,
    TRANSITION_PENALTY,
    WINDOW_HOP,
)
from src.features import signal_to_windows
from src.preprocessing import load_audio

K = len(CATEGORIES)


def load_bundle(path: str | Path = MODEL_BUNDLE_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Entrenar con `python -m src.train`.")
    return joblib.load(path)


def predict_proba(bundle: dict[str, Any], X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.zeros((0, K))
    return bundle["modelo"].predict_proba(X[:, bundle["feature_idx"]])


# ---------------------------------------------------------------------------
# De ventanas a celdas
# ---------------------------------------------------------------------------


def windows_to_cells(starts: np.ndarray, probs: np.ndarray, duration: float, cell: float) -> np.ndarray:
    """Probabilidad de cada celda [k*cell, (k+1)*cell) como promedio de las ventanas
    que contienen su centro. Con la ventana final alineada al fin del audio, ninguna
    celda queda sin cubrir; si el audio dura menos de 3 s la unica ventana cubre todo."""
    n_cells = max(1, int(np.ceil(duration / cell - 1e-9)))
    centers = (np.arange(n_cells) + 0.5) * cell
    centers = np.minimum(centers, max(duration - 1e-6, 0))
    acc = np.zeros((n_cells, K))
    cnt = np.zeros(n_cells)
    for s, p in zip(starts, probs):
        m = (centers >= s) & (centers < s + SEGMENT_DURATION)
        acc[m] += p
        cnt[m] += 1
    empty = cnt == 0
    if np.any(empty) and len(starts):  # por seguridad: ventana mas cercana
        for k in np.flatnonzero(empty):
            j = int(np.argmin(np.abs(starts + SEGMENT_DURATION / 2 - centers[k])))
            acc[k] = probs[j]
            cnt[k] = 1
    return acc / np.maximum(cnt, 1)[:, None]


# ---------------------------------------------------------------------------
# Suavizado temporal
# ---------------------------------------------------------------------------


def smooth_median(p: np.ndarray, kernel: int = MEDIAN_KERNEL) -> np.ndarray:
    if p.shape[0] < 2 or kernel <= 1:
        return p.copy()
    q = median_filter(p, size=(kernel, 1), mode="nearest")
    return q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)


def viterbi(p: np.ndarray, penalty: float = TRANSITION_PENALTY) -> np.ndarray:
    """Secuencia de estados que maximiza sum_t log p_t(s_t) - penalty * (cambios).

    Equivale a un HMM con P(quedarse) = 1 - e, P(cambiar a otra clase) = e / (K-1),
    tomando las probabilidades a posteriori del clasificador como verosimilitudes
    (el dataset es balanceado, asi que el prior es uniforme). La penalidad en nats
    es log((1-e)(K-1)/e); con 6 nats un cambio solo se acepta si la nueva clase es
    sostenidamente mas probable durante varias celdas.
    """
    T = p.shape[0]
    logp = np.log(np.maximum(p, 1e-9))
    trans = np.full((K, K), -penalty)
    np.fill_diagonal(trans, 0.0)
    score = logp[0].copy()
    back = np.zeros((T, K), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + trans          # cand[i, j]: venir de i y estar en j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(K)] + logp[t]
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Corridas (inicio, fin_exclusivo, etiqueta)."""
    out = []
    start = 0
    for t in range(1, labels.size + 1):
        if t == labels.size or labels[t] != labels[start]:
            out.append((start, t, int(labels[start])))
            start = t
    return out


def enforce_min_duration(labels: np.ndarray, p: np.ndarray, min_cells: int) -> np.ndarray:
    """Absorbe corridas mas cortas que `min_cells` en el vecino mas probable."""
    labels = labels.copy()
    while True:
        r = runs(labels)
        if len(r) <= 1:
            return labels
        lengths = [b - a for a, b, _ in r]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_cells:
            return labels
        a, b, _ = r[i]
        neigh = []
        if i > 0:
            neigh.append(r[i - 1][2])
        if i < len(r) - 1:
            neigh.append(r[i + 1][2])
        best = max(neigh, key=lambda k: p[a:b, k].mean())
        labels[a:b] = best


# ---------------------------------------------------------------------------
# Analisis completo
# ---------------------------------------------------------------------------


def analyze_signal(
    x: np.ndarray,
    bundle: dict[str, Any],
    hop: float = WINDOW_HOP,
    smoothing: str = SMOOTHING,
    penalty: float = TRANSITION_PENALTY,
    min_duration: float = MIN_SEGMENT_DURATION,
    keep_frames: bool = False,
) -> dict[str, Any]:
    duration = x.size / SAMPLE_RATE
    if x.size == 0:
        return {"duration": 0.0, "segments": [], "cells": {"t": [], "raw": [], "smooth": []}}
    X, starts, frames = signal_to_windows(x, hop_seconds=hop, include_tail=True)
    probs = predict_proba(bundle, X)
    p_cells = windows_to_cells(starts, probs, duration, hop)

    if smoothing == "viterbi":
        labels = viterbi(p_cells, penalty)
        p_smooth = p_cells
    elif smoothing == "mediana":
        p_smooth = smooth_median(p_cells)
        labels = np.argmax(p_smooth, axis=1)
    else:
        p_smooth = p_cells
        labels = np.argmax(p_cells, axis=1)
    min_cells = int(np.ceil(min_duration / hop - 1e-9))
    if min_cells > 1:
        labels = enforce_min_duration(labels, p_smooth, min_cells)

    segments = []
    for a, b, lab in runs(labels):
        mean_p = p_cells[a:b].mean(axis=0)
        segments.append({
            "start_time": round(a * hop, 3),
            "end_time": round(min(b * hop, duration), 3),
            "label": CATEGORIES[lab],
            "label_id": lab,
            "confidence": round(float(mean_p[lab]), 4),
            "probabilities": {c: round(float(v), 4) for c, v in zip(CATEGORIES, mean_p)},
        })
    result = {
        "duration": round(duration, 3),
        "hop": hop,
        "smoothing": smoothing,
        "penalty": penalty,
        "min_duration": min_duration,
        "segments": segments,
        "windows": {"start": np.round(starts, 3).tolist(), "probs": np.round(probs, 4).tolist()},
        "cells": {
            "t": np.round(np.arange(p_cells.shape[0]) * hop, 3).tolist(),
            "raw": np.round(p_cells, 4).tolist(),
            "labels": [CATEGORIES[k] for k in labels],
        },
    }
    if keep_frames:
        result["_X"] = X
        result["_starts"] = starts
        result["_frames"] = frames
    return result


def segment_file(path: str | Path, bundle: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
    bundle = bundle or load_bundle()
    return analyze_signal(load_audio(path), bundle, **kwargs)


def format_segments(result: dict[str, Any]) -> str:
    if not result["segments"]:
        return "Audio vacio."
    lines = [f"Duracion: {result['duration']:.2f} s | suavizado: {result['smoothing']}",
             "Inicio (s) | Fin (s)    | Categoria | Confianza", "-" * 50]
    for s in result["segments"]:
        lines.append(f"{s['start_time']:10.2f} | {s['end_time']:10.2f} | {s['label']:9s} | {s['confidence']:.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentacion automatica de un archivo de audio.")
    parser.add_argument("audio_path")
    parser.add_argument("--modelo", default=str(MODEL_BUNDLE_PATH))
    parser.add_argument("--salto", type=float, default=WINDOW_HOP)
    parser.add_argument("--suavizado", choices=["ninguno", "mediana", "viterbi"], default=SMOOTHING)
    parser.add_argument("--penalidad", type=float, default=TRANSITION_PENALTY)
    parser.add_argument("--duracion-minima", type=float, default=MIN_SEGMENT_DURATION)
    args = parser.parse_args()
    result = segment_file(args.audio_path, load_bundle(args.modelo), hop=args.salto,
                          smoothing=args.suavizado, penalty=args.penalidad, min_duration=args.duracion_minima)
    print(format_segments(result))


if __name__ == "__main__":
    main()
