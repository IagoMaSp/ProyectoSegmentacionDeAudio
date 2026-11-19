"""FFT radix-2 (Cooley-Tukey, decimacion en el tiempo) implementada a mano.

Version iterativa: primero se reordena la entrada por inversion de bits y despues
se aplican log2(N) etapas de mariposas. Cada etapa combina DFTs de largo m/2 en
DFTs de largo m:

    X[k]       = E[k] + W_m^k O[k]
    X[k + m/2] = E[k] - W_m^k O[k],    W_m = exp(-j 2 pi / m)

Costo O(N log N) contra O(N^2) de la DFT directa. Opera sobre el ultimo eje, asi que
se puede pasar una matriz de frames (M x N) y se transforman todos juntos.
"""
from __future__ import annotations

import numpy as np

_BITREV_CACHE: dict[int, np.ndarray] = {}
_TWIDDLE_CACHE: dict[int, np.ndarray] = {}


def _bit_reverse_indices(n: int) -> np.ndarray:
    if n not in _BITREV_CACHE:
        bits = n.bit_length() - 1
        idx = np.arange(n)
        rev = np.zeros(n, dtype=np.int64)
        for b in range(bits):
            rev |= ((idx >> b) & 1) << (bits - 1 - b)
        _BITREV_CACHE[n] = rev
    return _BITREV_CACHE[n]


def _twiddles(m: int) -> np.ndarray:
    if m not in _TWIDDLE_CACHE:
        _TWIDDLE_CACHE[m] = np.exp(-2j * np.pi * np.arange(m // 2) / m)
    return _TWIDDLE_CACHE[m]


def fft_radix2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError(f"El largo debe ser potencia de 2 (se recibio {n}).")
    a = x[..., _bit_reverse_indices(n)]
    lead = a.shape[:-1]
    m = 2
    while m <= n:
        half = m // 2
        blocks = a.reshape(*lead, n // m, m)
        even = blocks[..., :half]
        odd = blocks[..., half:] * _twiddles(m)
        blocks = np.concatenate([even + odd, even - odd], axis=-1)
        a = blocks.reshape(*lead, n)
        m *= 2
    return a


def rfft_radix2(x: np.ndarray, n: int | None = None) -> np.ndarray:
    """Mitad no redundante del espectro de una senal real (bins 0..N/2)."""
    x = np.asarray(x, dtype=np.float64)
    if n is not None and x.shape[-1] != n:
        if x.shape[-1] > n:
            x = x[..., :n]
        else:
            pad = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
            x = np.pad(x, pad)
    full = fft_radix2(x)
    return full[..., : x.shape[-1] // 2 + 1]


def dft_direct(x: np.ndarray) -> np.ndarray:
    """DFT por definicion, O(N^2). Solo para comparar en los tests."""
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[-1]
    k = np.arange(n)
    w = np.exp(-2j * np.pi * np.outer(k, k) / n)
    return x @ w.T
