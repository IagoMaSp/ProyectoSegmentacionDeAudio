"""Fuga espectral: ventana rectangular contra Hann, calculada con la FFT propia.

Uso:  python -m scripts.leakage_demo
Salida: docs/figuras/fuga_espectral.png
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, N_FFT, N_WIN, SAMPLE_RATE
from src.fft import rfft_radix2
from src.preprocessing import hann_window

INK, ORANGE, BLUE, MUTED = "#171717", "#D97706", "#2563EB", "#8A8983"


def spectrum_db(frame: np.ndarray, window: np.ndarray, n_fft: int) -> np.ndarray:
    X = rfft_radix2(frame * window, n=n_fft)
    mag = np.abs(X) / (window.sum() / 2)
    return 20 * np.log10(np.maximum(mag, 1e-9))


def main() -> None:
    fs, n = SAMPLE_RATE, N_WIN
    n_fft = 8192  # relleno de ceros grande solo para ver bien la forma de los lobulos
    t = np.arange(n) / fs
    f_bin = fs / N_FFT
    f_on = 32 * f_bin            # tono justo sobre un bin de la FFT de 512 (1000 Hz)
    f_off = 32.5 * f_bin         # tono entre dos bins: peor caso
    weak = 10 ** (-50 / 20)      # tono debil 50 dB por debajo, en 2.5 kHz
    x = np.sin(2 * np.pi * f_off * t) + weak * np.sin(2 * np.pi * 2500 * t)
    rect = np.ones(n)
    hann = hann_window(n)
    f = np.arange(n_fft // 2 + 1) * fs / n_fft

    s_rect = spectrum_db(x, rect, n_fft)
    s_hann = spectrum_db(x, hann, n_fft)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(f, s_rect, color=ORANGE, lw=0.9, label="Rectangular")
    ax.plot(f, s_hann, color=BLUE, lw=0.9, label="Hann periodica")
    ax.axvline(2500, color=MUTED, ls=":", lw=0.8)
    ax.set_xlim(0, 4000)
    ax.set_ylim(-110, 5)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("dB")
    ax.set_title(f"Tono de {f_off:.1f} Hz (entre bins) + tono de 2500 Hz a -50 dB, frame de {n} muestras")
    ax.legend(frameon=False)
    ax.grid(color="#E2E1DA", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "fuga_espectral.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    idx = np.argmin(np.abs(f - 2500))
    out = {
        "nivel_en_2500Hz_rect_dB": round(float(s_rect[idx]), 1),
        "nivel_en_2500Hz_hann_dB": round(float(s_hann[idx]), 1),
        "tono_debil_real_dB": -50.0,
        "tono_sobre_bin_Hz": f_on,
        "tono_entre_bins_Hz": f_off,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
