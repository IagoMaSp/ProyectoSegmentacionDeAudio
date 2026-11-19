"""Figuras del diseno de filtros y de la demo de aliasing.

Uso:  python -m scripts.filter_design
Salida: docs/figuras/acond_*.png y docs/figuras/aliasing_*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from src.config import (
    AA_ATTENUATION_DB,
    FIGURES_DIR,
    HP_CUTOFF_HZ,
    HP_ORDER,
    NOTCH_FREQS_HZ,
    NOTCH_Q,
    SAMPLE_RATE,
)
from src import filters as F

INK = "#171717"
MUTED = "#8A8983"
BLUE = "#2563EB"
ORANGE = "#D97706"
GREEN = "#059669"

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#E2E1DA",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def _db(h: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(h), 1e-12))


def fig_bode() -> None:
    fs = SAMPLE_RATE
    freqs = np.geomspace(5, fs / 2, 6000)
    hp = F.design_highpass(fs)
    notch = F.design_notch(fs, NOTCH_FREQS_HZ[0])
    total = F.design_conditioning(fs)
    _, h_hp = F.sos_frequency_response(hp, fs, freqs=freqs)
    _, h_n = F.sos_frequency_response(notch, fs, freqs=freqs)
    _, h_t = F.sos_frequency_response(total, fs, freqs=freqs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    ax1.semilogx(freqs, _db(h_hp), color=BLUE, lw=1.2, label=f"Pasa-altos Butterworth N={HP_ORDER}, fc={HP_CUTOFF_HZ:.0f} Hz")
    ax1.semilogx(freqs, _db(h_n), color=ORANGE, lw=1.2, label=f"Notch {NOTCH_FREQS_HZ[0]:.0f} Hz, Q={NOTCH_Q:.0f}")
    ax1.semilogx(freqs, _db(h_t), color=INK, lw=1.4, label="Cascada (una pasada)")
    ax1.semilogx(freqs, 2 * _db(h_t), color=INK, lw=1.0, ls="--", label="Cascada con filtfilt, |H|^2")
    ax1.axvline(HP_CUTOFF_HZ, color=MUTED, lw=0.8, ls=":")
    ax1.axhline(-3, color=MUTED, lw=0.8, ls=":")
    ax1.set_ylim(-90, 5)
    ax1.set_ylabel("Modulo [dB]")
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_title("Respuesta en frecuencia del acondicionamiento (fs = 16 kHz)")

    ax2.semilogx(freqs, np.degrees(np.unwrap(np.angle(h_hp))), color=BLUE, lw=1.2, label="Pasa-altos")
    ax2.semilogx(freqs, np.degrees(np.angle(h_n)), color=ORANGE, lw=1.2, label="Notch")
    ax2.semilogx(freqs, np.degrees(np.unwrap(np.angle(h_t))), color=INK, lw=1.4, label="Cascada")
    ax2.axhline(0, color=GREEN, lw=1.0, ls="--", label="filtfilt (fase nula)")
    ax2.set_ylabel("Fase [grados]")
    ax2.set_xlabel("Frecuencia [Hz]")
    ax2.legend(loc="upper right", frameon=False)
    fig.savefig(FIGURES_DIR / "acond_bode.png")
    plt.close(fig)

    # detalle lineal de la muesca
    f_lin = np.linspace(40, 60, 2000)
    _, h_lin = F.sos_frequency_response(notch, fs, freqs=f_lin)
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(f_lin, _db(h_lin), color=ORANGE, lw=1.3, label="una pasada")
    ax.plot(f_lin, 2 * _db(h_lin), color=INK, lw=1.0, ls="--", label="filtfilt")
    ax.axhline(-3, color=MUTED, lw=0.8, ls=":")
    bw = NOTCH_FREQS_HZ[0] / NOTCH_Q
    ax.axvspan(NOTCH_FREQS_HZ[0] - bw / 2, NOTCH_FREQS_HZ[0] + bw / 2, color="#F3F2EC")
    ax.set_ylim(-80, 3)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Modulo [dB]")
    ax.set_title(f"Notch de red: BW = f0/Q = {bw:.1f} Hz")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIGURES_DIR / "acond_notch_detalle.png")
    plt.close(fig)


def _zplane(ax, sos: np.ndarray, title: str, zoom: tuple[float, float, float] | None = None) -> None:
    z, p = F.sos_poles_zeros(sos)
    theta = np.linspace(0, 2 * np.pi, 600)
    ax.plot(np.cos(theta), np.sin(theta), color=MUTED, lw=0.8)
    ax.scatter(z.real, z.imag, s=46, facecolors="none", edgecolors=BLUE, linewidths=1.3, label="ceros")
    ax.scatter(p.real, p.imag, s=46, marker="x", color=ORANGE, linewidths=1.3, label="polos")
    ax.set_aspect("equal")
    ax.set_title(title)
    if zoom:
        cx, cy, r = zoom
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
    else:
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("Re{z}")
    ax.set_ylabel("Im{z}")


def fig_poles_zeros() -> None:
    fs = SAMPLE_RATE
    hp = F.design_highpass(fs)
    notch = F.design_notch(fs, NOTCH_FREQS_HZ[0])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    _zplane(axes[0], np.vstack([hp, notch]), "Plano z completo")
    _zplane(axes[1], hp, "Pasa-altos (zoom cerca de z = 1)", zoom=(0.985, 0.0, 0.03))
    _zplane(axes[2], notch, "Notch 50 Hz (zoom)", zoom=(0.9995, 0.0196, 0.004))
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "acond_polos_ceros.png")
    plt.close(fig)


def fig_group_delay() -> None:
    fs = SAMPLE_RATE
    n = 1 << 16
    f, gd_hp = F.sos_group_delay(F.design_highpass(fs), fs, n)
    _, gd_t = F.sos_group_delay(F.design_conditioning(fs), fs, n)
    mask = f >= 20
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.semilogx(f[mask], 1e3 * gd_hp[mask], color=BLUE, lw=1.2, label="Pasa-altos")
    ax.semilogx(f[mask], 1e3 * gd_t[mask], color=INK, lw=1.2, label="Cascada (una pasada causal)")
    ax.semilogx(f[mask], 2e3 * gd_t[mask], color=INK, lw=0.9, ls="--", label="Modo en vivo (dos pasadas causales)")
    ax.axhline(0, color=GREEN, lw=1.0, label="Offline con filtfilt")
    ax.set_ylim(-5, 60)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Retardo de grupo [ms]")
    ax.set_title("Retardo de grupo: concentrado por debajo de 150 Hz")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(FIGURES_DIR / "acond_retardo_grupo.png")
    plt.close(fig)


def fig_antialias_fir() -> dict:
    info = {}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for fs_in, color in ((48000, INK), (44100, BLUE), (22050, ORANGE)):
        h = F.antialias_filter(fs_in, SAMPLE_RATE)
        up, down = F.rational_factors(fs_in, SAMPLE_RATE)
        f, H = sps.freqz(h, worN=1 << 16, fs=fs_in * up)
        mask = f <= 12000
        ax.plot(f[mask], _db(H[mask]), color=color, lw=1.1, label=f"{fs_in} -> 16000 Hz (L={up}, M={down}, {h.size} coef.)")
        info[str(fs_in)] = {"L": up, "M": down, "coeficientes": int(h.size)}
    ax.axvline(8000, color=MUTED, ls=":", lw=0.8)
    ax.axvline(7200, color=MUTED, ls=":", lw=0.8)
    ax.axhline(-AA_ATTENUATION_DB, color=MUTED, ls=":", lw=0.8)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Modulo [dB]")
    ax.set_title("Filtro anti-alias FIR (Kaiser): paso hasta 7.2 kHz, rechazo de 80 dB desde 8 kHz")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIGURES_DIR / "acond_antialias_fir.png")
    plt.close(fig)
    return info


def aliasing_signals(fs_in: int = 48000, fs_out: int = 8000, seconds: float = 1.0):
    """Tonos de 1 kHz, 5 kHz y 7 kHz muestreados a 48 kHz y llevados a 8 kHz.

    5 kHz y 7 kHz superan la nueva Nyquist (4 kHz): sin anti-alias aparecen en
    |f - k fs_out| = 3 kHz y 1 kHz, donde no existian.
    """
    t = np.arange(int(fs_in * seconds)) / fs_in
    x = (np.sin(2 * np.pi * 1000 * t) + 0.7 * np.sin(2 * np.pi * 5000 * t)
         + 0.5 * np.sin(2 * np.pi * 7000 * t))
    naive = F.naive_resample(x, fs_in, fs_out)
    clean = F.resample(x, fs_in, fs_out)
    return x, naive, clean


def _spectrum_db(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(x.size, 1 / fs)
    mag = np.abs(X) / (w.sum() / 2)
    return f, 20 * np.log10(np.maximum(mag, 1e-9))


def fig_aliasing() -> dict:
    fs_in, fs_out = 48000, 8000
    x, naive, clean = aliasing_signals(fs_in, fs_out)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.4))
    f, s = _spectrum_db(x, fs_in)
    axes[0].plot(f, s, color=INK, lw=0.9)
    axes[0].axvline(fs_out / 2, color=ORANGE, ls="--", lw=0.9)
    axes[0].set_xlim(0, 10000)
    axes[0].set_title("Original a 48 kHz: tonos de 1, 5 y 7 kHz (linea: Nyquist de 8 kHz)")
    f, s = _spectrum_db(naive, fs_out)
    axes[1].plot(f, s, color=ORANGE, lw=0.9)
    axes[1].set_title("Diezmado x6 sin anti-alias: 5 kHz -> 3 kHz y 7 kHz -> 1 kHz (alias)")
    f, s = _spectrum_db(clean, fs_out)
    axes[2].plot(f, s, color=BLUE, lw=0.9)
    axes[2].set_title("resample_poly con FIR anti-alias: solo queda 1 kHz")
    for ax in axes:
        ax.set_ylim(-120, 5)
        ax.set_ylabel("dB")
    for ax in axes[1:]:
        ax.set_xlim(0, fs_out / 2)
    axes[2].set_xlabel("Frecuencia [Hz]")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "aliasing_tonos.png")
    plt.close(fig)

    # medir el nivel del alias en 3 kHz en cada caso
    def level_at(y, f0):
        f, s = _spectrum_db(y, fs_out)
        return float(s[np.argmin(np.abs(f - f0))])

    # barrido: chirp 0-24 kHz, espectrograma tras diezmar a 16 kHz
    t = np.arange(fs_in * 4) / fs_in
    chirp = sps.chirp(t, f0=0, f1=24000, t1=4, method="linear") * 0.5
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, y, title in ((axes[0], F.naive_resample(chirp, fs_in, 16000), "Sin anti-alias"),
                         (axes[1], F.resample(chirp, fs_in, 16000), "Con anti-alias")):
        f, tt, S = sps.spectrogram(y, fs=16000, nperseg=512, noverlap=384, window="hann")
        ax.pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="Greys", vmin=-110, vmax=-20)
        ax.set_title(f"Chirp 0-24 kHz llevado a 16 kHz: {title.lower()}")
        ax.set_xlabel("Tiempo [s]")
    axes[0].set_ylabel("Frecuencia [Hz]")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "aliasing_chirp.png")
    plt.close(fig)

    return {
        "alias_3kHz_sin_antialias_dB": round(level_at(naive, 3000), 1),
        "alias_3kHz_con_antialias_dB": round(level_at(clean, 3000), 1),
        "alias_1kHz_tono_real_dB": round(level_at(clean, 1000), 1),
    }


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_bode()
    fig_poles_zeros()
    fig_group_delay()
    aa = fig_antialias_fir()
    alias = fig_aliasing()
    sos = F.design_conditioning()
    z, p = F.sos_poles_zeros(sos)
    f, h = F.sos_frequency_response(sos, freqs=np.array([20.0, 50.0, 70.0, 100.0, 1000.0]))
    summary = {
        "pasa_altos": {"orden": HP_ORDER, "fc_Hz": HP_CUTOFF_HZ},
        "notch": {"f0_Hz": list(NOTCH_FREQS_HZ), "Q": NOTCH_Q, "BW_Hz": NOTCH_FREQS_HZ[0] / NOTCH_Q},
        "estable": F.is_stable(sos),
        "radio_max_polos": round(float(np.max(np.abs(p))), 6),
        "modulo_una_pasada_dB": {f"{fi:.0f}": round(float(v), 2) for fi, v in zip(f, _db(h))},
        "antialias": aa,
        "aliasing_demo": alias,
    }
    (FIGURES_DIR / "acond_resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figuras en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
