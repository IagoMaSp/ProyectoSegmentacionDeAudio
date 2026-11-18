"""Acondicionamiento de la senal: pasa-altos, notch de red y remuestreo con anti-alias.

Todo el filtrado se representa en secciones de segundo orden (SOS). Una cascada de
biquads es numericamente mucho mas robusta que la funcion de transferencia en forma
directa b/a: con polos cerca de z = 1 (pasa-altos de 70 Hz a 16 kHz) los coeficientes
del polinomio de orden alto pierden precision y el filtro puede volverse inestable.

Hay dos modos de aplicacion con la misma respuesta en modulo:

- offline (archivo completo, entrenamiento e inferencia): `sosfiltfilt`, filtrado
  ida y vuelta. Fase nula, modulo |H|^2.
- en vivo (microfono): la misma cascada aplicada dos veces en forma causal con
  `sosfilt` y estado interno. Modulo |H|^2 identico, fase no nula (retardo de grupo
  doble). Las caracteristicas usan solo el modulo del espectro por frame, por lo que
  ambos modos producen practicamente los mismos vectores.

Los parametros de diseno y su justificacion estan en `src/config.py` y en la seccion 3
de explicacion.md.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sps

from src.config import (
    AA_ATTENUATION_DB,
    AA_TRANSITION_FRACTION,
    HP_CUTOFF_HZ,
    HP_ORDER,
    NOTCH_FREQS_HZ,
    NOTCH_Q,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Diseno
# ---------------------------------------------------------------------------

def design_highpass(fs: float = SAMPLE_RATE, fc: float = HP_CUTOFF_HZ, order: int = HP_ORDER) -> np.ndarray:
    """Butterworth pasa-altos (maximamente plano en la banda de paso).

    Se disena por transformacion bilineal con pre-distorsion de frecuencia (scipy lo
    hace internamente al pasar `fs`), de modo que la frecuencia de -3 dB queda
    exactamente en `fc`.
    """
    return sps.butter(order, fc, btype="highpass", fs=fs, output="sos")


def design_notch(fs: float = SAMPLE_RATE, f0: float = 50.0, q: float = NOTCH_Q) -> np.ndarray:
    """Notch IIR de segundo orden: ceros sobre el circulo unidad en e^{+-j w0}.

    El ancho de banda a -3 dB es BW = f0 / Q. Los polos estan en el mismo angulo con
    radio r < 1, y cuanto mas cerca de 1 mas angosta es la muesca.
    """
    b, a = sps.iirnotch(f0, q, fs=fs)
    return sps.tf2sos(b, a)


def design_conditioning(fs: float = SAMPLE_RATE) -> np.ndarray:
    """Cascada completa de acondicionamiento: pasa-altos seguido de los notch."""
    sections = [design_highpass(fs)]
    for f0 in NOTCH_FREQS_HZ:
        if f0 < fs / 2:
            sections.append(design_notch(fs, f0))
    return np.vstack(sections)


def sos_frequency_response(sos: np.ndarray, fs: float = SAMPLE_RATE, n_points: int = 4096,
                           freqs: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    worN = n_points if freqs is None else freqs
    f, h = sps.sosfreqz(sos, worN=worN, fs=fs)
    return f, h


def sos_group_delay(sos: np.ndarray, fs: float = SAMPLE_RATE, n_points: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Retardo de grupo de la cascada, en segundos (suma del de cada seccion)."""
    total = np.zeros(n_points)
    f = None
    for section in sos:
        f, gd = sps.group_delay((section[:3], section[3:]), w=n_points, fs=fs)
        total += gd
    return f, total / fs


def sos_poles_zeros(sos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zeros, poles = [], []
    for section in sos:
        z, p, _ = sps.tf2zpk(section[:3], section[3:])
        zeros.append(z)
        poles.append(p)
    return np.concatenate(zeros), np.concatenate(poles)


def is_stable(sos: np.ndarray) -> bool:
    _, poles = sos_poles_zeros(sos)
    return bool(np.all(np.abs(poles) < 1.0))


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------

_SOS_CACHE: dict[float, np.ndarray] = {}


def conditioning_sos(fs: float = SAMPLE_RATE) -> np.ndarray:
    if fs not in _SOS_CACHE:
        _SOS_CACHE[fs] = design_conditioning(fs)
    return _SOS_CACHE[fs]


def condition_offline(x: np.ndarray, fs: float = SAMPLE_RATE) -> np.ndarray:
    """Filtrado de fase nula para un archivo completo (entrenamiento e inferencia)."""
    x = np.asarray(x, dtype=np.float64)
    sos = conditioning_sos(fs)
    # sosfiltfilt necesita una senal mas larga que su relleno de bordes
    padlen = min(3 * (2 * sos.shape[0] + 1), max(x.size - 1, 0))
    if x.size < 2:
        return x.copy()
    return sps.sosfiltfilt(sos, x, padlen=padlen)


class CausalConditioner:
    """Version causal para el modo en vivo.

    Aplica la cascada dos veces (ida e ida otra vez, no ida y vuelta), lo que da el
    mismo modulo |H|^2 que `condition_offline`. El estado `zi` se conserva entre
    bloques para que el resultado no dependa de como se trocea el flujo.
    """

    def __init__(self, fs: float = SAMPLE_RATE):
        base = conditioning_sos(fs)
        self.sos = np.vstack([base, base])
        self.zi = np.zeros((self.sos.shape[0], 2))

    def reset(self) -> None:
        self.zi[:] = 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        y, self.zi = sps.sosfilt(self.sos, np.asarray(chunk, dtype=np.float64), zi=self.zi)
        return y


# ---------------------------------------------------------------------------
# Remuestreo
# ---------------------------------------------------------------------------

def rational_factors(fs_in: int, fs_out: int) -> tuple[int, int]:
    frac = Fraction(int(fs_out), int(fs_in))
    return frac.numerator, frac.denominator


def design_antialias(fs_in: int, fs_out: int,
                     attenuation_db: float = AA_ATTENUATION_DB,
                     transition_fraction: float = AA_TRANSITION_FRACTION) -> np.ndarray:
    """FIR pasa-bajos de fase lineal (ventana de Kaiser) para `resample_poly`.

    El filtro trabaja a la tasa intermedia fs_in * L. La banda de rechazo empieza
    exactamente en la nueva frecuencia de Nyquist min(fs_in, fs_out) / 2, y la banda
    de transicion ocupa `transition_fraction` de esa frecuencia por debajo de ella.
    Asi nada por encima de fs_out / 2 sobrevive con mas de -`attenuation_db` dB y no
    puede plegarse dentro de la banda util.
    """
    up, down = rational_factors(fs_in, fs_out)
    fs_up = fs_in * up
    f_nyq = min(fs_in, fs_out) / 2.0
    width = transition_fraction * f_nyq
    numtaps, beta = sps.kaiserord(attenuation_db, width / (fs_up / 2.0))
    numtaps |= 1  # largo impar: retardo entero y fase lineal tipo I
    cutoff = f_nyq - width / 2.0
    return sps.firwin(numtaps, cutoff, window=("kaiser", beta), fs=fs_up)


_AA_CACHE: dict[tuple[int, int], np.ndarray] = {}


def antialias_filter(fs_in: int, fs_out: int) -> np.ndarray:
    key = (int(fs_in), int(fs_out))
    if key not in _AA_CACHE:
        _AA_CACHE[key] = design_antialias(*key)
    return _AA_CACHE[key]


def resample(x: np.ndarray, fs_in: int, fs_out: int = SAMPLE_RATE) -> np.ndarray:
    """Remuestreo racional L/M con filtro anti-alias explicito."""
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    up, down = rational_factors(fs_in, fs_out)
    h = antialias_filter(fs_in, fs_out)
    return sps.resample_poly(np.asarray(x, dtype=np.float64), up, down, window=h)


def naive_resample(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Remuestreo SIN anti-alias (solo para la demo y la evaluacion).

    Si el factor es entero se toma una de cada M muestras; si no, interpolacion lineal.
    Las componentes por encima de fs_out / 2 se pliegan sobre la banda util.
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    x = np.asarray(x, dtype=np.float64)
    if fs_in % fs_out == 0:
        return x[:: fs_in // fs_out].copy()
    n_out = int(np.floor(x.size * fs_out / fs_in))
    t_out = np.arange(n_out) * (fs_in / fs_out)
    return np.interp(t_out, np.arange(x.size), x)


class StreamingResampler:
    """Remuestreo por bloques equivalente a `resample` sobre la senal completa.

    Se guarda un historial de entrada que cubre el soporte del FIR; en cada bloque se
    remuestrea historial + bloque y solo se emiten las muestras de salida cuyo
    soporte quedo completo. El inicio del historial siempre es multiplo de M, asi la
    rejilla de salida no se corre entre bloques.
    """

    def __init__(self, fs_in: int, fs_out: int = SAMPLE_RATE):
        self.fs_in, self.fs_out = int(fs_in), int(fs_out)
        self.up, self.down = rational_factors(self.fs_in, self.fs_out)
        self.passthrough = self.up == self.down == 1
        if not self.passthrough:
            self.h = antialias_filter(self.fs_in, self.fs_out)
            half_len = (self.h.size - 1) // 2
            self.radius = int(np.ceil(half_len / self.up)) + 1  # en muestras de entrada
        self.buf = np.zeros(0)
        self.s0 = 0          # indice global de entrada del primer elemento de buf
        self.emitted = 0     # cantidad de muestras de salida ya entregadas

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float64)
        if self.passthrough:
            return chunk.copy()
        self.buf = np.concatenate([self.buf, chunk])
        if self.buf.size == 0:
            return np.zeros(0)
        y = sps.resample_poly(self.buf, self.up, self.down, window=self.h)
        o0 = self.s0 * self.up // self.down
        last_valid_in = self.buf.size - 1 - self.radius
        j_end = int(np.floor(last_valid_in * self.up / self.down)) + 1 if last_valid_in >= 0 else 0
        j_start = self.emitted - o0
        if j_end <= j_start:
            return np.zeros(0)
        out = y[j_start:j_end]
        self.emitted = o0 + j_end
        # descartar historial que ya no hace falta
        need_from = self.emitted * self.down / self.up - self.radius
        new_s0 = int(np.floor(max(need_from, 0) / self.down)) * self.down
        if new_s0 > self.s0:
            self.buf = self.buf[new_s0 - self.s0:]
            self.s0 = new_s0
        return out

